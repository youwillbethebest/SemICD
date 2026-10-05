"""Step 5: multi-task full fine-tuning with validation-based checkpoint selection."""
import importlib.metadata
import json
import math
import random
from pathlib import Path

import torch

from ..codebook.bundle import read_bundle
from ..utils import read_rows, seed_all, write_json
from .checkpoint import checkpoint_codec, restore_state, save_checkpoint
from .codec import register_bundle
from .evaluate import evaluate_rows
from .model import encode_rows, load_unsloth
from .tasks import validate_training_dataset


def windows(rows, config, epoch):
    count = math.ceil(len(rows) / (config.batch_size * config.gradient_accumulation_steps))
    groups = {task: [] for task in config.task_weights}
    for row in rows:
        groups[row['task']].append(row)
    if any(len(group) < count for group in groups.values()):
        raise ValueError('Increase batch/accumulation: every update needs each configured task')
    rng = random.Random(config.seed + epoch)
    for group in groups.values():
        rng.shuffle(group)
    return [{task: group[len(group) * i // count:len(group) * (i + 1) // count]
             for task, group in groups.items()} for i in range(count)]


def collate(rows, pad_id, device):
    length = max(len(r['prompt']) + len(r['response']) for r in rows)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(rows):
        start = len(row['prompt'])
        seq = row['prompt'] + row['response']
        ids[i, :len(seq)] = torch.tensor(seq, device=device)
        mask[i, :len(seq)] = 1
        labels[i, start:len(seq)] = ids[i, start:len(seq)]
    return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}


def make_optimizer(model, config, total_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 betas=(config.adam_beta1, config.adam_beta2),
                                 eps=config.adam_epsilon, weight_decay=config.weight_decay)
    warmup = math.ceil(total_steps * config.warmup_ratio)
    def scale(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = min(1., (step - warmup) / max(1, total_steps - warmup))
        if config.lr_scheduler_type == 'constant':
            return 1.
        if config.lr_scheduler_type == 'linear':
            return 1. - progress
        return .5 * (1 + math.cos(math.pi * progress))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train_epoch(model, tokenizer, rows, config, optimizer, scheduler, scaler, epoch):
    model.train()
    losses, sizes = [], []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    device = model.get_input_embeddings().weight.device
    use_fp16 = config.dtype == 'fp16' and device.type == 'cuda'
    for window in windows(rows, config, epoch):
        optimizer.zero_grad(set_to_none=True)
        value = 0.
        sizes.append(sum(len(group) for group in window.values()))
        for task, group in window.items():
            tokens = sum(len(row['response']) for row in group)
            for start in range(0, len(group), config.batch_size):
                batch = group[start:start + config.batch_size]
                weight = config.task_weights[task] * sum(len(row['response']) for row in batch) / tokens
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                    output = model(**collate(batch, tokenizer.pad_token_id, device), use_cache=False, return_dict=True)
                    loss = output.loss * weight
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                scaler.scale(loss).backward()
                value += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(value)
    return {'epoch': epoch, 'mean_update_loss': sum(losses) / len(losses), 'updates': len(losses),
            'actual_examples_per_update': sizes, 'objective': 'weighted_task_response_token_mean',
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None}


def train(config, resume=None, loader=load_unsloth):
    seed_all(config.seed)
    root = Path(config.output_dir)
    if root.exists() and any(root.iterdir()) and resume is None:
        raise ValueError('Use an empty output directory or explicitly resume')
    root.mkdir(parents=True, exist_ok=True)
    bundle = json.loads((Path(resume) / 'sid_bundle.json').read_text()) if resume else read_bundle(config.sid_bundle)
    if bundle['manifest'].get('representation', 'semantic') != config.representation:
        raise ValueError('Config representation differs from the selected bundle')
    train_rows, val_rows = read_rows(config.train_file), read_rows(config.validation_file)
    dataset_report = validate_training_dataset(config, bundle, train_rows, val_rows)
    model, tokenizer = loader(config, resume, training=True)
    codec = checkpoint_codec(resume, tokenizer) if resume else register_bundle(tokenizer, model, bundle)
    if codec.bundle != bundle:
        raise ValueError('Resume must use the checkpoint SID bundle')
    rows = encode_rows(train_rows, tokenizer, codec, config)
    steps = len(windows(rows, config, 1)) * config.epochs
    optimizer, scheduler = make_optimizer(model, config, steps)
    scaler = torch.amp.GradScaler('cuda', enabled=config.dtype == 'fp16' and torch.cuda.is_available())
    progress = {'epoch': 0, 'step': 0, 'best_micro_f1': -1., 'best_epoch': None, 'bad_epochs': 0, 'stopped': False}
    if resume:
        progress = restore_state(resume, config, bundle, optimizer, scheduler, scaler)
    if progress.get('stopped'):
        raise ValueError('Checkpoint already reached early stopping')
    write_json(root / 'resolved_config.json', config.to_dict())
    versions = {}
    for package in ('torch', 'transformers', 'unsloth', 'numpy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(root / 'runtime.json', {'packages': versions, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                                     'total_parameters': sum(p.numel() for p in model.parameters()),
                                     'device': str(model.get_input_embeddings().weight.device), 'resume': str(resume) if resume else None,
                                     'test_used': False, 'dataset_test_preparation_used': bool(dataset_report and dataset_report.get('test_used')),
                                     'note_truncation': 'head_prefix_character_bisection' if config.truncate_notes else None,
                                     'truncated_train_notes': sum(row['note_truncated'] for row in rows),
                                     'epoch_boundary_resume': True})
    for epoch in range(progress['epoch'] + 1, config.epochs + 1):
        if (root / f'epoch_{epoch:02}').exists():
            raise ValueError('A later checkpoint exists; use a new output directory to fork a resume')
        report = train_epoch(model, tokenizer, rows, config, optimizer, scheduler, scaler, epoch)
        write_json(root / f'train_epoch{epoch}.json', report)
        metrics = evaluate_rows(model, tokenizer, codec, val_rows, config, root / f'validation_epoch{epoch}')
        improved = metrics['micro_f1'] > progress['best_micro_f1']
        progress.update(epoch=epoch, step=progress['step'] + report['updates'],
                        bad_epochs=0 if improved else progress['bad_epochs'] + 1)
        if improved:
            progress.update(best_micro_f1=metrics['micro_f1'], best_epoch=epoch,
                            best_checkpoint=str((root / f'epoch_{epoch:02}').resolve()))
        progress['stopped'] = bool(config.early_stopping_patience and progress['bad_epochs'] >= config.early_stopping_patience)
        save_checkpoint(root / f'epoch_{epoch:02}', model, tokenizer, config, bundle, optimizer, scheduler, scaler, progress.copy())
        write_json(root / 'selection.json', {**progress, 'metric': 'validation/micro_f1', 'tie_break': 'earliest',
                                           'test_used': False, 'selected_checkpoint': progress['best_checkpoint']})
        if progress['stopped']:
            break
    return progress
