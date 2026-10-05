"""CPU check of the real training core, checkpoint round trip and tiny overfit.

Run through SLURM with an output directory argument. The production GPU check
uses the public CLI and Unsloth loader separately.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from semicd.codebook.bundle import read_bundle
from semicd.training.checkpoint import checkpoint_codec
from semicd.training.codec import register_bundle
from semicd.training.config import TrainConfig
from semicd.training.evaluate import evaluate_rows
from semicd.training.model import encode_rows
from semicd.training.train import make_optimizer, train, train_epoch
from semicd.utils import read_rows, seed_all, write_json


def cpu_loader(config, checkpoint=None, training=True):
    source = str(checkpoint or config.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(source)
    return model, tokenizer


def main(output):
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    config = TrainConfig(model_name_or_path=str(root / 'artifacts/tiny-model'),
        train_file=str(output / 'tasks/train.jsonl'),
        validation_file=str(root / 'examples/synthetic/validation.jsonl'),
        sid_bundle=str(root / 'examples/synthetic/bundle'), output_dir=str(output / 'cpu-run'),
        dtype='fp32', max_seq_length=256, max_new_tokens=12,
        gradient_accumulation_steps=8, epochs=2, learning_rate=0.01,
        gradient_checkpointing=False).validate()
    progress = train(config, loader=cpu_loader)
    checkpoint = Path(progress['best_checkpoint'])
    model, tokenizer = cpu_loader(config, checkpoint, training=False)
    codec = checkpoint_codec(checkpoint, tokenizer)
    metrics = evaluate_rows(model, tokenizer, codec, read_rows(config.validation_file), config, output / 'cpu-reloaded')
    saved = json.loads((output / f'cpu-run/validation_epoch{progress["best_epoch"]}/metrics.json').read_text())
    assert metrics == saved, 'Reloaded checkpoint changed validation results'
    seed_all(config.seed)
    model, tokenizer = cpu_loader(config)
    codec = register_bundle(tokenizer, model, read_bundle(config.sid_bundle))
    # Same three-task production objective; one fixed window, at most 100 updates.
    config.gradient_accumulation_steps = 32
    rows = encode_rows(read_rows(config.train_file), tokenizer, codec, config)
    optimizer, scheduler = make_optimizer(model, config, 100)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    losses = []
    for update in range(1, 101):
        report = train_epoch(model, tokenizer, rows, config, optimizer, scheduler, scaler, update)
        losses.append(report['mean_update_loss'])
        if len(losses) >= 2 and losses[-1] < losses[0] / 2:
            break
    assert losses[-1] < losses[0] / 2, 'Tiny overfit did not halve the loss'
    write_json(output / 'cpu-smoke.json', {'checkpoint_reload_metrics_identical': True,
        'validation_rows': metrics['rows'], 'overfit_updates': len(losses),
        'initial_loss': losses[0], 'final_loss': losses[-1], 'overfit_losses': losses,
        'loader': 'Transformers CPU, production train/train_epoch core', 'test_used': False})
    print(json.dumps({'cpu_smoke': 'passed', 'overfit_initial': losses[0], 'overfit_final': losses[-1]}))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
