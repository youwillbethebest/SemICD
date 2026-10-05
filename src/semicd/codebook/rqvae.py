"""RQ-VAE quantizer (LC-Rec model, K=128 / L=3): train it, then export SID paths.

The vendored model in third_party/lcrec_rqvae is unchanged.
"""
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..third_party.lcrec_rqvae.rqvae import RQVAE

MAIN_COMMIT = 'e3ffc2e5b2a7721154bddafa61824e72981155f0'


def default_config(in_dim):
    # main configs/qwen_rqvae_uid_seed0.json, with the K128 amendment.
    return {'model': dict(in_dim=in_dim, num_emb_list=[128]*3, e_dim=32,
        layers=[2048, 1024, 512, 256, 128, 64], dropout_prob=0.0, bn=True,
        loss_type='mse', quant_loss_weight=1.0, beta=0.25, kmeans_init=True,
        kmeans_iters=100, sk_epsilons=[0.003, 0.0, 0.0], sk_iters=50),
        'epochs': 10000, 'batch_size': 1024, 'lr': 0.001, 'weight_decay': 0.0001,
        'betas': [0.9, 0.999], 'eps': 1e-8, 'warmup_epochs': 50,
        'eval_step': 50, 'clip_norm': 1.0, 'export_batch_size': 64,
        'usm_max_rounds': 20, 'export_usm_last_epsilon': 0.003}


def collision_summary(tokens):
    _, counts = np.unique(tokens, axis=0, return_counts=True)
    return dict(unique_sids=len(counts), collision_rate=1-len(counts)/len(tokens),
                affected_codes=int(counts[counts > 1].sum()),
                collision_groups=int((counts > 1).sum()), max_group_size=int(counts.max()))


def collision_groups(tokens):
    groups = {}
    for i, token in enumerate(tokens):
        groups.setdefault(tuple(token), []).append(i)
    return [rows for rows in groups.values() if len(rows) > 1]


@torch.no_grad()
def export_arrays(model, data, batch_size=64, max_rounds=20):
    """Main fixed_prefix_v2: native NN paths plus last-layer collision repair."""
    model.eval()
    tokens, last_inputs = [], []
    for batch in data.split(batch_size):
        residual = model.encoder(batch)
        indices = []
        for level, quantizer in enumerate(model.rq.vq_layers):
            if level == len(model.rq.vq_layers)-1:
                last_inputs.append(residual.clone())
            quantized, _, chosen = quantizer(residual, use_sk=False)
            residual = residual-quantized
            indices.append(chosen)
        tokens.append(torch.stack(indices, dim=-1))
    raw = torch.cat(tokens).cpu().numpy()
    residuals = torch.cat(last_inputs)
    final = raw.copy()
    last = model.rq.vq_layers[-1]
    rounds = []
    for index in range(max_rounds):
        groups = collision_groups(final)
        if not groups:
            break
        changes = []
        for rows in groups:
            _, _, chosen = last(residuals[rows], use_sk=True)
            updated = final[rows].copy()
            updated[:, -1] = chosen.cpu().numpy()
            for row, before, after in zip(rows, final[rows], updated):
                if not np.array_equal(before, after):
                    changes.append(dict(row=row, before=before.tolist(), after=after.tolist()))
            final[rows] = updated
        rounds.append(dict(round=index+1, groups_processed=groups,
                           changes=changes, **collision_summary(final)))
    if not np.array_equal(raw[:, :-1], final[:, :-1]):
        raise ValueError('USM changed an earlier level')
    return raw, final, rounds


@torch.no_grad()
def native_paths(model, data, batch_size):
    model.eval()
    return torch.cat([model.get_indices(batch, use_sk=False)
                      for batch in data.split(batch_size)]).cpu().numpy()


def load_checkpoint(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('arm', 'baseline') != 'baseline':
        raise ValueError('RQ-VAE construction requires the LC-Rec baseline model')
    model = RQVAE(**saved['config']['model'])
    model.load_state_dict(saved['state_dict'])
    # Upstream initialization flags are ordinary attributes, absent from state_dict.
    for quantizer in model.rq.vq_layers:
        quantizer.initted = True
    return model.to(device).eval(), saved


def train_quantizer(vectors, config, seed, device, output_dir):
    from transformers import get_linear_schedule_with_warmup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    loader = DataLoader(torch.from_numpy(vectors), batch_size=config['batch_size'],
        shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    model = RQVAE(**config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
        weight_decay=config['weight_decay'], betas=tuple(config['betas']), eps=config['eps'])
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=config['warmup_epochs']*len(loader),
        num_training_steps=config['epochs']*len(loader))
    data = torch.from_numpy(vectors).to(device)
    best, selected_epoch = float('inf'), None
    checkpoint = output_dir/'best_collision_model.pth'
    with (output_dir/'training.jsonl').open('w', buffering=1) as log:
        for epoch in range(1, config['epochs']+1):
            model.train()
            totals = torch.zeros(2, device=device)
            for batch in loader:
                optimizer.zero_grad()
                batch = batch.to(device)
                out, quant_loss, _ = model(batch)
                loss, recon = model.compute_loss(out, quant_loss, xs=batch)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite RQ-VAE training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                totals += torch.stack([loss.detach(), recon.detach()])
            row = dict(epoch=epoch, train_loss_sum=float(totals[0]), recon_sum=float(totals[1]))
            if epoch % config['eval_step'] == 0:
                metrics = collision_summary(native_paths(model, data, config['export_batch_size']))
                if metrics['collision_rate'] < best:
                    best, selected_epoch = metrics['collision_rate'], epoch
                    torch.save(dict(config=config, seed=seed, epoch=epoch, arm='baseline',
                        metrics=metrics, state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), checkpoint)
                row.update(metrics, selected_epoch=selected_epoch, best_collision=best)
            log.write(json.dumps(row)+'\n')
    return checkpoint


def construct_rqvae(vectors, codes, args):
    config = json.loads(args.rqvae_config.read_text()) if args.rqvae_config else default_config(vectors.shape[1])
    if args.rqvae_checkpoint:
        model, saved = load_checkpoint(args.rqvae_checkpoint, args.device)
        if args.rqvae_config and saved['config']['model'] != config['model']:
            raise ValueError('RQ-VAE checkpoint/config model mismatch')
        config = saved['config']
    if config['model']['in_dim'] != vectors.shape[1] or config['model']['num_emb_list'] != [128]*3:
        raise ValueError('RQ-VAE requires matching embedding dimensions and K128/L3')
    if not args.rqvae_checkpoint:
        if config['epochs'] < config['eval_step'] or min(config['eval_step'], config['batch_size']) < 1:
            raise ValueError('RQ-VAE needs positive batch/evaluation settings and at least one checkpoint evaluation')
        if config['model']['bn'] and (len(codes) < 2 or len(codes) % config['batch_size'] == 1 or config['batch_size'] == 1):
            raise ValueError('BatchNorm requires at least two codes per batch; adjust rqvae-config batch_size')
        if config['model']['kmeans_init'] and min(len(codes), config['batch_size']) < 128:
            raise ValueError('K-means initialization requires at least 128 codes in the first batch')
    folder = args.output_dir.with_name(args.output_dir.name+'-rqvae')
    folder.mkdir(parents=True, exist_ok=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (folder/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    checkpoint = args.rqvae_checkpoint or train_quantizer(vectors, config, args.sid_seed, args.device, folder)
    model, saved = load_checkpoint(checkpoint, args.device)
    last = model.rq.vq_layers[-1]
    last.sk_epsilon = config.get('export_usm_last_epsilon', 0.003)
    native, final, rounds = export_arrays(model, torch.from_numpy(vectors).to(args.device),
        config['export_batch_size'], config['usm_max_rounds'])
    for name, paths in (('native', native), ('usm', final)):
        (folder/f'{name}_assignments.json').write_text(json.dumps(
            {code: path.tolist() for code, path in zip(codes, paths)}, indent=2)+'\n')
    (folder/'usm_rounds.json').write_text(json.dumps(dict(code_order=codes, rounds=rounds), indent=2)+'\n')
    report = dict(config=config, main_commit=MAIN_COMMIT, selected_epoch=saved['epoch'],
        seed=saved['seed'], checkpoint=str(checkpoint.resolve()), export=args.rqvae_export,
        exporter='fixed_prefix_v2', native=collision_summary(native), usm=collision_summary(final),
        native_assignments=str((folder/'native_assignments.json').resolve()),
        usm_assignments=str((folder/'usm_assignments.json').resolve()),
        test_used=False, patient_splits_accessed=[])
    (folder/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return native if args.rqvae_export == 'nn' else final, report
