"""Epoch checkpoints: save, verify the SID token mapping, and resume."""
import json
import random
from pathlib import Path

import numpy as np
import torch

from ..utils import digest, write_json
from .codec import make_codec


def contract(config, bundle):
    values = config.to_dict()
    values.pop('output_dir')
    # Old strict checkpoints predate this optional preprocessing flag.
    if not values['truncate_notes']:
        values.pop('truncate_notes')
    return {'config': values, 'bundle': bundle, 'input_sha256':
            {key: digest(getattr(config, key)) for key in ('train_file', 'validation_file')}}


def save_checkpoint(path, model, tokenizer, config, bundle, optimizer, scheduler, scaler, progress):
    path = Path(path)
    temp = path.with_name(path.name + '.incomplete')
    temp.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(temp, safe_serialization=True)
    tokenizer.save_pretrained(temp)
    write_json(temp / 'train_config.json', config.to_dict())
    write_json(temp / 'sid_bundle.json', bundle)
    codec = make_codec(tokenizer, bundle)
    write_json(temp / 'token_ids.json', {code: list(ids) for code, ids in codec.forward.items()})
    write_json(temp / 'progress.json', progress)
    state = {'contract': contract(config, bundle), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'progress': progress,
             'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
             'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    torch.save(state, temp / 'training_state.pt')
    write_json(temp / 'complete.json', {'complete': True, 'resume_boundary': 'epoch'})
    temp.rename(path)


def checkpoint_codec(path, tokenizer):
    path = Path(path)
    if not (path / 'complete.json').is_file():
        raise ValueError('Incomplete checkpoint')
    bundle = json.loads((path / 'sid_bundle.json').read_text())
    codec = make_codec(tokenizer, bundle)
    actual = {code: list(ids) for code, ids in codec.forward.items()}
    if actual != json.loads((path / 'token_ids.json').read_text()):
        raise ValueError('Checkpoint tokenizer/mapping token IDs changed')
    return codec


def restore_state(path, config, bundle, optimizer, scheduler, scaler):
    # Full optimizer/RNG checkpoints are trusted local artifacts, not arbitrary downloads.
    state = torch.load(Path(path) / 'training_state.pt', map_location='cpu', weights_only=False)
    if state['contract'] != contract(config, bundle):
        raise ValueError('Resume contract mismatch: config, mapping or input contents changed')
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        if not torch.cuda.is_available() or len(state['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA RNG device count mismatch')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['progress']
