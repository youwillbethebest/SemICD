"""Verify the saved outputs of the synthetic CLI walkthrough."""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from semicd.training.checkpoint import checkpoint_codec
from semicd.training.config import TrainConfig
from semicd.utils import digest, read_rows, write_json


def check(root):
    selection = json.loads((root / 'run/selection.json').read_text())
    checkpoint = Path(selection['selected_checkpoint'])
    config = TrainConfig(**json.loads((checkpoint / 'train_config.json').read_text())).validate()
    state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
    for key, expected in state['contract']['input_sha256'].items():
        assert digest(getattr(config, key)) == expected, f'{key} changed after checkpoint save'
    assert config.epochs == 2 and config.max_seq_length == 256 and config.max_new_tokens == 12
    assert config.gradient_accumulation_steps == 8 and config.learning_rate == 0.01
    assert not config.gradient_checkpointing
    assert selection['epoch'] == 2 and not selection['test_used']
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    codec = checkpoint_codec(checkpoint, tokenizer)
    for code in codec.forward:
        assert codec.decode(codec.encode([code])) == ([code], True, True)
    saved = json.loads((root / f'run/validation_epoch{selection["best_epoch"]}/metrics.json').read_text())
    reloaded = json.loads((root / 'reloaded-validation/metrics.json').read_text())
    assert saved == reloaded, 'CLI reload changed validation metrics'
    predictions = [json.loads(line) for line in (root / 'reloaded-validation/predictions.jsonl').read_text().splitlines()]
    gold = read_rows(config.validation_file)
    assert len(predictions) == len(gold) == reloaded['rows'] == 2
    assert [row['gold_codes'] for row in predictions] == [row['codes'] for row in gold]
    access = json.loads((root / 'reloaded-validation/access.json').read_text())
    assert access['status'] == 'completed' and not access['test_used']
    write_json(root / 'verification.json', {'passed': True, 'epochs': 2,
        'selected_epoch': selection['best_epoch'], 'reloaded_metrics_identical': True,
        'checkpoint_input_hashes_verified': True, 'checkpoint_token_ids_verified': True,
        'prediction_rows': len(predictions), 'test_used': False,
        'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'cuda_version': torch.version.cuda})
    print('Synthetic CLI checkpoint and evaluation verification passed')


if __name__ == '__main__':
    check(Path(sys.argv[1]).resolve())
