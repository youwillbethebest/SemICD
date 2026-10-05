"""Training configuration: defaults, CLI flags and inference of omitted paths."""
import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class TrainConfig:
    model_name_or_path: str = ''
    train_file: str = ''
    validation_file: str = ''
    sid_bundle: str = ''
    representation: str = 'semantic'
    dataset_manifest: str = ''
    output_dir: str = ''
    dtype: str = 'bf16'
    max_seq_length: int = 8192
    epochs: int = 6
    batch_size: int = 1
    gradient_accumulation_steps: int = 0  # 0: 16 for three-task training, 1 for Raw ICD
    learning_rate: float = 3e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    seed: int = 42
    num_beams: int = 4
    max_new_tokens: int = 384
    eval_batch_size: int = 1
    length_penalty: float = 1.0
    lr_scheduler_type: str = 'cosine'
    gradient_checkpointing: bool = True
    truncate_notes: bool = True
    task_weights: dict = None
    early_stopping_patience: int = 2

    def validate(self):
        if self.representation not in ('semantic', 'random', 'atomic', 'raw'):
            raise ValueError('Unknown representation')
        if not self.sid_bundle:
            raise ValueError('--sid-bundle is required unless --train-file comes from prepare-training')
        for key in ('model_name_or_path', 'train_file', 'validation_file', 'output_dir'):
            if not getattr(self, key):
                raise ValueError(f'{key} is required')
        if self.gradient_accumulation_steps == 0:
            self.gradient_accumulation_steps = 1 if self.representation == 'raw' else 16
        for key in ('epochs', 'batch_size', 'gradient_accumulation_steps', 'max_seq_length', 'num_beams', 'max_new_tokens', 'eval_batch_size'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        if self.dtype not in ('bf16', 'fp16', 'fp32'):
            raise ValueError('dtype must be bf16, fp16 or fp32')
        if self.lr_scheduler_type not in ('cosine', 'linear', 'constant'):
            raise ValueError('Unknown learning-rate scheduler')
        if not math.isfinite(self.length_penalty):
            raise ValueError('length_penalty must be finite')
        if self.max_new_tokens >= self.max_seq_length:
            raise ValueError('max_new_tokens must leave room for a prompt')
        for key in ('learning_rate', 'adam_epsilon', 'max_grad_norm'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not 0 <= self.warmup_ratio < 1 or not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError('invalid warmup/beta')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0 or self.early_stopping_patience < 0:
            raise ValueError('invalid weight decay or patience')
        if self.task_weights is None:
            self.task_weights = {'note_to_sid': 1.0} if self.representation == 'raw' else {'note_to_sid': 0.8, 'profile_to_sid': 0.1, 'sid_to_profile': 0.1}
        if self.representation == 'raw' and self.task_weights != {'note_to_sid': 1.0}:
            raise ValueError('Raw baseline uses note-only supervision')
        if not self.task_weights or any(not math.isfinite(w) or w <= 0 for w in self.task_weights.values()):
            raise ValueError('task weights must be finite and positive')
        if not math.isclose(sum(self.task_weights.values()), 1.0):
            raise ValueError('task weights must sum to one')
        return self

    def to_dict(self):
        return asdict(self)


# Flags shown in --help; every other TrainConfig field is still accepted (or set via --config).
PUBLIC = {'model_name_or_path', 'train_file', 'validation_file', 'sid_bundle', 'output_dir', 'epochs',
          'learning_rate', 'gradient_accumulation_steps', 'max_seq_length', 'max_new_tokens',
          'gradient_checkpointing', 'truncate_notes', 'num_beams', 'seed'}


def add_train_args(parser):
    parser.add_argument('--config', type=Path, help='JSON file with any TrainConfig fields')
    parser.add_argument('--resume-from-checkpoint', type=Path)
    for field in fields(TrainConfig):
        name = '--' + field.name.replace('_', '-')
        shown = None if field.name in PUBLIC else argparse.SUPPRESS
        if field.type is bool:
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help=shown)
        else:
            parser.add_argument(name, type=json.loads if field.type is dict else field.type,
                                default=argparse.SUPPRESS, help=shown)


def resolve_config(args):
    values = vars(args).copy()
    config = values.pop('config', None)
    if config is None and values.get('resume_from_checkpoint'):
        config = values['resume_from_checkpoint'] / 'train_config.json'
    allowed = {f.name for f in fields(TrainConfig)}
    raw = json.loads(config.read_text()) if config else {}
    if set(raw) - allowed:
        raise ValueError(f'Unknown config keys: {sorted(set(raw) - allowed)}')
    path_keys = {'train_file', 'validation_file', 'sid_bundle', 'output_dir', 'dataset_manifest'}
    if config:
        for key in path_keys:
            if raw.get(key):
                raw[key] = str((config.parent / raw[key]).resolve())
        # Hub IDs remain strings. Local model paths are explicitly ./, ../ or absolute.
        model = raw.get('model_name_or_path', '')
        if model.startswith(('./', '../', '/')):
            raw['model_name_or_path'] = str((config.parent / model).resolve())
    raw.update({k: v for k, v in values.items() if k in allowed})
    # prepare-training records its SID bundle in the manifest beside the training file.
    training_manifest = Path(raw.get('train_file') or '.').parent / 'manifest.json'
    if not raw.get('sid_bundle') and raw.get('train_file') and training_manifest.is_file():
        raw['sid_bundle'] = json.loads(training_manifest.read_text()).get('bundle_source', '')
    # The representation is fixed by the bundle; the dataset manifest sits beside the validation split.
    bundle_manifest = Path(raw.get('sid_bundle') or '.') / 'manifest.json'
    if 'representation' not in raw and bundle_manifest.is_file():
        raw['representation'] = json.loads(bundle_manifest.read_text()).get('representation', 'semantic')
    dataset_manifest = Path(raw.get('validation_file') or '.').parent / 'manifest.json'
    if not raw.get('dataset_manifest') and raw.get('validation_file') and dataset_manifest.is_file():
        raw['dataset_manifest'] = str(dataset_manifest)
    for key in path_keys:
        if raw.get(key):
            raw[key] = str(Path(raw[key]).resolve())
    return TrainConfig(**raw).validate()
