"""Lightweight command parser; model dependencies are imported after dispatch."""
import argparse
import json
from pathlib import Path

from .training.config import add_train_args, resolve_config


def parser():
    root = argparse.ArgumentParser(description='SemICD training and evaluation')
    commands = root.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='full fine-tuning; validation only, no implicit test')
    add_train_args(train)
    evaluate = commands.add_parser('evaluate', help='evaluate using checkpoint tokenizer, mapping and decode config')
    evaluate.add_argument('--checkpoint', type=Path, required=True,
                          help='a checkpoint, or a training run directory (uses its best checkpoint)')
    evaluate.add_argument('--eval-file', type=Path, required=True)
    evaluate.add_argument('--output-dir', type=Path, required=True)
    ontology = commands.add_parser('prepare-ontology', help='official FY2020 ICD-10-CM paths with optional FY2015 fallback')
    ontology.add_argument('--codes-file', type=Path, required=True)
    ontology.add_argument('--primary-dir', type=Path, required=True)
    ontology.add_argument('--supplement-dir', type=Path)
    ontology.add_argument('--output-dir', type=Path, required=True)
    umls = commands.add_parser('prepare-umls', help='extract English terms from a licensed local MRCONSO.RRF')
    umls.add_argument('--codes-file', type=Path, required=True)
    umls.add_argument('--umls-file', type=Path, required=True)
    umls.add_argument('--release', required=True)
    umls.add_argument('--output-dir', type=Path, required=True)
    icd9 = commands.add_parser('prepare-icd9-inputs', help='adapt local ICD-9 Full catalog/path and raw term exports')
    icd9.add_argument('--codes-file', type=Path, required=True)
    icd9.add_argument('--catalog-file', type=Path, required=True)
    icd9.add_argument('--paths-file', type=Path, required=True)
    icd9.add_argument('--umls-terms-file', type=Path, required=True)
    icd9.add_argument('--source-contract', type=Path, required=True)
    icd9.add_argument('--output-dir', type=Path, required=True)
    profiles = commands.add_parser('prepare-profiles', help='construct six-field profiles from authoritative path and UMLS exports')
    profiles.add_argument('--ontology-file', type=Path, required=True)
    profiles.add_argument('--umls-terms-file', type=Path, required=True)
    profiles.add_argument('--codes-file', type=Path, required=True)
    profiles.add_argument('--output-dir', type=Path, required=True)
    profiles.set_defaults(code_system=None, edition=None)
    dataset = commands.add_parser('prepare-dataset', help='convert Edin tables; ICD-9 Full uses diagnosis and procedure columns')
    dataset.add_argument('--dataset', choices=['mimiciii', 'mimiciv'], help='inferred from the diagnosis column')
    dataset.add_argument('--label-space', choices=['diagnosis', 'mixed'], help='default mixed for MIMIC-III, diagnosis for MIMIC-IV')
    dataset.add_argument('--upstream-file', type=Path, required=True)
    dataset.add_argument('--splits-file', type=Path, required=True)
    dataset.add_argument('--output-dir', type=Path, required=True)
    codebook = commands.add_parser('build-codebook', help='embeddings -> HKM/RK-Means/RQ-VAE -> UID -> bundle; or matched representation controls')
    codebook.add_argument('--representation', choices=['semantic', 'random', 'atomic', 'raw'], default='semantic')
    codebook.add_argument('--method', choices=['hkm', 'rkmeans', 'rqvae'], default='hkm')
    codebook.add_argument('--code-system', choices=['ICD-9', 'ICD-10-CM'], help='only for a standalone codes file; otherwise inferred')
    codebook.add_argument('--profiles-file', type=Path)
    codebook.add_argument('--embedding-dir', type=Path)
    codebook.add_argument('--embedding-model-name-or-path', default='Qwen/Qwen3-Embedding-4B')
    codebook.add_argument('--embedding-batch-size', type=int, default=8)
    codebook.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    codebook.add_argument('--sid-seed', type=int, default=0)
    codebook.add_argument('--n-init', type=int, default=10)
    codebook.add_argument('--max-iter', type=int, default=300)
    codebook.add_argument('--rqvae-config', type=Path, help='RQ-VAE model/training JSON; defaults to the paper K128/L3 recipe')
    codebook.add_argument('--rqvae-checkpoint', type=Path, help='export assignments from a trained RQ-VAE checkpoint instead of fitting')
    codebook.add_argument('--source-bundle', type=Path)
    codebook.add_argument('--assignment-seed', type=int, default=42)
    codebook.add_argument('--codes-file', type=Path)
    codebook.add_argument('--output-dir', type=Path, required=True)
    codebook.set_defaults(edition=None, rqvae_export='usm', raw_prompt_file=None)
    training = commands.add_parser('prepare-training', help='add both profile tasks using the selected bundle representation')
    training.add_argument('--train-file', type=Path, required=True)
    training.add_argument('--profiles-file', type=Path, required=True)
    training.add_argument('--sid-bundle', type=Path, required=True)
    training.add_argument('--output-dir', type=Path, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    # Each step imports its own dependencies, so `--help` and data preparation stay lightweight.
    if args.command == 'prepare-ontology':
        from .data.ontology import prepare_ontology
        print(json.dumps(prepare_ontology(args), indent=2))
    elif args.command == 'prepare-umls':
        from .data.umls import prepare_umls
        print(json.dumps(prepare_umls(args), indent=2))
    elif args.command == 'prepare-icd9-inputs':
        from .data.icd9_inputs import prepare_icd9_inputs
        print(json.dumps(prepare_icd9_inputs(args), indent=2))
    elif args.command == 'prepare-dataset':
        from .data.dataset import prepare_dataset
        print(json.dumps(prepare_dataset(args), indent=2))
    elif args.command == 'prepare-profiles':
        from .data.profiles import prepare_profiles
        print(json.dumps(prepare_profiles(args), indent=2))
    elif args.command == 'build-codebook':
        from .codebook.build import build_codebook
        print(json.dumps(build_codebook(args), indent=2))
    elif args.command == 'prepare-training':
        from .training.tasks import prepare_training
        print(json.dumps(prepare_training(args), indent=2))
    elif args.command == 'train':
        from .training.train import train
        train(resolve_config(args), args.resume_from_checkpoint)
    elif args.command == 'evaluate':
        from .training.evaluate import evaluate_checkpoint
        evaluate_checkpoint(args.checkpoint, args.eval_file, args.output_dir)
