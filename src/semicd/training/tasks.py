"""Step 4: build the note->SID, profile->SID and SID->profile training tasks."""
import json
from pathlib import Path

from ..data.dataset import PAPER_STATISTICS, ICD9_FULL_STATISTICS, ICD9_FULL_SPLITS, ICD9_FULL_TRAIN_CODES


def prepare_training(args):
    """Materialize both auxiliary tasks using the *selected* representation."""
    from ..codebook.bundle import read_bundle
    from .codec import representation_text
    bundle = read_bundle(args.sid_bundle)
    profiles = [json.loads(line) for line in args.profiles_file.read_text().splitlines() if line.strip()]
    by_code = {r['code']: r['profile'] for r in profiles}
    if len(by_code) != len(profiles) or set(by_code) != set(bundle['assignments']):
        raise ValueError('Profiles must match the train-only bundle catalog exactly')
    rows = [json.loads(line) for line in args.train_file.read_text().splitlines() if line.strip()]
    if not rows or any(set(row['codes']) - set(by_code) for row in rows):
        raise ValueError('Training notes contain unknown codes or are empty')
    if any(row.get('task', 'note_to_sid') != 'note_to_sid' for row in rows):
        raise ValueError('Supply clinical training rows before auxiliary construction')
    if bundle['manifest'].get('representation', 'semantic') != 'raw':
        for code in sorted(by_code):
            rows.append({'task': 'profile_to_sid', 'prompt': by_code[code], 'codes': [code]})
            rows.append({'task': 'sid_to_profile', 'prompt': representation_text(bundle, code), 'response': by_code[code]})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'train.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = {'representation': bundle['manifest'].get('representation', 'semantic'),
              'rows': len(rows), 'profiles': len(by_code), 'test_used': False,
              'train_source': str(args.train_file.resolve()), 'profiles_source': str(args.profiles_file.resolve()),
              'bundle_source': str(args.sid_bundle.resolve())}
    (args.output_dir / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def validate_training_dataset(config, bundle, train_rows, validation_rows):
    """Require the Table 1 preparation gate for real ICD paper training."""
    system = bundle['manifest']['code_system']
    if system == 'synthetic':
        return None
    if not config.dataset_manifest:
        raise ValueError('Paper training requires --dataset-manifest from prepare-dataset')
    report = json.loads(Path(config.dataset_manifest).read_text())
    dataset = {'ICD-9': 'mimiciii', 'ICD-10-CM': 'mimiciv'}[system]
    mixed = system == 'ICD-9' and bundle['manifest'].get('label_space') == 'mixed'
    expected = ICD9_FULL_STATISTICS if mixed else PAPER_STATISTICS[dataset]
    if (not report.get('passed') or report.get('code_system') != system
            or report.get('label_space') != bundle['manifest'].get('label_space')
            or any(report.get('statistics', {}).get(k) != v for k, v in expected.items())):
        raise ValueError('Dataset manifest does not pass the matching label-space gate')
    if mixed and (report.get('train_code_types') != ICD9_FULL_TRAIN_CODES
                  or {s: v['documents'] for s, v in report['splits'].items()} != ICD9_FULL_SPLITS):
        raise ValueError('ICD-9 Full accounting differs from the frozen mixed catalog')
    clinical = [row for row in train_rows if row.get('task', 'note_to_sid') == 'note_to_sid']
    if len(clinical) != report['splits']['train']['documents'] or len(validation_rows) != report['splits']['validation']['documents']:
        raise ValueError('Training/validation row counts differ from the preparation manifest')
    codes = {c for row in clinical for c in row['codes']}
    if codes != set(bundle['assignments']) or len(codes) != report['train_catalog_codes']:
        raise ValueError('Bundle catalog must equal the prepared train-only code universe')
    return report
