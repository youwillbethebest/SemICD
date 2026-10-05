"""Step 1: Edin et al. MIMIC tables + released splits -> exact ICD-set JSONL."""
import json
from pathlib import Path

from .icd import normalize_code, typed_icd9

PAPER_STATISTICS = {
    'mimiciii': {'documents': 52723, 'patients': 41126, 'codes': 8929},
    'mimiciv': {'documents': 122304, 'patients': 65675, 'codes': 16155},
}
ICD9_FULL_STATISTICS = {'documents': 52722, 'codes': 8929}
ICD9_FULL_SPLITS = {'train': 47719, 'validation': 1631, 'test': 3372}
ICD9_FULL_TRAIN_CODES = {'DIAG': 6724, 'PROC': 1968}

DATASETS = {'mimiciii': ('MIMIC-III v1.4', 'ICD-9', 'icd9_diag'),
            'mimiciv': ('MIMIC-IV v2.2', 'ICD-10-CM', 'icd10_diag')}


def read_table(path):
    import pandas as pd
    path = Path(path)
    if path.suffix == '.feather':
        return pd.read_feather(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix == '.jsonl':
        return pd.DataFrame(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    raise ValueError('Use upstream Feather/Parquet or explicit JSONL table exports')


def convert_tables(data, splits, code_system, diagnosis_column, expected, procedure_column=None):
    """No patient identifiers or note text are included in the aggregate report."""
    required = {'_id', 'subject_id', 'text', diagnosis_column}
    if procedure_column:
        required.add(procedure_column)
    if not required <= set(data.columns) or not {'_id', 'split'} <= set(splits.columns):
        raise ValueError('Missing upstream ID, subject, text, diagnosis or split columns')
    if data['_id'].isna().any() or data['subject_id'].isna().any() or splits['_id'].isna().any():
        raise ValueError('Missing admission or patient identity')
    if data['_id'].duplicated().any() or splits['_id'].duplicated().any():
        raise ValueError('Duplicate document or split membership')
    if set(splits['split']) != {'train', 'val', 'test'}:
        raise ValueError('Released splits must contain train, val and test')
    if set(data['_id']) != set(splits['_id']):
        raise ValueError('Upstream data and split document membership differ')
    joined = data.merge(splits[['_id', 'split']], on='_id', validate='one_to_one')
    patients = {split: set(joined.loc[joined['split'] == split, 'subject_id']) for split in ('train', 'val', 'test')}
    if any(patients[a] & patients[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Patient leakage across released splits')
    outputs = {split: [] for split in ('train', 'validation', 'test')}
    retained_patients, vocabulary, dropped = set(), set(), 0
    for row in joined.to_dict('records'):
        codes = set()
        for column, kind in [(diagnosis_column, 'DIAG')] + ([(procedure_column, 'PROC')] if procedure_column else []):
            labels = row[column]
            if labels is None or isinstance(labels, float) and labels != labels:
                labels = []
            if not isinstance(labels, (list, tuple)) and not hasattr(labels, 'tolist'):
                raise ValueError('Code columns must contain string lists')
            codes.update(typed_icd9(code, kind) if procedure_column else normalize_code(code, code_system)
                         for code in labels)
        codes = sorted(codes)
        if not codes:
            dropped += 1
            continue
        if not isinstance(row['text'], str) or not row['text'].strip():
            raise ValueError('Empty clinical note in retained document')
        split = 'validation' if row['split'] == 'val' else row['split']
        outputs[split].append({'prompt': row['text'], 'codes': codes})
        retained_patients.add(row['subject_id'])
        vocabulary.update(codes)
    observed = {'documents': sum(map(len, outputs.values())), 'patients': len(retained_patients), 'codes': len(vocabulary)}
    train_codes = sorted({c for row in outputs['train'] for c in row['codes']})
    train_catalog = set(train_codes)
    report = {'statistics': observed, 'expected': expected, 'passed': all(observed.get(k) == v for k, v in expected.items()),
              'diagnosis_only': not bool(procedure_column), 'dropped_no_labels': dropped, 'subject_overlap': 0,
              'splits': {s: {'documents': len(rows), 'unknown_gold_assignments': sum(c not in train_catalog for r in rows for c in r['codes'])}
                         for s, rows in outputs.items()}, 'train_catalog_codes': len(train_codes),
              'normalization': 'uppercase; remove decimal; preserve leading zeros; DIAG:/PROC: for mixed ICD-9'}
    if not procedure_column:
        report['dropped_no_diagnosis'] = dropped
    if procedure_column:
        report['train_code_types'] = {kind: sum(c.startswith(kind + ':') for c in train_codes) for kind in ('DIAG', 'PROC')}
    return outputs, train_codes, report


def prepare_dataset(args):
    upstream = read_table(args.upstream_file)
    if args.dataset is None:
        found = [name for name, (_, _, column) in DATASETS.items() if column in upstream.columns]
        if len(found) != 1:
            raise ValueError('Cannot infer the dataset from the diagnosis column; pass --dataset')
        args.dataset = found[0]
    label_space = getattr(args, 'label_space', None) or ('mixed' if args.dataset == 'mimiciii' else 'diagnosis')
    if label_space == 'mixed' and args.dataset != 'mimiciii':
        raise ValueError('Mixed input is supported for ICD-9 Full only')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    access = {'test_used': True, 'purpose': 'dataset preparation/statistics only', 'status': 'started'}
    (args.output_dir / 'access.json').write_text(json.dumps(access, indent=2) + '\n')
    try:
        dataset, system, diagnosis = DATASETS[args.dataset]
        mixed = label_space == 'mixed'
        expected = ICD9_FULL_STATISTICS if mixed else PAPER_STATISTICS[args.dataset]
        outputs, codes, report = convert_tables(upstream, read_table(args.splits_file),
                                               system, diagnosis, expected, 'icd9_proc' if mixed else None)
        if mixed:
            report['passed'] &= ({s: v['documents'] for s, v in report['splits'].items()} == ICD9_FULL_SPLITS
                                 and report['train_code_types'] == ICD9_FULL_TRAIN_CODES)
        report.update(dataset=dataset, code_system=system, label_space=label_space,
                      upstream_source=str(args.upstream_file.resolve()), split_source=str(args.splits_file.resolve()),
                      test_used=True)
        (args.output_dir / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        if not report['passed']:
            raise ValueError('Dataset statistics/accounting mismatch; see manifest.json; no dataset was exported')
        for split, rows in outputs.items():
            (args.output_dir / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (args.output_dir / 'train_codes.json').write_text(json.dumps(codes, indent=2) + '\n')
    except Exception:
        access['status'] = 'failed'
        (args.output_dir / 'access.json').write_text(json.dumps(access, indent=2) + '\n')
        raise
    access['status'] = 'completed'
    (args.output_dir / 'access.json').write_text(json.dumps(access, indent=2) + '\n')
    return report
