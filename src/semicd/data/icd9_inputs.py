"""Adapt local ICD-9 Full catalog/path exports and the original raw term cache.

This preserves supplied hierarchy records; it does not certify their edition or
reconstruct an official ontology. Catalog membership is distinct from billability.
"""
import csv
import json

from .icd import read_codes, typed_icd9


def indexed_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        rows = {}
        for row in csv.DictReader(stream):
            rows.setdefault(row['code_norm'], []).append(row)
    return rows


def prepare_icd9_inputs(args):
    codes = read_codes(args.codes_file, 'ICD-9')
    if any(not c.startswith(('DIAG:', 'PROC:')) for c in codes):
        raise ValueError('ICD-9 Full requires DIAG:/PROC: catalog keys')
    catalog, paths = indexed_csv(args.catalog_file), indexed_csv(args.paths_file)
    raw_terms = json.loads(args.umls_terms_file.read_text())
    if not isinstance(raw_terms, dict) or any(not isinstance(v, list) or any(not isinstance(t, str) for t in v) for v in raw_terms.values()):
        raise ValueError('Raw UMLS cache must map original dotted codes to string lists')
    contract = json.loads(args.source_contract.read_text())
    edition, release = contract.get('ontology_edition'), contract.get('umls_release')
    if edition is not None and (not isinstance(edition, str) or not edition.strip() or ',' in edition):
        raise ValueError('ontology_edition must be a nonempty single edition identifier or null')
    rows, terms, issues = [], {}, []
    metadata = {'ontology_edition': edition, 'umls_release': release,
                'hierarchy_verification': 'supplied legacy paths; not certified by this adapter',
                'billability_verified': False}
    for code in codes:
        records, ancestors = catalog.get(code, []), paths.get(code, [])
        if len(records) != 1 or len(ancestors) != 1:
            issues.append({'code': code, 'reason': 'missing_or_duplicate_catalog_or_path'})
            continue
        row, path = records[0], ancestors[0]
        kind = {'diagnosis': 'DIAG', 'procedure': 'PROC'}.get(row['code_kind'])
        official = row['official_code']
        if kind != code.split(':')[0] or typed_icd9(official, kind) != code or path['leaf_id'] != official:
            issues.append({'code': code, 'reason': 'code_type_or_leaf_mismatch'})
            continue
        values = {field: row[column].strip() for field, column in
                  [('chapter', 'chapter_text'), ('block', 'block_text'),
                   ('category', 'category_text'), ('description', 'description')]}
        if any(not value for value in values.values()) or any(not path.get(k, '').strip() for k in ('chapter_id', 'block_id', 'category_id')):
            issues.append({'code': code, 'reason': 'missing_path_or_description'})
            continue
        rows.append({'code': code, 'code_system': 'ICD-9', 'code_kind': row['code_kind'],
                     'edition': edition or 'unverified', 'billable': None, 'catalog_member': True,
                     'source_metadata': metadata, 'path_ids': {k: path[k] for k in ('chapter_id', 'block_id', 'category_id')},
                     **values})
        # Never reuse the already selected synonyms/preferred fields in the CSV.
        terms[code] = raw_terms.get(official, [])
    manifest = {'code_system': 'ICD-9', 'label_space': 'mixed', 'target_codes': len(codes),
                'matched_codes': len(rows), 'code_types': {kind: sum(c.startswith(kind + ':') for c in codes) for kind in ('DIAG', 'PROC')},
                'source_metadata': metadata, 'terms': 'original dotted-key raw cache; selection deferred to prepare-profiles',
                'sources': {name: str(getattr(args, name).resolve()) for name in
                            ('codes_file', 'catalog_file', 'paths_file', 'umls_terms_file', 'source_contract')},
                'unmatched_umls_codes': [r['code'] for r in rows if not terms[r['code']]],
                'passed': not issues, 'patient_splits_accessed': []}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output_dir / 'issues.json').write_text(json.dumps(issues, indent=2) + '\n')
    if issues:
        raise ValueError('ICD-9 input conversion failed; see issues.json')
    (args.output_dir / 'ontology.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (args.output_dir / 'umls_terms.json').write_text(json.dumps(terms, indent=2) + '\n')
    return manifest
