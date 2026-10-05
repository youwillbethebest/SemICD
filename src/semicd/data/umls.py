"""Extract raw English atoms from a user's licensed local MRCONSO.RRF."""
import json

from .icd import normalize_code, read_codes


def conso_rows(path):
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            fields = line.rstrip('\r\n').split('|')
            if len(fields) < 18:
                raise ValueError(f'MRCONSO row {number} has fewer than 18 fields')
            yield fields


def prepare_umls(args):
    codes = read_codes(args.codes_file)
    if not args.release.strip():
        raise ValueError('Provide the actual UMLS release')
    wanted = set(codes)
    by_cui = {}
    code_cuis = {code: set() for code in codes}
    for fields in conso_rows(args.umls_file):
        if fields[11] == 'ICD10CM':
            code = normalize_code(fields[13], 'ICD-10-CM')
            if code in wanted:
                by_cui.setdefault(fields[0], set()).add(code)
                code_cuis[code].add(fields[0])
    terms = {code: [] for code in codes}
    sources = set()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'umls_atoms.jsonl').open('w', encoding='utf-8') as output:
        for fields in conso_rows(args.umls_file):
            if fields[1] != 'ENG' or fields[0] not in by_cui:
                continue
            sources.add(fields[11])
            for code in sorted(by_cui[fields[0]]):
                terms[code].append(fields[14])
                output.write(json.dumps({'code': code, 'cui': fields[0], 'term': fields[14],
                    'sab': fields[11], 'tty': fields[12], 'suppress': fields[16]}) + '\n')
    manifest = {'release': args.release, 'release_source': 'user supplied; not inferred from MRCONSO',
                'umls_source': str(args.umls_file.resolve()), 'codes_source': str(args.codes_file.resolve()),
                'code_system': 'ICD-10-CM', 'code_matching_sab': 'ICD10CM', 'language': 'ENG',
                'include_suppressible': True, 'include_obsolete': True, 'atom_sabs': sorted(sources),
                'target_codes': len(codes), 'codes_with_terms': sum(bool(v) for v in terms.values()),
                'unmatched_codes': [code for code in codes if not code_cuis[code]],
                'codes_without_english_terms': [code for code in codes if code_cuis[code] and not terms[code]],
                'cuis_by_code': {code: sorted(values) for code, values in code_cuis.items()},
                'raw_terms': sum(map(len, terms.values())), 'term_selection': 'deferred to prepare-profiles',
                'patient_splits_accessed': []}
    (args.output_dir / 'umls_terms.json').write_text(json.dumps(terms, indent=2) + '\n')
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
