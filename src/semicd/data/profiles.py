"""Step 2: six-field code profiles from the ICD ontology and UMLS terms."""
import json
import re

FIELDS = ('CHAPTER', 'BLOCK', 'CATEGORY', 'DESCRIPTION', 'UMLS Terms', 'SYNONYMS')


def whitespace(text):
    return ' '.join(text.split())


def profile_metadata(rows, code_system=None, edition=None):
    """Reuse declared source metadata; never infer an ICD edition from a code."""
    if code_system is None:
        systems = {row.get('code_system') for row in rows}
        if len(systems) != 1 or not systems <= {'ICD-9', 'ICD-10-CM'}:
            raise ValueError('Input rows must declare one ICD code_system; use --code-system to select it explicitly')
        code_system = systems.pop()
    if edition is None:
        editions = {row.get('edition') for row in rows}
        if not editions or any(not isinstance(value, str) or not value.strip() for value in editions):
            raise ValueError('Input rows must declare edition; it cannot be inferred from code strings')
        edition = ','.join(sorted(editions))
    return code_system, edition


def select_terms(description, terms):
    """Select UMLS synonyms for a profile, keeping at most five (as in the paper)."""
    description = whitespace(description)
    unique = {}
    for term in terms:
        term = whitespace(term)
        if term and len(term) <= 160:
            key = term.casefold()
            unique.setdefault(key, term)
    if not unique:
        return description, []
    words = set(re.findall(r'[a-z0-9]+', description.casefold()))
    def score(key):
        other = set(re.findall(r'[a-z0-9]+', key))
        return len(words & other) / max(1, len(words | other))
    description_key = description.casefold()
    preferred_key = description_key if description_key in unique else min(
        unique, key=lambda key: (-score(key), abs(len(unique[key]) - len(description)), key))
    remaining = [key for key in unique if key not in (preferred_key, description_key)]
    synonyms = sorted(remaining, key=lambda key: (-score(key), len(unique[key]), key))[:5]
    return unique[preferred_key], [unique[key] for key in synonyms]


def prepare_profiles(args):
    from .icd import catalog_label_space
    codes = json.loads(args.codes_file.read_text())
    if not isinstance(codes, list) or not codes or any(not isinstance(c, str) or not c for c in codes) or len(set(codes)) != len(codes):
        raise ValueError('codes-file must be a nonempty unique list of code strings')
    terms = json.loads(args.umls_terms_file.read_text())
    if not isinstance(terms, dict) or any(not isinstance(v, list) or any(not isinstance(t, str) for t in v) for v in terms.values()):
        raise ValueError('UMLS export must map normalized code strings to term lists')
    ontology = {}
    for line in args.ontology_file.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        code = row['code']
        ontology.setdefault(code, []).append(row)
    selected = [row for code in codes for row in ontology.get(code, [])]
    args.code_system, args.edition = profile_metadata(selected, args.code_system, args.edition)
    label_space = catalog_label_space(codes, args.code_system)
    if terms and catalog_label_space(list(terms), args.code_system) != label_space:
        raise ValueError('UMLS and catalog label spaces differ')
    profiles, issues = [], []
    for code in sorted(codes):
        rows = ontology.get(code, [])
        if len(rows) != 1:
            issues.append({'code': code, 'reason': 'unmapped' if not rows else 'duplicate_or_multi_parent', 'paths': len(rows)})
            continue
        row = rows[0]
        if row.get('code_system') != args.code_system or row.get('edition') not in args.edition.split(','):
            issues.append({'code': code, 'reason': 'edition_or_system_mismatch'})
            continue
        if (label_space == 'mixed' and (row.get('catalog_member') is not True or row.get('billable') is False)
                or label_space != 'mixed' and row.get('billable') is not True):
            issues.append({'code': code, 'reason': 'invalid_or_unspecified_catalog_member' if label_space == 'mixed' else 'nonbillable_or_unspecified'})
            continue
        if any(not isinstance(row.get(field), str) or not whitespace(row[field]) for field in ('chapter', 'block', 'category', 'description')):
            issues.append({'code': code, 'reason': 'missing_path_or_description'})
            continue
        preferred, synonyms = select_terms(row['description'], terms.get(code, []))
        values = [whitespace(row[f]) for f in ('chapter', 'block', 'category', 'description')]
        values += [preferred, '; '.join(synonyms)]
        profiles.append({'code': code, 'code_system': args.code_system, 'edition': row['edition'], 'label_space': label_space,
                         'fields': dict(zip(FIELDS, values)),
                         'profile': '\n'.join(f'{field}: {value}' for field, value in zip(FIELDS, values))})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'code_system': args.code_system, 'edition': args.edition, 'label_space': label_space,
                'ontology_source': str(args.ontology_file.resolve()), 'umls_source': str(args.umls_terms_file.resolve()),
                'codes_source': str(args.codes_file.resolve()), 'overlap': 'jaccard', 'word_pattern': '[a-z0-9]+',
                'term_max_characters': 160, 'max_synonyms': 5, 'codes': len(codes),
                'source_editions': {edition: sum(row['edition'] == edition for row in profiles) for edition in args.edition.split(',')},
                'profiles': len(profiles), 'missing_umls_codes': sum(not any(whitespace(t) and len(whitespace(t)) <= 160 for t in terms.get(row['code'], [])) for row in profiles),
                'passed': not issues, 'patient_splits_accessed': [],
                'code_validation': 'source catalog membership; billability not verified' if label_space == 'mixed' else 'billable',
                'source_metadata': [row['source_metadata'] for row in selected if 'source_metadata' in row][:1]}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output_dir / 'issues.json').write_text(json.dumps(issues, indent=2) + '\n')
    if issues:
        raise ValueError('Profile construction gate failed; see issues.json')
    (args.output_dir / 'profiles.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in profiles))
    return manifest
