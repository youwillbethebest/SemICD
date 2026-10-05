"""Build edition-tagged ICD-10-CM paths from official order and tabular files."""
import json
import xml.etree.ElementTree as ET

from .icd import normalize_code, read_codes

FILES = {'FY2020': ('icd10cm_order_2020.txt', 'icd10cm_tabular_2020.xml'),
         'FY2015': ('icd10cm_order_2015.txt', 'FY15_Tabular.xml')}
BASE = 'https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/'


def source_files(directory, edition):
    result = []
    for name in FILES[edition]:
        matches = sorted(directory.rglob(name))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {name} under {directory}')
        result.append(matches[0])
    return result


def read_release(directory, edition):
    order_path, tabular_path = source_files(directory, edition)
    order, paths = {}, {}
    # Official fixed-width order format; adapted from ICDSID data.py at e3ffc2e5.
    for line in order_path.read_text(encoding='latin1').splitlines():
        if not line.strip():
            continue
        code = normalize_code(line[6:14], 'ICD-10-CM')
        flag = line[14:16].strip()
        if flag not in ('0', '1'):
            raise ValueError(f'Invalid order-file billable flag for {code}')
        record = {'billable': flag == '1', 'description': line[77:].strip() or line[16:77].strip()}
        order.setdefault(code, []).append(record)

    def visit(diag, path, extensions):
        name = normalize_code(diag.findtext('name') or '', 'ICD-10-CM')
        definition = diag.find('sevenChrDef')
        if definition is not None:
            extensions = [node.attrib['char'] for node in definition.findall('extension')]
        paths.setdefault(name, []).append(path)
        children = diag.findall('diag')
        # The XML declares seventh-character extensions separately from leaves.
        # Expand that declaration explicitly; do not infer ancestry from prefixes.
        if not children and extensions and len(name) < 7:
            for extension in extensions:
                paths.setdefault(name.ljust(6, 'X') + extension, []).append(path)
        for child in children:
            visit(child, path, extensions)

    tree = ET.parse(tabular_path)
    for chapter in tree.getroot().findall('chapter'):
        for section in chapter.findall('section'):
            for category in section.findall('diag'):
                path = {'chapter': (chapter.findtext('desc') or '').strip(),
                        'block': (section.findtext('desc') or '').strip(),
                        'category': (category.findtext('desc') or '').strip()}
                visit(category, path, [])
    source = {'edition': edition, 'order_file': str(order_path.resolve()),
              'tabular_file': str(tabular_path.resolve()), 'publisher': 'CDC/NCHS',
              'official_directory': BASE + edition[2:] + '/'}
    return order, paths, source


def prepare_ontology(args):
    codes = read_codes(args.codes_file)
    primary = read_release(args.primary_dir, 'FY2020')
    supplement = read_release(args.supplement_dir, 'FY2015') if args.supplement_dir else None
    rows, issues = [], []
    for code in codes:
        release = primary
        if code not in primary[0] and supplement is not None:
            release = supplement
        order, paths, source = release
        records, ancestors = order.get(code, []), paths.get(code, [])
        reason = None
        if not records:
            reason = 'unmatched'
        elif len(records) != 1:
            reason = 'duplicate_order'
        elif not records[0]['billable']:
            reason = 'nonbillable'
        elif not ancestors:
            reason = 'missing_path'
        elif len(ancestors) != 1:
            reason = 'duplicate_or_multi_parent'
        elif not records[0]['description'] or any(not value for value in ancestors[0].values()):
            reason = 'missing_description_or_ancestor'
        if reason:
            issues.append({'code': code, 'edition': source['edition'], 'reason': reason})
        else:
            rows.append({'code': code, 'code_system': 'ICD-10-CM', 'edition': source['edition'],
                         **records[0], **ancestors[0]})
    manifest = {'code_system': 'ICD-10-CM', 'target_codes': len(codes), 'matched_codes': len(rows),
                'source_editions': {edition: sum(r['edition'] == edition for r in rows) for edition in FILES},
                'sources': [primary[2]] + ([supplement[2]] if supplement else []),
                'codes_source': str(args.codes_file.resolve()), 'passed': not issues,
                'fallback': 'FY2015 only when code absent from FY2020 order', 'patient_splits_accessed': []}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output_dir / 'issues.json').write_text(json.dumps(issues, indent=2) + '\n')
    if issues:
        raise ValueError('Ontology preparation failed; see issues.json')
    (args.output_dir / 'ontology.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return manifest
