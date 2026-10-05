"""SID bundle format: a directory with manifest.json and assignments.json."""
import json
from pathlib import Path


def read_bundle(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    assignments = json.loads((path / 'assignments.json').read_text())
    representation = manifest.get('representation', 'semantic')
    if representation not in ('semantic', 'random', 'atomic', 'raw'):
        raise ValueError('Unknown representation')
    if representation in ('atomic', 'raw'):
        if not manifest.get('code_system') or manifest.get('label_space') not in ('diagnosis', 'mixed') or manifest.get('method') != representation:
            raise ValueError('Raw/Atomic bundle requires system and diagnosis/mixed catalog')
        if not assignments or any(not isinstance(c, str) or not c or not isinstance(v, list) or len(v) != 1 or type(v[0]) is not int or v[0] < 0 for c, v in assignments.items()):
            raise ValueError('Raw/Atomic requires one index per code')
        if len({v[0] for v in assignments.values()}) != len(assignments):
            raise ValueError('Duplicate Raw/Atomic index')
        if representation == 'raw' and (not manifest.get('raw_prompt') or manifest.get('raw_format') != 'code_tags_comma' or manifest.get('raw_end') != 'eos'):
            raise ValueError('Raw bundle requires prompt, code-tag format and native EOS contract')
        return {'manifest': manifest, 'assignments': assignments}
    if manifest.get('k') != 128 or manifest.get('depth') != 3:
        raise ValueError('This release recipe requires a K128/L3 bundle')
    if not manifest.get('code_system') or manifest.get('label_space') not in ('diagnosis', 'mixed'):
        raise ValueError('Bundle requires code_system and diagnosis/mixed label_space')
    if manifest.get('method') not in ('hkm', 'rkmeans', 'rqvae'):
        raise ValueError('Unknown SID method')
    if manifest.get('uid_policy') not in ('deterministic', 'none'):
        raise ValueError('Unknown UID policy')
    if not assignments:
        raise ValueError('Empty catalog')
    native_groups = {}
    tuples = set()
    for code, values in assignments.items():
        if not isinstance(code, str) or not code:
            raise ValueError('Codes must be nonempty strings')
        if len(values) not in (3, 4) or any(type(v) is not int or v < 0 for v in values):
            raise ValueError(f'Invalid SID tuple for {code}')
        if any(v >= manifest['k'] for v in values[:3]):
            raise ValueError('SID outside declared K')
        if manifest.get('uid_policy') == 'none' and len(values) != 3:
            raise ValueError('UID tokens are disabled by this bundle')
        native_groups.setdefault(tuple(values[:3]), []).append((code, values))
        if tuple(values) in tuples:
            raise ValueError('SID collisions must be resolved before generative training')
        tuples.add(tuple(values))

    if manifest.get('uid_policy') == 'deterministic':
        for path, members in native_groups.items():
            if len(members) == 1:
                code, values = members[0]
                if len(values) != 3:
                    raise ValueError(f'Unique native SID must not carry a UID: {code}')
            else:
                if any(len(values) != 4 for _, values in members):
                    raise ValueError(f'Colliding native SID requires a UID: {path}')
                uids = [values[3] for _, values in members]
                if len(set(uids)) != len(uids):
                    raise ValueError(f'UID collision within native SID: {path}')
    elif any(len(members) > 1 for members in native_groups.values()):
        raise ValueError('Native SID collisions require deterministic UID disambiguation')

    return {'manifest': manifest, 'assignments': assignments}
