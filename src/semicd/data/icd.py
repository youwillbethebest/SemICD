"""Normalize ICD-code strings without losing leading zeros.

Matches the normalization used in ICDSID common.py at e3ffc2e5b2a7721154bddafa61824e72981155f0.
The public data interface requires strings and an explicit diagnosis code system.
"""


def normalize_code(code, code_system):
    if code_system not in ('ICD-9', 'ICD-10-CM'):
        raise ValueError('Expected code system ICD-9 or ICD-10-CM')
    if not isinstance(code, str):
        raise TypeError('ICD codes must be strings to preserve leading zeros')
    normalized = code.strip().upper().replace('.', '')
    if not normalized:
        raise ValueError('ICD code must be nonempty')
    return normalized


def typed_icd9(code, kind):
    """Keep diagnosis/procedure identity when compact digit strings overlap."""
    if kind not in ('DIAG', 'PROC'):
        raise ValueError('Expected DIAG or PROC')
    import re
    code = normalize_code(code, 'ICD-9')
    pattern = r'(?:[0-9]{3,5}|V[0-9]{2,4}|E[0-9]{3,4})' if kind == 'DIAG' else r'[0-9]{2,4}'
    if not re.fullmatch(pattern, code):
        raise ValueError(f'Invalid ICD-9 {kind} code: {code}')
    return kind + ':' + code


def catalog_label_space(codes, code_system):
    if code_system == 'ICD-9' and any(':' in c for c in codes):
        for code in codes:
            kind, separator, value = code.partition(':')
            if not separator or typed_icd9(value, kind) != code:
                raise ValueError('Mixed ICD-9 catalogs require normalized DIAG:/PROC: keys')
        return 'mixed'
    if any(normalize_code(c, code_system) != c for c in codes):
        raise ValueError('Catalog codes must already be normalized')
    return 'diagnosis'


def read_codes(path, code_system='ICD-10-CM'):
    """Read a train-only, normalized ICD catalog for input preparation."""
    import json
    from pathlib import Path
    codes = json.loads(Path(path).read_text())
    if not isinstance(codes, list) or not codes or any(not isinstance(c, str) for c in codes):
        raise ValueError('Codes must be a nonempty list of strings')
    if len(set(codes)) != len(codes):
        raise ValueError('Codes must be unique and already normalized')
    catalog_label_space(codes, code_system)
    return sorted(codes)
