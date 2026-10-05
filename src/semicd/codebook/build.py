"""Step 3: profile embeddings -> quantizer -> UIDs for collisions -> SID bundle."""
import json


def assign_uids(codes, paths):
    if len(codes) != len(paths) or len(set(codes)) != len(codes) or not codes:
        raise ValueError('Code/path accounting mismatch or duplicate/empty catalog')
    groups = {}
    for code, path in zip(codes, paths):
        values = tuple(int(v) for v in path)
        if len(values) != 3 or any(v < 0 or v >= 128 for v in values):
            raise ValueError('Expected K128/L3 native path')
        groups.setdefault(values, []).append(code)
    assignments = {}
    for path, members in groups.items():
        for uid, code in enumerate(sorted(members)):
            assignments[code] = list(path) + ([uid] if len(members) > 1 else [])
    return dict(sorted(assignments.items())), {
        'native_unique_paths': len(groups), 'collision_groups': sum(len(v) > 1 for v in groups.values()),
        'uid_codes': sum(len(v) for v in groups.values() if len(v) > 1),
        'max_collision_group': max(map(len, groups.values())),
    }


def permute_bundle(bundle, seed):
    import copy
    import numpy as np
    result = copy.deepcopy(bundle)
    codes = sorted(bundle['assignments'])
    permutation = np.random.default_rng(seed).permutation(len(codes)).tolist()
    result['assignments'] = {code: list(bundle['assignments'][codes[j]]) for code, j in zip(codes, permutation)}
    result['manifest'].update(representation='random', assignment_seed=seed,
                              permutation='complete_path_multiset', fixed_points=sum(i == j for i, j in enumerate(permutation)))
    return result


def build_codebook(args):
    from .bundle import read_bundle
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    base = {'code_system': args.code_system, 'edition': args.edition, 'label_space': 'diagnosis',
            'representation': args.representation}
    if args.representation == 'random':
        if args.source_bundle is None or args.assignment_seed is None:
            raise ValueError('Random SID needs --source-bundle and --assignment-seed')
        source = read_bundle(args.source_bundle)
        if source['manifest'].get('representation', 'semantic') != 'semantic':
            raise ValueError('Random SID must permute a Semantic SID bundle')
        base['label_space'] = source['manifest']['label_space']
        for key in ('code_system', 'edition'):
            if base[key] is None:
                base[key] = source['manifest'].get(key)
        if any(source['manifest'].get(key) != base[key] for key in ('code_system', 'edition', 'label_space')):
            raise ValueError('Source bundle system/edition/label space mismatch')
        bundle = permute_bundle(source, args.assignment_seed)
        bundle['manifest']['source_bundle'] = str(args.source_bundle.resolve())
    elif args.representation in ('raw', 'atomic'):
        if args.codes_file is None:
            raise ValueError('Raw/Atomic construction needs the train-only --codes-file')
        codes = json.loads(args.codes_file.read_text())
        if not isinstance(codes, list) or not codes or any(not isinstance(c, str) or not c for c in codes) or len(set(codes)) != len(codes):
            raise ValueError('Codes must be nonempty unique strings')
        if args.code_system is None:
            manifest = args.codes_file.parent / 'manifest.json'
            args.code_system = json.loads(manifest.read_text()).get('code_system') if manifest.exists() else None
            if args.code_system not in ('ICD-9', 'ICD-10-CM'):
                raise ValueError('Standalone code catalogs need --code-system; prepared catalogs inherit it from manifest.json')
        base['code_system'] = args.code_system
        from ..data.icd import catalog_label_space
        base['label_space'] = catalog_label_space(codes, args.code_system)
        base.update(method=args.representation, codes_source=str(args.codes_file.resolve()))
        if args.representation == 'raw':
            from ..training.prompts import RAW_NOTE
            prompt = args.raw_prompt_file.read_text().strip() if args.raw_prompt_file else RAW_NOTE.format(code_system=args.code_system)
            if base['label_space'] == 'mixed' and not args.raw_prompt_file:
                prompt = prompt.replace('diagnosis codes', 'diagnosis and procedure codes')
                prompt += ' Use DIAG: and PROC: prefixes with dot-stripped codes to distinguish code types.'
            if not prompt:
                raise ValueError('Raw system prompt is empty')
            base.update(raw_prompt=prompt, raw_format='code_tags_comma', raw_end='eos')
        bundle = {'manifest': base, 'assignments': {code: [i] for i, code in enumerate(sorted(codes))}}
    else:
        import numpy as np
        if args.profiles_file is None:
            raise ValueError('Semantic construction needs --profiles-file')
        rows = [json.loads(line) for line in args.profiles_file.read_text().splitlines() if line.strip()]
        from ..data.profiles import profile_metadata
        args.code_system, args.edition = profile_metadata(rows, args.code_system, args.edition)
        from ..data.icd import catalog_label_space
        base.update(code_system=args.code_system, edition=args.edition,
                    label_space=catalog_label_space([r['code'] for r in rows], args.code_system))
        if not rows or any(r.get('code_system') != args.code_system or r.get('edition') not in args.edition.split(',') for r in rows):
            raise ValueError('Profiles must declare the matching code system and edition')
        embedding_dir = args.embedding_dir
        if embedding_dir is None:
            from argparse import Namespace
            from .embeddings import embed
            embedding_dir = args.output_dir.with_name(args.output_dir.name + '-embeddings')
            embed(Namespace(profiles_file=args.profiles_file, output_dir=embedding_dir,
                            embedding_model_name_or_path=args.embedding_model_name_or_path,
                            embedding_batch_size=args.embedding_batch_size, embedding_max_length=512,
                            pooling='last', normalize=True, device=args.device))
        codes = json.loads((embedding_dir / 'codes.json').read_text())
        if codes != [r['code'] for r in rows] or len(set(codes)) != len(codes):
            raise ValueError('Embedding code order must match the profile rows exactly')
        vectors = np.load(embedding_dir / 'embeddings.npy', allow_pickle=False)
        if vectors.ndim != 2 or vectors.shape[0] != len(codes) or not vectors.shape[1] or not np.isfinite(vectors).all():
            raise ValueError('Invalid embedding shape/accounting or nonfinite vectors')
        order = sorted(range(len(codes)), key=lambda i: codes[i])
        codes, vectors = [codes[i] for i in order], vectors[order]
        settings = dict(depth=3, seed=args.sid_seed, n_init=args.n_init,
                        max_iter=args.max_iter, tol=1e-4, algorithm='lloyd', fit_threads=1)
        if args.method == 'hkm':
            from .hkm import hierarchical_kmeans
            paths = hierarchical_kmeans(vectors, np.asarray(codes), branching_factor=128, **settings)
        elif args.method == 'rkmeans':
            from .rkmeans import residual_kmeans
            paths, _, _, _ = residual_kmeans(vectors, np.asarray(codes), codebook_size=128, **settings)
        else:
            from .rqvae import construct_rqvae
            paths, report = construct_rqvae(vectors, codes, args)
            settings = report['config']
            base.update(rqvae=report, sid_seed=report['seed'])
        assignments, collision = assign_uids(codes, paths)
        base.update(method=args.method, k=128, depth=3, uid_policy='deterministic',
                    sid_seed=base.get('sid_seed', args.sid_seed), quantizer_settings=settings,
                    profile_source=str(args.profiles_file.resolve()),
                    embedding_source=str(embedding_dir.resolve()),
                    source_editions={edition: sum(row['edition'] == edition for row in rows) for edition in args.edition.split(',')},
                    embedding=json.loads((embedding_dir / 'manifest.json').read_text()), **collision)
        bundle = {'manifest': base, 'assignments': assignments}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in bundle.items():
        (args.output_dir / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
    read_bundle(args.output_dir)
    return bundle['manifest']
