"""Embed code profiles with a separate embedding model (not the training backbone)."""
import json


def embed(args):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    if args.embedding_batch_size < 1 or args.embedding_max_length < 1:
        raise ValueError('Embedding batch size and length must be positive')
    rows = [json.loads(line) for line in args.profiles_file.read_text().splitlines() if line.strip()]
    codes = [row['code'] for row in rows]
    if not rows or len(set(codes)) != len(codes) or any(not isinstance(c, str) or not c for c in codes):
        raise ValueError('Profiles need unique string codes')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name_or_path)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(args.embedding_model_name_or_path).to(args.device).eval()
    chunks = []
    truncated = 0
    with torch.no_grad():
        for i in range(0, len(rows), args.embedding_batch_size):
            texts = [row['profile'] for row in rows[i:i+args.embedding_batch_size]]
            lengths = [len(tokenizer.encode(text)) for text in texts]
            truncated += sum(n > args.embedding_max_length for n in lengths)
            batch = tokenizer(texts, padding=True, truncation=True, max_length=args.embedding_max_length, return_tensors='pt').to(args.device)
            hidden = model(**batch).last_hidden_state.float()
            mask = batch['attention_mask']
            if args.pooling == 'last':
                vectors = hidden[torch.arange(len(texts), device=hidden.device), mask.sum(-1) - 1]
            else:
                vectors = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            if args.normalize:
                vectors = torch.nn.functional.normalize(vectors, dim=-1)
            if not torch.isfinite(vectors).all():
                raise ValueError('Nonfinite profile embeddings')
            chunks.append(vectors.cpu().numpy())
    array = np.concatenate(chunks)
    np.save(args.output_dir / 'embeddings.npy', array, allow_pickle=False)
    (args.output_dir / 'codes.json').write_text(json.dumps(codes, indent=2) + '\n')
    manifest = {'embedding_model_name_or_path': args.embedding_model_name_or_path,
                'embedding_revision': getattr(model.config, '_commit_hash', None), 'pooling': args.pooling,
                'normalize': args.normalize, 'max_length': args.embedding_max_length,
                'truncated_profiles': truncated, 'rows': len(rows), 'dimension': int(array.shape[1]),
                'profiles_source': str(args.profiles_file.resolve()), 'patient_splits_accessed': []}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
