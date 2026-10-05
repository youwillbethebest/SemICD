"""Step 6: constrained decoding and exact ICD-set metrics."""
import json
from pathlib import Path

import torch

from ..utils import read_rows, write_json
from .checkpoint import checkpoint_codec
from .config import TrainConfig
from .model import load_unsloth, note_prompt


def score_sets(predictions, golds, catalog):
    tp = fp = fn = exact = unknown = 0
    catalog = set(catalog)
    counts = {code: [0, 0, 0] for code in catalog}
    for predicted, gold in zip(predictions, golds):
        p, g = set(predicted), set(gold)
        tp += len(p & g); fp += len(p - g); fn += len(g - p)
        exact += p == g
        unknown += len(g - set(catalog))
        for code in p | g:
            if code in counts:
                counts[code][0] += code in p & g
                counts[code][1] += code in p - g
                counts[code][2] += code in g - p
    return {'micro_precision': tp / max(1, tp + fp), 'micro_recall': tp / max(1, tp + fn),
            'micro_f1': 2 * tp / max(1, 2 * tp + fp + fn),
            'macro_f1_all_catalog': sum(2*a / max(1, 2*a+b+c) for a,b,c in counts.values()) / len(counts),
            'exact_set_match': exact / len(golds), 'unknown_gold_assignments': unknown,
            'catalog_labels': len(catalog), 'rows': len(golds), 'tp': tp, 'fp': fp, 'fn': fn}


@torch.no_grad()
def evaluate_rows(model, tokenizer, codec, rows, config, output):
    model.eval()
    device = model.get_input_embeddings().weight.device
    predictions, golds, records = [], [], []
    for start in range(0, len(rows), config.eval_batch_size):
        batch_rows = rows[start:start + config.eval_batch_size]
        prompts = []
        for row in batch_rows:
            if not isinstance(row.get('codes'), list) or any(not isinstance(c, str) for c in row['codes']):
                raise ValueError('Evaluation requires gold codes as strings')
            prompt, truncated = note_prompt(tokenizer, row['prompt'], codec, config, config.max_new_tokens)
            if not prompt or len(prompt) + config.max_new_tokens > config.max_seq_length:
                raise ValueError('Evaluation prompt exceeds generation budget; prepare explicit truncation')
            prompts.append((prompt, truncated))
        truncated_notes = [truncated for _, truncated in prompts]
        prompts = [prompt for prompt, _ in prompts]
        width = max(map(len, prompts))
        ids = torch.full((len(prompts), width), tokenizer.pad_token_id, device=device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, prompt in enumerate(prompts):
            ids[i, -len(prompt):] = torch.tensor(prompt, device=device)
            mask[i, -len(prompt):] = 1
        def allowed_tokens(_, sequence):
            answer = sequence.tolist()[width:]
            # Beam search also invokes processors on finished, padded beam slots.
            # Keep them in an END sink; live prefixes retain strict codec checks.
            return [codec.end] if codec.end in answer else codec.allowed(answer)
        decode_options = {'prefix_allowed_tokens_fn': allowed_tokens} if codec.constrained else {}
        generated = model.generate(input_ids=ids, attention_mask=mask,
                                   max_new_tokens=config.max_new_tokens, num_beams=config.num_beams,
                                   length_penalty=config.length_penalty,
                                   do_sample=False, eos_token_id=codec.end, pad_token_id=tokenizer.pad_token_id,
                                   **decode_options,
                                   use_cache=True)
        for row, sequence, truncated in zip(batch_rows, generated, truncated_notes):
            answer = sequence[width:].tolist()
            if codec.end in answer:
                answer = answer[:answer.index(codec.end) + 1]
            codes, complete, valid = codec.decode(answer)
            predictions.append(codes); golds.append(row['codes'])
            records.append({'predicted_codes': codes, 'gold_codes': row['codes'], 'complete': complete,
                            'note_truncated': truncated,
                            'format_valid': valid, 'hit_output_token_budget': len(answer) >= config.max_new_tokens and not complete})
    metrics = score_sets(predictions, golds, codec.forward)
    for key in ('complete', 'format_valid', 'hit_output_token_budget'):
        metrics[key + '_rate'] = sum(row[key] for row in records) / len(records)
    Path(output).mkdir(parents=True, exist_ok=True)
    write_json(Path(output) / 'metrics.json', metrics)
    write_json(Path(output) / 'preprocessing.json', {
        'note_truncation': 'head_prefix_character_bisection' if config.truncate_notes else None,
        'truncated_notes': sum(row['note_truncated'] for row in records), 'rows': len(records)})
    (Path(output) / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    return metrics


def resolve_checkpoint(path):
    """Accept a checkpoint, or a training run directory and use its selected checkpoint."""
    path = Path(path)
    selection = path / 'selection.json'
    if selection.is_file():
        return Path(json.loads(selection.read_text())['selected_checkpoint'])
    return path


def evaluate_checkpoint(checkpoint, eval_file, output_dir):
    """Evaluate a saved checkpoint with its own tokenizer, SID mapping and decoding settings."""
    checkpoint, eval_file, output_dir = resolve_checkpoint(checkpoint), Path(eval_file), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    split = 'test' if 'test' in eval_file.stem else 'validation'
    access = {'split': split, 'test_used': split == 'test', 'status': 'started',
              'checkpoint': str(checkpoint.resolve()), 'eval_file': str(eval_file.resolve())}
    write_json(output_dir / 'access.json', access)
    try:
        config = TrainConfig(**json.loads((checkpoint / 'train_config.json').read_text())).validate()
        model, tokenizer = load_unsloth(config, checkpoint, training=False)
        codec = checkpoint_codec(checkpoint, tokenizer)
        metrics = evaluate_rows(model, tokenizer, codec, read_rows(eval_file), config, output_dir)
    except Exception:
        access['status'] = 'failed'
        write_json(output_dir / 'access.json', access)
        raise
    access['status'] = 'completed'
    write_json(output_dir / 'access.json', access)
    return metrics
