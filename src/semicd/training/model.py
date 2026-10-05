"""Load the backbone with Unsloth and turn rows into prompt/response token IDs."""
import torch

from .prompts import paper_messages


def load_unsloth(config, checkpoint=None, training=True):
    # Import Unsloth before Transformers so its model patches take effect.
    from unsloth import FastLanguageModel
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[config.dtype]
    # AMP fp16 needs FP32 master parameters for GradScaler/AdamW.
    if training and config.dtype == 'fp16':
        dtype = torch.float32
    model, processor = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint or config.model_name_or_path),
        max_seq_length=config.max_seq_length, dtype=dtype,
        load_in_4bit=False, load_in_8bit=False, full_finetuning=True,
        use_gradient_checkpointing='unsloth' if config.gradient_checkpointing else False,
    )
    tokenizer = getattr(processor, 'tokenizer', processor)
    if getattr(model, 'peft_config', None) or any('lora_' in name for name, _ in model.named_parameters()):
        raise ValueError('Full-finetuning runner does not accept adapter checkpoints')
    # Text-only checkpoint required in v1; multimodal towers need a separately tested path.
    if hasattr(model.config, 'vision_config'):
        raise ValueError('Use a validated text-only checkpoint; multimodal towers are not supported in v1')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if training:
        FastLanguageModel.for_training(model, use_gradient_checkpointing='unsloth' if config.gradient_checkpointing else False)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        FastLanguageModel.for_inference(model)
    return model, tokenizer


def prompt_ids(tokenizer, text, task, code_system, representation='semantic', raw_prompt=None, label_space='diagnosis'):
    if not tokenizer.chat_template:
        raise ValueError('A model chat template is required')
    encoded = tokenizer.apply_chat_template(
        paper_messages(task, text, code_system, representation, raw_prompt, label_space),
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    return list(encoded['input_ids'] if hasattr(encoded, 'keys') else encoded)


def note_prompt(tokenizer, text, codec, config, reserve):
    """Fit the note's head prefix, preserving the complete chat wrapper.
    """
    manifest = codec.bundle['manifest']
    def render(length):
        return prompt_ids(tokenizer, text[:length], 'note_to_sid', manifest['code_system'],
                          manifest.get('representation', 'semantic'), manifest.get('raw_prompt'),
                          manifest.get('label_space', 'diagnosis'))
    prompt = render(len(text))
    if len(prompt) + reserve <= config.max_seq_length or not config.truncate_notes:
        return prompt, False
    # Keep at least one non-whitespace character, including leading newlines.
    low, high = len(text) - len(text.lstrip()) + 1, len(text)
    if len(render(low)) + reserve > config.max_seq_length:
        raise ValueError('Chat wrapper and complete target/generation budget do not fit the context')
    while low < high:
        middle = (low + high + 1) // 2
        if len(render(middle)) + reserve <= config.max_seq_length:
            low = middle
        else:
            high = middle - 1
    return render(low), True


def encode_rows(rows, tokenizer, codec, config):
    encoded = []
    for row in rows:
        task = row.get('task', 'note_to_sid')
        if task not in config.task_weights:
            raise ValueError(f'Task {task} has no configured weight')
        if task == 'sid_to_profile':
            response = tokenizer.encode(row['response'], add_special_tokens=False) + [tokenizer.eos_token_id]
        else:
            response = codec.encode(row['codes'])
        truncated = False
        if task == 'note_to_sid':
            reserve = max(config.max_new_tokens, len(response)) if config.truncate_notes else len(response)
            prompt, truncated = note_prompt(tokenizer, row['prompt'], codec, config, reserve)
        else:
            manifest = codec.bundle['manifest']
            prompt = prompt_ids(tokenizer, row['prompt'], task, manifest['code_system'],
                                manifest.get('representation', 'semantic'), manifest.get('raw_prompt'),
                                manifest.get('label_space', 'diagnosis'))
        if not prompt or not response or len(prompt) + len(response) > config.max_seq_length:
            raise ValueError('Empty or overlength sample; prepare explicit truncation upstream')
        encoded.append({'prompt': prompt, 'response': response, 'task': task, 'note_truncated': truncated})
    if set(config.task_weights) != {r['task'] for r in encoded}:
        raise ValueError('Configured task weights must match observed training tasks')
    return encoded
