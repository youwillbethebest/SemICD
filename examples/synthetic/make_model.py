"""Create a tiny local Qwen3 checkpoint for the README's synthetic walkthrough."""
import argparse
from pathlib import Path


def make_model(output):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    words = ['[PAD]', '[UNK]', '[EOS]', 'user', 'assistant', 'synthetic', 'alpha', 'beta', 'gamma']
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                       unk_token='[UNK]', eos_token='[EOS]')
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                        head_dim=32, max_position_embeddings=256, tie_word_embeddings=True,
                        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id)
    Qwen3ForCausalLM(config).save_pretrained(output)
    tokenizer.save_pretrained(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/tiny-model'))
    make_model(parser.parse_args().output_dir)
