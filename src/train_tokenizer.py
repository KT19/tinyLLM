import json
import os

from models.Tokenizer import SPECIALS

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any

from datasets import load_from_disk
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


def train_tokenizer(output_dir: str, vocab_size: int, datasets: list[tuple[str, Any]], max_length: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()  # type: ignore
    tokenizer.decoder = decoders.ByteLevel()  # type: ignore
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)  # type: ignore

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    def batch_iterator(max_length: int):
        for key, dataset in datasets:
            counter = 0
            for ex in dataset:
                txt = ex.get(key)
                counter += 1
                if counter == max_length:
                    break
                if txt:
                    yield txt

    tokenizer.train_from_iterator(batch_iterator(max_length), trainer=trainer)

    save_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(save_path)
    meta = {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "system_token": "<|system|>",
        "user_token": "<|user|>",
        "assistant_token": "<|assistant|>",
        "end_of_turn_token": "<|end_of_turn|>",
        "sep_token": "<|sep|>",
        "begin_of_think_token": "<think>",
        "end_of_think_token": "</think>",
        "begin_of_answer_token": "<answer>",
        "end_of_answer_token": "</answer>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": 131072,
    }

    with open(os.path.join(output_dir, "tokenizer_meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"Saved tokenizer to {output_dir}")


def main():
    datasets = [
        ("text", load_from_disk("dataset_source/open-web-math_5gb/final")),
        ("text", load_from_disk("dataset_source/fineweb_5gb/final")),
    ]
    train_tokenizer(output_dir="tokenizer", vocab_size=32000, datasets=datasets, max_length=50000)


if __name__ == "__main__":
    main()
