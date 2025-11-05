from dataclasses import dataclass

import torch
import yaml

from models.Tokenizer import BPETokenizer
from utils import build_model


@dataclass
class GenerateConfig:
    seed: int
    device: str
    tokenizer_dir: str
    model_path: str
    max_new_token: int
    temperature: float
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0


def generate(cfg: GenerateConfig):
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    # Tokenizer
    tokenizer = BPETokenizer(cfg.tokenizer_dir)
    vocab = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab}")

    # build model
    model = build_model(config_path="configs/model_config.yaml")
    # load model
    checkpoint = torch.load(cfg.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    print(f"load model from {cfg.model_path}")

    test_inputs = [
        "The future of artificiall intelligence",
        "What is",
    ]
    for tst_input in test_inputs:
        input_ids = tokenizer.encode_ids(tst_input)
        input_ids = torch.LongTensor(input_ids).to(device)
        print("-" * 10)
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=cfg.max_new_token,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
        )
        print(f"{tokenizer.decode_ids(generated)}")


if __name__ == "__main__":
    with open("configs/pretrain_generate_config.yaml", "r") as f:
        yaml_data = yaml.safe_load(f)

    gen_cfg = GenerateConfig(**yaml_data)
    generate(gen_cfg)
