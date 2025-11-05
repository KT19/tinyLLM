from dataclasses import dataclass

import torch
import yaml

from models.Tokenizer import BPETokenizer
from my_datasets.sft_dataset import ChatRenderer
from utils import build_model


@dataclass
class ChatConfig:
    system_prompt: str
    seed: int
    device: str
    tokenizer_dir: str
    model_path: str
    max_new_token: int
    temperature: float
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0


def generate(cfg: ChatConfig):
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    # Tokenizer
    tokenizer = BPETokenizer(cfg.tokenizer_dir)
    vocab = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab}")

    # Get terminate tokens
    eot_id = tokenizer.eot_id

    # build model
    model = build_model(config_path="configs/model_config.yaml")
    # load model
    checkpoint = torch.load(cfg.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    print(f"load model from {cfg.model_path}")

    chat_render = ChatRenderer(tokenizer=tokenizer)
    messages = [{"role": tokenizer.get_system_tag(), "content": cfg.system_prompt}]
    while True:
        user_input = input("You: -> ")

        message = {"role": tokenizer.get_user_tag(), "content": user_input}
        messages.append(message)  # Add as an message

        input_ids = chat_render.chat_input(messages=messages, thinking=False)
        input_ids_size = len(input_ids)
        print("--" * 10)
        input_ids = torch.LongTensor(input_ids).to(device)
        # Add sequentially
        for _ in range(cfg.max_new_token):
            # There is an efficient inference method, like KV-cache
            next_token = model.generate(
                input_ids=input_ids,
                max_new_tokens=1,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
            )[-1]

            next_token_tensor = torch.LongTensor([next_token]).to(cfg.device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=0)

            if next_token == eot_id:
                break

        conversation = input_ids.tolist()
        response = tokenizer.decode_ids(conversation[input_ids_size:], skip_specials=False).lstrip()
        print(f"assistant -> {response}")
        messages.append({"role": tokenizer.get_assistant_tag(), "content": response})
        ## Debuggin purpose
        # print("=" * 10 + "Chat history" + "=" * 10)
        # print(tokenizer.decode_ids(conversation, skip_specials=False))


if __name__ == "__main__":
    with open("configs/chat_config.yaml", "r") as f:
        yaml_data = yaml.safe_load(f)

    gen_cfg = ChatConfig(**yaml_data)
    generate(gen_cfg)
