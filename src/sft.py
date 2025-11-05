import math
import os
import random

from my_datasets.sft_dataset import AlpacaDataset, MixedSFTDataset, MixedSFTDatasetConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from dataclasses import dataclass

import numpy as np
import schedulefree
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from models.Tokenizer import BPETokenizer
from utils import build_model


@dataclass(frozen=True)
class SFTConfig:
    seed: int
    device: str
    tokenizer_dir: str
    epochs: int
    micro_batch_size: int
    learning_rate: float
    weight_decay: float
    max_seq_len: int
    clip_norm: float
    tokens_per_step: int
    log_every_update: int
    save_per_step: int
    base_model_ckpt: str
    out_dir: str

    # Optional. Used on Mixed dataset
    mixed_dataset_config: list[MixedSFTDatasetConfig] | None = None
    early_terminate_step: int | None = None


def train(cfg: SFTConfig, dataset: Dataset):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"

    # build model
    model = build_model(config_path="configs/model_config.yaml")

    # load model
    checkpoint = torch.load(cfg.base_model_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(cfg.device)  # type: ignore

    # set dataloader
    dataloader = DataLoader(dataset, batch_size=cfg.micro_batch_size, num_workers=1)

    global_step = 0
    model.train()
    optimizer.train()
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1, 1):
        terminate = False
        grad_accum = max(1, cfg.tokens_per_step // (cfg.micro_batch_size * cfg.max_seq_len))
        print(
            f"Epoch: {epoch}/{cfg.epochs}, loader length: {len(dataloader)} Grad accum: {grad_accum}, tokens / step target: {cfg.tokens_per_step}"
        )

        optimizer.zero_grad()
        loss_accum = 0

        for step, data in enumerate(dataloader):
            global_step += 1

            input_ids, labels = (
                data["input_ids"].to(device, non_blocking=True),
                data["labels"].to(device, non_blocking=True),
            )

            with torch.amp.autocast(cfg.device, dtype=torch.bfloat16):  # type: ignore
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

            scaler.scale(loss / grad_accum).backward()
            loss_accum += loss.item()

            if ((step + 1) % grad_accum == 0) or ((step + 1) == len(dataloader)):
                # step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if global_step > 0 and global_step % cfg.log_every_update == 0:
                avg_loss = loss_accum / cfg.log_every_update  # average
                ppl = math.exp(min(avg_loss, 20))  # avoid overflow
                elapsed_global = time.time() - start_time
                print(
                    f"global_step = {global_step} | loss={avg_loss:.4f} | perplexity={ppl:.2f} | global elapsed: {elapsed_global / 60:.2f} min "
                )
                loss_accum = 0

            if global_step > 0 and global_step % cfg.save_per_step == 0:
                os.makedirs(cfg.out_dir, exist_ok=True)
                save_path = os.path.join(cfg.out_dir, f"sft_step_{global_step}.pt")
                torch.save(
                    {
                        "step": global_step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                print(f"model saved to {save_path}")

            if cfg.early_terminate_step is not None and global_step > cfg.early_terminate_step:
                print(f"End of training because global step reaches {cfg.early_terminate_step}")
                terminate = True
                break

        if terminate:
            break

    # Final save
    os.makedirs(cfg.out_dir, exist_ok=True)
    save_path = os.path.join(cfg.out_dir, "sft_final.pt")
    torch.save({"epoch": cfg.epochs, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, save_path)
    print("Done")


def load_config(path: str) -> SFTConfig:
    with open(path, "r") as f:
        yaml_data = yaml.safe_load(f)

    mixed_list = None
    if "mixed_dataset_config" in yaml_data:
        mixed_list = []
        for item in yaml_data["mixed_dataset_config"]:
            if "mixed_dataset" in item:
                cfg = item["mixed_dataset"]
                mixed_list.append(MixedSFTDatasetConfig(**cfg))

    return SFTConfig(
        seed=yaml_data["seed"],
        device=yaml_data["device"],
        tokenizer_dir=yaml_data["tokenizer_dir"],
        epochs=yaml_data["epochs"],
        micro_batch_size=yaml_data["micro_batch_size"],
        learning_rate=yaml_data["learning_rate"],
        weight_decay=yaml_data["weight_decay"],
        max_seq_len=yaml_data["max_seq_len"],
        clip_norm=yaml_data["clip_norm"],
        tokens_per_step=yaml_data["tokens_per_step"],
        log_every_update=yaml_data["log_every_update"],
        save_per_step=yaml_data["save_per_step"],
        base_model_ckpt=yaml_data["base_model_ckpt"],
        out_dir=yaml_data["out_dir"],
        mixed_dataset_config=mixed_list,
        early_terminate_step=yaml_data.get("early_terminate_step"),
    )


if __name__ == "__main__":
    # Phase-1
    trn_cfg = load_config(path="configs/post_training/sft-stage1.yaml")
    tokenizer = BPETokenizer(trn_cfg.tokenizer_dir)
    print("-" * 10 + "Stage-1 SFT" + "-" * 10)
    dataset = AlpacaDataset(tokenizer=tokenizer, max_seq_len=trn_cfg.max_seq_len)
    train(cfg=trn_cfg, dataset=dataset)

    # Phase-2
    trn_cfg = load_config(path="configs/post_training/sft-stage2.yaml")
    tokenizer = BPETokenizer(trn_cfg.tokenizer_dir)
    assert trn_cfg.mixed_dataset_config is not None, "Should be given"
    print("-" * 10 + "Stage-2 SFT" + "-" * 10)
    dataset = MixedSFTDataset(tokenizer=tokenizer, max_seq_len=trn_cfg.max_seq_len, cfgs=trn_cfg.mixed_dataset_config)
    train(cfg=trn_cfg, dataset=dataset)
