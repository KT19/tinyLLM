import math
import os
import random

from my_datasets.pretrain_dataset import MixedDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from dataclasses import dataclass
from itertools import islice

import numpy as np
import schedulefree
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from models.Tokenizer import BPETokenizer
from utils import build_model


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    device: str
    tokenizer_dir: str
    total_steps: int
    micro_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    seq_schedule: list[int]
    dataset_configs: list[str]
    clip_norm_schedule: list[float]
    tokens_per_step: int
    log_every: int
    ckpt_every: int
    out_dir: str


def train(cfg: TrainConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"

    # Tokenizer
    tokenizer = BPETokenizer(cfg.tokenizer_dir)
    vocab = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab}")

    # build model
    model = build_model(config_path="configs/model_config.yaml")
    model.to(device)

    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, warmup_steps=cfg.warmup_steps
    )
    scaler = torch.amp.GradScaler(cfg.device)  # type: ignore

    # set data
    dataset_configs = cfg.dataset_configs
    config_prefix = "configs/pretraining/"
    mixed_dataset = MixedDataset(
        tokenizer=tokenizer, max_seq_len=1024, cfg_path=config_prefix + "dataset_stage1.yaml"
    )
    dataloader = DataLoader(mixed_dataset, batch_size=cfg.micro_batch_size, num_workers=1)
    data_itr = islice(dataloader, 1, None)

    # seq len schedule
    seq_schedule = cfg.seq_schedule
    total_steps = cfg.total_steps
    steps_per_stage = total_steps // len(seq_schedule)  # for curriculum

    # hyperparameter schedule
    clip_norm_schedule = cfg.clip_norm_schedule

    global_step = 0
    model.train()
    optimizer.train()
    start_time = time.time()

    for stage, (config, cur_seq, clip_norm) in enumerate(zip(dataset_configs, seq_schedule, clip_norm_schedule)):
        print(f"\n=== Pretraining Stage {stage + 1}/{len(seq_schedule)}: seq_len={cur_seq} ===\n")

        grad_accum = max(1, cfg.tokens_per_step // (cfg.micro_batch_size * cur_seq))
        print(f"Grad accum: {grad_accum}, tokens / step target: {cfg.tokens_per_step}")

        mixed_dataset.update_dataset_config(config_prefix + config)
        mixed_dataset.update_max_seq_len_all(cur_seq)

        stage_start_time = time.time()
        while global_step < total_steps and (global_step - stage * steps_per_stage) < steps_per_stage:
            optimizer.zero_grad()
            loss_accum = 0.0
            global_step += 1

            for _ in range(grad_accum):
                try:
                    batch_data = next(data_itr)
                    input_ids, labels = batch_data["input_ids"], batch_data["labels"]
                except StopIteration:
                    data_itr = islice(dataloader, 1, None)
                    batch_data = next(data_itr)

                    input_ids, labels = batch_data["input_ids"], batch_data["labels"]

                input_ids, labels = input_ids.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                with torch.amp.autocast(cfg.device, dtype=torch.bfloat16):  # type: ignore
                    logits = model(input_ids)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

                scaler.scale(loss / grad_accum).backward()
                loss_accum += loss.item()

            # step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()

            if global_step > 0 and global_step % cfg.log_every == 0:
                avg_loss = loss_accum / grad_accum
                ppl = math.exp(min(avg_loss, 20))  # avoid overflow
                elapsed_global = time.time() - start_time
                elapsed_stage = time.time() - stage_start_time
                print(
                    f"Step {global_step} / {total_steps} | "
                    f"seq={cur_seq} | "
                    f"loss={avg_loss:.4f} | "
                    f"perplexity={ppl:.2f} | "
                    f"stage elapsed: {elapsed_stage / 60:.2f} min | "
                    f"global elapsed: {elapsed_global / 60:.2f} min "
                )

            if (global_step % cfg.ckpt_every == 0 and global_step > 0) or global_step == total_steps:
                os.makedirs(cfg.out_dir, exist_ok=True)
                save_path = os.path.join(cfg.out_dir, f"pretrain_step_{global_step}.pt")
                torch.save(
                    {
                        "step": global_step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                print(f"model saved to {save_path}")

            if global_step > total_steps:
                break

    # Final save
    os.makedirs(cfg.out_dir, exist_ok=True)
    save_path = os.path.join(cfg.out_dir, "pretrain_final.pt")
    torch.save({"step": global_step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, save_path)
    print("Done")


if __name__ == "__main__":
    with open("configs/train_config.yaml", "r") as f:
        yaml_data = yaml.safe_load(f)

    trn_cfg = TrainConfig(**yaml_data)
    train(trn_cfg)
