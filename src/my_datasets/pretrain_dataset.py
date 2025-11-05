import abc
import ctypes
import random
from dataclasses import dataclass
from itertools import islice
from multiprocessing import Manager, Value
from typing import Any

import torch
import yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader, IterableDataset

from models.Tokenizer import BPETokenizer


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    weight: float


class SharedConfig:
    def __init__(self, init_len: int, init_weights: dict[str, float]):
        self.max_len = Value(ctypes.c_int, init_len)
        mgr = Manager()
        self.weights = mgr.dict(init_weights)


class PretrainDatasetBase(abc.ABC, IterableDataset):
    """
    Base class. All dataset used in pretraining inherit this base class.
    """

    def __init__(self, tokenizer: BPETokenizer, shared: SharedConfig):
        self.tokenizer = tokenizer
        self.shared = shared
        self.dataset: Any

    def update_max_seq_len(self, max_seq_len: int) -> None:
        self.shared.max_len.value = max_seq_len
        print(f"updated to {max_seq_len}")

    @abc.abstractmethod
    def _parse(self, data: Any) -> str | None:
        raise NotImplementedError

    def __iter__(self):
        buf = []

        for data in self.dataset:
            text = self._parse(data)
            if text is None:
                continue
            ids = self.tokenizer.encode_ids(text, add_bos=False, add_eos=True)
            buf.extend(ids)

            while True:
                current_len = int(self.shared.max_len.value)
                if len(buf) < current_len + 1:
                    break
                x = buf[:current_len]
                y = buf[1 : current_len + 1]
                buf = buf[current_len:]
                yield {"input_ids": torch.LongTensor(x), "labels": torch.LongTensor(y)}


class MixedDataset(IterableDataset):
    """
    In practical, multiple data sources will be used.
    This class handles such use-case.
    """

    def __init__(self, tokenizer: Any, max_seq_len: int, cfg_path: str) -> None:
        self.datasets: dict[str, PretrainDatasetBase] = {}

        with open(cfg_path, "r") as f:
            configs = yaml.safe_load(f)
            init_weights = {c["name"]: float(c["weight"]) for c in configs}

        self.shared = SharedConfig(init_len=max_seq_len, init_weights=init_weights)

        for name in init_weights.keys():
            self.datasets[name] = self._get_dataset(name=name, tokenizer=tokenizer, shared=self.shared)

    def _get_dataset(self, name: str, tokenizer: Any, shared: SharedConfig) -> PretrainDatasetBase:
        if name == "HuggingFaceFW/fineweb-edu":
            dataset = FineWebEduDataset(tokenizer=tokenizer, shared=shared)
        elif name == "open-web-math/open-web-math":
            dataset = OpenWebMathDataset(tokenizer=tokenizer, shared=shared)
        else:
            raise ValueError("Unexpected dataset name")

        return dataset

    def update_dataset_config(self, cfg_path: str) -> None:
        with open(cfg_path, "r") as f:
            configs = yaml.safe_load(f)

            # read each dataset's name and mixed ratio
            for config in configs:
                cfg = DatasetConfig(name=config["name"], weight=config["weight"])
                self.shared.weights[cfg.name] = float(cfg.weight)

        print(f"Updated mixed data ratio using {cfg_path}")

    def update_max_seq_len_all(self, max_seq_len: int) -> None:
        self.shared.max_len.value = max_seq_len

    def __iter__(self):
        iters = {key: iter(ds) for key, ds in self.datasets.items()}

        while True:
            keys = list(self.shared.weights.keys())
            weights = [float(self.shared.weights[k]) for k in keys]
            sampled = random.choices(population=keys, weights=weights, k=1)[0]

            try:
                yield next(iters[sampled])
            except StopIteration:
                iters[sampled] = iter(self.datasets[sampled])


class FineWebEduDataset(PretrainDatasetBase):
    def __init__(self, tokenizer: Any, shared: SharedConfig) -> None:
        super().__init__(tokenizer=tokenizer, shared=shared)
        self.dataset = load_from_disk("dataset_source/fineweb_5gb/final")
        print(f"Fineweb Edu: data set len is {len(self.dataset)}")

    def _parse(self, data: Any) -> str | None:
        text = data.get("text")
        if not text:
            return None

        return text


class OpenWebMathDataset(PretrainDatasetBase):
    def __init__(self, tokenizer: Any, shared: SharedConfig) -> None:
        super().__init__(tokenizer=tokenizer, shared=shared)
        self.dataset = load_from_disk("dataset_source/open-web-math_5gb/final")
        print(f"Open-Web-Math: data set len is {len(self.dataset)}")

    def _parse(self, data: Any) -> str | None:
        text = data.get("text")
        if not text:
            return None

        return text


"""Used only debugging purpose"""
if __name__ == "__main__":
    dataset_configs = ["dataset_stage1.yaml", "dataset_stage2.yaml", "dataset_stage3.yaml"]
    seq_lens = [128, 512, 1024]

    # Tokenizer
    tokenizer = BPETokenizer("tokenizer")
    config_prefix = "configs/pretraining/"

    dataset = MixedDataset(tokenizer=tokenizer, max_seq_len=16, cfg_path=config_prefix + "dataset_stage1-2.yaml")
    dataloader = DataLoader(dataset, 1, num_workers=1)
    data_itr = islice(dataloader, 1, None)

    for config, seq_len in zip(dataset_configs, seq_lens):
        print("-" * 10 + f"{config}" + "-" * 10)
        dataset.update_dataset_config(config_prefix + config)
        dataset.update_max_seq_len_all(seq_len)

        data = next(data_itr)

        print(f"input -> {data['input_ids'][0]}")
        print(f"labels -> {data['labels'][0]}")
        print(tokenizer.decode_ids(data["input_ids"][0], skip_specials=False))
