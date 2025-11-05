import random
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset

from models.Tokenizer import BPETokenizer

# This system prompt is used in the Alpaca and UltraChat
SYSTEM_PROMPT = "You are a helpful assistant."
# This system prompt is used in the Open-R1-Math
REASONING_SYSTEM_PROMPT = (
    "You are an AI assistant. Solve the problem step by step inside "
    "<think> ... </think>, "
    "then provide only the final answer inside <answer> ... </answer>."
)


def _padding(ids: list[int], pad_v: int, max_seq_len: int) -> list[int]:
    if len(ids) < max_seq_len:
        padding = [pad_v] * (max_seq_len - len(ids))
        return ids + padding

    return ids


class ChatRenderer:
    """
    The basic chat renderer to make the text consistent
    """

    def __init__(self, tokenizer: BPETokenizer, add_bos=True, add_eot=True):
        self.tokenizer = tokenizer
        self.add_bos = add_bos
        self.add_eot = add_eot

        self.user_token = tokenizer.get_user_tag()
        self.assistant_token = tokenizer.get_assistant_tag()
        self.system_token = tokenizer.get_system_tag()
        self.eot_token = tokenizer.get_eot_tag()
        self.bos_token = tokenizer.get_bos_tag()
        self.eos_token = tokenizer.get_eos_tag()

        self.pad_id = tokenizer.pad_id
        self.eos_id = tokenizer.eos_id

    def chat_input(self, messages: list[dict[str, str]], thinking: bool) -> list[int]:
        prev_eot = self.add_eot  # store current state
        self.add_eot = False  # Always false

        ids = self.render(messages)[0]
        assistant_role = f"{self.assistant_token}\n"
        if thinking:
            assistant_role += "<think>"
        role_ids = self.tokenizer.encode_ids(assistant_role)

        full_ids = ids + role_ids

        self.add_eot = prev_eot

        return full_ids

    def render(self, messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
        """
        Format:
            <|begin_of_text|>
            <|system|>\n...content...\n<|end_of_turn|>\n
            <|user|>\n...content...\n<|end_of_turn|>\n
            <|assistant|>\n...content...\n<|end_of_turn|>\n
            ...
            <|end_of_text|>
        Return:
            ids (full conversation)
            masks (`1` in assistant parts)
        """
        ids: list[int] = []
        mask: list[int] = []

        if self.add_bos:
            bos = self.tokenizer.encode_ids(self.bos_token, add_bos=False, add_eos=False)
            ids.extend(bos)
            mask.extend([0] * len(bos))

        for m in messages:
            role = m["role"]
            content = m["content"]

            role_prefix = f"{role}\n"
            eot_suffix = f"{self.eot_token}\n"

            # role token + newline + content + newline + EOT + newline
            seg_str = role_prefix + content + eot_suffix
            seg_ids = self.tokenizer.encode_ids(seg_str, add_bos=False, add_eos=False)
            ids.extend(seg_ids)

            role_ids = self.tokenizer.encode_ids(role_prefix, add_bos=False, add_eos=False)
            eot_ids = self.tokenizer.encode_ids(eot_suffix, add_bos=False, add_eos=False)

            if role == self.assistant_token:
                content_len = len(seg_ids) - len(role_ids) - len(eot_ids)
                mask.extend([0] * len(role_ids))  # filter role
                mask.extend([1] * max(0, content_len))  # i.e., the response from assistant
                mask.extend([0] * len(eot_ids))
            else:
                mask.extend([0] * len(seg_ids))  # others

        if self.add_eot:
            e = self.tokenizer.encode_ids(self.eos_token, add_bos=False, add_eos=False)
            ids.extend(e)
            mask.extend([0] * len(e))

        return ids, mask

    def make_labels(self, ids: list[int], mask: list[int], max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Truncate based on the max_seq_len
        ids = ids[:max_seq_len]
        mask = mask[:max_seq_len]

        eot_id = self.tokenizer.eot_id
        eos_id = self.tokenizer.eos_id
        specials = {eos_id, self.pad_id}

        # Shift labels by 1
        labels = [-100] * len(ids)
        for i in range(len(ids) - 1):
            if mask[i] != 1:
                continue
            target = ids[i + 1]

            if target == eot_id:
                labels[i] = eot_id
            elif target in specials:
                labels[i] = -100
            else:
                labels[i] = target

        ids = _padding(ids, self.pad_id, max_seq_len)
        labels = _padding(labels, -100, max_seq_len)

        return torch.LongTensor(ids), torch.LongTensor(labels)


class AlpacaDataset(Dataset):
    """
    I used publicly available (HF) cleaned alpaca dataset
    """

    def __init__(self, tokenizer: BPETokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.dataset = load_dataset("yahma/alpaca-cleaned", split="train")

        self.renderer = ChatRenderer(tokenizer=tokenizer, add_bos=True, add_eot=True)
        self.user_token = tokenizer.get_user_tag()
        self.assistant_token = tokenizer.get_assistant_tag()
        self.system_token = tokenizer.get_system_tag()

        print(f"[Alpaca] System: `{self.system_token}` | User `{self.user_token}` | Assistant: `{self.assistant_token}")

    def _format_instruction(self, instruction: str, input: str | None) -> str:
        if input is not None and len(input) > 0:
            content = f"{instruction}\n\n{input}"
        else:
            content = f"{instruction}"

        return content

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data: dict[str, str] = self.dataset[idx]  # type: ignore

        instrunction = data.get("instruction") or ""
        input = data.get("input")
        output = data.get("output") or ""

        user_content = self._format_instruction(instruction=instrunction, input=input)

        messages = [
            {"role": self.system_token, "content": SYSTEM_PROMPT},
            {"role": self.user_token, "content": user_content},
            {"role": self.assistant_token, "content": output},
        ]

        ids, mask = self.renderer.render(messages=messages)
        input_ids, labels = self.renderer.make_labels(ids=ids, mask=mask, max_seq_len=self.max_seq_len)

        return {"input_ids": input_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore


class DollyDataset(Dataset):
    """
    Category is also used for system prompt
    """

    def __init__(self, tokenizer: BPETokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

        self.renderer = ChatRenderer(tokenizer=tokenizer, add_bos=True, add_eot=True)
        self.user_token = tokenizer.get_user_tag()
        self.assistant_token = tokenizer.get_assistant_tag()
        self.system_token = tokenizer.get_system_tag()

        print(f"[Dolly] System: `{self.system_token}` | User `{self.user_token}` | Assistant: `{self.assistant_token}")

    def _format_instruction(self, instruction: str, context: str) -> str:
        if len(context) > 0:
            content = f"{instruction}\n\nContext:\n{context}"
        else:
            content = instruction

        return content

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data: dict[str, str] = self.dataset[idx]  # type: ignore

        instrunction = (data.get("instruction") or "").strip()
        context = (data.get("context") or "").strip()
        response = (data.get("response") or "").strip()

        # get category to add system prompt
        category = data.get("category")
        system_prompt = f"You are assisting {category} tasks."

        user_content = self._format_instruction(instruction=instrunction, context=context)

        messages = [
            {"role": self.system_token, "content": system_prompt},
            {"role": self.user_token, "content": user_content},
            {"role": self.assistant_token, "content": response},
        ]

        ids, mask = self.renderer.render(messages=messages)
        input_ids, labels = self.renderer.make_labels(ids=ids, mask=mask, max_seq_len=self.max_seq_len)

        return {"input_ids": input_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore


class UltraChatDataset(Dataset):
    def __init__(self, tokenizer: BPETokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.dataset = load_dataset("HuggingFaceH4/ultrachat_200k", "default")["train_sft"]

        self.renderer = ChatRenderer(tokenizer=tokenizer, add_bos=True, add_eot=True)
        self.user_token = tokenizer.get_user_tag()
        self.assistant_token = tokenizer.get_assistant_tag()
        self.system_token = tokenizer.get_system_tag()

        print(
            f"[UltraChat] System: `{self.system_token}` | User `{self.user_token}`| Assistant: `{self.assistant_token}`"
        )

    def _clean_multiturn_conversation(self, conversations: list[dict[str, str]]) -> list[dict[str, str]]:
        """Clean roles"""
        clean_chats = []

        for turn in conversations:
            role = (turn.get("role") or turn.get("from") or "").lower()
            content = (turn.get("content") or turn.get("value") or "").strip()

            if role in ["human", "user", "prompter"]:
                role = self.user_token

            elif role in ["gpt", "assistant", "bot", "model"]:
                role = self.assistant_token

            elif "system" in role:
                role = self.system_token

            else:
                role = self.user_token

            clean_chats.append({"role": role, "content": content})

        return clean_chats

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data: dict[str, list[dict[str, str]]] = self.dataset[idx]  # type: ignore

        # Clean the conversation
        conversation = self._clean_multiturn_conversation(data["messages"])

        # add system prompt
        messages: list[dict[str, str]] = [{"role": self.system_token, "content": SYSTEM_PROMPT}]
        messages.extend(conversation)

        ids, mask = self.renderer.render(messages=messages)
        input_ids, labels = self.renderer.make_labels(ids=ids, mask=mask, max_seq_len=self.max_seq_len)

        return {"input_ids": input_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore


class OpenR1MathDataset(Dataset):
    """
    Experimental.
    Instead of pure chat format, convert messages for reasoning model
    """

    def __init__(self, tokenizer: BPETokenizer, max_seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Currently "default". Other options, like "all" can be possible
        self.dataset = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")

        self.renderer = ChatRenderer(tokenizer=tokenizer, add_bos=True, add_eot=True)
        self.user_token = tokenizer.get_user_tag()
        self.assistant_token = tokenizer.get_assistant_tag()
        self.system_token = tokenizer.get_system_tag()

        print(
            f"[OpenR1-Math Data] System: `{self.system_token}` | User `{self.user_token}`| Assistant: `{self.assistant_token}`"
        )

    def _format_reasoning(self, problem: str, solution: str, answer: str) -> list[dict[str, str]]:
        """
        Convert to reasoning format.
        Note that, we don't want to inject <think></think> tags in ideal case.
        The reasoning will be emerged via post-training but here, to introduce reasoning ability
        """
        messages = []

        system_msg = {"role": self.system_token, "content": REASONING_SYSTEM_PROMPT}
        user_msg = {"role": self.user_token, "content": problem}

        messages.append(system_msg)
        messages.append(user_msg)

        # Formatting reasoning
        reasoning_answer = f"<think>{solution}</think>\n<answer>{answer}</answer>"
        assistant_msg = {"role": self.assistant_token, "content": reasoning_answer}

        messages.append(assistant_msg)

        return messages

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data: dict[str, str] = self.dataset[idx]  # type: ignore

        # Format reasoning
        messages = self._format_reasoning(problem=data["problem"], solution=data["solution"], answer=data["answer"])

        ids, mask = self.renderer.render(messages=messages)
        input_ids, labels = self.renderer.make_labels(ids=ids, mask=mask, max_seq_len=self.max_seq_len)

        return {"input_ids": input_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore


@dataclass(frozen=True)
class MixedSFTDatasetConfig:
    name: str
    weight: float


class MixedSFTDataset(IterableDataset):
    """
    Handling multiple sources in SFT
    """

    def __init__(self, tokenizer: BPETokenizer, max_seq_len: int, cfgs: list[MixedSFTDatasetConfig]) -> None:
        # set ratio
        self.weights = {cfg.name: cfg.weight for cfg in cfgs}

        self.datasets: dict[str, Dataset] = {}
        self._lengths: dict[str, int] = {}
        self._total_size = 0

        for name in self.weights.keys():
            self.datasets[name] = self._get_dataset(name=name, tokenizer=tokenizer, max_seq_len=max_seq_len)
            self._lengths[name] = len(self.datasets[name])  # type: ignore
            self._total_size += len(self.datasets[name])  # type: ignore

    def _get_dataset(self, name: str, tokenizer: BPETokenizer, max_seq_len: int) -> Dataset:
        if name == "Alpaca":
            dataset = AlpacaDataset(tokenizer=tokenizer, max_seq_len=max_seq_len)
        elif name == "Dolly":
            dataset = DollyDataset(tokenizer=tokenizer, max_seq_len=max_seq_len)
        elif name == "UltraChat":
            dataset = UltraChatDataset(tokenizer=tokenizer, max_seq_len=max_seq_len)
        elif name == "OpenR1Math":
            dataset = OpenR1MathDataset(tokenizer=tokenizer, max_seq_len=max_seq_len)
        else:
            raise ValueError("Unexpected dataset name")

        return dataset

    def __iter__(self):
        iters = {key: iter(ds) for key, ds in self.datasets.items()}
        remain = {k: self._lengths[k] for k in self.datasets.keys()}

        keys = list(self.weights.keys())
        weights = [float(self.weights[k]) for k in keys]

        active_ds = set(keys)

        while active_ds:
            idx = [i for i, k in enumerate(keys) if k in active_ds]
            active_keys = [keys[i] for i in idx]
            active_weights = [weights[i] for i in idx]

            sampled = random.choices(population=active_keys, weights=active_weights, k=1)[0]

            try:
                item = next(iters[sampled])

                remain[sampled] -= 1
                if remain[sampled] <= 0:
                    active_ds.remove(sampled)

                yield item
            except StopIteration:
                active_ds.discard(sampled)

    def __len__(self) -> int:
        return self._total_size


"""used only debugging purpose"""
if __name__ == "__main__":
    # Tokenizer
    tokenizer = BPETokenizer("tokenizer")

    configs = [
        MixedSFTDatasetConfig(name="Alpaca", weight=0.5),
        MixedSFTDatasetConfig(name="Dolly", weight=0.5),
    ]
    dataset = MixedSFTDataset(tokenizer=tokenizer, max_seq_len=512, cfgs=configs)
    dataset = OpenR1MathDataset(tokenizer=tokenizer, max_seq_len=512)
    dataloader = DataLoader(dataset, 1, num_workers=1)

    for i, data in enumerate(dataloader):
        if i == 5:
            break
        print("-" * 10)
        print(data["input_ids"].size())
        input_ids = data["input_ids"][0].tolist()
        print(tokenizer.decode_ids(input_ids, skip_specials=False))

        # check one-step shift
        for input_id, target in zip(input_ids, data["labels"][0].tolist()):
            print(f"{input_id}, {target}")
