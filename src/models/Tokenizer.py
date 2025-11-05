import json

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors

SPECIALS = [
    "<pad>",
    "<unk>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_of_turn|>",
    "<|sep|>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
]


class BPETokenizer:
    def __init__(self, tokenizer_dir: str = "tokenizer"):
        # setup tokenizer
        self.tokenizer = Tokenizer.from_file(f"{tokenizer_dir}/tokenizer.json")

        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)  # type: ignore

        meta = json.load(open(f"{tokenizer_dir}/tokenizer_meta.json"))

        self._pad = self.tokenizer.token_to_id(meta["pad_token"])
        self._unk = self.tokenizer.token_to_id(meta["unk_token"])
        self._unk_str = meta.get("unk_token", "<unk>")
        self._bos = self.tokenizer.token_to_id(meta["bos_token"])
        self._eos = self.tokenizer.token_to_id(meta["eos_token"])
        self._eot = self.tokenizer.token_to_id(meta["end_of_turn_token"])

        self.model_max_length = meta.get("model_max_length")

        self._vocab = self.tokenizer.get_vocab()
        self._id2tok = {i: t for t, i in self._vocab.items()}
        self._special_tokens = {v for k, v in meta.items() if "token" in k}
        self._special_ids = {
            self.tokenizer.token_to_id(s) for s in self._special_tokens if self.tokenizer.token_to_id(s) is not None
        }

    def get_sep_tag(self) -> str:
        return "<|sep|>"

    def get_user_tag(self) -> str:
        return "<|user|>"

    def get_assistant_tag(self) -> str:
        return "<|assistant|>"

    def get_bos_tag(self) -> str:
        return "<|begin_of_text|>"

    def get_eos_tag(self) -> str:
        return "<|end_of_text|>"

    def get_eot_tag(self) -> str:
        return "<|end_of_turn|>"

    def get_system_tag(self) -> str:
        return "<|system|>"

    def encode_ids(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """
        Encode input (string) -> ids
        """
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [self._bos] + ids
        if add_eos:
            ids = ids + [self._eos]
        return ids

    def decode_ids(self, ids: list[int], skip_specials: bool = True, remove_specials_id: list[int] | None = [0]) -> str:
        """
        Decode input (ids) -> string
        """
        if remove_specials_id:
            ids = [i for i in ids if i not in remove_specials_id]

        text = self.tokenizer.decode(ids, skip_special_tokens=skip_specials)

        return text

    def id2token(self, idx: int) -> str:
        """
        Single step mapping
        id -> token
        """
        try:
            return self.tokenizer.id_to_token(idx)
        except Exception:
            print(f"{idx} cannot be converted to token. Return <unk>")
            return self._id2tok.get(idx, self._unk_str)

    def token2id(self, token: str) -> int:
        """
        Single step mapping
        token -> id
        """
        id = self.tokenizer.token_to_id(token)
        if id is None:
            print(f"{token} cannot be converted to id. Return <unk> id")
            return self._unk

        return id

    def get_vocab_size(self) -> int:
        return len(self._id2tok)

    @property
    def pad_id(self) -> int:
        return self._pad

    @property
    def bos_id(self) -> int:
        return self._bos

    @property
    def eos_id(self) -> int:
        return self._eos

    @property
    def eot_id(self) -> int:
        return self._eot
