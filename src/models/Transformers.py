from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TransformerBlock import TransformerBlock
from modules.CustomHRM import HRM, HRMState
from modules.RMSNorm import RMSNorm
from modules.RoPE import RoPE


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    dim: int
    n_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    expansion: float
    rope_theta: float
    rope_scale: float
    max_seq_len: int
    window_size: int
    attn_dropout: float
    residual_dropout: float
    # FOR custom HRM
    hrm_modules: int
    hrm_N: int
    hrm_T: int


class TinyLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # tie embedding
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        rope = RoPE(rope_dim=cfg.head_dim, base_theta=cfg.rope_theta, scaling=cfg.rope_scale)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=cfg.dim,
                    num_heads=cfg.num_heads,
                    num_kv_heads=cfg.num_kv_heads,
                    head_dim=cfg.head_dim,
                    expansion=cfg.expansion,
                    rope=rope,
                    window_size=cfg.window_size,
                    attn_dropout=cfg.attn_dropout,
                    residual_dropout=cfg.residual_dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.latent_reasoning = HRM(
            dim=cfg.dim,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            expansion=cfg.expansion,
            rope=rope,
            window_size=cfg.window_size,
            attn_dropout=cfg.attn_dropout,
            residual_dropout=cfg.residual_dropout,
            hrm_modules=cfg.hrm_modules,
            N=cfg.hrm_N,
            T=cfg.hrm_T,
        )

        self.final_norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # tie embed

    def _reasoning_forward(self, x: torch.Tensor, abs_pos_start: int) -> torch.Tensor:
        state = HRMState(x)

        state = self.latent_reasoning(x, state, abs_pos_start)

        return state.get_zH()

    def forward(self, input_ids: torch.Tensor, abs_pos_start: int | None = None) -> torch.Tensor:
        B, T = input_ids.shape

        x = self.token_embed(input_ids)

        if abs_pos_start is None:
            abs_pos_start = 0

        # dense transformer block
        for block in self.blocks:
            x = block(x, abs_pos_start=abs_pos_start)
        # custom hrm block
        x = self._reasoning_forward(x, abs_pos_start=abs_pos_start)

        # convert to output
        x = self.final_norm(x)

        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> list[int]:
        """
        The return always contains input_ids
        Args:
            input_ids: (T)
        """
        """Adding functions"""

        def apply_repetition_penalty(logits: torch.Tensor, prev_ids: torch.Tensor, penalty: float = 1.1):
            """
            Args:
                logits: (batch_size, vocab)
                prev_ids: (batch_size, length)
            """
            if penalty <= 1.0 or prev_ids.numel() == 0:
                return logits

            logits = logits.clone()

            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, prev_ids.clamp_max(logits.size(-1) - 1), True)

            logits[mask] /= penalty

            return logits

        def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
            if k <= 0 or k >= logits.size(-1):
                return logits

            v, _ = torch.topk(logits, k, dim=-1)
            logits[logits < v[:, [-1]]] = -float("inf")
            return logits

        def top_p_filter(logits: torch.Tensor, p: float, min_keep: int = 1) -> torch.Tensor:
            if p <= 0.0 or p >= 1.0:
                return logits

            # sort
            sorted_logits, sorted_ids = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)

            # mask tokens where cumulative prob exceeds p
            mask = cumprobs > p
            mask[..., :min_keep] = False
            mask = mask.roll(1, dims=-1)
            mask[..., 0] = False

            sorted_logits[mask] = -float("inf")

            logits.scatter_(dim=-1, index=sorted_ids, src=sorted_logits)

            return logits

        """Until HERE"""
        self.eval()
        out = input_ids.unsqueeze(0)

        for _ in range(max_new_tokens):
            if out.size(1) > self.cfg.window_size:
                ctx = out[:, -self.cfg.window_size :]
                abs_pos_start = out.size(1) - ctx.size(1)
            else:
                ctx = out
                abs_pos_start = 0

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # type: ignore
                logits = self(ctx, abs_pos_start)

            logits = logits[:, -1, :] / max(1e-5, temperature)

            if top_k is not None:
                logits = top_k_filter(logits=logits, k=top_k)

            if top_p is not None:
                logits = top_p_filter(logits=logits, p=top_p, min_keep=1)

            logits = apply_repetition_penalty(logits=logits, prev_ids=out, penalty=repetition_penalty)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_id], dim=1)

        return out.squeeze(0).tolist()
