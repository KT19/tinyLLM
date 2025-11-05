import torch
import torch.nn as nn
from flash_attn import flash_attn_func

from modules.RoPE import RoPE


class GQSWAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope: RoPE,
        window_size: int = 128,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be multiple of num_kv_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope = rope
        self.window_size = window_size
        self.attn_dropout = attn_dropout

        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, kv_heads, T, D) -> (B, num_heads, T, D)
        repeat = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeat, dim=1)

    def _flash_attn_local(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int, dropout_p: float, causal: bool
    ) -> torch.Tensor:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        left = max(0, window_size - 1)
        right = 0

        y = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=None,
            causal=causal,
            window_size=(left, right),
        )
        return y.transpose(1, 2).contiguous()  # type: ignore

    def forward(self, x: torch.Tensor, abs_pos_start: int) -> torch.Tensor:
        """
        x: input tensor (B, T, C)
        pos_id: (T,)
        cache & is_prefill: for inference
        """
        B, T, _ = x.shape
        device = x.device

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, KH, T, D)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, KT, T, D)

        # Repeat KV for GQA
        k_rep = self._repeat_kv(k)  # (B, H, Tk, D)
        v_rep = self._repeat_kv(v)

        # 1. Apply RoPE
        pos_ids = torch.arange(abs_pos_start, abs_pos_start + T, device=device, dtype=torch.long)
        q = self.rope(q, pos_ids)
        k_rep = self.rope(k_rep, pos_ids)

        # Calculate attention
        y = self._flash_attn_local(
            q=q,
            k=k_rep,
            v=v_rep,
            window_size=self.window_size,
            dropout_p=self.attn_dropout if self.training else 0.0,
            causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.o_proj(y)

        return y
