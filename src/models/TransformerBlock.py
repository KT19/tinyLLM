import torch
import torch.nn as nn

from modules.Attentions import GQSWAttention
from modules.RMSNorm import RMSNorm
from modules.RoPE import RoPE
from modules.SwiGLU import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        expansion: float,
        rope: RoPE,
        window_size: int,
        attn_dropout: float = 0.0,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size=dim)
        self.attn = GQSWAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope=rope,
            window_size=window_size,
            attn_dropout=attn_dropout,
        )
        self.residual_dropout = nn.Dropout(residual_dropout)
        self.mlp_norm = RMSNorm(hidden_size=dim)
        self.mlp = SwiGLU(dim=dim, expansion=expansion, dropout=residual_dropout)

    def forward(self, x: torch.Tensor, abs_pos_start: int) -> torch.Tensor:
        a = self.attn(self.attn_norm(x), abs_pos_start=abs_pos_start)
        x = x + self.residual_dropout(a)
        m = self.mlp(self.mlp_norm(x))
        x = x + m
        return x
