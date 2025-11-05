import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, rope_dim: int, base_theta: float = 10000.0, scaling: float = 1.0):
        super().__init__()
        self.rope_dim = rope_dim
        self.base_theta = base_theta * scaling
        inv_freq = 1.0 / (self.base_theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.rope_dim // 2]
        x2 = x[..., self.rope_dim // 2 :]

        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, T, D=rope_dim)
        positions: (T,) position id
        """

        # sinusoid
        freqs = torch.einsum("t,d->td", positions.float(), self.inv_freq)  # (T, D / 2)
        embed = torch.cat([freqs, freqs], dim=-1)
        cos = embed.cos()[None, None, :, :]  # (1, 1, T, D)
        sin = embed.sin()[None, None, :, :]  # (1, 1, T, D)

        x1, x2 = x[..., : self.rope_dim], x[..., self.rope_dim :]

        x1 = x1 * cos + self._rotate_half(x1) * sin

        return torch.cat([x1, x2], dim=-1)
