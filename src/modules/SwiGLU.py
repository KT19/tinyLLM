import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, dim: int, expansion: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.w3(F.silu(self.w1(x)) * self.w2(x))

        return self.dropout(out)
