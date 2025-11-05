import torch
import torch.nn as nn

from models.TransformerBlock import TransformerBlock
from modules.RoPE import RoPE

"""
Custom implementation of Hierarchical Reasoning Model
See: 
https://arxiv.org/abs/2506.21734
"""


class HRMState(nn.Module):
    def __init__(self, x_input: torch.Tensor) -> None:
        super().__init__()
        self.zL = torch.zeros_like(x_input, device=x_input.device)
        self.zH = torch.zeros_like(x_input, device=x_input.device)

    def set_state(self, zL: torch.Tensor, zH: torch.Tensor) -> None:
        self.zL = zL
        self.zH = zH

    def detach(self) -> None:
        self.zL = self.zL.detach()
        self.zH = self.zH.detach()

    def set_zL(self, zL: torch.Tensor) -> None:
        self.zL = zL

    def get_zL(self) -> torch.Tensor:
        return self.zL

    def get_zH(self) -> torch.Tensor:
        return self.zH


class HRMBlock(nn.Module):
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
        hrm_modules: int = 2,
    ) -> None:
        super().__init__()

        # the layer is represented as transformer block
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    expansion=expansion,
                    rope=rope,
                    window_size=window_size,
                    attn_dropout=attn_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(hrm_modules)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, abs_pos_start: int) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, D)
            input_injection: (B, T, D)
        """

        for layer in self.layers:
            output = layer(hidden_states + input_injection, abs_pos_start=abs_pos_start)

        return output


class HRM(nn.Module):
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
        hrm_modules: int = 2,
        N: int = 2,
        T: int = 2,
    ) -> None:
        super().__init__()

        self.L_net = HRMBlock(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            expansion=expansion,
            rope=rope,
            window_size=window_size,
            attn_dropout=attn_dropout,
            residual_dropout=residual_dropout,
            hrm_modules=hrm_modules,
        )
        self.H_net = HRMBlock(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            expansion=expansion,
            rope=rope,
            window_size=window_size,
            attn_dropout=attn_dropout,
            residual_dropout=residual_dropout,
            hrm_modules=hrm_modules,
        )
        self.N = N
        self.T = T

    def forward(self, x_input: torch.Tensor, state: HRMState, abs_pos_start: int) -> HRMState:
        zL = state.get_zL()
        zH = state.get_zH()
        B, T, _ = x_input.size()

        # empirically, w/o 1-step approx is better
        for _i in range(self.N * self.T - 1):
            zL = self.L_net(zL, zH + x_input, abs_pos_start=abs_pos_start)
            if (_i + 1) % self.T == 0:
                zH = self.H_net(zH, zL, abs_pos_start=abs_pos_start)

        zL = self.L_net(zL, zH + x_input, abs_pos_start=abs_pos_start)
        zH = self.H_net(zH, zL, abs_pos_start=abs_pos_start)

        state.set_state(zL=zL, zH=zH)

        return state
