from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
from torch import nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softplus":
        return nn.Softplus(beta=5.0)
    raise ValueError(f"Unsupported activation: {name}")


@dataclass(frozen=True)
class PNTField2DConfig:
    in_dim: int = 2
    hidden_dim: int = 128
    num_blocks: int = 4
    fourier_dim: int = 32
    fourier_scale: float = 6.0
    activation: str = "silu"
    positive_output: bool = True


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, activation: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.fc1(x))
        y = self.fc2(y)
        return self.act(x + y)


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, fourier_dim: int, scale: float) -> None:
        super().__init__()
        if fourier_dim <= 0:
            raise ValueError("fourier_dim must be positive")
        b = torch.randn(in_dim, fourier_dim) * float(scale)
        self.register_buffer("b_matrix", b)

    @property
    def out_dim(self) -> int:
        return int(self.b_matrix.shape[1] * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * x @ self.b_matrix
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PNTField2D(nn.Module):
    """A 2D P-NTFields-style scalar field model for fair baseline comparison."""

    def __init__(self, cfg: Optional[PNTField2DConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or PNTField2DConfig()
        self.features = FourierFeatures(
            in_dim=self.cfg.in_dim,
            fourier_dim=self.cfg.fourier_dim,
            scale=self.cfg.fourier_scale,
        )
        self.input_layer = nn.Linear(self.features.out_dim, self.cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(self.cfg.hidden_dim, self.cfg.activation) for _ in range(self.cfg.num_blocks)]
        )
        self.output_layer = nn.Linear(self.cfg.hidden_dim, 1)
        self.act = _make_activation(self.cfg.activation)
        self.softplus = nn.Softplus(beta=1.0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)

    def field(self, xy: torch.Tensor) -> torch.Tensor:
        z = self.features(xy)
        z = self.act(self.input_layer(z))
        for block in self.blocks:
            z = block(z)
        out = self.output_layer(z)
        if self.cfg.positive_output:
            out = self.softplus(out) + 1e-3
        return out

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.field(xy)

    def gradient(self, xy: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        xy_req = xy.clone().detach().requires_grad_(True)
        val = self.forward(xy_req)[:, 0]
        grad = torch.autograd.grad(
            outputs=val.sum(),
            inputs=xy_req,
            create_graph=create_graph,
            retain_graph=create_graph,
            only_inputs=True,
        )[0]
        return grad
