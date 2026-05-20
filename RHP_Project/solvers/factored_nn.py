from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch import nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        act = _make_activation(activation)
        layers = [nn.Linear(in_dim, hidden_dim), act]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), _make_activation(activation)])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class Factoring:
    start_xy: Tuple[float, float]
    dist_eps: float = 1e-6

    def dist(self, xy: torch.Tensor) -> torch.Tensor:
        start = torch.tensor(self.start_xy, dtype=xy.dtype, device=xy.device).view(1, 2)
        d2 = torch.sum((xy - start) ** 2, dim=-1, keepdim=True)
        return torch.sqrt(d2 + (self.dist_eps**2))


class FactoredTimeNN(nn.Module):
    def __init__(
        self,
        start_xy: Tuple[float, float],
        hidden_dim: int = 128,
        num_layers: int = 4,
        activation: str = "tanh",
        dist_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.factoring = Factoring(start_xy=start_xy, dist_eps=dist_eps)
        self.mlp = MLP(
            in_dim=2,
            out_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
        )
        self.softplus = nn.Softplus(beta=1.0)

    def tau(self, xy: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(xy)
        return self.softplus(raw) + 1e-6

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        dist = self.factoring.dist(xy)
        return dist * self.tau(xy)


class VanillaTimeNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            in_dim=2,
            out_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
        )
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.mlp(xy)) + 1e-6

