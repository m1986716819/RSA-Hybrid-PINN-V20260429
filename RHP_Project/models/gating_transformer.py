from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class GatingTransformerConfig:
    input_dim: int = 14
    seq_len: int = 10
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0


class GatingTransformer(nn.Module):
    def __init__(self, cfg: Optional[GatingTransformerConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or GatingTransformerConfig()
        if self.cfg.d_model % self.cfg.nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.embed = nn.Linear(self.cfg.input_dim, self.cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.cfg.seq_len, self.cfg.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.nhead,
            dim_feedforward=self.cfg.dim_feedforward,
            dropout=float(self.cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.cfg.num_layers)

        self.head = nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.GELU(),
            nn.Linear(self.cfg.d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Input must have shape (batch, seq_len, input_dim).")
        if x.shape[1] != self.cfg.seq_len:
            raise ValueError(f"Expected seq_len={self.cfg.seq_len}, got {x.shape[1]}.")
        if x.shape[2] != self.cfg.input_dim:
            raise ValueError(f"Expected input_dim={self.cfg.input_dim}, got {x.shape[2]}.")

        h = self.embed(x)
        h = h + self.pos
        h = self.encoder(h)
        last = h[:, -1, :]
        value = self.head(last)[:, 0]
        return value

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def predict_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


__all__ = ["GatingTransformer", "GatingTransformerConfig"]
