from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def apply_sdf_surface_boost(
    weight: np.ndarray,
    sdf_grid: np.ndarray,
    band: float = 0.05,
    boost: float = 0.5,
) -> np.ndarray:
    w = weight.astype(np.float32)
    sdf = sdf_grid.astype(np.float32)
    surf = (sdf >= 0.0) & (sdf < float(band))
    w = w * (1.0 + float(boost) * surf.astype(np.float32))
    return w.astype(np.float32)


def sample_adaptive_xy(
    xs: np.ndarray,
    ys: np.ndarray,
    weight: np.ndarray,
    n: int,
    rng: np.random.Generator,
    device: torch.device,
    jitter: bool = True,
) -> torch.Tensor:
    h, w = weight.shape
    p = weight.reshape(-1).astype(np.float64)
    s = float(np.sum(p))
    if not np.isfinite(s) or s <= 0.0:
        p = np.ones_like(p, dtype=np.float64) / float(p.size)
    else:
        p = p / s

    idx = rng.choice(h * w, size=(n,), replace=True, p=p)
    ii = idx // w
    jj = idx % w

    if jitter:
        dx = float(xs[1] - xs[0]) if w > 1 else 1.0
        dy = float(ys[1] - ys[0]) if h > 1 else 1.0
        x0 = xs[jj]
        y0 = ys[ii]
        x = x0 + rng.uniform(-0.5, 0.5, size=(n,)).astype(np.float32) * dx
        y = y0 + rng.uniform(-0.5, 0.5, size=(n,)).astype(np.float32) * dy
    else:
        x = xs[jj]
        y = ys[ii]

    xy = np.stack([x, y], axis=1).astype(np.float32)
    return torch.from_numpy(xy).to(device=device, dtype=torch.float32)

