"""Continuous coupling between RSA and PINN fields.

Provides three coupling strategies, all producing alpha(x) in [0, 1]:

  1. ConstantCoupling     — baseline, alpha = constant (ignores input)
  2. PhysicsGuidedCoupling — zero-parameter physics formula:
       alpha = sigmoid(w_resid * Delta + w_conf * C + w_safety * S + b)
     where Delta = | ||grad_T_pinn|| - 1/V |  (PDE residual)
           C     = 1 - cos(grad_T_pinn, grad_T_rsa)  (gradient conflict)
           S     = exp(-k * SDF(x))  (safety risk)
  3. NeuralResidualCoupling — tiny 2-layer MLP on [Delta, C, S]

All classes accept a (batch, 3) feature tensor [Delta, C, S] in forward().
ConstantCoupling ignores the input and returns its stored constant.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


class ConstantCoupling(nn.Module):
    """Constant blending factor — baseline, ignores input features.
        
    alpha=0.0 -> pure PINN rollout
    alpha=1.0 -> pure RSA rollout
    """

    def __init__(self, alpha: float = 0.0) -> None:
        super().__init__()
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.full((features.shape[0], 1), self.alpha, dtype=features.dtype, device=features.device)


class PhysicsGuidedCoupling(nn.Module):
    """Physics-guided adaptive coupling via smooth sigmoid formula.

    Computes alpha from three physically interpretable features:
      Delta = | ||grad_T_pinn|| - 1/V |   — Eikonal residual
      C     = 1 - cos(grad_T_pinn, grad_T_rsa)  — gradient conflict
      S     = exp(-k * SDF(x))            — safety risk

    alpha = sigmoid(w_resid * Delta + w_conf * C + w_safety * S + b)

    All weights are passed as constructor arguments (not learned by default),
    making the formula fully interpretable and zero-parameter at runtime.
    """

    def __init__(
        self,
        w_resid: float = 1.0,
        w_conf: float = 2.0,
        w_safety: float = 3.0,
        b: float = -1.0,
        k: float = 8.0,
    ) -> None:
        super().__init__()
        self.w_resid = float(w_resid)
        self.w_conf = float(w_conf)
        self.w_safety = float(w_safety)
        self.b = float(b)
        self.k = float(k)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute alpha from physics features.

        Args:
            features: (batch, 3) tensor = [Delta, C, S]
                Delta: | ||grad_T_pinn|| - 1/V |  (non-negative)
                C:     1 - cos(v_pinn, v_rsa)      (in [0, 2])
                S:     exp(-k * SDF)               (in [0, 1])

        Returns:
            alpha: (batch, 1) tensor in [0, 1]
        """
        resid = features[:, 0:1]
        conflict = features[:, 1:2]
        safety = features[:, 2:3]
        logit = self.w_resid * resid + self.w_conf * conflict + self.w_safety * safety + self.b
        return torch.sigmoid(logit)


class NeuralResidualCoupling(nn.Module):
    """Lightweight learned coupling via a 2-layer MLP on [Delta, C, S].

    Architecture: Linear(3, H) -> Tanh -> Linear(H, H) -> Tanh -> Linear(H, 1) -> Sigmoid
    Default H=16, so the model has only ~350 parameters.

    This serves as the learnable interface for future end-to-end training
    or meta-learning of the coupling policy.
    """

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


# ---------------------------------------------------------------
# Blending utilities
# ---------------------------------------------------------------


def blend_directions(
    v_pinn: torch.Tensor,
    v_rsa: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Blend PINN and RSA directions using coupling factor alpha.

    v = (1 - alpha) * v_pinn + alpha * v_rsa, then renormalised.

    Args:
        v_pinn: direction from PINN gradient (-grad_T / ||grad_T||)
        v_rsa: direction from RSA path lookahead
        alpha: coupling factor in [0, 1], shape (batch, 1)

    Returns:
        Blended unit direction vector, same shape as v_pinn
    """
    a = alpha.view(-1, 1) if alpha.dim() == 1 else alpha
    a = a.expand(-1, v_pinn.shape[-1]) if a.shape[-1] == 1 else a
    v = (1.0 - a) * v_pinn + a * v_rsa
    norm = torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-12
    return v / norm


def blend_time_fields(
    T_pinn: torch.Tensor,
    T_rsa: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Blend PINN and RSA time fields using coupling factor alpha.

    T = (1 - alpha) * T_pinn + alpha * T_rsa
    """
    a = alpha.view(-1, 1) if alpha.dim() == 1 else alpha
    return (1.0 - a) * T_pinn + a * T_rsa


# ---------------------------------------------------------------
# Physics feature computation utility
# ---------------------------------------------------------------


def compute_coupling_features(
    pinn_grad_norm: torch.Tensor,
    v_pinn: torch.Tensor,
    v_rsa: torch.Tensor,
    sdf_val: torch.Tensor,
    speed_val: torch.Tensor,
    k: float = 8.0,
    delta_clip: float = 2.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute the three physics features for adaptive coupling.

    Args:
        pinn_grad_norm: ||grad_T_pinn||, shape (batch, 1)
        v_pinn: normalised PINN step direction, shape (batch, 2)
        v_rsa: normalised RSA lookahead direction, shape (batch, 2)
        sdf_val: signed distance at current point, shape (batch, 1)
        speed_val: wave speed V(x) at current point, shape (batch, 1)
        k: steepness of safety risk exponential
        delta_clip: upper bound for Delta to prevent logit overflow
        eps: small constant for numerical stability

    Returns:
        features: (batch, 3) tensor = [Delta, C, S]
    """
    # Delta = | ||grad_T|| - 1/V |, clamped to prevent explosion near V->0
    inv_speed = 1.0 / torch.clamp(speed_val, min=eps)
    residual = torch.abs(pinn_grad_norm - inv_speed)
    if delta_clip > 0.0:
        residual = torch.clamp(residual, max=float(delta_clip))

    # C = 1 - cos(v_pinn, v_rsa)  in [0, 2]
    cos_sim = torch.sum(v_pinn * v_rsa, dim=-1, keepdim=True)
    conflict = 1.0 - cos_sim

    # S = exp(-k * SDF)  in [0, 1]
    sdf_clamped = torch.clamp(sdf_val, min=0.0)
    safety = torch.exp(-k * sdf_clamped)

    return torch.cat([residual, conflict, safety], dim=-1)


__all__ = [
    "ConstantCoupling",
    "PhysicsGuidedCoupling",
    "NeuralResidualCoupling",
    "blend_directions",
    "blend_time_fields",
    "compute_coupling_features",
]
