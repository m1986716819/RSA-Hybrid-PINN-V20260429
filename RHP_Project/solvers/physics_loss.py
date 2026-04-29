from __future__ import annotations

from typing import Callable, Tuple

import torch


def _safe_speed(speed: torch.Tensor, min_speed: float, max_speed: float = 1.0) -> torch.Tensor:
    return torch.clamp(speed, min=float(min_speed), max=float(max_speed))


def eikonal_residual(
    model: torch.nn.Module,
    xy: torch.Tensor,
    speed_fn: Callable[[torch.Tensor], torch.Tensor],
    grad_eps: float = 1e-12,
    speed_eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xy = xy.requires_grad_(True)
    t = model(xy)
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    grad = torch.autograd.grad(
        outputs=t.sum(),
        inputs=xy,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm = torch.sqrt(torch.sum(grad**2, dim=-1) + grad_eps)

    f = _safe_speed(speed_fn(xy).detach(), min_speed=max(float(speed_eps), 1e-3), max_speed=1.0)
    mask = (f > 0.0).to(dtype=xy.dtype)
    inv_f = 1.0 / f
    res = grad_norm - inv_f
    return res, mask


def physics_loss(
    model: torch.nn.Module,
    xy: torch.Tensor,
    speed_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    res, mask = eikonal_residual(model=model, xy=xy, speed_fn=speed_fn)
    return torch.mean((res**2) * mask)


def upwind_physics_loss(
    model: torch.nn.Module,
    xy: torch.Tensor,
    speed_fn: Callable[[torch.Tensor], torch.Tensor],
    fd_step: float = 0.01,
    speed_eps: float = 1e-6,
) -> torch.Tensor:
    xy = xy.detach()
    h = float(fd_step)
    t0 = model(xy)
    if t0.ndim == 2:
        t0 = t0[:, 0]

    ex = torch.tensor([h, 0.0], dtype=xy.dtype, device=xy.device).view(1, 2)
    ey = torch.tensor([0.0, h], dtype=xy.dtype, device=xy.device).view(1, 2)
    tpx = model(xy + ex)[:, 0]
    tmx = model(xy - ex)[:, 0]
    tpy = model(xy + ey)[:, 0]
    tmy = model(xy - ey)[:, 0]

    dmx = (t0 - tmx) / h
    dpx = (tpx - t0) / h
    dmy = (t0 - tmy) / h
    dpy = (tpy - t0) / h

    ax = torch.maximum(torch.relu(dmx), torch.relu(-dpx)) ** 2
    ay = torch.maximum(torch.relu(dmy), torch.relu(-dpy)) ** 2
    grad_up = torch.sqrt(ax + ay + 1e-12)

    f = _safe_speed(speed_fn(xy).detach(), min_speed=max(float(speed_eps), 1e-3), max_speed=1.0)
    mask = (f > 0.0).to(dtype=xy.dtype)
    inv_f = 1.0 / f
    res = grad_up - inv_f
    return torch.mean((res**2) * mask)


def start_bc_loss(
    model: torch.nn.Module,
    start_xy: torch.Tensor,
) -> torch.Tensor:
    if start_xy.ndim == 1:
        start_xy = start_xy.unsqueeze(0)
    t0 = model(start_xy)
    if t0.ndim == 2:
        t0 = t0[:, 0]
    return torch.mean(t0**2)


def obstacle_loss(
    model: torch.nn.Module,
    xy: torch.Tensor,
    env_sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    speed_fn: Callable[[torch.Tensor], torch.Tensor],
    sdf_band: float = 0.05,
    lambda_int: float = 100.0,
    lambda_grad: float = 10.0,
    lambda_dir: float = 50.0,
    lambda_vort: float = 20.0,
    sdf_scale: float = 10.0,
    speed_eps: float = 1e-6,
) -> torch.Tensor:
    xy = xy.requires_grad_(True)
    t = model(xy)
    if t.ndim == 2:
        t = t[:, 0]

    sdf = env_sdf_fn(xy)
    if sdf.ndim == 2:
        sdf = sdf[:, 0]

    inside = (sdf < 0.0).to(dtype=xy.dtype)
    if torch.any(inside > 0):
        w_in = torch.exp(torch.clamp(-sdf, min=0.0) * float(sdf_scale))
        loss_int = torch.mean(inside * w_in * (t**2))
    else:
        loss_int = torch.zeros((), device=xy.device, dtype=xy.dtype)

    edge = ((sdf >= 0.0) & (sdf < float(sdf_band))).to(dtype=xy.dtype)
    if torch.any(edge > 0):
        grad = torch.autograd.grad(
            outputs=t.sum(),
            inputs=xy,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_norm = torch.sqrt(torch.sum(grad**2, dim=-1) + 1e-12)
        inv_f = (1.0 / _safe_speed(speed_fn(xy).detach(), min_speed=max(float(speed_eps), 1e-3), max_speed=1.0)).to(
            dtype=xy.dtype
        )
        penalty = torch.relu(inv_f - grad_norm) ** 2
        loss_edge = torch.mean(edge * penalty)
    else:
        loss_edge = torch.zeros((), device=xy.device, dtype=xy.dtype)

    if torch.any(edge > 0):
        sdf_for_grad = sdf
        if not sdf_for_grad.requires_grad:
            sdf_for_grad = sdf_for_grad.requires_grad_(True)
        grad_sdf = torch.autograd.grad(
            outputs=sdf_for_grad.sum(),
            inputs=xy,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        dot = torch.sum(grad * grad_sdf, dim=-1)
        loss_dir = torch.mean(edge * (torch.relu(dot) ** 2))
    else:
        loss_dir = torch.zeros((), device=xy.device, dtype=xy.dtype)

    if torch.any(edge > 0):
        v = -grad / (grad_norm.unsqueeze(-1) + 1e-12)
        dvx = torch.autograd.grad(v[:, 0].sum(), xy, create_graph=True, retain_graph=True, only_inputs=True)[0]
        dvy = torch.autograd.grad(v[:, 1].sum(), xy, create_graph=True, retain_graph=True, only_inputs=True)[0]
        omega = dvy[:, 0] - dvx[:, 1]
        loss_vort = torch.mean(edge * (omega**2))
    else:
        loss_vort = torch.zeros((), device=xy.device, dtype=xy.dtype)

    return (
        float(lambda_int) * loss_int
        + float(lambda_grad) * loss_edge
        + float(lambda_dir) * loss_dir
        + float(lambda_vort) * loss_vort
    )


def loss_gate_forcing(
    model: torch.nn.Module,
    xy: torch.Tensor,
    d_target: torch.Tensor,
    env_sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    lambda_gate: float = 150.0,
) -> torch.Tensor:
    xy = xy.requires_grad_(True)
    t = model(xy)
    if t.ndim == 2:
        t = t[:, 0]
    grad = torch.autograd.grad(
        outputs=t.sum(),
        inputs=xy,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gnorm = torch.sqrt(torch.sum(grad**2, dim=-1) + 1e-12)
    move = (-grad) / gnorm.unsqueeze(-1)

    d = d_target.to(device=xy.device, dtype=xy.dtype)
    if d.ndim == 1:
        d = d.view(1, 2)
    dnorm = torch.sqrt(torch.sum(d**2, dim=-1, keepdim=True) + 1e-12)
    d = d / dnorm

    cos = torch.sum(move * d, dim=-1)
    sdf = env_sdf_fn(xy).detach()
    if sdf.ndim == 2:
        sdf = sdf[:, 0]
    free = (sdf >= 0.0).to(dtype=xy.dtype)
    loss = torch.mean(free * (1.0 - cos))
    return float(lambda_gate) * loss


def loss_monotonicity(
    model: torch.nn.Module,
    xy: torch.Tensor,
    d_target: torch.Tensor,
    env_sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    alpha: float = 0.5,
) -> torch.Tensor:
    xy = xy.requires_grad_(True)
    t = model(xy)
    if t.ndim == 2:
        t = t[:, 0]
    grad = torch.autograd.grad(
        outputs=t.sum(),
        inputs=xy,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    v = -grad

    d = d_target.to(device=xy.device, dtype=xy.dtype)
    if d.ndim == 1:
        d = d.view(1, 2)
    dnorm = torch.sqrt(torch.sum(d**2, dim=-1, keepdim=True) + 1e-12)
    d = d / dnorm

    proj = torch.sum(v * d, dim=-1)
    sdf = env_sdf_fn(xy).detach()
    if sdf.ndim == 2:
        sdf = sdf[:, 0]
    free = (sdf >= 0.0).to(dtype=xy.dtype)
    loss = torch.relu(torch.tensor(float(alpha), device=xy.device, dtype=xy.dtype) - proj)
    denom = torch.mean(free) + 1e-12
    return torch.mean(loss * free) / denom


def loss_curl(
    model: torch.nn.Module,
    xy: torch.Tensor,
) -> torch.Tensor:
    xy = xy.requires_grad_(True)
    t = model(xy)
    if t.ndim == 2:
        t = t[:, 0]
    grad = torch.autograd.grad(
        outputs=t.sum(),
        inputs=xy,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    tx = grad[:, 0]
    ty = grad[:, 1]
    dty = torch.autograd.grad(
        outputs=ty.sum(),
        inputs=xy,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    dtx = torch.autograd.grad(
        outputs=tx.sum(),
        inputs=xy,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    dty_dx = dty[:, 0]
    dtx_dy = dtx[:, 1]
    curl = dty_dx - dtx_dy
    return torch.mean(curl**2)
