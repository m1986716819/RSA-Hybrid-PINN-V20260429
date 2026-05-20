from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from .envs.maze_2d import Bounds2D, Maze2DEnv
from .evaluator.metrics import EvalMetrics, evaluate_methods
from .evaluator.plotter import save_field_and_paths_plot, save_field_quiver_plot
from .solvers.coupling import PhysicsGuidedCoupling, ConstantCoupling
from .solvers.factored_nn import FactoredTimeNN, VanillaTimeNN
from .solvers.physics_loss import (
    loss_monotonicity,
    obstacle_loss,
    physics_loss,
    start_bc_loss,
    upwind_physics_loss,
)
from .solvers.pntfield_2d import PNTField2D, PNTField2DConfig
from .solvers.rsa_engine import RSAEngine, RSAResult
from .utils.sampler import apply_sdf_surface_boost, sample_adaptive_xy


def _set_seed(seed: int) -> np.random.Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def _load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _archive_root(cfg: Dict[str, Any]) -> Path:
    root = _project_root()
    archive_root = cfg.get("eval", {}).get("archive_root", "test_result")
    return (root / str(archive_root)).resolve()


def _date_stamp(cfg: Dict[str, Any]) -> str:
    stamp = cfg.get("eval", {}).get("archive_date")
    if stamp is not None and str(stamp).strip():
        return str(stamp).strip()
    return dt.date.today().isoformat()


def _next_run_dir(cfg: Dict[str, Any], create: bool = True) -> Path:
    date_dir = _archive_root(cfg) / _date_stamp(cfg)
    if create:
        date_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted([p for p in date_dir.glob("run_*") if p.is_dir()])
    next_idx = len(run_dirs) + 1
    run_dir = date_dir / f"run_{next_idx:02d}"
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _latest_run_dir(cfg: Dict[str, Any], create_if_missing: bool = False) -> Path | None:
    date_dir = _archive_root(cfg) / _date_stamp(cfg)
    if not date_dir.exists():
        return _next_run_dir(cfg, create=True) if create_if_missing else None
    run_dirs = sorted([p for p in date_dir.glob("run_*") if p.is_dir()])
    if not run_dirs:
        return _next_run_dir(cfg, create=True) if create_if_missing else None
    return run_dirs[-1]


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _archive_outputs(cfg: Dict[str, Any], out_dir: str) -> Path:
    out_path = Path(out_dir).resolve()
    run_dir = _next_run_dir(cfg, create=True)
    benchmark_dir = run_dir / "benchmark"
    field_dir = run_dir / "field_visualizations"
    reports_dir = run_dir / "reports"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    field_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(out_path / "results.json", benchmark_dir / "results.json")
    _copy_if_exists(out_path / "benchmark_metrics.png", benchmark_dir / "benchmark_metrics.png")

    for img in sorted(out_path.glob("fields*.png")):
        _copy_if_exists(img, field_dir / img.name)

    _copy_tree_if_exists(out_path / "report_plots", reports_dir / "report_plots")
    return run_dir


def _make_env(cfg: Dict[str, Any], device: torch.device) -> Maze2DEnv:
    bcfg = cfg["env"]["bounds"]
    bounds = Bounds2D(**bcfg)
    name = cfg["env"]["name"]
    common = dict(
        bounds=bounds,
        obstacle_inflation=float(cfg["env"].get("obstacle_inflation", 0.0)),
        speed_free=float(cfg["rsa"].get("speed_free", 1.0)),
        speed_obstacle=float(cfg["rsa"].get("speed_obstacle", 0.0)),
        device=device,
    )
    if name == "u_maze":
        p = cfg["env"]["u_maze"]
        env = Maze2DEnv.make_u_maze(
            **common,
            wall_thickness=float(p["wall_thickness"]),
            inner_gap=float(p["inner_gap"]),
            depth=float(p["depth"]),
            center=tuple(p.get("center", [0.0, 0.0])),
            rotation_deg=float(p.get("rotation_deg", 0.0)),
        )
    elif name == "narrow_passage":
        p = cfg["env"]["narrow_passage"]
        env = Maze2DEnv.make_narrow_passage(
            **common,
            passage_width=float(p["passage_width"]),
            block_width=float(p["block_width"]),
            block_height=float(p["block_height"]),
            center=tuple(p.get("center", [0.0, 0.0])),
            rotation_deg=float(p.get("rotation_deg", 0.0)),
        )
    elif name == "trap_u_shape":
        p = cfg["env"]["trap_u_shape"]
        env = Maze2DEnv.make_trap_u_shape(
            **common,
            wall_thickness=float(p.get("wall_thickness", 0.03)),
            left_x=float(p.get("left_x", 0.35)),
            u_depth=float(p.get("u_depth", 0.28)),
            u_height=float(p.get("u_height", 0.55)),
            center_y=float(p.get("center_y", 0.5)),
        )
    elif name == "trap_heterogeneous_vf":
        p = cfg["env"]["trap_heterogeneous_vf"]
        env = Maze2DEnv.make_trap_heterogeneous_vf(
            **common,
            wall_thickness=float(p.get("wall_thickness", 0.03)),
            left_x=float(p.get("left_x", 0.35)),
            u_depth=float(p.get("u_depth", 0.28)),
            u_height=float(p.get("u_height", 0.55)),
            center_y=float(p.get("center_y", 0.5)),
            v_slow=float(p.get("v_slow", 0.3)),
            v_fast=float(p.get("v_fast", 1.0)),
            sigmoid_beta=float(p.get("sigmoid_beta", 8.0)),
        )
    elif name == "cost_pit":
        p = cfg["env"]["cost_pit"]
        env = Maze2DEnv.make_cost_pit(
            bounds=bounds,
            pit_center=tuple(p.get("pit_center", [0.5, 0.5])),
            pit_radius=float(p.get("pit_radius", 0.35)),
            v_slow=float(p.get("v_slow", 0.05)),
            v_fast=float(p.get("v_fast", 1.0)),
            sigmoid_beta=float(p.get("sigmoid_beta", 10.0)),
            chokepoint_width=float(p.get("chokepoint_width", 0.10)),
            chokepoint_x=float(p.get("chokepoint_x", 0.38)),
            chokepoint_y_low=float(p.get("chokepoint_y_low", 0.28)),
            chokepoint_y_high=float(p.get("chokepoint_y_high", 0.72)),
            obstacle_inflation=common["obstacle_inflation"],
            speed_free=common["speed_free"],
            speed_obstacle=common["speed_obstacle"],
            device=device,
        )
    elif name == "open_space":
        env = Maze2DEnv.make_open_space(**common)
    elif name == "random_maze":
        from .envs.maze_2d import make_random_maze
        rm = cfg["env"]["random_maze"]
        rng = np.random.default_rng(int(rm.get("seed", cfg.get("seed", 0))))
        env = make_random_maze(
            bounds=bounds,
            rng=rng,
            start_xy=tuple(cfg["env"]["start"]),
            goal_xy=tuple(cfg["env"]["goal"]),
            num_obstacles=int(rm.get("num_obstacles", 10)),
            box_ratio=float(rm.get("box_ratio", 0.5)),
            size_range=(float(rm.get("size_min", 0.05)), float(rm.get("size_max", 0.18))),
            clearance=float(rm.get("clearance", 0.06)),
            speed_free=common["speed_free"],
            speed_obstacle=common["speed_obstacle"],
            obstacle_inflation=common["obstacle_inflation"],
            device=device,
        )
    elif name == "random_maze_vp":
        from .envs.random_generator import generate_random_environment
        rm = cfg["env"]["random_maze_vp"]
        env_rng = np.random.default_rng(int(rm.get("seed", cfg.get("seed", 0))))
        env = generate_random_environment(
            bounds=bounds,
            start_xy=tuple(cfg["env"]["start"]),
            goal_xy=tuple(cfg["env"]["goal"]),
            num_obstacles=int(rm.get("num_obstacles", 4)),
            min_radius=float(rm.get("min_radius", 0.06)),
            max_radius=float(rm.get("max_radius", 0.18)),
            poly_vertices=int(rm.get("poly_vertices", 5)),
            clearance=float(rm.get("clearance", 0.06)),
            vf_grid_size=tuple(rm.get("vf_grid_size", [40, 40])),
            vf_noise_scale=float(rm.get("vf_noise_scale", 3.0)),
            v_min=float(rm.get("v_min", 0.2)),
            v_max=float(rm.get("v_max", 1.0)),
            speed_free=common["speed_free"],
            speed_obstacle=common["speed_obstacle"],
            obstacle_inflation=common["obstacle_inflation"],
            device=device,
            rng=env_rng,
        )
    else:
        raise ValueError(f"Unknown env.name: {name}")

    vf_cfg = cfg["env"].get("velocity_field", {})
    if isinstance(vf_cfg, dict) and vf_cfg.get("enabled", False):
        vf_type = str(vf_cfg.get("type", "half_space"))
        if vf_type == "half_space":
            v_upper = float(vf_cfg.get("v_upper", 1.0))
            v_lower = float(vf_cfg.get("v_lower", 0.2))
            boundary_y = float(vf_cfg.get("boundary_y", 0.0))
            def _half_space_velocity(xy: torch.Tensor) -> torch.Tensor:
                return torch.where(
                    xy[:, 1] >= boundary_y,
                    torch.full((xy.shape[0],), v_upper, device=xy.device, dtype=xy.dtype),
                    torch.full((xy.shape[0],), v_lower, device=xy.device, dtype=xy.dtype),
                )
            env.set_velocity_field(_half_space_velocity)
        elif vf_type == "perlin":
            from .envs.maze_2d import make_perlin_velocity_field
            vf_c = vf_cfg.get("perlin", {})
            noise_grid = tuple(vf_c.get("grid_size", [40, 40]))
            v_min = float(vf_c.get("v_min", 0.3))
            v_max = float(vf_c.get("v_max", 1.0))
            noise_scale = float(vf_c.get("scale", 3.0))
            vf_seed = int(vf_c.get("seed", cfg.get("seed", 0)))
            vf_fn = make_perlin_velocity_field(
                bounds=bounds,
                grid_size=noise_grid,
                v_min=v_min,
                v_max=v_max,
                noise_scale=noise_scale,
                seed=vf_seed,
            )
            env.set_velocity_field(vf_fn)
        else:
            raise ValueError(f"Unknown velocity_field.type: {vf_type}")

    return env


def _grid_coords(bounds: Bounds2D, grid_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = int(grid_size[0]), int(grid_size[1])
    xs = np.linspace(bounds.x_min, bounds.x_max, w, dtype=np.float32)
    ys = np.linspace(bounds.y_min, bounds.y_max, h, dtype=np.float32)
    return xs, ys


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    denom = torch.mean(mask) + 1e-12
    return torch.mean(x * mask) / denom


def _laplacian_smoothness_loss(model: torch.nn.Module, xy: torch.Tensor, env: Maze2DEnv) -> torch.Tensor:
    xy_req = xy.clone().detach().requires_grad_(True)
    pred = model(xy_req)[:, 0]
    grad_xy = torch.autograd.grad(
        outputs=pred.sum(),
        inputs=xy_req,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    lap = torch.zeros_like(pred)
    for dim in range(xy_req.shape[1]):
        second = torch.autograd.grad(
            outputs=grad_xy[:, dim].sum(),
            inputs=xy_req,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0][:, dim]
        lap = lap + second
    lap = torch.clamp(lap, min=-10.0, max=10.0)
    with torch.no_grad():
        free_mask = (env.sdf(xy_req) >= 0.0).to(dtype=xy_req.dtype)
    return _masked_mean(lap**2, free_mask)


def _goal_bc_loss(model: torch.nn.Module, goal_xy: torch.Tensor) -> torch.Tensor:
    if goal_xy.ndim == 1:
        goal_xy = goal_xy.unsqueeze(0)
    t_goal = model(goal_xy)
    if t_goal.ndim == 2:
        t_goal = t_goal[:, 0]
    return torch.mean(t_goal**2)


def _gradient_floor_loss(
    model: torch.nn.Module,
    xy: torch.Tensor,
    env: Maze2DEnv,
    grad_floor: float,
) -> torch.Tensor:
    xy_req = xy.clone().detach().requires_grad_(True)
    pred = model(xy_req)[:, 0]
    grad_xy = torch.autograd.grad(
        outputs=pred.sum(),
        inputs=xy_req,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm = torch.sqrt(torch.sum(grad_xy**2, dim=-1) + 1e-12)
    with torch.no_grad():
        free_mask = (env.sdf(xy_req) >= 0.0).to(dtype=xy_req.dtype)
    return _masked_mean(torch.relu(float(grad_floor) - grad_norm) ** 2, free_mask)


def _distance_ranking_loss(
    pred: torch.Tensor,
    dist_to_goal: torch.Tensor,
    free_mask: torch.Tensor,
    margin: float,
    rng: np.random.Generator,
    max_pairs: int = 512,
) -> torch.Tensor:
    free_idx = torch.nonzero(free_mask > 0.5, as_tuple=False).squeeze(-1)
    if free_idx.numel() < 2:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    num_pairs = int(min(max_pairs, free_idx.numel() // 2))
    if num_pairs <= 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    idx_np = free_idx.detach().cpu().numpy()
    i_idx = torch.from_numpy(rng.choice(idx_np, size=num_pairs, replace=True)).to(device=pred.device, dtype=torch.long)
    j_idx = torch.from_numpy(rng.choice(idx_np, size=num_pairs, replace=True)).to(device=pred.device, dtype=torch.long)
    dist_i = dist_to_goal[i_idx]
    dist_j = dist_to_goal[j_idx]
    pred_i = pred[i_idx]
    pred_j = pred[j_idx]
    farther = (dist_i > dist_j).to(dtype=pred.dtype)
    nearer = (dist_j > dist_i).to(dtype=pred.dtype)
    margin_t = torch.tensor(float(margin), device=pred.device, dtype=pred.dtype)
    loss_farther = farther * torch.relu(margin_t - (pred_i - pred_j)) ** 2
    loss_nearer = nearer * torch.relu(margin_t - (pred_j - pred_i)) ** 2
    active = farther + nearer
    denom = torch.mean(active) + 1e-12
    return torch.mean(loss_farther + loss_nearer) / denom


def _interp_vec_bilinear(
    xs: np.ndarray,
    ys: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    xy: np.ndarray,
) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    w = xs.shape[0]
    h = ys.shape[0]

    x = np.clip(x, xs[0], xs[-1])
    y = np.clip(y, ys[0], ys[-1])

    j = np.searchsorted(xs, x) - 1
    i = np.searchsorted(ys, y) - 1
    j = np.clip(j, 0, w - 2)
    i = np.clip(i, 0, h - 2)

    x0 = xs[j]
    x1 = xs[j + 1]
    y0 = ys[i]
    y1 = ys[i + 1]
    tx = (x - x0) / (x1 - x0 + 1e-12)
    ty = (y - y0) / (y1 - y0 + 1e-12)

    def _interp(f: np.ndarray) -> np.ndarray:
        f00 = f[i, j]
        f10 = f[i, j + 1]
        f01 = f[i + 1, j]
        f11 = f[i + 1, j + 1]
        fill = 0.0
        f00 = np.where(np.isfinite(f00), f00, fill)
        f10 = np.where(np.isfinite(f10), f10, fill)
        f01 = np.where(np.isfinite(f01), f01, fill)
        f11 = np.where(np.isfinite(f11), f11, fill)
        f0 = f00 * (1 - tx) + f10 * tx
        f1 = f01 * (1 - tx) + f11 * tx
        return f0 * (1 - ty) + f1 * ty

    out_x = _interp(vx)
    out_y = _interp(vy)
    return np.stack([out_x, out_y], axis=1).astype(np.float32)


def _gate_boost_weight(
    weight: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    gate_pts: np.ndarray,
    radius: float,
    boost: float,
) -> np.ndarray:
    w = weight.astype(np.float32, copy=True)
    if gate_pts.size == 0:
        return w
    dx = float(xs[1] - xs[0]) if xs.shape[0] > 1 else 1.0
    dy = float(ys[1] - ys[0]) if ys.shape[0] > 1 else 1.0
    r_cells = int(np.ceil(float(radius) / max(1e-12, min(dx, dy))))
    h, ww = w.shape

    for p in gate_pts:
        j0 = int(np.argmin(np.abs(xs - float(p[0]))))
        i0 = int(np.argmin(np.abs(ys - float(p[1]))))
        i1 = max(0, i0 - r_cells)
        i2 = min(h, i0 + r_cells + 1)
        j1 = max(0, j0 - r_cells)
        j2 = min(ww, j0 + r_cells + 1)
        w[i1:i2, j1:j2] *= 1.0 + float(boost)

    return w.astype(np.float32)


def _get_convergence_cfg(cfg: Dict[str, Any]) -> Tuple[float, int, int]:
    tcfg = cfg.get("train", {})
    ccfg = tcfg.get("convergence", {}) if isinstance(tcfg, dict) else {}
    tol = float(ccfg.get("tol", 1e-3))
    patience = int(ccfg.get("patience", 50))
    check_every = int(ccfg.get("check_every", 1))
    return tol, patience, check_every


def sample_path_tube(
    path_xy: np.ndarray,
    radius: float,
    num_samples: int,
    rng: np.random.Generator,
    env: Maze2DEnv,
    device: torch.device,
    lookahead: int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path_xy = np.asarray(path_xy, dtype=np.float32)
    if path_xy.ndim != 2 or path_xy.shape[1] != 2:
        raise ValueError("path_xy must have shape [M,2].")
    if path_xy.shape[0] < 2:
        raise ValueError("path_xy must contain at least 2 points.")
    lookahead = int(max(1, lookahead))

    m = path_xy.shape[0]
    idx = rng.integers(0, m, size=(int(num_samples),), endpoint=False)
    base = path_xy[idx]

    ang = rng.uniform(0.0, 2 * np.pi, size=(int(num_samples),)).astype(np.float32)
    rad = np.sqrt(rng.uniform(0.0, 1.0, size=(int(num_samples),)).astype(np.float32)) * float(radius)
    offset = np.stack([np.cos(ang) * rad, np.sin(ang) * rad], axis=1).astype(np.float32)
    pts = (base + offset).astype(np.float32)

    d2 = np.sum((pts[:, None, :] - path_xy[None, :, :]) ** 2, axis=-1)
    nn_idx = np.argmin(d2, axis=1).astype(np.int32)
    look_idx = np.minimum(nn_idx + lookahead, m - 1).astype(np.int32)
    p_look = path_xy[look_idx]
    d_look = (p_look - pts).astype(np.float32)
    dn = np.linalg.norm(d_look, axis=1, keepdims=True) + 1e-12
    d_look = (d_look / dn).astype(np.float32)

    pts_t = torch.from_numpy(pts).to(device=device, dtype=torch.float32).requires_grad_(True)
    sdf = env.sdf(pts_t)
    if sdf.ndim == 2:
        sdf = sdf[:, 0]
    grad_sdf = torch.autograd.grad(sdf.sum(), pts_t, create_graph=False, retain_graph=False)[0]
    n = grad_sdf / (torch.linalg.norm(grad_sdf, dim=-1, keepdim=True) + 1e-12)
    d_look_t = torch.from_numpy(d_look).to(device=device, dtype=torch.float32)
    dot = torch.sum(d_look_t * n, dim=-1, keepdim=True)
    mask = dot < 0.0
    d_proj = d_look_t - dot * n
    d_proj = d_proj / (torch.linalg.norm(d_proj, dim=-1, keepdim=True) + 1e-12)
    d_look_t = torch.where(mask, d_proj, d_look_t)
    d_look = d_look_t.detach().cpu().numpy().astype(np.float32)

    return pts, d_look, p_look


def _train_rhp(
    env: Maze2DEnv,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    rsa_guidance: RSAResult,
    rsa_path_xy: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    weight: np.ndarray,
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    device: torch.device,
    coupling_alpha: float = 0.3,
) -> Tuple[FactoredTimeNN, int, int]:
    """Train RHP-PINN with RSA distillation + Eikonal regularization.

    Loss terms:
      - L_Eikonal: ||del T|| - 1/V (physics residual)
      - L_distill: ||del T_pred - del T_RSA||^2 along RSA path (gradient alignment)
      - L_mono:   T along path must be monotonically decreasing toward goal
      - L_obs:    obstacle boundary constraints (SDF-informed)
      - L_bc:     T(start) = 0 boundary condition
    """
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    model = FactoredTimeNN(
        start_xy=start_xy,
        hidden_dim=int(mcfg["hidden_dim"]),
        num_layers=int(mcfg["num_layers"]),
        activation=str(mcfg.get("activation", "tanh")),
        dist_eps=float(mcfg.get("dist_eps", 1e-6)),
    ).to(device=device)

    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))
    start_t = torch.tensor(start_xy, dtype=torch.float32, device=device)

    tol, patience, check_every = _get_convergence_cfg(cfg)
    ok_count = 0
    converged_step = -1
    train_steps = 0
    ocfg = tcfg.get("obstacle", {}) if isinstance(tcfg, dict) else {}
    sdf_band = float(ocfg.get("sdf_band", 0.05))
    lambda_int = float(ocfg.get("lambda_int", 100.0))
    lambda_grad = float(ocfg.get("lambda_grad", 10.0))
    lambda_dir = float(ocfg.get("lambda_dir", 50.0))
    lambda_vort = float(ocfg.get("lambda_vort", 20.0))
    sdf_scale = float(ocfg.get("sdf_scale", 10.0))
    lambda_obs = float(tcfg.get("lambda_obs", 1.0))
    pcfg = tcfg.get("path_anchor", {}) if isinstance(tcfg, dict) else {}
    lambda_path = float(pcfg.get("lambda_path", 20.0))
    lambda_distill = float(pcfg.get("lambda_distill", 50.0))
    lambda_mono_dir = float(pcfg.get("lambda_mono_dir", 30.0))
    path_radius = float(pcfg.get("radius", 0.05))
    path_batch = int(pcfg.get("batch_size", 512))
    lambda_mono = float(pcfg.get("lambda_mono", 20.0))
    lambda_phys_start = float(tcfg.get("lambda_phys_start", 0.1))
    lambda_phys_end = float(tcfg.get("lambda_phys", 1.0))
    phcfg = tcfg.get("physics", {}) if isinstance(tcfg, dict) else {}
    physics_mode = str(phcfg.get("mode", "autograd")).lower()
    fd_step = float(phcfg.get("fd_step", 0.01))
    training_speed_fn = env.wave_speed if env.has_velocity_field() else env.speed

    if rsa_path_xy.ndim != 2 or rsa_path_xy.shape[1] != 2:
        raise ValueError("rsa_path_xy must have shape [M,2]")
    rsa_path_xy = rsa_path_xy.astype(np.float32)
    if rsa_path_xy.shape[0] < 2:
        raise ValueError("rsa_path_xy must contain at least 2 points.")

    tang = np.zeros_like(rsa_path_xy, dtype=np.float32)
    tang[:-1] = rsa_path_xy[1:] - rsa_path_xy[:-1]
    tang[-1] = tang[-2]
    tang_norm = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    tang = tang / tang_norm
    goal_target = float(rsa_guidance.interpolate_t(np.asarray([goal_xy], dtype=np.float32))[0])
    lambda_goal = float(tcfg.get("lambda_goal", 5.0))
    lambda_path_dir = float(pcfg.get("lambda_dir", 10.0))

    warmup_steps = int(tcfg.get("warmup_steps", 0))
    batch_size = int(tcfg["batch_size"])

    dx = float(xs[1] - xs[0]) if xs.shape[0] > 1 else 1.0
    dy = float(ys[1] - ys[0]) if ys.shape[0] > 1 else 1.0
    tgrid = rsa_guidance.t.astype(np.float32)
    finite = np.isfinite(tgrid)
    fill = float(np.max(tgrid[finite])) if np.any(finite) else 1.0
    tgrid = np.where(finite, tgrid, fill)
    dtdy, dtdx = np.gradient(tgrid, dy, dx)
    dtdx = dtdx.astype(np.float32)
    dtdy = dtdy.astype(np.float32)

    tube_radius = float(pcfg.get("tube_radius", 0.1))
    tube_samples = int(pcfg.get("tube_samples", 256))
    tube_batch_frac = float(pcfg.get("tube_batch_frac", 0.85))
    tube_batch_frac = float(np.clip(tube_batch_frac, 0.75, 0.95))

    for _ in range(warmup_steps):
        n_tube = int(max(256, min(batch_size, int(round(batch_size * tube_batch_frac)))))
        n_global = int(max(1, batch_size - n_tube))
        xy_global = sample_adaptive_xy(xs=xs, ys=ys, weight=weight, n=n_global, rng=rng, device=device)
        xy_tube_np, _, _ = sample_path_tube(
            path_xy=rsa_path_xy,
            radius=tube_radius,
            num_samples=n_tube,
            rng=rng,
            env=env,
            device=device,
            lookahead=25,
        )
        xy_tube = torch.from_numpy(xy_tube_np).to(device=device, dtype=torch.float32)
        xy = torch.cat([xy_global, xy_tube], dim=0)

        n_boundary = int(max(64, min(batch_size // 4, 512)))
        boundary_xy = env.sample_near_obstacles(n_boundary, sdf_band=0.08, rng=rng, device=device)
        if boundary_xy.shape[0] > 0:
            xy = torch.cat([xy, boundary_xy], dim=0)

        sdf = env.sdf(xy).detach()
        free_mask = (sdf >= 0.0).to(dtype=xy.dtype)
        t_target = torch.from_numpy(rsa_guidance.interpolate_t(xy.detach().cpu().numpy())).to(device=device)

        pred = model(xy)[:, 0]
        loss_sup = _masked_mean((pred - t_target) ** 2, free_mask)
        loss_bc = start_bc_loss(model, start_t)
        loss_obs = obstacle_loss(
            model=model,
            xy=xy,
            env_sdf_fn=env.sdf,
            speed_fn=training_speed_fn,
            sdf_band=sdf_band,
            lambda_int=lambda_int,
            lambda_grad=lambda_grad,
            lambda_dir=lambda_dir,
            lambda_vort=lambda_vort,
            sdf_scale=sdf_scale,
        )
        loss = (
            float(tcfg.get("lambda_sup", 1.0)) * loss_sup
            + float(tcfg.get("lambda_bc", 0.0)) * loss_bc
            + float(lambda_obs) * loss_obs
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_steps += 1

    max_steps = int(tcfg["max_steps"])
    for step in range(max_steps):
        n_tube = int(max(256, min(batch_size, int(round(batch_size * tube_batch_frac)))))
        n_global = int(max(1, batch_size - n_tube))
        xy_global = sample_adaptive_xy(xs=xs, ys=ys, weight=weight, n=n_global, rng=rng, device=device)
        xy_tube_np, tube_d_look_np, tube_p_look_np = sample_path_tube(
            path_xy=rsa_path_xy,
            radius=tube_radius,
            num_samples=n_tube,
            rng=rng,
            env=env,
            device=device,
            lookahead=25,
        )
        xy_tube = torch.from_numpy(xy_tube_np).to(device=device, dtype=torch.float32)
        xy = torch.cat([xy_global, xy_tube], dim=0)
        sdf = env.sdf(xy).detach()
        free_mask = (sdf >= 0.0).to(dtype=xy.dtype)
        t_target = torch.from_numpy(rsa_guidance.interpolate_t(xy.detach().cpu().numpy())).to(device=device)
        pred = model(xy)[:, 0]
        loss_sup = torch.zeros((), device=device, dtype=torch.float32)
        progress = float(step + 1) / float(max(1, max_steps))
        lambda_phys_now = lambda_phys_start + (lambda_phys_end - lambda_phys_start) * progress
        lambda_phys_now = 0.8 * lambda_phys_now
        loss_phys_global = (
            upwind_physics_loss(model=model, xy=xy_global, speed_fn=training_speed_fn, fd_step=fd_step)
            if physics_mode == "upwind"
            else physics_loss(model=model, xy=xy_global, speed_fn=training_speed_fn)
        )
        loss_phys_tube = (
            upwind_physics_loss(model=model, xy=xy_tube, speed_fn=training_speed_fn, fd_step=fd_step)
            if physics_mode == "upwind"
            else physics_loss(model=model, xy=xy_tube, speed_fn=training_speed_fn)
        )
        loss_phys = loss_phys_global + 0.1 * loss_phys_tube
        loss_bc = start_bc_loss(model, start_t)
        loss_obs = obstacle_loss(
            model=model,
            xy=xy,
            env_sdf_fn=env.sdf,
            speed_fn=training_speed_fn,
            sdf_band=sdf_band,
            lambda_int=lambda_int,
            lambda_grad=lambda_grad,
            lambda_dir=lambda_dir,
            lambda_vort=lambda_vort,
            sdf_scale=sdf_scale,
        )
        idx = rng.integers(0, rsa_path_xy.shape[0], size=(path_batch,), endpoint=False)
        base = rsa_path_xy[idx]
        base_tang = tang[idx]
        ang = rng.uniform(0.0, 2 * np.pi, size=(path_batch,)).astype(np.float32)
        rad = np.sqrt(rng.uniform(0.0, 1.0, size=(path_batch,)).astype(np.float32)) * float(path_radius)
        offset = np.stack([np.cos(ang) * rad, np.sin(ang) * rad], axis=1).astype(np.float32)
        anchor_xy = (base + offset).astype(np.float32)
        anchor_t = torch.from_numpy(rsa_guidance.interpolate_t(anchor_xy)).to(device=device)
        anchor_xy_t = torch.from_numpy(anchor_xy).to(device=device, dtype=torch.float32).requires_grad_(True)
        with torch.no_grad():
            anchor_free = (env.sdf(anchor_xy_t) >= 0.0).to(dtype=anchor_xy_t.dtype)
        anchor_pred = model(anchor_xy_t)[:, 0]
        loss_path = _masked_mean((anchor_pred - anchor_t) ** 2, anchor_free)
        grad_anchor = torch.autograd.grad(
            outputs=anchor_pred.sum(),
            inputs=anchor_xy_t,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gnorm = torch.sqrt(torch.sum(grad_anchor**2, dim=-1) + 1e-12)
        move_dir = (-grad_anchor) / gnorm.unsqueeze(-1)
        tang_t = torch.from_numpy(base_tang).to(device=device, dtype=torch.float32)
        cos = torch.sum(move_dir * tang_t, dim=-1)
        loss_path_dir = _masked_mean((1.0 - cos) ** 2, anchor_free)

        # RSA path distillation: align PINN gradient with RSA reference gradient
        rsa_grad_anchor = _interp_vec_bilinear(xs=xs, ys=ys, vx=dtdx, vy=dtdy, xy=anchor_xy)
        rsa_grad_anchor_t = torch.from_numpy(rsa_grad_anchor).to(device=device, dtype=torch.float32)
        loss_distill = _masked_mean(torch.sum((grad_anchor - rsa_grad_anchor_t) ** 2, dim=-1), anchor_free)

        # Enhanced monotonicity constraint: gradient should point toward goal
        goal_dir_np = (np.asarray(goal_xy, dtype=np.float32) - anchor_xy).astype(np.float32)
        goal_dir_np = goal_dir_np / (np.linalg.norm(goal_dir_np, axis=1, keepdims=True) + 1e-12)
        goal_dir_t = torch.from_numpy(goal_dir_np).to(device=device, dtype=torch.float32)
        cos_goal = torch.sum((-grad_anchor) * goal_dir_t, dim=-1)
        # Use a tighter threshold for monotonicity (0.8 instead of 0.5)
        loss_mono_dir = _masked_mean(torch.relu(0.8 - cos_goal) ** 2, anchor_free)

        goal_xy_t = torch.tensor(goal_xy, dtype=torch.float32, device=device).view(1, 2)
        goal_pred = model(goal_xy_t)[:, 0]
        loss_goal = torch.mean((goal_pred - goal_target) ** 2)

        idx_m = rng.integers(0, rsa_path_xy.shape[0] - 1, size=(path_batch,), endpoint=False)
        p0 = rsa_path_xy[idx_m]
        p1 = rsa_path_xy[idx_m + 1]
        xy01 = np.concatenate([p0, p1], axis=0).astype(np.float32)
        xy01_t = torch.from_numpy(xy01).to(device=device, dtype=torch.float32)
        t01 = model(xy01_t)[:, 0]
        t0 = t01[: p0.shape[0]]
        t1 = t01[p0.shape[0] :]
        ds = np.linalg.norm(p1 - p0, axis=1).astype(np.float32)
        f0 = float(cfg.get("rsa", {}).get("speed_free", 1.0))
        min_dt = torch.from_numpy(ds).to(device=device, dtype=torch.float32) / float(max(1e-6, f0))
        loss_mono = torch.mean(torch.relu(min_dt - (t1 - t0)) ** 2)

        # Distill weight should be active from the start to guide topology
        distill_weight = lambda_distill * (0.3 + 0.7 * float(progress))
        mono_dir_weight = lambda_mono_dir * float(progress)
        loss = (
            float(tcfg.get("lambda_sup", 1.0)) * loss_sup
            + float(lambda_phys_now) * loss_phys
            + float(tcfg.get("lambda_bc", 0.0)) * loss_bc
            + float(lambda_obs) * loss_obs
            + float(lambda_path) * loss_path
            + float(lambda_path_dir) * loss_path_dir
            + float(lambda_goal) * loss_goal
            + float(lambda_mono) * loss_mono
            + float(distill_weight) * loss_distill
            + float(mono_dir_weight) * loss_mono_dir
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_steps += 1

        if check_every > 0 and ((step + 1) % check_every == 0):
            if float(loss.detach().item()) <= tol:
                ok_count += 1
            else:
                ok_count = 0
            if ok_count >= patience:
                converged_step = train_steps
                break

    if converged_step < 0:
        converged_step = train_steps
    return model, converged_step, train_steps


def _train_vanilla_pinn(
    env: Maze2DEnv,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[VanillaTimeNN, int, int]:
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    model = VanillaTimeNN(
        hidden_dim=int(mcfg["hidden_dim"]),
        num_layers=int(mcfg["num_layers"]),
        activation=str(mcfg.get("activation", "tanh")),
    ).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))
    start_t = torch.tensor(start_xy, dtype=torch.float32, device=device)
    goal_t = torch.tensor(goal_xy, dtype=torch.float32, device=device)
    batch_size = int(tcfg["batch_size"])

    tol, patience, check_every = _get_convergence_cfg(cfg)
    ok_count = 0
    converged_step = -1
    train_steps = 0
    ocfg = tcfg.get("obstacle", {}) if isinstance(tcfg, dict) else {}
    sdf_band = float(ocfg.get("sdf_band", 0.05))
    lambda_int = float(ocfg.get("lambda_int", 100.0))
    lambda_grad = float(ocfg.get("lambda_grad", 10.0))
    lambda_dir = float(ocfg.get("lambda_dir", 50.0))
    lambda_vort = float(ocfg.get("lambda_vort", 20.0))
    sdf_scale = float(ocfg.get("sdf_scale", 10.0))
    lambda_obs = float(tcfg.get("lambda_obs", 1.0))
    lambda_goal = float(tcfg.get("lambda_goal", 5.0))
    phcfg = tcfg.get("physics", {}) if isinstance(tcfg, dict) else {}
    physics_mode = str(phcfg.get("mode", "autograd")).lower()
    fd_step = float(phcfg.get("fd_step", 0.01))
    max_steps = int(tcfg["max_steps"])
    warmup_steps = int(tcfg.get("warmup_steps", 0))
    curriculum_cfg = tcfg.get("curriculum", {}) if isinstance(tcfg, dict) else {}
    phys_ramp_start = float(curriculum_cfg.get("phys_ramp_start", 0.3))
    phys_ramp_end = float(curriculum_cfg.get("phys_ramp_end", 0.7))
    phys_hold = float(curriculum_cfg.get("phys_hold", 1.5))

    # Warmup: use Euclidean distance to goal as geometric supervision
    for step in range(warmup_steps):
        xy = env.sample_uniform(batch_size, rng=rng, device=device)
        with torch.no_grad():
            dist_target = torch.linalg.norm(xy - goal_t.view(1, 2), dim=-1)
        pred = model(xy)[:, 0]
        loss_geom = torch.mean((pred - dist_target) ** 2)
        loss_goal_bc = start_bc_loss(model, goal_t)
        loss = loss_geom + 5.0 * loss_goal_bc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_steps += 1

    # Curriculum: phase 1 = boundary-only, phase 2 = ramp physics, phase 3 = full
    for step in range(max_steps):
        progress = float(step + 1) / float(max(1, max_steps))
        if progress < phys_ramp_start:
            lambda_phys_now = 0.0
            lambda_bc_scale = 3.0
        elif progress < phys_ramp_end:
            frac = (progress - phys_ramp_start) / (phys_ramp_end - phys_ramp_start + 1e-12)
            lambda_phys_now = frac * float(tcfg.get("lambda_phys", 1.0))
            lambda_bc_scale = 3.0 - 1.5 * frac
        else:
            lambda_phys_now = phys_hold * float(tcfg.get("lambda_phys", 1.0))
            lambda_bc_scale = 1.5
        xy = env.sample_uniform(batch_size, rng=rng, device=device)
        loss_phys = (
            upwind_physics_loss(model=model, xy=xy, speed_fn=env.speed, fd_step=fd_step)
            if physics_mode == "upwind"
            else physics_loss(model=model, xy=xy, speed_fn=env.speed)
        )
        loss_goal_bc = start_bc_loss(model, goal_t)
        loss_obs = obstacle_loss(
            model=model,
            xy=xy,
            env_sdf_fn=env.sdf,
            speed_fn=env.speed,
            sdf_band=sdf_band,
            lambda_int=lambda_int,
            lambda_grad=lambda_grad,
            lambda_dir=lambda_dir,
            lambda_vort=lambda_vort,
            sdf_scale=sdf_scale,
        )
        loss = (
            float(lambda_phys_now) * loss_phys
            + float(tcfg.get("lambda_bc", 5.0)) * lambda_bc_scale * loss_goal_bc
            + float(lambda_obs) * loss_obs
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_steps += 1

        if check_every > 0 and ((step + 1) % check_every == 0):
            if float(loss.detach().item()) <= tol:
                ok_count += 1
            else:
                ok_count = 0
            if ok_count >= patience:
                converged_step = train_steps
                break

    if converged_step < 0:
        converged_step = train_steps
    return model, converged_step, train_steps


def _train_pntfield_2d(
    env: Maze2DEnv,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[PNTField2D, int, int]:
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    pcfg = mcfg.get("pntfield_2d", {}) if isinstance(mcfg, dict) else {}
    ptcfg = tcfg.get("pntfield_2d", {}) if isinstance(tcfg, dict) else {}
    model = PNTField2D(
        PNTField2DConfig(
            hidden_dim=int(pcfg.get("hidden_dim", mcfg["hidden_dim"])),
            num_blocks=int(pcfg.get("num_blocks", max(2, int(mcfg["num_layers"]) - 1))),
            fourier_dim=int(pcfg.get("fourier_dim", 32)),
            fourier_scale=float(pcfg.get("fourier_scale", 6.0)),
            activation=str(pcfg.get("activation", "silu")),
            # The pure neural baseline needs an exact zero-value goal boundary;
            # allowing signed output avoids collapsing into a near-constant positive floor.
            positive_output=False,
        )
    ).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=float(ptcfg.get("lr", tcfg["lr"])))
    start_t = torch.tensor(start_xy, dtype=torch.float32, device=device)
    goal_t = torch.tensor(goal_xy, dtype=torch.float32, device=device)
    batch_size = int(ptcfg.get("batch_size", tcfg["batch_size"]))

    tol, patience, check_every = _get_convergence_cfg(cfg)
    ok_count = 0
    converged_step = -1
    train_steps = 0
    ocfg = tcfg.get("obstacle", {}) if isinstance(tcfg, dict) else {}
    sdf_band = float(ocfg.get("sdf_band", 0.05))
    lambda_int = float(ocfg.get("lambda_int", 100.0))
    lambda_grad = float(ocfg.get("lambda_grad", 10.0))
    lambda_dir = float(ocfg.get("lambda_dir", 50.0))
    lambda_vort = float(ocfg.get("lambda_vort", 20.0))
    sdf_scale = float(ocfg.get("sdf_scale", 10.0))
    lambda_obs_end = float(ptcfg.get("lambda_obs", tcfg.get("lambda_obs", 1.0)))
    lambda_obs_start = float(ptcfg.get("lambda_obs_start", 0.25 * lambda_obs_end))
    lambda_phys_end = float(ptcfg.get("lambda_phys", tcfg.get("lambda_phys", 1.0)))
    lambda_phys_start = float(ptcfg.get("lambda_phys_start", 0.2 * lambda_phys_end))
    lambda_visc_start = float(ptcfg.get("lambda_visc_start", 1e-3))
    lambda_visc_end = float(ptcfg.get("lambda_visc_end", 1e-4))
    lambda_geom_start = float(ptcfg.get("lambda_geom_start", 1.0))
    lambda_geom_end = float(ptcfg.get("lambda_geom_end", 0.1))
    lambda_start_anchor = float(ptcfg.get("lambda_start_anchor", 5.0))
    lambda_dir_start = float(ptcfg.get("lambda_dir_start", 2.0))
    lambda_dir_end = float(ptcfg.get("lambda_dir_end", 0.5))
    alpha_dir = float(ptcfg.get("alpha_dir", 0.2))
    grad_clip_norm = float(ptcfg.get("grad_clip_norm", 1.0))
    lambda_goal = float(ptcfg.get("lambda_goal", 10.0))
    warmup_steps = int(ptcfg.get("warmup_steps", 500))
    lambda_antiflat_start = float(ptcfg.get("lambda_antiflat_start", 8.0))
    lambda_antiflat_end = float(ptcfg.get("lambda_antiflat_end", 2.0))
    grad_floor_start = float(ptcfg.get("grad_floor_start", 0.75))
    grad_floor_end = float(ptcfg.get("grad_floor_end", 0.20))
    lambda_rank_start = float(ptcfg.get("lambda_rank_start", 6.0))
    lambda_rank_end = float(ptcfg.get("lambda_rank_end", 2.0))
    rank_margin_start = float(ptcfg.get("rank_margin_start", 0.90))
    rank_margin_end = float(ptcfg.get("rank_margin_end", 0.30))
    lambda_warm_keep = float(ptcfg.get("lambda_warm_keep", 0.5))
    lambda_start_value = float(ptcfg.get("lambda_start_value", 25.0))
    lambda_shell = float(ptcfg.get("lambda_shell", 8.0))
    lambda_phys_local = float(ptcfg.get("lambda_phys_local", 1.5))
    lambda_global_rank_start = float(ptcfg.get("lambda_global_rank_start", 6.0))
    lambda_global_rank_end = float(ptcfg.get("lambda_global_rank_end", 2.0))
    global_rank_margin_start = float(ptcfg.get("global_rank_margin_start", 0.40))
    global_rank_margin_end = float(ptcfg.get("global_rank_margin_end", 0.15))
    phcfg = tcfg.get("physics", {}) if isinstance(tcfg, dict) else {}
    physics_mode = str(phcfg.get("mode", "autograd")).lower()
    fd_step = float(phcfg.get("fd_step", 0.01))
    local_frac = float(ptcfg.get("local_frac", 0.15))
    start_radius = float(ptcfg.get("start_radius", 0.18))
    goal_radius = float(ptcfg.get("goal_radius", 0.12))
    shell_radius = float(ptcfg.get("shell_radius", min(0.10, start_radius)))
    shell_count = int(ptcfg.get("shell_count", 256))
    lambda_visc_start *= 0.25
    lambda_visc_end *= 0.25
    lambda_obs_start *= 0.5
    lambda_obs_end *= 0.5
    # Supervised-first recovery stage: keep physics/obstacle weak until rollout becomes usable.
    lambda_phys_start *= 0.2
    lambda_phys_end *= 0.2
    lambda_phys_local *= 0.2
    lambda_obs_start *= 0.2
    lambda_obs_end *= 0.2
    grad_clip_norm = max(1.5, grad_clip_norm)
    start_value_target = float(np.linalg.norm(np.asarray(start_xy, dtype=np.float32) - np.asarray(goal_xy, dtype=np.float32)))

    def _sample_disk(center_xy: torch.Tensor, radius: float, count: int) -> torch.Tensor:
        if count <= 0:
            return torch.empty((0, 2), dtype=torch.float32, device=device)
        ang = rng.uniform(0.0, 2.0 * np.pi, size=(count,)).astype(np.float32)
        rad = np.sqrt(rng.uniform(0.0, 1.0, size=(count,)).astype(np.float32)) * float(radius)
        center_np = center_xy.view(1, 2).detach().cpu().numpy().astype(np.float32)
        pts = np.stack([np.cos(ang) * rad, np.sin(ang) * rad], axis=1).astype(np.float32) + center_np
        return torch.from_numpy(pts).to(device=device, dtype=torch.float32)

    def _sample_training_xy() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_local = int(min(batch_size // 3, max(64, round(batch_size * local_frac))))
        n_global = int(max(1, batch_size - 2 * n_local))
        xy_global = env.sample_uniform(n_global, rng=rng, device=device)
        xy_start = _sample_disk(start_t, radius=start_radius, count=n_local)
        xy_goal = _sample_disk(goal_t, radius=goal_radius, count=n_local)
        return torch.cat([xy_global, xy_start, xy_goal], dim=0), xy_start, xy_goal

    for _ in range(warmup_steps):
        xy, _, _ = _sample_training_xy()
        with torch.no_grad():
            euclid_target = torch.linalg.norm(xy - goal_t.view(1, 2), dim=-1)
        pred_xy = model(xy)[:, 0]
        loss_warm = torch.mean((pred_xy - euclid_target) ** 2)
        opt.zero_grad(set_to_none=True)
        loss_warm.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        opt.step()
        train_steps += 1

    for step in range(int(tcfg["max_steps"])):
        xy, xy_start_local, xy_goal_local = _sample_training_xy()
        progress = float(step + 1) / float(max(1, int(tcfg["max_steps"])))
        lambda_phys_now = lambda_phys_start + (lambda_phys_end - lambda_phys_start) * progress
        lambda_obs_now = lambda_obs_start + (lambda_obs_end - lambda_obs_start) * progress
        lambda_visc_now = lambda_visc_start + (lambda_visc_end - lambda_visc_start) * progress
        lambda_geom_now = lambda_geom_start + (lambda_geom_end - lambda_geom_start) * progress
        lambda_dir_now = lambda_dir_start + (lambda_dir_end - lambda_dir_start) * progress
        lambda_antiflat_now = lambda_antiflat_start + (lambda_antiflat_end - lambda_antiflat_start) * progress
        grad_floor_now = grad_floor_start + (grad_floor_end - grad_floor_start) * progress
        lambda_rank_now = lambda_rank_start + (lambda_rank_end - lambda_rank_start) * progress
        rank_margin_now = rank_margin_start + (rank_margin_end - rank_margin_start) * progress
        lambda_global_rank_now = lambda_global_rank_start + (lambda_global_rank_end - lambda_global_rank_start) * progress
        global_rank_margin_now = global_rank_margin_start + (global_rank_margin_end - global_rank_margin_start) * progress
        loss_phys = (
            upwind_physics_loss(model=model, xy=xy, speed_fn=env.speed, fd_step=fd_step)
            if physics_mode == "upwind"
            else physics_loss(model=model, xy=xy, speed_fn=env.speed)
        )
        loss_phys_local = (
            upwind_physics_loss(model=model, xy=xy_start_local, speed_fn=env.speed, fd_step=fd_step)
            if physics_mode == "upwind"
            else physics_loss(model=model, xy=xy_start_local, speed_fn=env.speed)
        )
        loss_bc = start_bc_loss(model, goal_t)
        loss_obs = obstacle_loss(
            model=model,
            xy=xy,
            env_sdf_fn=env.sdf,
            speed_fn=env.speed,
            sdf_band=sdf_band,
            lambda_int=lambda_int,
            lambda_grad=lambda_grad,
            lambda_dir=lambda_dir,
            lambda_vort=lambda_vort,
            sdf_scale=sdf_scale,
        )
        loss_visc = _laplacian_smoothness_loss(model=model, xy=xy, env=env)
        with torch.no_grad():
            free_mask = (env.sdf(xy) >= 0.0).to(dtype=xy.dtype)
            euclid_target = torch.linalg.norm(xy - goal_t.view(1, 2), dim=-1)
        pred_xy = model(xy)[:, 0]
        loss_geom = _masked_mean((pred_xy - euclid_target) ** 2, free_mask)
        with torch.no_grad():
            start_free = (env.sdf(xy_start_local) >= 0.0).to(dtype=xy_start_local.dtype)
            start_target = torch.linalg.norm(xy_start_local - goal_t.view(1, 2), dim=-1)
            goal_free = (env.sdf(xy_goal_local) >= 0.0).to(dtype=xy_goal_local.dtype)
        pred_start = model(xy_start_local)[:, 0]
        loss_start_anchor = _masked_mean((pred_start - start_target) ** 2, start_free)
        pred_goal_local = model(xy_goal_local)[:, 0]
        loss_goal_local = _masked_mean(pred_goal_local**2, goal_free)
        shell_xy = _sample_disk(start_t, radius=shell_radius, count=shell_count)
        with torch.no_grad():
            shell_free = (env.sdf(shell_xy) >= 0.0).to(dtype=shell_xy.dtype)
            shell_target = torch.linalg.norm(shell_xy - goal_t.view(1, 2), dim=-1)
        pred_shell = model(shell_xy)[:, 0]
        loss_shell = _masked_mean((pred_shell - shell_target) ** 2, shell_free)
        pred_start_exact = model(start_t.view(1, 2))[:, 0]
        start_target_exact = torch.full_like(pred_start_exact, start_value_target)
        loss_start_value = torch.mean((pred_start_exact - start_target_exact) ** 2)
        loss_warm_keep = loss_start_anchor + loss_goal_local + 0.5 * loss_shell + 0.25 * loss_geom
        loss_antiflat = _gradient_floor_loss(
            model=model,
            xy=xy_start_local,
            env=env,
            grad_floor=grad_floor_now,
        )
        mean_start = _masked_mean(pred_start, start_free)
        mean_goal = _masked_mean(pred_goal_local, goal_free)
        loss_rank = torch.relu(float(rank_margin_now) - (mean_start - mean_goal)) ** 2
        loss_global_rank = _distance_ranking_loss(
            pred=pred_xy,
            dist_to_goal=euclid_target,
            free_mask=free_mask,
            margin=global_rank_margin_now,
            rng=rng,
            max_pairs=512,
        )
        local_xy = xy_start_local
        with torch.no_grad():
            d_goal_local = goal_t.view(1, 2) - local_xy
            d_goal_local = d_goal_local / (torch.linalg.norm(d_goal_local, dim=-1, keepdim=True) + 1e-12)
        loss_dir_geom = loss_monotonicity(
            model=model,
            xy=local_xy,
            d_target=d_goal_local,
            env_sdf_fn=env.sdf,
            alpha=alpha_dir,
        )
        loss = (
            float(lambda_phys_now) * loss_phys
            + float(lambda_phys_local) * loss_phys_local
            + float(lambda_goal) * loss_bc
            + float(lambda_obs_now) * loss_obs
            + float(lambda_visc_now) * loss_visc
            + float(lambda_geom_now) * loss_geom
            + float(lambda_start_anchor) * loss_start_anchor
            + float(lambda_start_value) * loss_start_value
            + float(lambda_shell) * loss_shell
            + float(lambda_warm_keep) * loss_warm_keep
            + float(lambda_antiflat_now) * loss_antiflat
            + float(lambda_rank_now) * loss_rank
            + float(lambda_global_rank_now) * loss_global_rank
            + float(lambda_dir_now) * loss_dir_geom
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        opt.step()
        train_steps += 1

        if check_every > 0 and ((step + 1) % check_every == 0):
            if float(loss.detach().item()) <= tol:
                ok_count += 1
            else:
                ok_count = 0
            if ok_count >= patience:
                converged_step = train_steps
                break

    if converged_step < 0:
        converged_step = train_steps
    return model, converged_step, train_steps


def _summarize(metrics_all: List[Dict[str, EvalMetrics]]) -> Dict[str, Dict[str, float]]:
    names = sorted({k for m in metrics_all for k in m.keys()})
    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        success = np.asarray([float(m[name].success) for m in metrics_all], dtype=np.float32)
        gap = np.asarray([m[name].optimality_gap for m in metrics_all], dtype=np.float32)
        gns = np.asarray(
            [m[name].grad_norm_start if m[name].grad_norm_start is not None else np.nan for m in metrics_all],
            dtype=np.float32,
        )
        csteps = np.asarray(
            [m[name].converge_steps if m[name].converge_steps is not None else np.nan for m in metrics_all],
            dtype=np.float32,
        )
        time_cost_arr = np.asarray(
            [m[name].time_cost if np.isfinite(m[name].time_cost) else np.nan for m in metrics_all],
            dtype=np.float32,
        )
        eff_arr = np.asarray(
            [m[name].efficiency_ratio if np.isfinite(m[name].efficiency_ratio) else np.nan for m in metrics_all],
            dtype=np.float32,
        )
        phys_arr = np.asarray(
            [m[name].physical_consistency if np.isfinite(m[name].physical_consistency) else np.nan for m in metrics_all],
            dtype=np.float32,
        )
        curv_arr = np.asarray(
            [m[name].curvature_sharpness if np.isfinite(m[name].curvature_sharpness) else np.nan for m in metrics_all],
            dtype=np.float32,
        )
        gap = np.where(np.isfinite(gap), gap, np.nan)
        gns = np.where(np.isfinite(gns), gns, np.nan)
        csteps = np.where(np.isfinite(csteps), csteps, np.nan)
        gap_mean = float(np.nanmean(gap)) if np.any(np.isfinite(gap)) else float("nan")
        gns_mean = float(np.nanmean(gns)) if np.any(np.isfinite(gns)) else float("nan")
        csteps_mean = float(np.nanmean(csteps)) if np.any(np.isfinite(csteps)) else float("nan")
        time_cost_mean = float(np.nanmean(time_cost_arr)) if np.any(np.isfinite(time_cost_arr)) else float("nan")
        eff_mean = float(np.nanmean(eff_arr)) if np.any(np.isfinite(eff_arr)) else float("nan")
        phys_mean = float(np.nanmean(phys_arr)) if np.any(np.isfinite(phys_arr)) else float("nan")
        curv_mean = float(np.nanmean(curv_arr)) if np.any(np.isfinite(curv_arr)) else float("nan")
        out[name] = {
            "SR_mean": float(np.mean(success)),
            "OptimalityGap_mean": gap_mean,
            "GradNormStart_mean": gns_mean,
            "ConvergeSteps_mean": csteps_mean,
            "TimeCost_mean": time_cost_mean,
            "EfficiencyRatio_mean": eff_mean,
            "PhysicalConsistency_mean": phys_mean,
            "CurvatureSharpness_mean": curv_mean,
        }
    return out


def generate_random_obstacles(
    seed: int,
    num_obstacles: int = 8,
    bounds: Optional[Bounds2D] = None,
) -> List[Dict]:
    """Generate a random obstacle layout for distribution-shift testing.

    This is a stub / interface placeholder. Returns a list of obstacle dicts
    compatible with Maze2DEnv. Actual random generation logic to be filled in.

    Args:
        seed: random seed for reproducibility
        num_obstacles: number of random box obstacles to generate
        bounds: environment bounds (used to constrain obstacle placement)

    Returns:
        List of obstacle dicts, each with keys: kind, center, half_size, rotation_rad
    """
    rng = np.random.default_rng(seed)
    obs = []
    for _ in range(num_obstacles):
        w = rng.uniform(0.05, 0.15)
        h = rng.uniform(0.05, 0.15)
        if bounds is not None:
            cx = rng.uniform(bounds.x_min + 0.15, bounds.x_max - 0.15)
            cy = rng.uniform(bounds.y_min + 0.15, bounds.y_max - 0.15)
        else:
            cx = rng.uniform(0.1, 0.9)
            cy = rng.uniform(0.1, 0.9)
        obs.append({
            "kind": "box",
            "center": [float(cx), float(cy)],
            "half_size": [float(w), float(h)],
        })
    return obs


def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    device = torch.device(cfg.get("device", "cpu"))
    env = _make_env(cfg, device=device)

    start_xy = tuple(cfg["env"]["start"])
    goal_xy = tuple(cfg["env"]["goal"])

    out_dir = str(cfg["eval"].get("out_dir", "outputs"))
    os.makedirs(out_dir, exist_ok=True)

    num_seeds = int(cfg.get("num_seeds", 30))
    metrics_all: List[Dict[str, EvalMetrics]] = []

    # Build coupling model for RHP-PINN rollout blending
    coupling_cfg = cfg.get("coupling", {})
    coupling_type = str(coupling_cfg.get("type", "physics"))
    coupling_alpha = 0.3  # fallback (only relevant for constant mode)
    if coupling_type == "constant":
        coupling_alpha = float(coupling_cfg.get("alpha", 0.3))
        coupling_model = ConstantCoupling(alpha=coupling_alpha)
    else:
        coupling_model = PhysicsGuidedCoupling(
            w_resid=float(coupling_cfg.get("w_resid", 1.0)),
            w_conf=float(coupling_cfg.get("w_conf", 2.0)),
            w_safety=float(coupling_cfg.get("w_safety", 3.0)),
            b=float(coupling_cfg.get("b", -1.0)),
            k=float(coupling_cfg.get("k", 8.0)),
        )

    for s in range(num_seeds):
        rng = _set_seed(int(cfg.get("seed", 0)) + s)

        xs, ys = _grid_coords(env.bounds, tuple(cfg["rsa"]["grid_size"]))
        speed = env.speed_grid(tuple(cfg["rsa"]["grid_size"]))
        free_mask = speed > 0.0

        rsa_engine = RSAEngine(xs=xs, ys=ys, connectivity=int(cfg["rsa"].get("connectivity", 8)))
        rsa_low = rsa_engine.solve(speed=speed, start_xy=start_xy, goal_xy=goal_xy)
        weight = RSAEngine.wavefront_weight(
            t=rsa_low.t,
            free_mask=free_mask,
            base_weight=float(cfg["train"]["adaptive_sampling"].get("base_weight", 0.2)),
            wavefront_power=float(cfg["train"]["adaptive_sampling"].get("wavefront_power", 1.0)),
            obstacle_band=float(cfg["train"]["adaptive_sampling"].get("obstacle_band", 0.0)),
            xs=xs,
            ys=ys,
        )
        xx, yy = np.meshgrid(xs, ys)
        xy_grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)
        with torch.no_grad():
            sdf_grid = env.sdf(torch.from_numpy(xy_grid).to(device=device)).cpu().numpy().reshape(speed.shape)
        weight = apply_sdf_surface_boost(
            weight=weight,
            sdf_grid=sdf_grid,
            band=float(cfg["train"].get("sdf_surface_band", 0.05)),
            boost=float(cfg["train"].get("sdf_surface_boost", 0.5)),
        )

        # --- RHP-PINN: Enhanced Adaptive Sampling ---
        # 1. Boost sampling near velocity field gradients (sharp changes)
        if env.has_velocity_field():
            v_grid = env.velocity_grid(tuple(cfg["rsa"]["grid_size"]))
            if v_grid is not None:
                vg_y, vg_x = np.gradient(v_grid)
                v_grad_mag = np.sqrt(vg_x**2 + vg_y**2)
                v_boost = (v_grad_mag / (np.max(v_grad_mag) + 1e-12)) * 2.0
                weight = weight * (1.0 + v_boost)

        # 2. Boost sampling in narrow channels (low positive SDF)
        narrow_mask = (sdf_grid > 0.0) & (sdf_grid < 0.04)
        weight = np.where(narrow_mask, weight * 3.0, weight)

        xs_ref, ys_ref = _grid_coords(env.bounds, tuple(cfg["eval"]["reference_grid_size"]))
        speed_ref = env.speed_grid(tuple(cfg["eval"]["reference_grid_size"]))
        rsa_ref = RSAEngine(xs=xs_ref, ys=ys_ref, connectivity=int(cfg["rsa"].get("connectivity", 8))).solve(
            speed=speed_ref,
            start_xy=start_xy,
            goal_xy=goal_xy,
        )
        ref_path_xy = rsa_ref.backtrack_path_xy()

        vanilla, vanilla_conv, vanilla_steps = _train_vanilla_pinn(env, start_xy, goal_xy, cfg, rng, device=device)
        pntfield, pntfield_conv, pntfield_steps = _train_pntfield_2d(env, start_xy, goal_xy, cfg, rng, device=device)
        rhp, rhp_conv, rhp_steps = _train_rhp(
            env,
            start_xy,
            goal_xy,
            rsa_ref,
            ref_path_xy,
            xs,
            ys,
            weight,
            cfg,
            rng,
            device=device,
            coupling_alpha=coupling_alpha,
        )

        models = {"vanilla_pinn": vanilla, "pntfield_2d": pntfield, "rhp_pinn": rhp}
        metrics, paths = evaluate_methods(
            env=env,
            start_xy=start_xy,
            goal_xy=goal_xy,
            rsa_low=rsa_low,
            rsa_ref=rsa_ref,
            models=models,
            path_step=float(cfg["eval"]["path_step"]),
            max_path_steps=int(cfg["eval"]["max_path_steps"]),
            goal_tol=float(cfg["eval"]["goal_tol"]),
            grad_probe_radius=float(cfg["eval"]["grad_probe_radius"]),
            grad_probe_points=int(cfg["eval"]["grad_probe_points"]),
            strict_goal_tol=bool(cfg["eval"].get("strict_goal_tol", False)),
            coupling_model=coupling_model,
            coupling_default_alpha=coupling_alpha,
        )
        metrics["vanilla_pinn"] = EvalMetrics(
            success=metrics["vanilla_pinn"].success,
            length=metrics["vanilla_pinn"].length,
            smoothness=metrics["vanilla_pinn"].smoothness,
            optimality_gap=metrics["vanilla_pinn"].optimality_gap,
            regret=metrics["vanilla_pinn"].regret,
            coupling_alpha=metrics["vanilla_pinn"].coupling_alpha,
            grad_norm_start=metrics["vanilla_pinn"].grad_norm_start,
            converge_steps=int(vanilla_conv),
            train_steps=int(vanilla_steps),
            time_cost=metrics["vanilla_pinn"].time_cost,
            efficiency_ratio=metrics["vanilla_pinn"].efficiency_ratio,
            physical_consistency=metrics["vanilla_pinn"].physical_consistency,
            curvature_sharpness=metrics["vanilla_pinn"].curvature_sharpness,
        )
        metrics["rhp_pinn"] = EvalMetrics(
            success=metrics["rhp_pinn"].success,
            length=metrics["rhp_pinn"].length,
            smoothness=metrics["rhp_pinn"].smoothness,
            optimality_gap=metrics["rhp_pinn"].optimality_gap,
            regret=metrics["rhp_pinn"].regret,
            coupling_alpha=metrics["rhp_pinn"].coupling_alpha,
            grad_norm_start=metrics["rhp_pinn"].grad_norm_start,
            converge_steps=int(rhp_conv),
            train_steps=int(rhp_steps),
            time_cost=metrics["rhp_pinn"].time_cost,
            efficiency_ratio=metrics["rhp_pinn"].efficiency_ratio,
            physical_consistency=metrics["rhp_pinn"].physical_consistency,
            curvature_sharpness=metrics["rhp_pinn"].curvature_sharpness,
        )
        metrics["pntfield_2d"] = EvalMetrics(
            success=metrics["pntfield_2d"].success,
            length=metrics["pntfield_2d"].length,
            smoothness=metrics["pntfield_2d"].smoothness,
            optimality_gap=metrics["pntfield_2d"].optimality_gap,
            regret=metrics["pntfield_2d"].regret,
            coupling_alpha=metrics["pntfield_2d"].coupling_alpha,
            grad_norm_start=metrics["pntfield_2d"].grad_norm_start,
            converge_steps=int(pntfield_conv),
            train_steps=int(pntfield_steps),
            time_cost=metrics["pntfield_2d"].time_cost,
            efficiency_ratio=metrics["pntfield_2d"].efficiency_ratio,
            physical_consistency=metrics["pntfield_2d"].physical_consistency,
            curvature_sharpness=metrics["pntfield_2d"].curvature_sharpness,
        )
        metrics_all.append(metrics)

        if bool(cfg["eval"].get("save_plots", True)) and (s == 0):
            out_path = os.path.join(out_dir, f"fields_paths_seed{int(cfg.get('seed', 0))}.png")
            v_grid = env.velocity_grid(tuple(cfg["rsa"]["grid_size"]))
            save_field_and_paths_plot(
                env=env,
                rsa=rsa_low,
                start_xy=start_xy,
                goal_xy=goal_xy,
                models=models,
                paths=paths,
                out_path=out_path,
                velocity_grid=v_grid,
            )
            max_len = float(cfg["eval"]["path_step"]) * float(cfg["eval"]["max_path_steps"])
            if (not metrics["rhp_pinn"].success) and (float(metrics["rhp_pinn"].length) >= 0.98 * max_len):
                qpath = os.path.join(out_dir, f"fields_quiver_seed{int(cfg.get('seed', 0))}.png")
                save_field_quiver_plot(
                    env=env,
                    rsa=rsa_low,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    model=models["rhp_pinn"],
                    out_path=qpath,
                    stride=4,
                )

    summary = _summarize(metrics_all)
    results = {
        "config": cfg,
        "summary": summary,
        "per_seed": [
            {k: asdict(v) for k, v in metrics.items()}
            for metrics in metrics_all
        ],
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    archive_dir = _archive_outputs(cfg=cfg, out_dir=out_dir)
    results["archive_dir"] = str(archive_dir)
    return results


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Save model state dict to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> bool:
    """Load model state dict from disk. Returns True if successful."""
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    ap.add_argument("--exp-dir", type=str, default=None,
                    help="If set, write results directly to this directory "
                         "(used by ExperimentRunner).")
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.exp_dir is not None:
        cfg.setdefault("eval", {})["out_dir"] = os.path.join(args.exp_dir, "rhp_outputs")
    results = run(cfg)
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
