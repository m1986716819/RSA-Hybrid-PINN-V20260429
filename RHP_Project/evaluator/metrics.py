"""Evaluation metrics and gradient rollout for path planning.

Provides:
- Path geometry metrics (length, smoothness, curvature)
- Time-aware metrics (time cost, efficiency, physical consistency)
- Gradient rollout with optional RSA / PINN coupling
- Multi-method evaluation harness
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from ..envs.maze_2d import Maze2DEnv
from ..solvers.coupling import blend_directions, compute_coupling_features
from ..solvers.rsa_engine import RSAResult


# ---------------------------------------------------------------
# Path geometry helpers
# ---------------------------------------------------------------


def path_length(path_xy: np.ndarray) -> float:
    if path_xy.shape[0] < 2:
        return float("inf")
    diffs = path_xy[1:] - path_xy[:-1]
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def path_smoothness(path_xy: np.ndarray) -> float:
    if path_xy.shape[0] < 3:
        return float("inf")
    v1 = path_xy[1:-1] - path_xy[:-2]
    v2 = path_xy[2:] - path_xy[1:-1]
    n1 = np.linalg.norm(v1, axis=1) + 1e-12
    n2 = np.linalg.norm(v2, axis=1) + 1e-12
    c = np.sum(v1 * v2, axis=1) / (n1 * n2)
    c = np.clip(c, -1.0, 1.0)
    ang = np.arccos(c)
    return float(np.mean(np.abs(ang)))


def path_time_cost(path_xy: np.ndarray, velocity_field_np_fn: Optional[Callable] = None) -> float:
    if path_xy.shape[0] < 2:
        return float("inf")
    if velocity_field_np_fn is None:
        return path_length(path_xy)
    total = 0.0
    for i in range(path_xy.shape[0] - 1):
        seg = path_xy[i + 1] - path_xy[i]
        ds = float(np.linalg.norm(seg))
        mid = (path_xy[i] + path_xy[i + 1]) * 0.5
        v = float(velocity_field_np_fn(mid))
        if v > 1e-8:
            total += ds / v
        else:
            return float("inf")
    return total


def path_eikonal_residual(
    path_xy: np.ndarray,
    model: torch.nn.Module,
    env: Maze2DEnv,
    n_samples: int = 100,
) -> float:
    if path_xy.shape[0] < 3:
        return float("nan")
    n = min(n_samples, path_xy.shape[0])
    idx = np.linspace(0, path_xy.shape[0] - 1, n, dtype=np.int64)
    device = next(model.parameters()).device
    xy = torch.from_numpy(path_xy[idx].astype(np.float32)).to(device=device)
    xy.requires_grad_(True)
    t = model(xy)
    if t.ndim == 2:
        t = t[:, 0]
    grad = torch.autograd.grad(t.sum(), xy, create_graph=False, retain_graph=False)[0]
    grad_norm = torch.sqrt(torch.sum(grad**2, dim=-1) + 1e-12)
    with torch.no_grad():
        speed_val = env.wave_speed(xy.detach())
        inv_speed = 1.0 / torch.clamp(speed_val, min=1e-6)
    residual = torch.abs(grad_norm - inv_speed).mean().detach().cpu().item()
    return float(residual)


def path_curvature(path_xy: np.ndarray) -> Tuple[float, float, float]:
    if path_xy.shape[0] < 4:
        return (float("nan"), float("nan"), float("nan"))
    v = path_xy[1:] - path_xy[:-1]
    speed = np.linalg.norm(v, axis=1)
    a = v[1:] - v[:-1]
    accel = np.linalg.norm(a, axis=1)
    speed_trough = float(np.percentile(speed, 10))
    sharpness = float(np.percentile(accel, 90))
    avg_accel = float(np.mean(accel))
    return (speed_trough, sharpness, avg_accel)


def grad_norm_near_start(
    model: torch.nn.Module,
    start_xy: Tuple[float, float],
    radius: float,
    num_points: int,
) -> float:
    device = next(model.parameters()).device
    rng = np.random.default_rng(0)
    ang = rng.uniform(0.0, 2 * np.pi, size=(num_points,)).astype(np.float32)
    pts = np.stack([np.cos(ang), np.sin(ang)], axis=1) * float(radius) + np.asarray(start_xy, dtype=np.float32)
    xy = torch.from_numpy(pts).to(device=device, dtype=torch.float32).requires_grad_(True)
    t = model(xy)
    if t.ndim == 2:
        t = t[:, 0]
    grad = torch.autograd.grad(t.sum(), xy, create_graph=False, retain_graph=False)[0]
    gnorm = torch.linalg.norm(grad, dim=-1).detach().cpu().numpy()
    gnorm = gnorm[np.isfinite(gnorm)]
    if gnorm.size == 0:
        return float("nan")
    return float(np.mean(gnorm))


# ---------------------------------------------------------------
# Gradient rollout with optional coupling
# ---------------------------------------------------------------


def integrate_path_by_grad(
    model: torch.nn.Module,
    env: Maze2DEnv,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    step_size: float,
    max_steps: int,
    goal_tol: float,
    rsa_field: Optional[RSAResult] = None,
    rsa_path_xy: Optional[np.ndarray] = None,
    coupling_model: Optional[torch.nn.Module] = None,
    default_alpha: float = 0.3,
    strict_goal_tol: bool = False,
) -> Tuple[np.ndarray, bool, float]:
    """Roll out a path by following the model's gradient field.

    When rsa_path_xy and coupling_model are provided, the step direction is
    a continuous blend of PINN and RSA directions:

        v = (1 - alpha) * v_pinn + alpha * v_rsa

    where alpha = coupling_model(x). Without coupling, pure PINN rollout is used
    with a simple safety slide that follows obstacle boundaries.

    Returns:
        path_xy: (N, 2) array of rollout waypoints
        success: whether the final point reached the goal
        mean_alpha: average coupling factor over the rollout (NaN if pure PINN)
    """
    device = next(model.parameters()).device
    x = torch.tensor(start_xy, dtype=torch.float32, device=device).view(1, 2)
    goal = torch.tensor(goal_xy, dtype=torch.float32, device=device).view(1, 2)
    pts = [x.detach().cpu().numpy()[0]]

    has_coupling = coupling_model is not None
    has_rsa = rsa_path_xy is not None and rsa_path_xy.ndim == 2 and rsa_path_xy.shape[0] >= 2
    if has_rsa:
        path_arr = np.asarray(rsa_path_xy, dtype=np.float32)
    else:
        path_arr = None

    alpha_accum = []
    goal_threshold = float(goal_tol) if bool(strict_goal_tol) else max(float(goal_tol), 0.17)

    # ── Fix #3: slide momentum buffer to prevent oscillation ──
    _prev_slide_dir: Optional[np.ndarray] = None
    _oscillation_counter: int = 0

    def _normalize(v: torch.Tensor) -> torch.Tensor:
        return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-12)

    def _line_search(x_cur: torch.Tensor, v_dir: torch.Tensor, base_h: float) -> Tuple[torch.Tensor, bool]:
        for i in range(8):
            h = float(base_h) * (0.5**i)
            x_try = x_cur + h * v_dir
            with torch.no_grad():
                if bool(env.is_free(x_try).item()):
                    return x_try, True
        return x_cur, False

    def _slide_step(x_cur: torch.Tensor, v_dir: torch.Tensor, sdf_val: float) -> Tuple[torch.Tensor, bool]:
        nonlocal _prev_slide_dir, _oscillation_counter

        # Fix #4: adaptive step size based on SDF
        base_h = float(step_size)
        if sdf_val < 0.03:
            base_h = float(step_size) * 0.5
        elif sdf_val < 0.08:
            base_h = float(step_size) * 0.75

        # First try direct line search
        x_next, ok = _line_search(x_cur, v_dir, base_h)
        if ok:
            _oscillation_counter = max(0, _oscillation_counter - 1)
            return x_next, True

        # Slide along obstacle boundary
        x_s = x_cur.detach().requires_grad_(True)
        sdf_s = env.sdf(x_s)
        if sdf_s.ndim == 2:
            sdf_s = sdf_s[:, 0]
        if not torch.isfinite(sdf_s).all():
            return x_cur, False
        grad_sdf = torch.autograd.grad(sdf_s.sum(), x_s, create_graph=False, retain_graph=False)[0]
        n = _normalize(grad_sdf)
        dot = torch.sum(v_dir * n, dim=-1, keepdim=True)
        v_slide_raw = v_dir - dot * n
        v_slide = _normalize(v_slide_raw)

        # Fix #3: anti-oscillation with momentum
        slide_np = v_slide.detach().cpu().numpy()[0].astype(np.float32)
        if _prev_slide_dir is not None:
            cos_sim = float(np.dot(slide_np, _prev_slide_dir))
            if cos_sim < -0.7:
                # Direction reversal detected — we're oscillating.
                # Force a forward-biased direction using the momentum buffer.
                _oscillation_counter += 1
                v_forward = _normalize((torch.from_numpy(_prev_slide_dir).to(device=device, dtype=torch.float32).view(1, 2) + v_dir) * 0.5)
                x_next, ok = _line_search(x_cur, v_forward, base_h * 0.5)
                if ok:
                    return x_next, True
                # If oscillation persists, try larger forward momentum step
                if _oscillation_counter >= 3:
                    v_forced = v_dir * 0.5 + torch.from_numpy(_prev_slide_dir).to(device=device, dtype=torch.float32).view(1, 2) * 0.5
                    v_forced = _normalize(v_forced)
                    x_next, ok = _line_search(x_cur, v_forced, base_h * 0.25)
                    if ok:
                        _oscillation_counter = 0
                        return x_next, True
            else:
                _oscillation_counter = max(0, _oscillation_counter - 1)

        _prev_slide_dir = slide_np.copy()

        x_next, ok = _line_search(x_cur, v_slide, base_h)
        if ok:
            return x_next, True

        # Last resort: try the original direction with a tiny step
        for i in range(6):
            h = float(step_size) * (0.25 ** i)
            x_try = x_cur + h * v_dir
            with torch.no_grad():
                if bool(env.is_free(x_try).item()):
                    return x_try, True

        return x_cur, False

    def _lookahead_dir(cur_xy_np: np.ndarray) -> Tuple[np.ndarray, float, int]:
        if path_arr is None:
            return np.zeros(2, dtype=np.float32), 0.0, 0
        d2 = np.sum((path_arr - cur_xy_np.reshape(1, 2)) ** 2, axis=1)
        idx = int(np.argmin(d2))
        dist_to_path = float(np.sqrt(d2[idx]))

        # Fix #2: SDF gating — if the nearest path point is too close to obstacles,
        # step back along the path to a safer point.
        with torch.no_grad():
            nearest_pt = path_arr[idx]
            sdf_check = env.sdf(torch.from_numpy(nearest_pt.reshape(1, 2).astype(np.float32)).to(device))
            if sdf_check.ndim == 2:
                sdf_check = sdf_check[:, 0]
            nearest_sdf = float(sdf_check.detach().cpu().numpy()[0])

        if nearest_sdf < 0.015:
            # Find a safer point (SDF >= 0.03) by walking backwards along the path
            for offset in range(1, min(20, idx + 1)):
                safer_pt = path_arr[idx - offset]
                with torch.no_grad():
                    sdf_check2 = env.sdf(torch.from_numpy(safer_pt.reshape(1, 2).astype(np.float32)).to(device))
                    if sdf_check2.ndim == 2:
                        sdf_check2 = sdf_check2[:, 0]
                    if float(sdf_check2.detach().cpu().numpy()[0]) >= 0.03:
                        idx = idx - offset
                        break

        lookahead = min(40, path_arr.shape[0] - 1)
        j = min(idx + lookahead, path_arr.shape[0] - 1)
        d = (path_arr[j] - cur_xy_np).astype(np.float32)
        dn = float(np.linalg.norm(d) + 1e-12)
        return (d / dn).astype(np.float32), dist_to_path, idx

    # State for decision chain
    rsa_streak = 0
    progress_stall_count = 0
    last_path_idx = 0
    vf_active = env.has_velocity_field()

    for step in range(int(max_steps)):
        x = x.detach().requires_grad_(True)
        t = model(x)
        if t.ndim == 2:
            t = t[:, 0]
        sdf = env.sdf(x)
        if sdf.ndim == 2:
            sdf = sdf[:, 0]
        sdf_item = float(sdf.detach().cpu().numpy()[0])
        if not torch.isfinite(t).all() or not torch.isfinite(sdf).all():
            pts.append(x.detach().cpu().numpy()[0])
            break

        joint = torch.stack([t.sum(), sdf.sum()], dim=0)
        grad_out = torch.eye(2, dtype=joint.dtype, device=device)
        grads = torch.autograd.grad(
            outputs=joint,
            inputs=x,
            grad_outputs=grad_out,
            is_grads_batched=True,
            create_graph=False,
            retain_graph=False,
        )[0]
        grad_t = grads[0]
        if not torch.isfinite(grad_t).all():
            pts.append(x.detach().cpu().numpy()[0])
            break

        gnorm = torch.linalg.norm(grad_t, dim=-1, keepdim=True) + 1e-12
        v_pinn = -grad_t / gnorm

        if has_coupling and has_rsa:
            cur_xy_np = x.detach().cpu().numpy()[0].astype(np.float32)
            d_rsa_np, d_path, path_idx = _lookahead_dir(cur_xy_np)
            v_rsa = torch.tensor(d_rsa_np, dtype=torch.float32, device=device).view(1, 2)

            # Compute physics features for adaptive coupling
            speed_val = env.wave_speed(x.detach()).view(-1, 1)
            sdf_for_feat = sdf.view(-1, 1) if sdf.dim() == 1 else sdf
            features = compute_coupling_features(
                pinn_grad_norm=gnorm,
                v_pinn=v_pinn,
                v_rsa=v_rsa,
                sdf_val=sdf_for_feat,
                speed_val=speed_val,
            )
            with torch.no_grad():
                alpha = coupling_model(features).view(1, 1)

            # --- RHP-PINN Decision Chain (Multiple Coverage Gating) ---
            gating_ratio = float(alpha.item())
            
            # 1. gate_entry_protect: Narrow channel force RSA
            # Original threshold 0.06, reduced to 0.03 per instructions
            if d_path < 0.03:
                gating_ratio = 1.0
            
            # 2. progress_stall_count: Force RSA rescue if stuck
            if path_idx <= last_path_idx:
                progress_stall_count += 1
            else:
                progress_stall_count = 0
            if progress_stall_count >= 5:
                gating_ratio = 1.0
            
            # 3. rsa_streak: Cooldown to force PINN attempt
            if gating_ratio > 0.5:
                rsa_streak += 1
            else:
                rsa_streak = 0
            if rsa_streak >= 15:
                gating_ratio = 0.0 # Force PINN attempt
                rsa_streak = 0
            
            # 4. vf_autonomy: Velocity field gain awareness
            if vf_active:
                # Calculate velocity gain: V(next_point_pinn) - V(current_point)
                with torch.no_grad():
                    v_gain_h = 0.05
                    x_pinn_next = x + v_gain_h * v_pinn
                    v_next = float(env.wave_speed(x_pinn_next).item())
                    v_cur = float(speed_val.item())
                    vf_gain = v_next - v_cur
                
                if vf_gain > 0.01:
                    # Favor PINN in fast zones (+0.2 bias for PINN means -0.2 for gating_ratio)
                    gating_ratio = max(0.0, gating_ratio - 0.2)
            
            last_path_idx = path_idx
            alpha_accum.append(gating_ratio)
            v = blend_directions(v_pinn, v_rsa, torch.tensor([[gating_ratio]], device=device))
        else:
            v = v_pinn

        x_next, ok_free = _slide_step(x, v, sdf_item)

        if not ok_free:
            pts.append(x_next.detach().cpu().numpy()[0])
            break
        if not torch.isfinite(x_next).all():
            pts.append(x_next.detach().cpu().numpy()[0])
            break
        with torch.no_grad():
            if not bool(env.is_free(x_next).item()):
                pts.append(x_next.detach().cpu().numpy()[0])
                break
            if torch.linalg.norm(x_next - goal).item() <= goal_threshold:
                pts.append(x_next.detach().cpu().numpy()[0])
                return _make_result(pts, True, alpha_accum)

        x = x_next
        pts.append(x.detach().cpu().numpy()[0])

    ok = np.linalg.norm(np.asarray(pts[-1], dtype=np.float32) - np.asarray(goal_xy, dtype=np.float32)) <= goal_threshold
    return _make_result(pts, ok, alpha_accum)


def _make_result(
    pts: list,
    success: bool,
    alpha_accum: list,
) -> Tuple[np.ndarray, bool, float]:
    path = np.asarray(pts, dtype=np.float32)
    mean_alpha = float(np.mean(alpha_accum)) if alpha_accum else float("nan")
    return path, success, mean_alpha


# ---------------------------------------------------------------
# EvalMetrics and evaluate_methods
# ---------------------------------------------------------------


@dataclass(frozen=True)
class EvalMetrics:
    success: bool
    length: float
    smoothness: float
    optimality_gap: float
    regret: float = 0.0
    coupling_alpha: Optional[float] = None
    grad_norm_start: Optional[float] = None
    converge_steps: Optional[int] = None
    train_steps: Optional[int] = None
    time_cost: float = 0.0
    efficiency_ratio: float = 1.0
    physical_consistency: float = 0.0
    curvature_sharpness: float = 0.0


def evaluate_methods(
    env: Maze2DEnv,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    rsa_low: RSAResult,
    rsa_ref: RSAResult,
    models: Dict[str, torch.nn.Module],
    path_step: float,
    max_path_steps: int,
    goal_tol: float,
    grad_probe_radius: float,
    grad_probe_points: int,
    strict_goal_tol: bool = False,
    coupling_model: Optional[torch.nn.Module] = None,
    coupling_default_alpha: float = 0.3,
) -> Tuple[Dict[str, EvalMetrics], Dict[str, np.ndarray]]:
    """Evaluate all methods on a single environment seed.

    Args:
        coupling_model: optional continuous coupling model. When provided,
            the RHP-PINN method uses blended RSA + PINN rollout.
        coupling_default_alpha: default alpha for ConstantCoupling if no
            coupling_model is provided but coupling is desired.

    Returns:
        metrics: dict of method_name -> EvalMetrics
        paths: dict of method_name -> (N, 2) path array
    """
    ref_path = rsa_ref.backtrack_path_xy()
    ref_len = path_length(ref_path)

    velocity_field_np_fn = env.velocity_field_np if env.has_velocity_field() else None

    metrics: Dict[str, EvalMetrics] = {}
    paths: Dict[str, np.ndarray] = {"rsa": rsa_low.backtrack_path_xy(), "ref": ref_path}

    rsa_path = paths["rsa"]
    rsa_threshold = float(goal_tol) if bool(strict_goal_tol) else max(float(goal_tol), 0.17)
    rsa_ok = np.linalg.norm(rsa_path[-1] - np.asarray(goal_xy, dtype=np.float32)) <= rsa_threshold
    rsa_len = path_length(rsa_path)
    rsa_time_cost = path_time_cost(rsa_path, velocity_field_np_fn)
    rsa_eff = rsa_len / max(1e-12, rsa_time_cost)
    _, rsa_curv_sharp, _ = path_curvature(rsa_path)
    metrics["rsa"] = EvalMetrics(
        success=bool(rsa_ok),
        length=rsa_len,
        smoothness=path_smoothness(rsa_path),
        optimality_gap=(rsa_len - ref_len) / (ref_len + 1e-12),
        regret=(rsa_len - ref_len),
        grad_norm_start=None,
        time_cost=rsa_time_cost,
        efficiency_ratio=rsa_eff,
        curvature_sharpness=rsa_curv_sharp,
    )

    for name, model in models.items():
        use_coupling = coupling_model is not None and name in ("rhp_pinn",)
        path, ok, mean_alpha = integrate_path_by_grad(
            model=model,
            env=env,
            start_xy=start_xy,
            goal_xy=goal_xy,
            step_size=path_step,
            max_steps=max_path_steps,
            goal_tol=goal_tol,
            rsa_field=rsa_ref,
            rsa_path_xy=ref_path,
            coupling_model=coupling_model if use_coupling else None,
            default_alpha=coupling_default_alpha,
            strict_goal_tol=strict_goal_tol,
        )
        plen = path_length(path)
        tc = path_time_cost(path, velocity_field_np_fn)
        eff = plen / max(1e-12, tc)
        smooth = path_smoothness(path)
        gap = (plen - ref_len) / (ref_len + 1e-12)
        _, curv_sharp, _ = path_curvature(path)
        phys = path_eikonal_residual(path, model, env)

        paths[name] = path
        metrics[name] = EvalMetrics(
            success=bool(ok),
            length=plen,
            smoothness=smooth,
            optimality_gap=gap,
            regret=(plen - ref_len),
            coupling_alpha=mean_alpha if use_coupling else None,
            grad_norm_start=grad_norm_near_start(model, start_xy, grad_probe_radius, grad_probe_points),
            time_cost=tc,
            efficiency_ratio=eff,
            physical_consistency=phys,
            curvature_sharpness=curv_sharp,
        )

    return metrics, paths
