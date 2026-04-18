from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

from .envs.maze_2d import Bounds2D, Maze2DEnv
from .main_bench import _grid_coords, _make_env
from .solvers.factored_nn import FactoredTimeNN
from .solvers.physics_loss import start_bc_loss, upwind_physics_loss
from .solvers.rsa_engine import RSAEngine, RSAResult


@dataclass(frozen=True)
class CollectConfig:
    seq_len: int = 10
    feat_dim: int = 17
    lookahead: int = 8
    rollout_step: float = 0.02
    rollout_max_steps: int = 400
    record_stride: int = 1
    line_search_tries: int = 5
    train_steps: int = 300
    train_batch: int = 1024
    lr: float = 1e-3
    lambda_sup: float = 1.0
    lambda_phys: float = 0.3
    lambda_bc: float = 5.0
    fd_step: float = 0.01
    explore_noise_std: float = 0.35
    rollouts_per_seed: int = 2
    simple_extra_seeds: int = 50
    compare_horizon: int = 80
    future_horizon_short: int = 10
    future_horizon_long: int = 30
    progress_eps: float = 1e-3
    value_tau: float = 8.0
    danger_cos_threshold: float = -0.8
    danger_sdf_threshold: float = 0.05
    danger_goal_stall_steps: int = 5
    failed_tail_steps: int = 20
    failed_oversample_factor: int = 5
    failed_seed_extra_rollouts: int = 2
    failure_priority_seed: int = 0
    rubble_extra_seeds: int = 20
    rubble_blockage_radius: float = 0.10
    rubble_blockage_samples: int = 12
    rubble_blockage_threshold: float = 0.30
    success_choke_sdf: float = 0.18
    success_choke_blockage: float = 0.15
    success_weight_scale: float = 2.5
    success_oversample_factor: int = 4
    success_turn_weight_scale: float = 2.0
    success_turn_angle_deg: float = 25.0
    success_exit_path_frac: float = 0.45
    success_exit_sdf: float = 0.18
    success_exit_blockage: float = 0.10
    success_exit_goal_align: float = 0.75
    success_exit_progress_norm: float = 0.01
    success_exit_value_scale: float = 0.70
    success_exit_weight_scale: float = 1.5
    success_exit_oversample_factor: int = 3
    balance_classic_frac: float = 0.4
    balance_rubble_success_frac: float = 0.3
    balance_rubble_failure_frac: float = 0.3


def _set_seed(seed: int) -> np.random.Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def _nearest_path_dist_and_idx(path_xy: np.ndarray, x: np.ndarray) -> Tuple[float, int]:
    d2 = np.sum((path_xy - x.reshape(1, 2)) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return float(np.sqrt(d2[idx])), idx


def _rsa_lookahead_dir(path_xy: np.ndarray, x: np.ndarray, idx: int, lookahead: int) -> np.ndarray:
    j = min(idx + int(lookahead), path_xy.shape[0] - 1)
    d = (path_xy[j] - x).astype(np.float32)
    dn = float(np.linalg.norm(d) + 1e-12)
    return (d / dn).astype(np.float32)


def _sdf_and_grad(env: Maze2DEnv, x: torch.Tensor) -> Tuple[float, np.ndarray]:
    x = x.detach().requires_grad_(True)
    sdf = env.sdf(x)
    if sdf.ndim == 2:
        sdf = sdf[:, 0]
    g = torch.autograd.grad(sdf.sum(), x, create_graph=False, retain_graph=False)[0]
    g_np = g.detach().cpu().numpy()[0].astype(np.float32)
    return float(sdf.detach().cpu().numpy()[0]), g_np


def _pinn_dir(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    x = x.detach().requires_grad_(True)
    t = model(x)
    if t.ndim == 2:
        t = t[:, 0]
    g = torch.autograd.grad(t.sum(), x, create_graph=False, retain_graph=False)[0]
    g_np = g.detach().cpu().numpy()[0].astype(np.float32)
    n = float(np.linalg.norm(g_np) + 1e-12)
    return (-g_np / n).astype(np.float32)


def _pinn_dir_and_grad_norm(model: torch.nn.Module, x: torch.Tensor) -> Tuple[np.ndarray, float]:
    x = x.detach().requires_grad_(True)
    t = model(x)
    if t.ndim == 2:
        t = t[:, 0]
    g = torch.autograd.grad(t.sum(), x, create_graph=False, retain_graph=False)[0]
    g_np = g.detach().cpu().numpy()[0].astype(np.float32)
    grad_norm = float(np.linalg.norm(g_np) + 1e-12)
    return (-g_np / grad_norm).astype(np.float32), grad_norm


def _line_search_step(env: Maze2DEnv, x: torch.Tensor, v: np.ndarray, step: float, tries: int) -> Tuple[torch.Tensor, bool]:
    v_t = torch.tensor(v, dtype=torch.float32, device=x.device).view(1, 2)
    for i in range(tries + 1):
        h = float(step) * (0.5**i)
        x_try = x + h * v_t
        with torch.no_grad():
            ok = bool(env.is_free(x_try).item())
        if ok:
            return x_try, True
    return x, False


def _make_simple_env(cfg: Dict[str, Any], device: torch.device) -> Maze2DEnv:
    bcfg = cfg["env"]["bounds"]
    bounds = Bounds2D(**bcfg)
    return Maze2DEnv(
        bounds=bounds,
        obstacles=[],
        obstacle_inflation=float(cfg["env"].get("obstacle_inflation", 0.0)),
        speed_free=float(cfg["rsa"].get("speed_free", 1.0)),
        speed_obstacle=float(cfg["rsa"].get("speed_obstacle", 0.0)),
        device=device,
    )


def _make_rubble_env(cfg: Dict[str, Any], seed: int, device: torch.device) -> Maze2DEnv:
    bcfg = cfg["env"]["bounds"]
    bounds = Bounds2D(**bcfg)
    rng = np.random.default_rng(int(seed))
    return Maze2DEnv.make_rubble_field(
        bounds=bounds,
        rng=rng,
        start_xy=tuple(cfg["env"]["start"]),
        goal_xy=tuple(cfg["env"]["goal"]),
        num_clusters=int(rng.integers(4, 8)),
        circles_per_cluster=(int(rng.integers(4, 7)), int(rng.integers(8, 12))),
        radius_range=(float(rng.uniform(0.02, 0.035)), float(rng.uniform(0.055, 0.085))),
        cluster_spread=float(rng.uniform(0.07, 0.14)),
        clearance=float(rng.uniform(0.035, 0.06)),
        obstacle_inflation=float(cfg["env"].get("obstacle_inflation", 0.0)),
        speed_free=float(cfg["rsa"].get("speed_free", 1.0)),
        speed_obstacle=float(cfg["rsa"].get("speed_obstacle", 0.0)),
        device=device,
    )


def _path_length(path_xy: np.ndarray) -> float:
    if path_xy.shape[0] < 2:
        return 0.0
    diffs = path_xy[1:] - path_xy[:-1]
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _goal_stall_flag(goal_hist: List[float], stall_steps: int) -> bool:
    need = int(max(2, stall_steps))
    if len(goal_hist) < need:
        return False
    tail = np.asarray(goal_hist[-need:], dtype=np.float32)
    diffs = tail[:-1] - tail[1:]
    return bool(np.all(diffs <= 1e-4))


def _danger_flag(
    cos_sim: float,
    sdf_val: float,
    blockage: float,
    goal_hist: List[float],
    ccfg: CollectConfig,
) -> bool:
    votes = 0
    votes += int(cos_sim < float(ccfg.danger_cos_threshold))
    votes += int(sdf_val < float(ccfg.danger_sdf_threshold))
    votes += int(blockage > float(ccfg.rubble_blockage_threshold))
    votes += int(_goal_stall_flag(goal_hist, stall_steps=int(ccfg.danger_goal_stall_steps)))
    return bool(votes >= 2)


def _progress_features(
    dist_hist: List[float],
    grad_norm: float,
    stall_count: int,
    eps: float,
) -> Tuple[float, float, float, int]:
    d_t = float(dist_hist[-1])
    d_prev = float(dist_hist[-2]) if len(dist_hist) >= 2 else d_t
    d_prev5 = float(dist_hist[-6]) if len(dist_hist) >= 6 else float(dist_hist[0])
    progress_5 = float(d_prev5 - d_t)
    progress_norm = float((d_prev - d_t) / (float(grad_norm) + 1e-12))
    hist = np.asarray(dist_hist[-6:], dtype=np.float32)
    if hist.shape[0] >= 2:
        t = np.arange(hist.shape[0], dtype=np.float32)
        trend = float(np.polyfit(t, hist, deg=1)[0])
    else:
        trend = 0.0
    stall_next = int(stall_count + 1) if progress_norm < float(eps) else 0
    return progress_5, progress_norm, trend, stall_next


def _rebalance_433(
    X: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    bucket: np.ndarray,
    seed: np.ndarray,
    rng: np.random.Generator,
    target_fracs: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idxs = [np.flatnonzero(bucket == k) for k in range(3)]
    if any(arr.size == 0 for arr in idxs):
        return X, y, weight, bucket, seed
    total_target = int(
        min(
            idxs[0].size / max(1e-12, target_fracs[0]),
            idxs[1].size / max(1e-12, target_fracs[1]),
            idxs[2].size / max(1e-12, target_fracs[2]),
        )
    )
    total_target = max(32, total_target)
    selected = []
    for k, frac in enumerate(target_fracs):
        n_k = max(1, int(round(total_target * frac)))
        replace = idxs[k].size < n_k
        sel = rng.choice(idxs[k], size=n_k, replace=replace)
        selected.append(sel)
    keep = np.concatenate(selected, axis=0)
    rng.shuffle(keep)
    return X[keep], y[keep], weight[keep], bucket[keep], seed[keep]


def _build_feature(
    ccfg: CollectConfig,
    d_step: np.ndarray,
    d_rsa: np.ndarray,
    sdf_val: float,
    sdf_g_unit: np.ndarray,
    dist_goal: float,
    blockage: float,
    progress_5: float,
    progress_norm: float,
    trend: float,
    stall_count: int,
    goal_align: float,
    path_progress: float,
    exit_confidence: float,
) -> np.ndarray:
    rel_pinn = (float(ccfg.rollout_step) * d_step).astype(np.float32)
    rel_rsa = (float(ccfg.rollout_step) * d_rsa).astype(np.float32)
    cos_sim = float(np.dot(d_step, d_rsa))
    return np.asarray(
        [
            rel_pinn[0],
            rel_pinn[1],
            rel_rsa[0],
            rel_rsa[1],
            float(sdf_val),
            float(sdf_g_unit[0]),
            float(sdf_g_unit[1]),
            float(cos_sim),
            float(dist_goal),
            float(blockage),
            float(progress_5),
            float(progress_norm),
            float(trend),
            float(stall_count),
            float(goal_align),
            float(path_progress),
            float(exit_confidence),
        ],
        dtype=np.float32,
    )


def _exit_confidence(
    *,
    sdf_val: float,
    blockage: float,
    goal_align: float,
    progress_norm: float,
    path_progress: float,
    exit_sdf: float,
    exit_blockage: float,
    exit_goal_align: float,
    exit_progress_norm: float,
    exit_path_frac: float,
) -> float:
    open_score = float(
        np.clip(
            (float(sdf_val) - float(exit_sdf)) / max(1e-6, 0.35 - float(exit_sdf)),
            0.0,
            1.0,
        )
    )
    low_block_score = float(
        np.clip(
            1.0 - float(blockage) / max(float(exit_blockage), 1e-6),
            0.0,
            1.0,
        )
    )
    align_score = float(
        np.clip(
            (float(goal_align) - float(exit_goal_align)) / max(1e-6, 1.0 - float(exit_goal_align)),
            0.0,
            1.0,
        )
    )
    progress_score = float(
        np.clip(
            float(progress_norm) / max(float(exit_progress_norm), 1e-6),
            0.0,
            1.0,
        )
    )
    path_score = float(
        np.clip(
            (float(path_progress) - float(exit_path_frac)) / max(1e-6, 1.0 - float(exit_path_frac)),
            0.0,
            1.0,
        )
    )
    return float(min(open_score, low_block_score, align_score, max(progress_score, path_score)))


def _local_blockage(env: Maze2DEnv, cur_xy: np.ndarray, radius: float, samples: int) -> float:
    n = int(max(4, samples))
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1) * float(radius)
    pts = ring + cur_xy.reshape(1, 2)
    pts_t = torch.from_numpy(pts.astype(np.float32)).to(device=env.device)
    with torch.no_grad():
        free = env.is_free(pts_t).detach().cpu().numpy().astype(np.float32)
    return float(1.0 - np.mean(free))


def _simulate_value_cost(
    env: Maze2DEnv,
    model: torch.nn.Module,
    rsa_path_xy: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: Tuple[float, float],
    ccfg: CollectConfig,
    device: torch.device,
    horizon: int,
    policy: str,
) -> float:
    x = torch.tensor(start_xy, dtype=torch.float32, device=device).view(1, 2)
    goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)
    dist_hist: List[float] = []
    stall_count = 0
    total_cost = 0.0

    for _ in range(int(horizon)):
        cur = x.detach().cpu().numpy()[0].astype(np.float32)
        dist_goal = float(np.linalg.norm(goal - cur))
        dist_hist.append(dist_goal)
        if dist_goal <= 0.05:
            break
        _, idx = _nearest_path_dist_and_idx(rsa_path_xy, cur)
        d_rsa = _rsa_lookahead_dir(rsa_path_xy, cur, idx=idx, lookahead=ccfg.lookahead)
        d_pinn, grad_norm = _pinn_dir_and_grad_norm(model, x)
        d_step = d_rsa if str(policy) == "rsa" else d_pinn
        x_next, ok = _line_search_step(env, x, d_step, step=ccfg.rollout_step, tries=ccfg.line_search_tries)
        if not ok:
            total_cost += 10.0
            break
        nxt = x_next.detach().cpu().numpy()[0].astype(np.float32)
        dist_next = float(np.linalg.norm(goal - nxt))
        progress = float(dist_goal - dist_next)
        progress_norm = float(progress / (float(grad_norm) + 1e-12))
        stall_count = stall_count + 1 if progress_norm < float(ccfg.progress_eps) else 0
        total_cost += 1.0 + 5.0 * float(stall_count > 0) - 2.0 * max(progress, 0.0)
        x = x_next
    else:
        total_cost += 10.0
    return float(total_cost)


def _state_value_target(
    env: Maze2DEnv,
    model: torch.nn.Module,
    rsa_path_xy: np.ndarray,
    cur_xy: np.ndarray,
    goal_xy: Tuple[float, float],
    ccfg: CollectConfig,
    device: torch.device,
) -> float:
    cost10 = min(
        _simulate_value_cost(env, model, rsa_path_xy, cur_xy, goal_xy, ccfg, device, horizon=int(ccfg.future_horizon_short), policy="pinn"),
        _simulate_value_cost(env, model, rsa_path_xy, cur_xy, goal_xy, ccfg, device, horizon=int(ccfg.future_horizon_short), policy="rsa"),
    )
    cost30 = min(
        _simulate_value_cost(env, model, rsa_path_xy, cur_xy, goal_xy, ccfg, device, horizon=int(ccfg.future_horizon_long), policy="pinn"),
        _simulate_value_cost(env, model, rsa_path_xy, cur_xy, goal_xy, ccfg, device, horizon=int(ccfg.future_horizon_long), policy="rsa"),
    )
    return float(0.5 * cost10 + 0.5 * cost30)


def _rollout_collect(
    env: Maze2DEnv,
    model: torch.nn.Module,
    rsa_path_xy: np.ndarray,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    rng: np.random.Generator,
    ccfg: CollectConfig,
    device: torch.device,
    controller: str,
    bucket_id: int,
    action_noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray]:
    seq_len = int(ccfg.seq_len)
    feat_dim = int(ccfg.feat_dim)
    hist: List[np.ndarray] = []
    xs: List[np.ndarray] = []
    ys: List[float] = []
    weights: List[float] = []
    buckets: List[int] = []
    sample_steps: List[int] = []
    goal_hist: List[float] = []
    dist_hist: List[float] = []
    traj_pts: List[np.ndarray] = []
    stall_count = 0

    x = torch.tensor(start_xy, dtype=torch.float32, device=device).view(1, 2)
    goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)
    success = False
    path_len = max(1, int(rsa_path_xy.shape[0]) - 1)

    for t in range(int(ccfg.rollout_max_steps)):
        cur = x.detach().cpu().numpy()[0].astype(np.float32)
        traj_pts.append(cur.copy())
        _, idx = _nearest_path_dist_and_idx(rsa_path_xy, cur)
        d_rsa = _rsa_lookahead_dir(rsa_path_xy, cur, idx=idx, lookahead=ccfg.lookahead)
        d_pinn, grad_norm = _pinn_dir_and_grad_norm(model, x)
        if str(controller) == "rsa":
            d_step = d_rsa
        elif action_noise_std > 0.0:
            d_step = d_pinn + float(action_noise_std) * rng.normal(size=(2,)).astype(np.float32)
            d_step = d_step / (float(np.linalg.norm(d_step)) + 1e-12)
        else:
            d_step = d_pinn

        sdf_val, sdf_g = _sdf_and_grad(env, x)
        sdf_gn = float(np.linalg.norm(sdf_g) + 1e-12)
        sdf_g_unit = (sdf_g / sdf_gn).astype(np.float32)

        dist_goal = float(np.linalg.norm(goal - cur))
        goal_hist.append(dist_goal)
        dist_hist.append(dist_goal)
        cos_sim = float(np.dot(d_step, d_rsa))
        blockage = _local_blockage(
            env=env,
            cur_xy=cur,
            radius=float(ccfg.rubble_blockage_radius),
            samples=int(ccfg.rubble_blockage_samples),
        )
        progress_5, progress_norm, trend, stall_count = _progress_features(
            dist_hist=dist_hist,
            grad_norm=grad_norm,
            stall_count=stall_count,
            eps=float(ccfg.progress_eps),
        )
        d_goal = (goal - cur).astype(np.float32)
        d_goal = d_goal / (float(np.linalg.norm(d_goal)) + 1e-12)
        goal_align = float(np.dot(d_pinn, d_goal))
        path_progress = float(idx) / float(path_len)
        exit_conf = _exit_confidence(
            sdf_val=sdf_val,
            blockage=blockage,
            goal_align=goal_align,
            progress_norm=progress_norm,
            path_progress=path_progress,
            exit_sdf=float(ccfg.success_exit_sdf),
            exit_blockage=float(ccfg.success_exit_blockage),
            exit_goal_align=float(ccfg.success_exit_goal_align),
            exit_progress_norm=float(ccfg.success_exit_progress_norm),
            exit_path_frac=float(ccfg.success_exit_path_frac),
        )
        danger_now = int(
            _danger_flag(
                cos_sim=cos_sim,
                sdf_val=sdf_val,
                blockage=blockage,
                goal_hist=goal_hist,
                ccfg=ccfg,
            )
        )

        feat = _build_feature(
            ccfg=ccfg,
            d_step=d_step,
            d_rsa=d_rsa,
            sdf_val=sdf_val,
            sdf_g_unit=sdf_g_unit,
            dist_goal=dist_goal,
            blockage=blockage,
            progress_5=progress_5,
            progress_norm=progress_norm,
            trend=trend,
            stall_count=stall_count,
            goal_align=goal_align,
            path_progress=path_progress,
            exit_confidence=exit_conf,
        )

        hist.append(feat)
        if len(hist) > seq_len:
            hist = hist[-seq_len:]

        if (t % int(max(1, ccfg.record_stride))) == 0:
            seq = np.zeros((seq_len, feat_dim), dtype=np.float32)
            h = np.stack(hist, axis=0)
            seq[-h.shape[0] :] = h
            cur_xy = x.detach().cpu().numpy()[0].astype(np.float32)
            value_target = _state_value_target(
                env=env,
                model=model,
                rsa_path_xy=rsa_path_xy,
                cur_xy=cur_xy,
                goal_xy=goal_xy,
                ccfg=ccfg,
                device=device,
            )
            xs.append(seq)
            ys.append(value_target)
            weights.append(1.0 + 0.5 * float(danger_now))
            buckets.append(int(bucket_id))
            sample_steps.append(int(t))

        if dist_goal <= 0.05:
            success = True
            break

        x_next, ok = _line_search_step(env, x, d_step, step=ccfg.rollout_step, tries=ccfg.line_search_tries)
        if not ok:
            break
        x = x_next

    if len(xs) == 0:
        return (
            np.zeros((0, seq_len, feat_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
            False,
            np.zeros((0, 2), dtype=np.float32),
        )
    x_arr = np.stack(xs, axis=0)
    y_arr = np.asarray(ys, dtype=np.float32)
    w_arr = np.asarray(weights, dtype=np.float32)
    bucket_arr = np.asarray(buckets, dtype=np.int64)
    step_arr = np.asarray(sample_steps, dtype=np.int32)
    if (not success) and x_arr.shape[0] > 0:
        delta = (step_arr.shape[0] - 1 - np.arange(step_arr.shape[0], dtype=np.float32))
        w_arr = w_arr * np.exp(-delta / float(ccfg.value_tau))
    if (not success) and x_arr.shape[0] > 0:
        tail = min(int(ccfg.failed_tail_steps), x_arr.shape[0])
        factor = int(ccfg.failed_oversample_factor)
        if factor > 1 and tail > 0:
            tail_slice = slice(x_arr.shape[0] - tail, x_arr.shape[0])
            repeat_idx = np.repeat(np.arange(tail), factor - 1)
            x_tail = x_arr[tail_slice][repeat_idx]
            y_tail = y_arr[tail_slice][repeat_idx]
            w_tail = w_arr[tail_slice][repeat_idx]
            bucket_tail = bucket_arr[tail_slice][repeat_idx]
            x_arr = np.concatenate([x_arr, x_tail], axis=0)
            y_arr = np.concatenate([y_arr, y_tail], axis=0)
            w_arr = np.concatenate([w_arr, w_tail], axis=0)
            bucket_arr = np.concatenate([bucket_arr, bucket_tail], axis=0)
    return (
        x_arr,
        y_arr,
        w_arr,
        bucket_arr,
        np.full((x_arr.shape[0],), int(not success), dtype=np.int32),
        success,
        np.asarray(traj_pts, dtype=np.float32),
    )


def _collect_rsa_path_samples(
    env: Maze2DEnv,
    model: torch.nn.Module,
    rsa_path_xy: np.ndarray,
    goal_xy: Tuple[float, float],
    ccfg: CollectConfig,
    device: torch.device,
    bucket_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq_len = int(ccfg.seq_len)
    feat_dim = int(ccfg.feat_dim)
    hist: List[np.ndarray] = []
    xs: List[np.ndarray] = []
    ys: List[float] = []
    ws: List[float] = []
    buckets: List[int] = []
    goal_hist: List[float] = []
    stall_count = 0
    goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)
    path_len = max(1, int(rsa_path_xy.shape[0]) - 1)

    for i in range(int(rsa_path_xy.shape[0])):
        cur = rsa_path_xy[i].astype(np.float32)
        x = torch.tensor(cur, dtype=torch.float32, device=device).view(1, 2)
        _, idx = _nearest_path_dist_and_idx(rsa_path_xy, cur)
        d_rsa = _rsa_lookahead_dir(rsa_path_xy, cur, idx=idx, lookahead=ccfg.lookahead)
        d_pinn, grad_norm = _pinn_dir_and_grad_norm(model, x)
        sdf_val, sdf_g = _sdf_and_grad(env, x)
        sdf_gn = float(np.linalg.norm(sdf_g) + 1e-12)
        sdf_g_unit = (sdf_g / sdf_gn).astype(np.float32)
        dist_goal = float(np.linalg.norm(goal - cur))
        goal_hist.append(dist_goal)
        blockage = _local_blockage(
            env=env,
            cur_xy=cur,
            radius=float(ccfg.rubble_blockage_radius),
            samples=int(ccfg.rubble_blockage_samples),
        )
        progress_5, progress_norm, trend, stall_count = _progress_features(
            goal_hist,
            grad_norm=grad_norm,
            stall_count=stall_count,
            eps=float(ccfg.progress_eps),
        )
        d_goal = (goal - cur).astype(np.float32)
        d_goal = d_goal / (float(np.linalg.norm(d_goal)) + 1e-12)
        goal_align = float(np.dot(d_pinn, d_goal))
        path_progress = float(idx) / float(path_len)
        exit_conf = _exit_confidence(
            sdf_val=sdf_val,
            blockage=blockage,
            goal_align=goal_align,
            progress_norm=progress_norm,
            path_progress=path_progress,
            exit_sdf=float(ccfg.success_exit_sdf),
            exit_blockage=float(ccfg.success_exit_blockage),
            exit_goal_align=float(ccfg.success_exit_goal_align),
            exit_progress_norm=float(ccfg.success_exit_progress_norm),
            exit_path_frac=float(ccfg.success_exit_path_frac),
        )
        feat = _build_feature(
            ccfg=ccfg,
            d_step=d_pinn,
            d_rsa=d_rsa,
            sdf_val=sdf_val,
            sdf_g_unit=sdf_g_unit,
            dist_goal=dist_goal,
            blockage=blockage,
            progress_5=progress_5,
            progress_norm=progress_norm,
            trend=trend,
            stall_count=stall_count,
            goal_align=goal_align,
            path_progress=path_progress,
            exit_confidence=exit_conf,
        )
        hist.append(feat)
        if len(hist) > seq_len:
            hist = hist[-seq_len:]
        seq = np.zeros((seq_len, feat_dim), dtype=np.float32)
        h = np.stack(hist, axis=0)
        seq[-h.shape[0] :] = h
        value_target = _state_value_target(
            env=env,
            model=model,
            rsa_path_xy=rsa_path_xy,
            cur_xy=cur,
            goal_xy=goal_xy,
            ccfg=ccfg,
            device=device,
        )
        cos_sim = float(np.dot(d_pinn, d_rsa))
        if 0 < i < int(rsa_path_xy.shape[0]) - 1:
            d_prev = (rsa_path_xy[i] - rsa_path_xy[i - 1]).astype(np.float32)
            d_next = (rsa_path_xy[i + 1] - rsa_path_xy[i]).astype(np.float32)
            d_prev = d_prev / (float(np.linalg.norm(d_prev)) + 1e-12)
            d_next = d_next / (float(np.linalg.norm(d_next)) + 1e-12)
            turn_cos = float(np.clip(np.dot(d_prev, d_next), -1.0, 1.0))
            turn_angle = float(np.arccos(turn_cos) * 180.0 / np.pi)
        else:
            turn_angle = 0.0
        narrow_score = max(0.0, float(ccfg.success_choke_sdf) - float(sdf_val)) / max(float(ccfg.success_choke_sdf), 1e-6)
        blockage_score = float(blockage) / max(float(ccfg.success_choke_blockage), 1e-6)
        blockage_score = float(np.clip(blockage_score, 0.0, 1.0))
        disagreement_score = 0.5 * float(np.clip(1.0 - cos_sim, 0.0, 2.0))
        choke_score = float(np.clip(max(narrow_score, blockage_score), 0.0, 1.0))
        turn_score = float(
            np.clip(
                max(0.0, turn_angle - float(ccfg.success_turn_angle_deg))
                / max(1.0, 90.0 - float(ccfg.success_turn_angle_deg)),
                0.0,
                1.0,
            )
        )
        focus_score = float(np.clip(max(choke_score, turn_score), 0.0, 1.0))
        sample_weight = (
            1.0
            + float(ccfg.success_weight_scale) * choke_score
            + float(ccfg.success_turn_weight_scale) * turn_score
            + 0.5 * disagreement_score
        )
        repeat = 1 + int(round(max(0.0, focus_score) * max(0, int(ccfg.success_oversample_factor) - 1)))
        for _ in range(max(1, repeat)):
            xs.append(seq.copy())
            ys.append(value_target)
            ws.append(sample_weight)
            buckets.append(int(bucket_id))

        if int(bucket_id) == 0:
            exit_score = float(exit_conf)
            if exit_score > 0.0:
                pinn_cost10 = _simulate_value_cost(
                    env,
                    model,
                    rsa_path_xy,
                    cur,
                    goal_xy,
                    ccfg,
                    device,
                    horizon=int(ccfg.future_horizon_short),
                    policy="pinn",
                )
                pinn_cost30 = _simulate_value_cost(
                    env,
                    model,
                    rsa_path_xy,
                    cur,
                    goal_xy,
                    ccfg,
                    device,
                    horizon=int(ccfg.future_horizon_long),
                    policy="pinn",
                )
                exit_target = float(float(ccfg.success_exit_value_scale) * (0.5 * pinn_cost10 + 0.5 * pinn_cost30))
                exit_target = float(min(value_target, exit_target))
                exit_weight = float(1.0 + float(ccfg.success_exit_weight_scale) * exit_score)
                exit_repeat = 1 + int(round(exit_score * max(0, int(ccfg.success_exit_oversample_factor) - 1)))
                for _ in range(max(1, exit_repeat)):
                    xs.append(seq.copy())
                    ys.append(exit_target)
                    ws.append(exit_weight)
                    buckets.append(int(bucket_id))

    if len(xs) == 0:
        return (
            np.zeros((0, seq_len, feat_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
        )
    x_arr = np.stack(xs, axis=0)
    return (
        x_arr,
        np.asarray(ys, dtype=np.float32),
        np.asarray(ws, dtype=np.float32),
        np.asarray(buckets, dtype=np.int64),
        np.zeros((x_arr.shape[0],), dtype=np.int32),
    )


def _train_quick_rhp(
    env: Maze2DEnv,
    start_xy: Tuple[float, float],
    rsa_ref: RSAResult,
    rng: np.random.Generator,
    ccfg: CollectConfig,
    device: torch.device,
) -> FactoredTimeNN:
    model = FactoredTimeNN(start_xy=start_xy).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=float(ccfg.lr))
    start_t = torch.tensor(start_xy, dtype=torch.float32, device=device)

    for _ in range(int(ccfg.train_steps)):
        xy = env.sample_uniform(int(ccfg.train_batch), rng=rng, device=device)
        with torch.no_grad():
            free = env.is_free(xy)
        xy = xy[free]
        if xy.shape[0] < 64:
            continue
        xy_np = xy.detach().cpu().numpy()
        t_rsa = torch.from_numpy(rsa_ref.interpolate_t(xy_np)).to(device=device, dtype=torch.float32)
        pred = model(xy)[:, 0]
        loss_sup = torch.mean((pred - t_rsa) ** 2)
        loss_phys = upwind_physics_loss(model=model, xy=xy, speed_fn=env.speed, fd_step=float(ccfg.fd_step))
        loss_bc = start_bc_loss(model=model, start_xy=start_t)
        loss = (
            float(ccfg.lambda_sup) * loss_sup
            + float(ccfg.lambda_phys) * loss_phys
            + float(ccfg.lambda_bc) * loss_bc
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return model


def collect(cfg: Dict[str, Any], out_path: str, num_seeds: int, device: str, ccfg: CollectConfig) -> None:
    dev = torch.device(device)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_w: List[np.ndarray] = []
    all_bucket: List[np.ndarray] = []
    all_failed: List[np.ndarray] = []
    all_seed: List[np.ndarray] = []
    base_seed = int(cfg.get("seed", 0))

    for s in range(int(num_seeds)):
        rng = _set_seed(base_seed + s)
        env = _make_env(cfg, device=dev)
        start_xy = tuple(cfg["env"]["start"])
        goal_xy = tuple(cfg["env"]["goal"])

        xs_ref, ys_ref = _grid_coords(env.bounds, tuple(cfg["eval"]["reference_grid_size"]))
        speed_ref = env.speed_grid(tuple(cfg["eval"]["reference_grid_size"]))
        rsa_ref = RSAEngine(xs=xs_ref, ys=ys_ref, connectivity=int(cfg["rsa"].get("connectivity", 8))).solve(
            speed=speed_ref,
            start_xy=start_xy,
            goal_xy=goal_xy,
        )
        rsa_path = rsa_ref.backtrack_path_xy()

        model = _train_quick_rhp(env, start_xy, rsa_ref, rng, ccfg, device=dev)
        x, y, w, bucket, failed = _collect_rsa_path_samples(
            env=env,
            model=model,
            rsa_path_xy=rsa_path,
            goal_xy=goal_xy,
            ccfg=ccfg,
            device=dev,
            bucket_id=0,
        )
        if x.shape[0] > 0:
            all_x.append(x)
            all_y.append(y)
            all_w.append(w)
            all_bucket.append(bucket)
            all_failed.append(failed)
            episode_id = base_seed + s * 100
            all_seed.append(np.full((x.shape[0],), episode_id, dtype=np.int32))

    for s in range(int(ccfg.rubble_extra_seeds)):
        rubble_seed = base_seed + 20000 + s
        rng = _set_seed(rubble_seed)
        rubble_env = _make_rubble_env(cfg, seed=rubble_seed, device=dev)
        rubble_start_xy = tuple(cfg["env"]["start"])
        rubble_goal_xy = tuple(cfg["env"]["goal"])
        xs_ref, ys_ref = _grid_coords(rubble_env.bounds, tuple(cfg["eval"]["reference_grid_size"]))
        speed_ref = rubble_env.speed_grid(tuple(cfg["eval"]["reference_grid_size"]))
        rsa_ref = RSAEngine(xs=xs_ref, ys=ys_ref, connectivity=int(cfg["rsa"].get("connectivity", 8))).solve(
            speed=speed_ref,
            start_xy=rubble_start_xy,
            goal_xy=rubble_goal_xy,
        )
        rsa_path = rsa_ref.backtrack_path_xy()
        model = _train_quick_rhp(rubble_env, rubble_start_xy, rsa_ref, rng, ccfg, device=dev)

        x, y, w, bucket, failed = _collect_rsa_path_samples(
            env=rubble_env,
            model=model,
            rsa_path_xy=rsa_path,
            goal_xy=rubble_goal_xy,
            ccfg=ccfg,
            device=dev,
            bucket_id=1,
        )
        if x.shape[0] > 0:
            all_x.append(x)
            all_y.append(y)
            all_w.append(w)
            all_bucket.append(bucket)
            all_failed.append(failed)
            episode_id = rubble_seed * 100
            all_seed.append(np.full((x.shape[0],), episode_id, dtype=np.int32))

        fail_collected = 0
        attempts = max(2, int(ccfg.rollouts_per_seed) + int(ccfg.failed_seed_extra_rollouts))
        for rollout_id in range(attempts):
            x, y, w, bucket, failed, success, _ = _rollout_collect(
                rubble_env,
                model,
                rsa_path,
                rubble_start_xy,
                rubble_goal_xy,
                rng,
                ccfg,
                device=dev,
                controller="pinn",
                bucket_id=2,
                action_noise_std=float(ccfg.explore_noise_std),
            )
            if x.shape[0] == 0 or success:
                continue
            all_x.append(x)
            all_y.append(y)
            all_w.append(w)
            all_bucket.append(bucket)
            all_failed.append(failed)
            episode_id = rubble_seed * 100 + 50 + rollout_id
            all_seed.append(np.full((x.shape[0],), episode_id, dtype=np.int32))
            fail_collected += 1
            if fail_collected >= int(ccfg.rollouts_per_seed):
                break

    if len(all_x) == 0:
        raise RuntimeError("No samples collected.")

    X = np.concatenate(all_x, axis=0)
    Y = np.concatenate(all_y, axis=0)
    W = np.concatenate(all_w, axis=0)
    B = np.concatenate(all_bucket, axis=0)
    FAILED = np.concatenate(all_failed, axis=0)
    S = np.concatenate(all_seed, axis=0)
    X, Y, W, B, S = _rebalance_433(
        X=X,
        y=Y,
        weight=W,
        bucket=B,
        seed=S,
        rng=np.random.default_rng(base_seed + 4242),
        target_fracs=(
            float(ccfg.balance_classic_frac),
            float(ccfg.balance_rubble_success_frac),
            float(ccfg.balance_rubble_failure_frac),
        ),
    )
    FAILED = (B == 2).astype(np.int32)
    bucket_frac = [float(np.mean(B == k)) for k in range(3)]
    print(
        f"[gating_dataset] size={int(Y.shape[0])} "
        f"value_mean={float(np.mean(Y)):.3f} "
        f"value_std={float(np.std(Y)):.3f} "
        f"bucket_frac=({bucket_frac[0]:.3f},{bucket_frac[1]:.3f},{bucket_frac[2]:.3f}) "
        f"failed_frac={float(np.mean(FAILED)):.3f}"
    )

    np.savez_compressed(out_path, X=X, y=Y, weight=W, bucket=B, failed=FAILED, seed=S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    ap.add_argument("--out", type=str, default=os.path.join("data", "gating_dataset.npz"))
    ap.add_argument("--num_seeds", type=int, default=200)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--lookahead", type=int, default=8)
    ap.add_argument("--rollout_step", type=float, default=0.02)
    ap.add_argument("--rollout_max_steps", type=int, default=400)
    ap.add_argument("--record_stride", type=int, default=1)
    ap.add_argument("--train_steps", type=int, default=300)
    ap.add_argument("--train_batch", type=int, default=1024)
    ap.add_argument("--explore_noise_std", type=float, default=0.35)
    ap.add_argument("--rollouts_per_seed", type=int, default=2)
    ap.add_argument("--future_horizon_short", type=int, default=10)
    ap.add_argument("--future_horizon_long", type=int, default=30)
    ap.add_argument("--progress_eps", type=float, default=1e-3)
    ap.add_argument("--value_tau", type=float, default=8.0)
    ap.add_argument("--failed_tail_steps", type=int, default=20)
    ap.add_argument("--failed_oversample_factor", type=int, default=5)
    ap.add_argument("--failed_seed_extra_rollouts", type=int, default=2)
    ap.add_argument("--failure_priority_seed", type=int, default=0)
    ap.add_argument("--rubble_extra_seeds", type=int, default=20)
    ap.add_argument("--rubble_blockage_radius", type=float, default=0.10)
    ap.add_argument("--rubble_blockage_samples", type=int, default=12)
    ap.add_argument("--rubble_blockage_threshold", type=float, default=0.30)
    ap.add_argument("--success_choke_sdf", type=float, default=0.18)
    ap.add_argument("--success_choke_blockage", type=float, default=0.15)
    ap.add_argument("--success_weight_scale", type=float, default=2.5)
    ap.add_argument("--success_oversample_factor", type=int, default=4)
    ap.add_argument("--success_turn_weight_scale", type=float, default=2.0)
    ap.add_argument("--success_turn_angle_deg", type=float, default=25.0)
    ap.add_argument("--success_exit_path_frac", type=float, default=0.45)
    ap.add_argument("--success_exit_sdf", type=float, default=0.18)
    ap.add_argument("--success_exit_blockage", type=float, default=0.10)
    ap.add_argument("--success_exit_goal_align", type=float, default=0.75)
    ap.add_argument("--success_exit_progress_norm", type=float, default=0.01)
    ap.add_argument("--success_exit_value_scale", type=float, default=0.70)
    ap.add_argument("--success_exit_weight_scale", type=float, default=1.5)
    ap.add_argument("--success_exit_oversample_factor", type=int, default=3)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ccfg = CollectConfig(
        seq_len=int(args.seq_len),
        lookahead=int(args.lookahead),
        rollout_step=float(args.rollout_step),
        rollout_max_steps=int(args.rollout_max_steps),
        record_stride=int(args.record_stride),
        train_steps=int(args.train_steps),
        train_batch=int(args.train_batch),
        explore_noise_std=float(args.explore_noise_std),
        rollouts_per_seed=int(args.rollouts_per_seed),
        future_horizon_short=int(args.future_horizon_short),
        future_horizon_long=int(args.future_horizon_long),
        progress_eps=float(args.progress_eps),
        value_tau=float(args.value_tau),
        failed_tail_steps=int(args.failed_tail_steps),
        failed_oversample_factor=int(args.failed_oversample_factor),
        failed_seed_extra_rollouts=int(args.failed_seed_extra_rollouts),
        failure_priority_seed=int(args.failure_priority_seed),
        rubble_extra_seeds=int(args.rubble_extra_seeds),
        rubble_blockage_radius=float(args.rubble_blockage_radius),
        rubble_blockage_samples=int(args.rubble_blockage_samples),
        rubble_blockage_threshold=float(args.rubble_blockage_threshold),
        success_choke_sdf=float(args.success_choke_sdf),
        success_choke_blockage=float(args.success_choke_blockage),
        success_weight_scale=float(args.success_weight_scale),
        success_oversample_factor=int(args.success_oversample_factor),
        success_turn_weight_scale=float(args.success_turn_weight_scale),
        success_turn_angle_deg=float(args.success_turn_angle_deg),
        success_exit_path_frac=float(args.success_exit_path_frac),
        success_exit_sdf=float(args.success_exit_sdf),
        success_exit_blockage=float(args.success_exit_blockage),
        success_exit_goal_align=float(args.success_exit_goal_align),
        success_exit_progress_norm=float(args.success_exit_progress_norm),
        success_exit_value_scale=float(args.success_exit_value_scale),
        success_exit_weight_scale=float(args.success_exit_weight_scale),
        success_exit_oversample_factor=int(args.success_exit_oversample_factor),
    )

    collect(cfg=cfg, out_path=args.out, num_seeds=int(args.num_seeds), device=str(args.device), ccfg=ccfg)


if __name__ == "__main__":
    main()
