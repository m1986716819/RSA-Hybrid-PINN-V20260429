from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import torch

from ..envs.maze_2d import Maze2DEnv
from ..models.gating_transformer import GatingTransformer, GatingTransformerConfig
from ..solvers.rsa_engine import RSAResult


_GATING_MODEL_CACHE: Dict[str, GatingTransformer] = {}


def _load_gating_model(device: torch.device) -> GatingTransformer:
    cache_key = str(device)
    if cache_key in _GATING_MODEL_CACHE:
        return _GATING_MODEL_CACHE[cache_key]

    ckpt_path = Path(__file__).resolve().parents[2] / "checkpoints" / "gating_transformer_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Gating checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device)
    cfg_dict = payload.get("cfg", {})
    cfg = GatingTransformerConfig(**cfg_dict) if isinstance(cfg_dict, dict) else GatingTransformerConfig()
    gate_model = GatingTransformer(cfg).to(device=device)
    gate_model.load_state_dict(payload["model_state"])
    gate_model.eval()
    _GATING_MODEL_CACHE[cache_key] = gate_model
    return gate_model


def _build_gating_feature(
    goal_xy: Tuple[float, float],
    step_size: float,
    d_pinn_np: np.ndarray,
    d_rsa_np: np.ndarray,
    cur_xy: np.ndarray,
    sdf_val: float,
    sdf_grad_np: np.ndarray,
    blockage: float,
    progress_5: float,
    progress_norm: float,
    trend: float,
    stall_count: int,
    goal_align: float,
    path_progress: float,
    exit_confidence: float,
) -> np.ndarray:
    sdf_g_norm = float(np.linalg.norm(sdf_grad_np) + 1e-12)
    sdf_g_unit = (sdf_grad_np / sdf_g_norm).astype(np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)
    rel_pinn = (float(step_size) * d_pinn_np).astype(np.float32)
    rel_rsa = (float(step_size) * d_rsa_np).astype(np.float32)
    cos_sim = float(np.dot(d_pinn_np, d_rsa_np))
    dist_goal = float(np.linalg.norm(goal - cur_xy))

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


def _local_blockage(env: Maze2DEnv, cur_xy: np.ndarray, radius: float = 0.10, samples: int = 12) -> float:
    n = int(max(4, samples))
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1) * float(radius)
    pts = ring + cur_xy.reshape(1, 2)
    pts_t = torch.from_numpy(pts.astype(np.float32)).to(device=env.device)
    with torch.no_grad():
        free = env.is_free(pts_t).detach().cpu().numpy().astype(np.float32)
    return float(1.0 - np.mean(free))


def _progress_features(dist_hist: list[float], grad_norm: float, stall_count: int, eps: float = 1e-3) -> tuple[float, float, float, int]:
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


def _exit_confidence(
    *,
    sdf_val: float,
    blockage: float,
    goal_align: float,
    progress_norm: float,
    path_progress: float,
    exit_sdf: float = 0.18,
    exit_blockage: float = 0.10,
    exit_goal_align: float = 0.75,
    exit_progress_norm: float = 0.01,
    exit_path_frac: float = 0.45,
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
    goal_attract_eps: float = 0.05,
    strict_goal_tol: bool = False,
    use_gating: bool = True,
) -> Tuple[np.ndarray, bool, float]:
    device = next(model.parameters()).device
    _ = rsa_field, goal_attract_eps
    gating_model = _load_gating_model(device) if bool(use_gating) else None
    x = torch.tensor(start_xy, dtype=torch.float32, device=device).view(1, 2)
    goal = torch.tensor(goal_xy, dtype=torch.float32, device=device).view(1, 2)
    pts = [x.detach().cpu().numpy()[0]]
    backtrack_steps = 0
    slide_steps = 0
    rsa_usage_count = 0
    history_buffer: Deque[np.ndarray] = deque(maxlen=10)
    goal_hist: Deque[float] = deque(maxlen=6)
    lookahead = 40
    last_path_idx = 0
    stall_steps = 0
    progress_stall_count = 0
    rsa_streak = 0
    rsa_exit_cooldown = 0
    gate_recenter_countdown = 0
    path = None
    if rsa_path_xy is not None:
        path = np.asarray(rsa_path_xy, dtype=np.float32)
        if not (path.ndim == 2 and path.shape[0] >= 2 and path.shape[1] == 2):
            path = None
    obstacle_count = int(len(getattr(env, "obstacles", [])))
    stall_trigger = 15 if obstacle_count >= 10 else 8
    stall_force_trigger = stall_trigger + 4
    stall_bias_trigger = max(4, stall_trigger - 3)
    def _lookahead_dir(cur_xy_np: np.ndarray) -> Tuple[np.ndarray, float, int]:
        if path is None:
            return np.asarray([0.0, 0.0], dtype=np.float32), float("inf"), 0
        d2 = np.sum((path - cur_xy_np.reshape(1, 2)) ** 2, axis=1)
        idx = int(np.argmin(d2))
        dist = float(np.sqrt(d2[idx]))
        j = min(idx + lookahead, path.shape[0] - 1)
        p_look = path[j]
        d = (p_look - cur_xy_np).astype(np.float32)
        dn = float(np.linalg.norm(d) + 1e-12)
        return (d / dn).astype(np.float32), dist, idx

    def _path_target_dir(cur_xy_np: np.ndarray, base_idx: int, extra_ahead: int) -> np.ndarray:
        if path is None:
            return np.asarray([0.0, 0.0], dtype=np.float32)
        j = min(max(0, int(base_idx)) + max(1, int(extra_ahead)), path.shape[0] - 1)
        d = (path[j] - cur_xy_np).astype(np.float32)
        dn = float(np.linalg.norm(d) + 1e-12)
        return (d / dn).astype(np.float32)

    def _sdf_normal(sdf_grad: torch.Tensor) -> torch.Tensor:
        nrm = torch.linalg.norm(sdf_grad, dim=-1, keepdim=True) + 1e-12
        return sdf_grad / nrm

    def _line_search(x_cur: torch.Tensor, v_dir: torch.Tensor) -> Tuple[torch.Tensor, bool, bool]:
        used_backtrack = False
        for i in range(0, 6):
            if i == 0:
                h = float(step_size)
            else:
                h = float(step_size) * (0.5**i)
                used_backtrack = True
            x_try = x_cur + h * v_dir
            with torch.no_grad():
                ok_free = bool(env.is_free(x_try).item())
            if ok_free:
                return x_try, True, used_backtrack
        return x_cur, False, used_backtrack

    def _candidate_step(x_cur: torch.Tensor, v_dir: torch.Tensor, sdf_grad_cur: torch.Tensor) -> Tuple[torch.Tensor, bool, bool, bool]:
        x_next, ok_free, used_backtrack = _line_search(x_cur, v_dir)
        used_slide = False
        if ok_free:
            return x_next, True, used_backtrack, used_slide
        n = _sdf_normal(sdf_grad_cur.detach())
        dot = torch.sum(v_dir * n, dim=-1, keepdim=True)
        v_slide = v_dir - dot * n
        v_slide = v_slide / (torch.linalg.norm(v_slide, dim=-1, keepdim=True) + 1e-12)
        x_next, ok_free2, used_backtrack2 = _line_search(x_cur, v_slide)
        return x_next, bool(ok_free2), bool(used_backtrack or used_backtrack2), bool(ok_free2)

    def _predict_value(hist_items: list[np.ndarray]) -> float:
        seq = np.zeros((gating_model.cfg.seq_len, gating_model.cfg.input_dim), dtype=np.float32)
        hist_np = np.stack(hist_items[-gating_model.cfg.seq_len :], axis=0)
        seq[-hist_np.shape[0] :] = hist_np
        seq_t = torch.from_numpy(seq).to(device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return float(gating_model.predict_value(seq_t)[0].item())

    def _candidate_value(x_candidate: torch.Tensor) -> tuple[float, float, float, float]:
        x_eval = x_candidate.detach().clone().requires_grad_(True)
        t_eval = model(x_eval)
        if t_eval.ndim == 2:
            t_eval = t_eval[:, 0]
        sdf_eval = env.sdf(x_eval)
        if sdf_eval.ndim == 2:
            sdf_eval = sdf_eval[:, 0]
        joint_eval = torch.stack([t_eval.sum(), sdf_eval.sum()], dim=0)
        grad_out_eval = torch.eye(2, dtype=joint_eval.dtype, device=device)
        grads_eval = torch.autograd.grad(
            outputs=joint_eval,
            inputs=x_eval,
            grad_outputs=grad_out_eval,
            is_grads_batched=True,
            create_graph=False,
            retain_graph=False,
        )[0]
        grad_t_eval = grads_eval[0]
        grad_sdf_eval = grads_eval[1]
        if not torch.isfinite(grad_t_eval).all() or not torch.isfinite(grad_sdf_eval).all():
            return float("inf"), 0.0, float("-inf"), -1.0

        cand_xy_np = x_eval.detach().cpu().numpy()[0].astype(np.float32)
        grad_t_eval_np = grad_t_eval.detach().cpu().numpy()[0].astype(np.float32)
        grad_sdf_eval_np = grad_sdf_eval.detach().cpu().numpy()[0].astype(np.float32)
        grad_norm_eval = float(np.linalg.norm(grad_t_eval_np) + 1e-12)
        d_pinn_eval_np = (-grad_t_eval_np / grad_norm_eval).astype(np.float32)
        sdf_eval_val = float(sdf_eval.detach().cpu().numpy()[0])
        d_rsa_eval_np, _, path_idx_eval = _lookahead_dir(cand_xy_np)
        d_goal_eval_np = (np.asarray(goal_xy, dtype=np.float32) - cand_xy_np).astype(np.float32)
        d_goal_eval_np = (d_goal_eval_np / (float(np.linalg.norm(d_goal_eval_np)) + 1e-12)).astype(np.float32)
        goal_align_eval = float(np.dot(d_pinn_eval_np, d_goal_eval_np))
        dist_goal_eval = float(np.linalg.norm(np.asarray(goal_xy, dtype=np.float32) - cand_xy_np))
        blockage_eval = _local_blockage(env=env, cur_xy=cand_xy_np)
        goal_hist_eval = list(goal_hist) + [dist_goal_eval]
        progress_5_eval, progress_norm_eval, trend_eval, progress_stall_eval = _progress_features(
            goal_hist_eval,
            grad_norm_eval,
            progress_stall_count,
        )
        path_progress_eval = float(path_idx_eval) / float(max(1, 0 if path is None else path.shape[0] - 1))
        exit_conf_eval = _exit_confidence(
            sdf_val=sdf_eval_val,
            blockage=blockage_eval,
            goal_align=goal_align_eval,
            progress_norm=progress_norm_eval,
            path_progress=path_progress_eval,
        )
        feat_eval = _build_gating_feature(
            goal_xy=goal_xy,
            step_size=step_size,
            d_pinn_np=d_pinn_eval_np,
            d_rsa_np=d_rsa_eval_np,
            cur_xy=cand_xy_np,
            sdf_val=sdf_eval_val,
            sdf_grad_np=grad_sdf_eval_np,
            blockage=blockage_eval,
            progress_5=progress_5_eval,
            progress_norm=progress_norm_eval,
            trend=trend_eval,
            stall_count=progress_stall_eval,
            goal_align=goal_align_eval,
            path_progress=path_progress_eval,
            exit_confidence=exit_conf_eval,
        )
        value_eval = _predict_value(list(history_buffer) + [feat_eval])
        return value_eval, exit_conf_eval, sdf_eval_val, goal_align_eval

    def _finalize(success: bool) -> Tuple[np.ndarray, bool, float]:
        n_moves = max(1, len(pts) - 1)
        if (backtrack_steps + slide_steps) > 0:
            print(
                f"[Integrate] backtrack_ratio={backtrack_steps/float(n_moves):.3f} "
                f"slide_ratio={slide_steps/float(n_moves):.3f}"
            )
        if gating_model is None:
            return np.asarray(pts, dtype=np.float32), bool(success), float("nan")
        rsa_ratio = rsa_usage_count / float(n_moves)
        print(f"[RSAUsage] ratio={rsa_ratio:.3f} success={bool(success)}")
        return np.asarray(pts, dtype=np.float32), bool(success), float(rsa_ratio)

    goal_threshold = float(goal_tol) if bool(strict_goal_tol) else max(float(goal_tol), 0.17)

    for step in range(int(max_steps)):
        x = x.detach().requires_grad_(True)
        t = model(x)
        if t.ndim == 2:
            t = t[:, 0]
        sdf = env.sdf(x)
        if sdf.ndim == 2:
            sdf = sdf[:, 0]
        if not torch.isfinite(t).all():
            return _finalize(False)
        if not torch.isfinite(sdf).all():
            return _finalize(False)
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
        grad_sdf = grads[1]
        if not torch.isfinite(grad_t).all() or not torch.isfinite(grad_sdf).all():
            return _finalize(False)
        gnorm = torch.linalg.norm(grad_t, dim=-1, keepdim=True) + 1e-12
        v_pinn = -grad_t / gnorm
        if gating_model is None:
            v = v_pinn
            x_next, ok_free, used_backtrack, used_slide = _candidate_step(x, v, grad_sdf)
            if used_backtrack:
                backtrack_steps += 1
            if used_slide:
                slide_steps += 1
            if not ok_free:
                pts.append(x_next.detach().cpu().numpy()[0])
                return _finalize(False)
            if not torch.isfinite(x_next).all():
                pts.append(x_next.detach().cpu().numpy()[0])
                return _finalize(False)
            with torch.no_grad():
                free = env.is_free(x_next).item()
                if not free:
                    pts.append(x_next.detach().cpu().numpy()[0])
                    return _finalize(False)
                if torch.linalg.norm(x_next - goal).item() <= goal_threshold:
                    pts.append(x_next.detach().cpu().numpy()[0])
                    return _finalize(True)
            x = x_next
            pts.append(x.detach().cpu().numpy()[0])
            continue

        cur_xy_np = x.detach().cpu().numpy()[0].astype(np.float32)
        sdf_curr = float(sdf.detach().cpu().numpy()[0])
        d_look_np, d_path, path_idx = _lookahead_dir(cur_xy_np)
        if path is not None:
            effective_idx = max(path_idx, last_path_idx)
            if d_path > 0.20:
                p_recover = path[effective_idx]
                d_recover = (p_recover - cur_xy_np).astype(np.float32)
                d_recover_norm = float(np.linalg.norm(d_recover) + 1e-12)
                d_look_np = (d_recover / d_recover_norm).astype(np.float32)
            elif d_path > 0.06 or (sdf_curr < 0.18 and d_path > 0.035) or (rsa_streak >= 4 and d_path > 0.03):
                # Re-center early in narrow corridors before the path follower starts wall-hugging.
                d_look_np = _path_target_dir(cur_xy_np, effective_idx, max(8, lookahead // 2))
        stall_steps = stall_steps + 1 if path_idx <= last_path_idx else 0
        if stall_steps >= stall_trigger and path is not None:
            extra_ahead = lookahead + min(stall_steps, 2 * lookahead)
            d_look_np = _path_target_dir(cur_xy_np, last_path_idx, extra_ahead)
        last_path_idx = max(last_path_idx, path_idx)
        d_pinn_np = v_pinn.detach().cpu().numpy()[0].astype(np.float32)
        d_goal_np = (np.asarray(goal_xy, dtype=np.float32) - cur_xy_np).astype(np.float32)
        d_goal_np = (d_goal_np / (float(np.linalg.norm(d_goal_np)) + 1e-12)).astype(np.float32)
        goal_align = float(np.dot(d_pinn_np, d_goal_np))
        dist_goal = float(np.linalg.norm(np.asarray(goal_xy, dtype=np.float32) - cur_xy_np))
        blockage = _local_blockage(env=env, cur_xy=cur_xy_np)
        goal_hist.append(dist_goal)
        progress_5, progress_norm, trend, progress_stall_count = _progress_features(
            list(goal_hist),
            float(torch.linalg.norm(grad_t, dim=-1).item() + 1e-12),
            progress_stall_count,
        )
        path_progress = float(path_idx) / float(max(1, 0 if path is None else path.shape[0] - 1))
        exit_conf = _exit_confidence(
            sdf_val=sdf_curr,
            blockage=blockage,
            goal_align=goal_align,
            progress_norm=progress_norm,
            path_progress=path_progress,
        )
        feat = _build_gating_feature(
            goal_xy=goal_xy,
            step_size=step_size,
            d_pinn_np=d_pinn_np,
            d_rsa_np=d_look_np,
            cur_xy=cur_xy_np,
            sdf_val=float(sdf.detach().cpu().numpy()[0]),
            sdf_grad_np=grad_sdf.detach().cpu().numpy()[0].astype(np.float32),
            blockage=blockage,
            progress_5=progress_5,
            progress_norm=progress_norm,
            trend=trend,
            stall_count=progress_stall_count,
            goal_align=goal_align,
            path_progress=path_progress,
            exit_confidence=exit_conf,
        )
        history_buffer.append(feat)
        sdf_curr = float(feat[4])
        v_pinn_t = torch.tensor(d_pinn_np, dtype=torch.float32, device=device).view(1, 2)
        v_rsa_t = torch.tensor(d_look_np, dtype=torch.float32, device=device).view(1, 2)
        x_pinn, ok_pinn, used_backtrack_pinn, used_slide_pinn = _candidate_step(x, v_pinn_t, grad_sdf)
        x_rsa, ok_rsa, used_backtrack_rsa, used_slide_rsa = _candidate_step(x, v_rsa_t, grad_sdf)
        value_pinn = float("inf")
        value_rsa = float("inf")
        exit_conf_pinn = 0.0
        sdf_pinn = float("-inf")
        goal_align_pinn = -1.0
        if ok_pinn:
            value_pinn, exit_conf_pinn, sdf_pinn, goal_align_pinn = _candidate_value(x_pinn)
        if ok_rsa:
            value_rsa, _, _, _ = _candidate_value(x_rsa)
        if sdf_curr < 0.15 or blockage > 0.6:
            current_factor = 1.0
        elif exit_conf > 0.3 or path_progress > 0.8:
            current_factor = 0.4
        else:
            current_factor = 0.95
        rsa_value_gate = bool(ok_rsa and ok_pinn and value_rsa < value_pinn * current_factor)

        if sdf_curr < 0.05 and ok_rsa:
            use_rsa = True
        elif ok_pinn and ok_rsa:
            score_pinn = float(value_pinn)
            score_rsa = float(value_rsa)
            if sdf_pinn > 0.18 and goal_align_pinn > 0.75:
                score_pinn -= float(0.5 + exit_conf_pinn)
                score_rsa += float(0.25 + 0.75 * exit_conf_pinn)
            use_rsa = bool(rsa_value_gate and (score_rsa < score_pinn))
        elif ok_rsa:
            use_rsa = True
        else:
            use_rsa = False

        if ok_rsa and ok_pinn:
            rsa_bonus = 0.0
            if blockage > 0.25:
                rsa_bonus += 1.0
            elif blockage > 0.10:
                rsa_bonus += 0.35
            if progress_stall_count >= 2:
                rsa_bonus += 0.75
            if stall_steps >= stall_bias_trigger and d_path < 0.03:
                rsa_bonus += 1.5
            if d_path > 0.08:
                rsa_bonus += 0.75
            if rsa_streak > 0 and (blockage > 0.10 or sdf_curr < 0.12):
                rsa_bonus += 0.75
            if rsa_streak >= 8 and d_path < 0.06:
                rsa_bonus += 0.75
            if rsa_streak >= 16 and d_path < 0.04:
                rsa_bonus += 1.5
            score_pinn = float(value_pinn)
            score_rsa = float(value_rsa - rsa_bonus)
            if sdf_pinn > 0.18 and goal_align_pinn > 0.75:
                score_pinn -= float(0.5 + exit_conf_pinn)
                score_rsa += float(0.25 + 0.75 * exit_conf_pinn)
            use_rsa = bool(rsa_value_gate and (score_rsa < score_pinn))
        gate_entry_protect = bool(
            ok_rsa
            and path_progress < 0.72
            and (
                (
                    d_path < 0.06
                    and (
                        sdf_curr < 0.24
                        or blockage > 0.08
                        or rsa_streak > 0
                        or progress_stall_count > 0
                    )
                )
                or (
                    path_progress < 0.62
                    and d_path < 0.09
                    and sdf_curr < 0.28
                    and value_rsa <= value_pinn + 0.35
                )
                or (
                    path_progress < 0.72
                    and d_path < 0.04
                    and sdf_curr < 0.30
                    and exit_conf < 0.20
                    and value_rsa <= value_pinn + 1.20
                )
            )
        )
        if gate_entry_protect:
            use_rsa = True
            gate_recenter_countdown = max(gate_recenter_countdown, 6)
        gate_recenter_protect = bool(
            gate_recenter_countdown > 0
            and ok_rsa
            and path_progress < 0.68
            and 0.04 < d_path < 0.075
            and sdf_curr < 0.24
            and blockage < 0.18
            and exit_conf < 0.18
            and value_rsa <= value_pinn + 0.10
        )
        if gate_recenter_protect:
            use_rsa = True
        gate_mid_rescue = bool(
            ok_rsa
            and path_progress < 0.95
            and d_path > 0.16
            and sdf_curr < 0.34
            and value_rsa + 0.05 < value_pinn
        )
        if gate_mid_rescue:
            use_rsa = True
        gate_late_rescue = bool(
            ok_rsa
            and path_progress < 0.82
            and 0.20 < d_path < 0.42
            and sdf_curr < 0.22
            and blockage < 0.28
            and value_rsa + 0.20 < value_pinn
        )
        if gate_late_rescue:
            use_rsa = True
        gate_hard_rescue = bool(
            ok_rsa
            and path_progress < 0.95
            and d_path > 0.34
            and sdf_curr < 0.08
        )
        if gate_hard_rescue:
            use_rsa = True
        if ok_rsa and stall_steps >= stall_force_trigger and d_path < 0.03 and (not ok_pinn or rsa_value_gate):
            use_rsa = True
        gate_release = bool(
            ok_pinn
            and path_progress > 0.45
            and d_path < 0.05
            and sdf_curr > 0.20
            and blockage < 0.15
            and (
                exit_conf_pinn > 0.25
                or (goal_align_pinn > 0.85 and value_pinn + 0.25 < value_rsa)
            )
        )
        if gate_release:
            use_rsa = False
            rsa_exit_cooldown = max(rsa_exit_cooldown, 10)
        hard_exit = bool(exit_conf > 0.75 and sdf_curr > 0.2)
        if hard_exit:
            use_rsa = False
            rsa_exit_cooldown = max(rsa_exit_cooldown, 10)
        elif rsa_exit_cooldown > 0:
            if sdf_curr < 0.1 or d_path > 0.08 or blockage > 0.25:
                rsa_exit_cooldown = 0
            else:
                use_rsa = False
        soft_exit = bool(sdf_pinn > 0.25 and goal_align_pinn > 0.9)
        if obstacle_count <= 3 and rsa_streak >= 6 and sdf_curr > 0.18 and goal_align > 0.75 and d_path < 0.06:
            soft_exit = True
        if obstacle_count <= 3 and rsa_streak >= 4 and stall_steps == 0 and blockage < 0.12 and sdf_curr > 0.18 and goal_align > 0.65 and d_path < 0.08:
            soft_exit = True
        if use_rsa and ok_pinn and soft_exit:
            use_rsa = False

        if use_rsa:
            rsa_usage_count += 1
            rsa_streak += 1
            x_next = x_rsa
            ok_free = ok_rsa
            used_backtrack = used_backtrack_rsa
            used_slide = used_slide_rsa
        else:
            rsa_streak = 0
            x_next = x_pinn
            ok_free = ok_pinn
            used_backtrack = used_backtrack_pinn
            used_slide = used_slide_pinn

        if used_backtrack:
            backtrack_steps += 1
        if used_slide:
            slide_steps += 1
        if step % 25 == 0 or step < 5:
            print(
                f"[VDM] step={step} choose={'rsa' if use_rsa else 'pinn'} "
                f"value_pinn={value_pinn:.3f} value_rsa={value_rsa:.3f} "
                f"exit={exit_conf:.3f} sdf={sdf_curr:.3f} d_path={d_path:.3f} "
                f"factor={current_factor:.2f} cooldown={rsa_exit_cooldown} "
                f"gate={int(gate_entry_protect)}/{int(gate_release)}"
            )

        if not ok_free:
            pts.append(x_next.detach().cpu().numpy()[0])
            return _finalize(False)

        if not torch.isfinite(x_next).all():
            pts.append(x_next.detach().cpu().numpy()[0])
            return _finalize(False)

        with torch.no_grad():
            free = env.is_free(x_next).item()
            if not free:
                pts.append(x_next.detach().cpu().numpy()[0])
                return _finalize(False)
            if torch.linalg.norm(x_next - goal).item() <= goal_threshold:
                pts.append(x_next.detach().cpu().numpy()[0])
                return _finalize(True)

        x = x_next
        pts.append(x.detach().cpu().numpy()[0])
        if rsa_exit_cooldown > 0:
            rsa_exit_cooldown -= 1
        if gate_recenter_countdown > 0:
            gate_recenter_countdown -= 1

    ok = np.linalg.norm(pts[-1] - np.asarray(goal_xy, dtype=np.float32)) <= goal_threshold
    return _finalize(bool(ok))


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


@dataclass(frozen=True)
class EvalMetrics:
    success: bool
    length: float
    smoothness: float
    optimality_gap: float
    regret: float = 0.0
    gating_ratio: Optional[float] = None
    gating_efficiency: Optional[float] = None
    grad_norm_start: Optional[float] = None
    converge_steps: Optional[int] = None
    train_steps: Optional[int] = None


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
) -> Tuple[Dict[str, EvalMetrics], Dict[str, np.ndarray]]:
    ref_path = rsa_ref.backtrack_path_xy()
    ref_len = path_length(ref_path)

    metrics: Dict[str, EvalMetrics] = {}
    paths: Dict[str, np.ndarray] = {"rsa": rsa_low.backtrack_path_xy(), "ref": ref_path}

    rsa_path = paths["rsa"]
    rsa_threshold = float(goal_tol) if bool(strict_goal_tol) else max(float(goal_tol), 0.17)
    rsa_ok = np.linalg.norm(rsa_path[-1] - np.asarray(goal_xy, dtype=np.float32)) <= rsa_threshold
    rsa_len = path_length(rsa_path)
    metrics["rsa"] = EvalMetrics(
        success=bool(rsa_ok),
        length=rsa_len,
        smoothness=path_smoothness(rsa_path),
        optimality_gap=(rsa_len - ref_len) / (ref_len + 1e-12),
        regret=(rsa_len - ref_len),
        grad_norm_start=None,
    )

    for name, model in models.items():
        path, ok, gating_ratio = integrate_path_by_grad(
            model=model,
            env=env,
            start_xy=start_xy,
            goal_xy=goal_xy,
            step_size=path_step,
            max_steps=max_path_steps,
            goal_tol=goal_tol,
            rsa_field=rsa_ref,
            rsa_path_xy=ref_path,
            strict_goal_tol=strict_goal_tol,
            use_gating=(name != "vanilla_pinn"),
        )
        paths[name] = path
        l = path_length(path)
        regret = l - ref_len
        gating_efficiency = regret / max(1e-12, gating_ratio) if np.isfinite(gating_ratio) else None
        metrics[name] = EvalMetrics(
            success=bool(ok),
            length=l,
            smoothness=path_smoothness(path),
            optimality_gap=(l - ref_len) / (ref_len + 1e-12),
            regret=regret,
            gating_ratio=float(gating_ratio) if np.isfinite(gating_ratio) else None,
            gating_efficiency=gating_efficiency,
            grad_norm_start=grad_norm_near_start(
                model=model,
                start_xy=start_xy,
                radius=grad_probe_radius,
                num_points=grad_probe_points,
            ),
        )

    return metrics, paths
