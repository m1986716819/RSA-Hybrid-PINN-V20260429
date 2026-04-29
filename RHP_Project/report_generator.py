from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

from .main_bench import (
    _copy_if_exists,
    _grid_coords,
    _latest_run_dir,
    _make_env,
    _set_seed,
    _train_rhp,
    _train_vanilla_pinn,
)
from .models.gating_transformer import GatingTransformer, GatingTransformerConfig
from .solvers.rsa_engine import RSAEngine, extract_gateway_segment
from .utils.sampler import apply_sdf_surface_boost

matplotlib.use("Agg")


def _load_gating_model(root: Path, device: torch.device) -> GatingTransformer:
    ckpt_path = root / "checkpoints" / "gating_transformer_best.pth"
    payload = torch.load(ckpt_path, map_location=device)
    cfg_dict = payload.get("cfg", {})
    cfg = GatingTransformerConfig(**cfg_dict) if isinstance(cfg_dict, dict) else GatingTransformerConfig()
    model = GatingTransformer(cfg).to(device=device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _build_gating_feature(
    goal_xy: Tuple[float, float],
    step_size: float,
    d_pinn_np: np.ndarray,
    d_rsa_np: np.ndarray,
    cur_xy: np.ndarray,
    sdf_val: float,
    sdf_grad_np: np.ndarray,
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
        ],
        dtype=np.float32,
    )


def _path_length(path_xy: np.ndarray) -> float:
    if path_xy.shape[0] < 2:
        return float("inf")
    diffs = path_xy[1:] - path_xy[:-1]
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _line_search(env, x_cur: torch.Tensor, v_dir: torch.Tensor, step_size: float) -> Tuple[torch.Tensor, bool]:
    for i in range(0, 6):
        h = float(step_size) if i == 0 else float(step_size) * (0.5**i)
        x_try = x_cur + h * v_dir
        with torch.no_grad():
            ok_free = bool(env.is_free(x_try).item())
        if ok_free:
            return x_try, True
    return x_cur, False


def _integrate_with_trace(
    model: torch.nn.Module,
    env,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    step_size: float,
    max_steps: int,
    goal_tol: float,
    rsa_path_xy: np.ndarray,
    gating_model: Optional[GatingTransformer] = None,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    x = torch.tensor(start_xy, dtype=torch.float32, device=device).view(1, 2)
    goal = torch.tensor(goal_xy, dtype=torch.float32, device=device).view(1, 2)
    pts = [x.detach().cpu().numpy()[0]]
    path = np.asarray(rsa_path_xy, dtype=np.float32)
    lookahead = 40
    last_path_idx = 0
    stall_steps = 0
    history_buffer: Deque[np.ndarray] = deque(maxlen=10)
    goal_hist: Deque[float] = deque(maxlen=5)
    gate_alpha = []
    gate_prob = []
    cos_trace = []
    sdf_trace = []
    path_dist_trace = []
    takeover_pts = []

    def _lookahead_dir(cur_xy_np: np.ndarray) -> Tuple[np.ndarray, float, int]:
        d2 = np.sum((path - cur_xy_np.reshape(1, 2)) ** 2, axis=1)
        idx = int(np.argmin(d2))
        dist = float(np.sqrt(d2[idx]))
        j = min(idx + lookahead, path.shape[0] - 1)
        d = (path[j] - cur_xy_np).astype(np.float32)
        dn = float(np.linalg.norm(d) + 1e-12)
        return (d / dn).astype(np.float32), dist, idx

    for _ in range(int(max_steps)):
        x = x.detach().requires_grad_(True)
        t = model(x)
        if t.ndim == 2:
            t = t[:, 0]
        sdf = env.sdf(x)
        if sdf.ndim == 2:
            sdf = sdf[:, 0]
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
        gnorm = torch.linalg.norm(grad_t, dim=-1, keepdim=True) + 1e-12
        v_pinn = -grad_t / gnorm
        d_pinn_np = v_pinn.detach().cpu().numpy()[0].astype(np.float32)
        cur_xy_np = x.detach().cpu().numpy()[0].astype(np.float32)

        if gating_model is None:
            v = v_pinn
            gate_alpha.append(0.0)
            gate_prob.append(0.0)
            cos_trace.append(1.0)
            sdf_trace.append(float(sdf.detach().cpu().numpy()[0]))
            path_dist_trace.append(float("nan"))
        else:
            d_look_np, d_path, path_idx = _lookahead_dir(cur_xy_np)
            effective_idx = max(path_idx, last_path_idx)
            if d_path > 0.20:
                p_recover = path[effective_idx]
                d_recover = (p_recover - cur_xy_np).astype(np.float32)
                d_recover_norm = float(np.linalg.norm(d_recover) + 1e-12)
                d_look_np = (d_recover / d_recover_norm).astype(np.float32)
            regressed = bool(path_idx < last_path_idx)
            if regressed:
                j = min(last_path_idx + lookahead, path.shape[0] - 1)
                d_fix = (path[j] - cur_xy_np).astype(np.float32)
                d_fix_norm = float(np.linalg.norm(d_fix) + 1e-12)
                d_look_np = (d_fix / d_fix_norm).astype(np.float32)
            stall_steps = stall_steps + 1 if path_idx <= last_path_idx else 0
            if stall_steps >= 6:
                j = min(last_path_idx + lookahead + min(stall_steps, lookahead), path.shape[0] - 1)
                d_push = (path[j] - cur_xy_np).astype(np.float32)
                d_push_norm = float(np.linalg.norm(d_push) + 1e-12)
                d_look_np = (d_push / d_push_norm).astype(np.float32)
            last_path_idx = max(last_path_idx, path_idx)

            dist_goal = float(np.linalg.norm(np.asarray(goal_xy, dtype=np.float32) - cur_xy_np))
            goal_hist.append(dist_goal)
            feat = _build_gating_feature(
                goal_xy=goal_xy,
                step_size=step_size,
                d_pinn_np=d_pinn_np,
                d_rsa_np=d_look_np,
                cur_xy=cur_xy_np,
                sdf_val=float(sdf.detach().cpu().numpy()[0]),
                sdf_grad_np=grad_sdf.detach().cpu().numpy()[0].astype(np.float32),
            )
            history_buffer.append(feat)
            seq = np.zeros((gating_model.cfg.seq_len, gating_model.cfg.input_dim), dtype=np.float32)
            hist_np = np.stack(list(history_buffer), axis=0)
            seq[-hist_np.shape[0] :] = hist_np
            seq_t = torch.from_numpy(seq).to(device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prob = (
                    float(gating_model.predict_proba(seq_t)[0].item())
                    if len(history_buffer) >= gating_model.cfg.seq_len
                    else 0.0
                )

            cos_align = float(np.dot(d_pinn_np, d_look_np))
            sdf_curr = float(feat[4])
            dist_term = float(np.clip((0.25 - d_path) / 0.25, 0.0, 1.0))
            wall_term = float(np.clip((0.25 - sdf_curr) / 0.25, 0.0, 1.0))
            conflict_term = float(np.clip((0.40 - cos_align) / 1.40, 0.0, 1.0))
            goal_stall = bool(
                len(goal_hist) == goal_hist.maxlen and np.all(np.diff(np.asarray(goal_hist, dtype=np.float32)) >= -1e-4)
            )
            danger_count = int(cos_align < -0.8) + int(goal_stall) + int(sdf_curr < 0.05)
            alpha_tube = 0.20 * dist_term * wall_term
            alpha_conflict = 0.95 * dist_term * wall_term * (0.20 + 0.80 * conflict_term)
            alpha_stall = 1.0 if (stall_steps >= 6 and dist_term > 0.70) else 0.0
            regret_prior = max(alpha_tube, alpha_conflict, alpha_stall)
            alpha = float(np.clip(max(0.15 * prob, regret_prior), 0.0, 1.0))
            if danger_count >= 2:
                alpha = float(max(alpha, 0.80))
            elif prob > 0.9 and danger_count >= 1:
                alpha = float(max(alpha, 0.66))
            if regressed:
                alpha = float(np.clip(alpha + 0.3, 0.0, 1.0))
            if stall_steps > 10:
                alpha = float(max(alpha, 0.86))

            gate_alpha.append(alpha)
            gate_prob.append(prob)
            cos_trace.append(cos_align)
            sdf_trace.append(sdf_curr)
            path_dist_trace.append(d_path)
            if alpha > 0.35:
                takeover_pts.append(cur_xy_np.copy())
            d_mix_np = alpha * d_look_np + (1.0 - alpha) * d_pinn_np
            d_mix_np = (d_mix_np / (float(np.linalg.norm(d_mix_np)) + 1e-12)).astype(np.float32)
            v = torch.tensor(d_mix_np, dtype=torch.float32, device=device).view(1, 2)

        x_next, ok_free = _line_search(env, x, v, step_size)
        if not ok_free:
            n = grad_sdf.detach() / (torch.linalg.norm(grad_sdf.detach(), dim=-1, keepdim=True) + 1e-12)
            dot = torch.sum(v * n, dim=-1, keepdim=True)
            v_slide = v - dot * n
            v_slide = v_slide / (torch.linalg.norm(v_slide, dim=-1, keepdim=True) + 1e-12)
            x_next, ok_free = _line_search(env, x, v_slide, step_size)
        pts.append(x_next.detach().cpu().numpy()[0])
        if not ok_free:
            break
        with torch.no_grad():
            if torch.linalg.norm(x_next - goal).item() <= max(float(goal_tol), 0.17):
                return {
                    "path": np.asarray(pts, dtype=np.float32),
                    "success": True,
                    "gating_ratio": float(np.mean(gate_alpha)) if len(gate_alpha) > 0 else None,
                    "gate_prob_mean": float(np.mean(gate_prob)) if len(gate_prob) > 0 else None,
                    "takeover_points": np.asarray(takeover_pts, dtype=np.float32),
                    "alpha_trace": np.asarray(gate_alpha, dtype=np.float32),
                    "prob_trace": np.asarray(gate_prob, dtype=np.float32),
                    "cos_trace": np.asarray(cos_trace, dtype=np.float32),
                    "sdf_trace": np.asarray(sdf_trace, dtype=np.float32),
                    "path_dist_trace": np.asarray(path_dist_trace, dtype=np.float32),
                }
        x = x_next

    ok = np.linalg.norm(pts[-1] - np.asarray(goal_xy, dtype=np.float32)) <= max(float(goal_tol), 0.17)
    return {
        "path": np.asarray(pts, dtype=np.float32),
        "success": bool(ok),
        "gating_ratio": float(np.mean(gate_alpha)) if len(gate_alpha) > 0 else None,
        "gate_prob_mean": float(np.mean(gate_prob)) if len(gate_prob) > 0 else None,
        "takeover_points": np.asarray(takeover_pts, dtype=np.float32),
        "alpha_trace": np.asarray(gate_alpha, dtype=np.float32),
        "prob_trace": np.asarray(gate_prob, dtype=np.float32),
        "cos_trace": np.asarray(cos_trace, dtype=np.float32),
        "sdf_trace": np.asarray(sdf_trace, dtype=np.float32),
        "path_dist_trace": np.asarray(path_dist_trace, dtype=np.float32),
    }


def _plot_trajectory_comparison(
    out_path: Path,
    env,
    sdf_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    rsa_path: np.ndarray,
    vanilla_trace: Dict[str, Any],
    rhp_trace: Dict[str, Any],
    gate_xy: np.ndarray,
) -> None:
    xx, yy = np.meshgrid(xs, ys)
    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    ax.contour(xx, yy, sdf_grid, levels=[0.0], colors="black", linewidths=2.0)
    ax.plot(rsa_path[:, 0], rsa_path[:, 1], "--", color="#55A868", linewidth=2.5, label="RSA")
    ax.plot(vanilla_trace["path"][:, 0], vanilla_trace["path"][:, 1], "-", color="#C44E52", linewidth=2.0, label="Vanilla PINN")
    ax.plot(rhp_trace["path"][:, 0], rhp_trace["path"][:, 1], "-", color="#4C72B0", linewidth=2.4, label="RHP-PINN")
    ax.scatter([start_xy[0]], [start_xy[1]], c="gold", edgecolors="black", s=90, marker="o", label="Start", zorder=5)
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="lime", edgecolors="black", s=120, marker="*", label="Goal", zorder=5)
    if not bool(vanilla_trace["success"]):
        fail_xy = vanilla_trace["path"][-1]
        ax.scatter([fail_xy[0]], [fail_xy[1]], c="#C44E52", s=120, marker="X", label="Vanilla fail", zorder=6)
    takeover_pts = rhp_trace["takeover_points"]
    if takeover_pts.size > 0:
        step = max(1, takeover_pts.shape[0] // 25)
        show_pts = takeover_pts[::step]
        ax.scatter(show_pts[:, 0], show_pts[:, 1], c="#8172B2", s=28, alpha=0.85, label="RHP takeover", zorder=6)
    if gate_xy.shape[0] > 0:
        ax.plot(gate_xy[:, 0], gate_xy[:, 1], color="#DD8452", linewidth=4.0, alpha=0.85, label="Gateway segment")
    ax.set_title("U-Maze Seed 0 Trajectory Comparison")
    ax.set_aspect("equal")
    ax.set_xlim(env.bounds.x_min, env.bounds.x_max)
    ax.set_ylim(env.bounds.y_min, env.bounds.y_max)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_gate_heatmap(
    out_path: Path,
    gating_model: GatingTransformer,
    device: torch.device,
    cos_trace: np.ndarray,
    sdf_trace: np.ndarray,
    prob_trace: np.ndarray,
    dist_goal: float,
) -> None:
    cos_vals = np.linspace(-1.0, 1.0, 161, dtype=np.float32)
    sdf_vals = np.linspace(-0.05, 0.30, 141, dtype=np.float32)
    heat = np.zeros((sdf_vals.shape[0], cos_vals.shape[0]), dtype=np.float32)
    for i, sdf_val in enumerate(sdf_vals):
        for j, cos_val in enumerate(cos_vals):
            feat = np.asarray([0.02, 0.0, 0.02, 0.0, float(sdf_val), 1.0, 0.0, float(cos_val), float(dist_goal)], dtype=np.float32)
            seq = np.tile(feat.reshape(1, -1), (gating_model.cfg.seq_len, 1))
            seq_t = torch.from_numpy(seq).to(device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                heat[i, j] = float(gating_model.predict_proba(seq_t)[0].item())
    fig, ax = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
    im = ax.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=[cos_vals[0], cos_vals[-1], sdf_vals[0], sdf_vals[-1]],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P_gate")
    if cos_trace.size > 0 and sdf_trace.size > 0 and prob_trace.size > 0:
        ax.scatter(cos_trace, sdf_trace, c=prob_trace, cmap="cool", s=16, alpha=0.55, edgecolors="none")
    ax.set_xlabel("cos_sim(PINN, RSA)")
    ax.set_ylabel("SDF")
    ax.set_title("Transformer Gate Probability")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_performance_bars(out_path: Path, per_seed_metrics: Dict[str, Any]) -> None:
    methods = ["vanilla_pinn", "rsa", "rhp_pinn"]
    labels = ["Vanilla PINN", "RSA", "RHP-PINN"]
    sr = [1.0 if bool(per_seed_metrics[m]["success"]) else 0.0 for m in methods]
    gap = [float(per_seed_metrics[m]["optimality_gap"]) for m in methods]
    colors = ["#C44E52", "#55A868", "#4C72B0"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    axes[0].bar(labels, sr, color=colors)
    axes[0].set_ylim(0.0, 1.1)
    axes[0].set_title("Success Rate")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(labels, gap, color=colors)
    axes[1].set_title("Optimality Gap")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_benchmark_group_summary(out_path: Path, summary: Dict[str, Any]) -> None:
    groups = ["classic", "random", "rubble", "overall"]
    labels = ["Classic", "Random", "Rubble", "Overall"]
    sr = [float(summary[g]["rhp_pinn"]["success"]["mean"]) for g in groups]
    gap = [float(summary[g]["rhp_pinn"]["optimality_gap"]["mean"]) for g in groups]
    ratio = [float(summary[g]["rhp_pinn"]["gating_ratio"]["mean"]) for g in groups]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)

    axes[0].bar(labels, sr, color="#4C72B0")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("RHP-PINN Success Rate")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(labels, gap, color="#55A868")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_title("Optimality Gap")
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].bar(labels, ratio, color="#C44E52")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title("RSA Usage Ratio")
    axes[2].grid(True, axis="y", alpha=0.25)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_method_comparison(out_path: Path, summary: Dict[str, Any]) -> None:
    methods = ["vanilla_pinn", "rsa", "rhp_pinn"]
    labels = ["Vanilla PINN", "RSA", "RHP-PINN"]
    groups = ["classic", "random", "rubble", "overall"]
    group_labels = ["Classic", "Random", "Rubble", "Overall"]
    colors = ["#C44E52", "#55A868", "#4C72B0"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    width = 0.24
    x = np.arange(len(groups), dtype=np.float32)

    for i, method in enumerate(methods):
        sr = [float(summary[g][method]["success"]["mean"]) for g in groups]
        gap = [float(summary[g][method]["optimality_gap"]["mean"]) for g in groups]
        axes[0].bar(x + (i - 1) * width, sr, width=width, color=colors[i], label=labels[i])
        axes[1].bar(x + (i - 1) * width, gap, width=width, color=colors[i], label=labels[i])

    axes[0].set_xticks(x, group_labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Success Rate by Group")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].set_xticks(x, group_labels)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_title("Optimality Gap by Group")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def generate_report(
    root_dir: str = ".",
    results_rel_path: str = "outputs/temp_benchmark.json",
    out_rel_dir: str = "outputs/report_plots",
) -> Dict[str, Any]:
    root = Path(root_dir).resolve()
    results_path = root / results_rel_path
    out_dir = root / out_rel_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if "summary" in results and "seed_results" in results and "config" not in results:
        summary = results["summary"]
        failed = []
        for item in results["seed_results"]:
            rhp = item.get("aggregate", {}).get("rhp_pinn", {})
            success_mean = rhp.get("success_mean")
            if success_mean is not None and float(success_mean) < 0.5:
                failed.append(
                    {
                        "group": item.get("group"),
                        "seed": item.get("seed"),
                        "env_kind": item.get("env_kind"),
                        "success_mean": float(success_mean),
                        "optimality_gap_mean": float(rhp.get("optimality_gap_mean")),
                        "gating_ratio_mean": float(rhp.get("gating_ratio_mean")),
                    }
                )

        group_path = out_dir / "benchmark_group_summary.png"
        compare_path = out_dir / "benchmark_method_comparison.png"
        _plot_benchmark_group_summary(group_path, summary)
        _plot_method_comparison(compare_path, summary)

        report = {
            "results_source": str(results_path),
            "summary": summary,
            "num_seed_results": int(len(results["seed_results"])),
            "num_failed_rhp_pinn": int(len(failed)),
            "failed_rhp_pinn": failed,
            "plots": {
                "benchmark_group_summary": str(group_path),
                "benchmark_method_comparison": str(compare_path),
            },
        }

        out_json = out_dir / "report_stats.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        archive_run_dir = _latest_run_dir({"archive": {"enabled": True}}, create_if_missing=False)
        if archive_run_dir is not None:
            report_archive_dir = archive_run_dir / "reports" / "report_plots"
            report_archive_dir.mkdir(parents=True, exist_ok=True)
            _copy_if_exists(out_json, report_archive_dir / out_json.name)
            _copy_if_exists(group_path, report_archive_dir / group_path.name)
            _copy_if_exists(compare_path, report_archive_dir / compare_path.name)
            report["archive_report_dir"] = str(report_archive_dir)
        return report

    cfg = results["config"]
    device = torch.device(cfg.get("device", "cpu"))
    env = _make_env(cfg, device=device)
    start_xy = tuple(cfg["env"]["start"])
    goal_xy = tuple(cfg["env"]["goal"])
    rng = _set_seed(int(cfg.get("seed", 0)))

    xs, ys = _grid_coords(env.bounds, tuple(cfg["rsa"]["grid_size"]))
    speed = env.speed_grid(tuple(cfg["rsa"]["grid_size"]))
    free_mask = speed > 0.0
    rsa_low = RSAEngine(xs=xs, ys=ys, connectivity=int(cfg["rsa"].get("connectivity", 8))).solve(
        speed=speed,
        start_xy=start_xy,
        goal_xy=goal_xy,
    )
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

    xs_ref, ys_ref = _grid_coords(env.bounds, tuple(cfg["eval"]["reference_grid_size"]))
    speed_ref = env.speed_grid(tuple(cfg["eval"]["reference_grid_size"]))
    rsa_ref = RSAEngine(xs=xs_ref, ys=ys_ref, connectivity=int(cfg["rsa"].get("connectivity", 8))).solve(
        speed=speed_ref,
        start_xy=start_xy,
        goal_xy=goal_xy,
    )
    ref_path_xy = rsa_ref.backtrack_path_xy()
    with torch.no_grad():
        sdf_path_ref = env.sdf(torch.from_numpy(ref_path_xy).to(device=device, dtype=torch.float32)).cpu().numpy()
    gate_info = extract_gateway_segment(
        path_xy=ref_path_xy,
        sdf_values=sdf_path_ref,
        segment_len=int(cfg["train"].get("gate", {}).get("segment_len", 16)),
        sdf_weight=float(cfg["train"].get("gate", {}).get("sdf_weight", 1.0)),
        diff_weight=float(cfg["train"].get("gate", {}).get("diff_weight", 1.0)),
    )

    vanilla, vanilla_conv, vanilla_steps = _train_vanilla_pinn(env, start_xy, goal_xy, cfg, rng, device=device)
    rhp, rhp_conv, rhp_steps = _train_rhp(
        env,
        start_xy,
        goal_xy,
        rsa_ref,
        ref_path_xy,
        gate_info.gate_xy,
        gate_info.d_target,
        gate_info.d_targets,
        xs,
        ys,
        weight,
        cfg,
        rng,
        device=device,
    )

    gating_model = _load_gating_model(root, device)
    vanilla_trace = _integrate_with_trace(
        model=vanilla,
        env=env,
        start_xy=start_xy,
        goal_xy=goal_xy,
        step_size=float(cfg["eval"]["path_step"]),
        max_steps=int(cfg["eval"]["max_path_steps"]),
        goal_tol=float(cfg["eval"]["goal_tol"]),
        rsa_path_xy=ref_path_xy,
        gating_model=None,
    )
    rhp_trace = _integrate_with_trace(
        model=rhp,
        env=env,
        start_xy=start_xy,
        goal_xy=goal_xy,
        step_size=float(cfg["eval"]["path_step"]),
        max_steps=int(cfg["eval"]["max_path_steps"]),
        goal_tol=float(cfg["eval"]["goal_tol"]),
        rsa_path_xy=ref_path_xy,
        gating_model=gating_model,
    )
    rsa_path = rsa_low.backtrack_path_xy()

    traj_path = out_dir / "trajectory_comparison_seed0.png"
    heat_path = out_dir / "gate_probability_heatmap.png"
    bars_path = out_dir / "performance_bars.png"

    _plot_trajectory_comparison(
        out_path=traj_path,
        env=env,
        sdf_grid=sdf_grid,
        xs=xs,
        ys=ys,
        start_xy=start_xy,
        goal_xy=goal_xy,
        rsa_path=rsa_path,
        vanilla_trace=vanilla_trace,
        rhp_trace=rhp_trace,
        gate_xy=gate_info.gate_xy,
    )
    _plot_gate_heatmap(
        out_path=heat_path,
        gating_model=gating_model,
        device=device,
        cos_trace=rhp_trace["cos_trace"],
        sdf_trace=rhp_trace["sdf_trace"],
        prob_trace=rhp_trace["prob_trace"],
        dist_goal=float(np.linalg.norm(np.asarray(goal_xy, dtype=np.float32) - np.asarray(start_xy, dtype=np.float32))),
    )
    _plot_performance_bars(out_path=bars_path, per_seed_metrics=results["per_seed"][0])

    per_seed0 = results["per_seed"][0]
    report = {
        "results_json_summary": results.get("summary", {}),
        "seed0_results_json": per_seed0,
        "seed0_live": {
            "rsa_success": True,
            "rsa_length": float(_path_length(rsa_path)),
            "vanilla_success": bool(vanilla_trace["success"]),
            "vanilla_length": float(_path_length(vanilla_trace["path"])),
            "vanilla_converge_steps": int(vanilla_conv),
            "vanilla_train_steps": int(vanilla_steps),
            "rhp_success": bool(rhp_trace["success"]),
            "rhp_length": float(_path_length(rhp_trace["path"])),
            "rhp_converge_steps": int(rhp_conv),
            "rhp_train_steps": int(rhp_steps),
            "transformer_gate_ratio": float(rhp_trace["gating_ratio"]) if rhp_trace["gating_ratio"] is not None else None,
            "transformer_gate_prob_mean": float(rhp_trace["gate_prob_mean"]) if rhp_trace["gate_prob_mean"] is not None else None,
            "takeover_count_alpha_gt_035": int(np.sum(rhp_trace["alpha_trace"] > 0.35)),
            "plots": {
                "trajectory_comparison": str(traj_path),
                "gate_probability_heatmap": str(heat_path),
                "performance_bars": str(bars_path),
            },
        },
        "models": {
            "vanilla": {
                "converge_steps": int(vanilla_conv),
                "train_steps": int(vanilla_steps),
            },
            "rhp": {
                "converge_steps": int(rhp_conv),
                "train_steps": int(rhp_steps),
            },
        },
    }

    out_json = out_dir / "report_stats.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    archive_run_dir = _latest_run_dir(cfg, create_if_missing=True)
    if archive_run_dir is not None:
        report_archive_dir = archive_run_dir / "reports" / "report_plots"
        report_archive_dir.mkdir(parents=True, exist_ok=True)
        _copy_if_exists(out_json, report_archive_dir / out_json.name)
        _copy_if_exists(traj_path, report_archive_dir / traj_path.name)
        _copy_if_exists(heat_path, report_archive_dir / heat_path.name)
        _copy_if_exists(bars_path, report_archive_dir / bars_path.name)
        report["archive_report_dir"] = str(report_archive_dir)

    return report


def main() -> None:
    report = generate_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
