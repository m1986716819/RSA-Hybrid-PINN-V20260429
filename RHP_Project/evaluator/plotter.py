from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt

from ..envs.maze_2d import Maze2DEnv
from ..solvers.rsa_engine import RSAResult


def _safe_contour_levels(
    field: np.ndarray,
    max_levels: int = 30,
) -> tuple[np.ndarray, float, float]:
    finite = np.isfinite(field)
    if np.any(finite):
        vmin = float(np.nanmin(field[finite]))
        vmax = float(np.nanpercentile(field[finite], 98))
        vmax = max(vmax, float(np.nanmax(field[finite])))
    else:
        vmin, vmax = 0.0, 1.0
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)):
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin + 1e-8:
        center = 0.5 * (vmin + vmax)
        span = max(1e-3, 1e-3 * max(1.0, abs(center)))
        vmin = center - span
        vmax = center + span
    levels = np.linspace(vmin, vmax, max(8, min(max_levels, 40)))
    return levels, vmin, vmax


def _eval_model_on_grid(
    model: torch.nn.Module,
    xs: np.ndarray,
    ys: np.ndarray,
    device: torch.device,
    batch: int = 65536,
) -> np.ndarray:
    xx, yy = np.meshgrid(xs, ys)
    xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)
    out = np.empty((xy.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for k in range(0, xy.shape[0], batch):
            xb = torch.from_numpy(xy[k : k + batch]).to(device=device)
            tb = model(xb)
            if tb.ndim == 2:
                tb = tb[:, 0]
            out[k : k + batch] = tb.detach().cpu().numpy()
    return out.reshape(ys.shape[0], xs.shape[0])


def save_field_and_paths_plot(
    env: Maze2DEnv,
    rsa: RSAResult,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    models: Dict[str, torch.nn.Module],
    paths: Dict[str, np.ndarray],
    out_path: str,
    max_levels: int = 30,
    velocity_grid: Optional[np.ndarray] = None,
) -> None:
    xs = rsa.xs
    ys = rsa.ys
    h, w = len(ys), len(xs)

    xx, yy = np.meshgrid(xs, ys)
    sdf_grid = env.sdf(torch.from_numpy(np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1))).cpu().numpy()
    sdf_grid = sdf_grid.reshape(h, w)

    fields: Dict[str, np.ndarray] = {"rsa": rsa.t}
    for name, model in models.items():
        device = next(model.parameters()).device
        fields[name] = _eval_model_on_grid(model=model, xs=xs, ys=ys, device=device)

    has_velocity = velocity_grid is not None and velocity_grid.size > 0
    names = ["rsa"] + list(models.keys())
    ncols = len(names) + (1 if has_velocity else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.6), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    elif not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for ax, name in zip(axes, names):
        field = fields[name]
        levels, vmin, vmax = _safe_contour_levels(field=field, max_levels=max_levels)
        c = ax.contourf(xx, yy, np.clip(field, vmin, vmax), levels=levels, cmap="viridis")
        ax.contour(xx, yy, sdf_grid, levels=[0.0], colors="white", linewidths=1.5)
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)

        ax.plot(start_xy[0], start_xy[1], "ro", markersize=6)
        ax.plot(goal_xy[0], goal_xy[1], "rx", markersize=7, mew=2)

        for pname, path in paths.items():
            if pname in ["ref"]:
                continue
            if pname == name or (name == "rsa" and pname == "rsa"):
                ax.plot(path[:, 0], path[:, 1], "-", linewidth=2.2, color="orange")

        if "ref" in paths:
            ref = paths["ref"]
            ax.plot(ref[:, 0], ref[:, 1], "--", linewidth=1.5, color="cyan")

        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlim(env.bounds.x_min, env.bounds.x_max)
        ax.set_ylim(env.bounds.y_min, env.bounds.y_max)

    if has_velocity:
        ax_v = axes[-1]
        v_display = np.where(np.isfinite(velocity_grid), velocity_grid, 0.0)
        vmin_v = float(np.nanmin(v_display))
        vmax_v = float(np.nanmax(v_display))
        if vmax_v <= vmin_v + 1e-8:
            vmin_v, vmax_v = 0.0, 1.0
        c_v = ax_v.contourf(xx, yy, v_display, levels=np.linspace(vmin_v, vmax_v, 20), cmap="coolwarm")
        ax_v.contour(xx, yy, sdf_grid, levels=[0.0], colors="white", linewidths=1.5)
        fig.colorbar(c_v, ax=ax_v, fraction=0.046, pad=0.04)

        ax_v.plot(start_xy[0], start_xy[1], "ro", markersize=6)
        ax_v.plot(goal_xy[0], goal_xy[1], "rx", markersize=7, mew=2)

        path_colors = {"rsa": "orange", "vanilla_pinn": "cyan", "pntfield_2d": "lime", "rhp_pinn": "magenta"}
        for pname, path in paths.items():
            if pname == "ref":
                continue
            ax_v.plot(path[:, 0], path[:, 1], "-", linewidth=2.0, color=path_colors.get(pname, "white"), label=pname)

        ax_v.legend(fontsize=7, loc="upper right")
        ax_v.set_title("velocity + paths")
        ax_v.set_aspect("equal")
        ax_v.set_xlim(env.bounds.x_min, env.bounds.x_max)
        ax_v.set_ylim(env.bounds.y_min, env.bounds.y_max)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_benchmark_metrics_plot(
    results_json_path: str,
    out_path: Optional[str] = None,
) -> str:
    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_seed = data.get("per_seed", [])
    if not isinstance(per_seed, list) or len(per_seed) == 0:
        raise ValueError("results.json missing per_seed")

    methods = sorted({k for m in per_seed for k in m.keys()})
    sr = {}
    gap = {}
    csteps = {}
    time_cost = {}

    has_time_cost = False
    for name in methods:
        s = np.asarray([bool(m[name].get("success", False)) for m in per_seed], dtype=np.float32)
        g = np.asarray([m[name].get("optimality_gap", np.nan) for m in per_seed], dtype=np.float32)
        c = np.asarray([m[name].get("converge_steps", np.nan) for m in per_seed], dtype=np.float32)
        t = np.asarray([m[name].get("time_cost", np.nan) for m in per_seed], dtype=np.float32)
        sr[name] = float(np.mean(s))
        gap[name] = g[np.isfinite(g)]
        csteps[name] = c[np.isfinite(c)]
        time_cost[name] = t[np.isfinite(t)]
        if np.any(np.isfinite(t)):
            has_time_cost = True

    if out_path is None:
        out_dir = os.path.dirname(results_json_path) or "."
        out_path = os.path.join(out_dir, "benchmark_metrics.png")

    n_plots = 4 if has_time_cost else 3
    fig = plt.figure(figsize=(4.2 * n_plots, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, n_plots)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    names = methods
    x = np.arange(len(names))

    ax0.bar(x, [sr[n] for n in names], color="#4C72B0")
    ax0.set_xticks(x, names, rotation=20, ha="right")
    ax0.set_ylim(0.0, 1.0)
    ax0.set_title("SR")
    ax0.grid(True, axis="y", alpha=0.25)

    gap_data = [gap[n] if gap[n].size > 0 else np.asarray([np.nan], dtype=np.float32) for n in names]
    ax1.boxplot(gap_data, labels=names, showfliers=False)
    ax1.tick_params(axis="x", rotation=20)
    ax1.set_title("Optimality Gap")
    ax1.grid(True, axis="y", alpha=0.25)

    c_data = [csteps[n] if csteps[n].size > 0 else np.asarray([np.nan], dtype=np.float32) for n in names]
    ax2.boxplot(c_data, labels=names, showfliers=False)
    ax2.tick_params(axis="x", rotation=20)
    ax2.set_title("Converge Steps")
    ax2.grid(True, axis="y", alpha=0.25)

    if has_time_cost:
        ax3 = fig.add_subplot(gs[0, 3])
        t_data = [time_cost[n] if time_cost[n].size > 0 else np.asarray([np.nan], dtype=np.float32) for n in names]
        ax3.boxplot(t_data, labels=names, showfliers=False)
        ax3.tick_params(axis="x", rotation=20)
        ax3.set_title("Time Cost")
        ax3.grid(True, axis="y", alpha=0.25)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_field_quiver_plot(
    env: Maze2DEnv,
    rsa: RSAResult,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    model: torch.nn.Module,
    out_path: str,
    stride: int = 4,
    max_levels: int = 30,
) -> None:
    xs = rsa.xs
    ys = rsa.ys
    h, w = len(ys), len(xs)

    xx, yy = np.meshgrid(xs, ys)
    sdf_grid = env.sdf(torch.from_numpy(np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1))).cpu().numpy()
    sdf_grid = sdf_grid.reshape(h, w)

    device = next(model.parameters()).device
    field = _eval_model_on_grid(model=model, xs=xs, ys=ys, device=device)
    levels, vmin, vmax = _safe_contour_levels(field=field, max_levels=max_levels)

    ii = np.arange(0, h, max(1, int(stride)))
    jj = np.arange(0, w, max(1, int(stride)))
    qx, qy = np.meshgrid(xs[jj], ys[ii])
    qxy = np.stack([qx.reshape(-1), qy.reshape(-1)], axis=-1).astype(np.float32)
    qxy_t = torch.from_numpy(qxy).to(device=device, dtype=torch.float32).requires_grad_(True)
    t = model(qxy_t)
    if t.ndim == 2:
        t = t[:, 0]
    grad = torch.autograd.grad(t.sum(), qxy_t, create_graph=False, retain_graph=False)[0]
    g = grad.detach().cpu().numpy()
    gnorm = np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
    u = (-g[:, 0:1] / gnorm).reshape(qx.shape)
    v = (-g[:, 1:2] / gnorm).reshape(qy.shape)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.4), constrained_layout=True)
    c = ax.contourf(xx, yy, np.clip(field, vmin, vmax), levels=levels, cmap="viridis")
    ax.contour(xx, yy, sdf_grid, levels=[0.0], colors="white", linewidths=1.5)
    fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    ax.quiver(qx, qy, u, v, color="black", alpha=0.7, pivot="mid", scale=35.0, width=0.003)

    ax.plot(start_xy[0], start_xy[1], "ro", markersize=6)
    ax.plot(goal_xy[0], goal_xy[1], "rx", markersize=7, mew=2)

    ax.set_title("rhp_pinn_grad_quiver")
    ax.set_aspect("equal")
    ax.set_xlim(env.bounds.x_min, env.bounds.x_max)
    ax.set_ylim(env.bounds.y_min, env.bounds.y_max)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
