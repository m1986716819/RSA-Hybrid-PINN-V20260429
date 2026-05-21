#!/usr/bin/env python3
"""Generate coupling_alpha decision behaviour figures (Paper Figure 1 & 2).

Runs a single-seed experiment on the specified config and produces:
  - figure1_decision_path.pdf   (path overlay with alpha colouring)
  - figure2_alpha_trajectory.pdf (alpha vs progress curve)

Usage:
  python scripts/run_decision_plot.py --config configs/trap_heterogeneous_vf.yaml
  python scripts/run_decision_plot.py --config configs/ablation_test1_topology.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from RHP_Project.envs.maze_2d import Bounds2D
from RHP_Project.solvers.rsa_engine import RSAEngine
from RHP_Project.solvers.coupling import PhysicsGuidedCoupling
from RHP_Project.solvers.factored_nn import FactoredTimeNN
from RHP_Project.evaluator.metrics import evaluate_methods, path_length
from RHP_Project.benchmark.visualization.decision_plotter import DecisionPlotter


def _load_cfg(path: str) -> dict:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config: {path}")
    return cfg


def _make_env(cfg: dict, device: torch.device):
    """Delegate to main_bench's _make_env to handle all scene types."""
    from RHP_Project.main_bench import _make_env as _mb_make_env
    env = _mb_make_env(cfg, device)
    return env, tuple(cfg["env"]["start"]), tuple(cfg["env"]["goal"])


def main():
    ap = argparse.ArgumentParser(description="Generate coupling_alpha decision figures")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--device", type=str, default="cpu", help="Device (cpu/mps/cuda)")
    ap.add_argument("--output", type=str, default="paper_figures", help="Output directory")
    args = ap.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(args.config)
    env_cfg = cfg["env"]
    start_xy = tuple(env_cfg["start"])
    goal_xy = tuple(env_cfg["goal"])
    bounds = Bounds2D(**env_cfg["bounds"])
    gs = tuple(cfg.get("rsa", {}).get("grid_size", [100, 100]))
    scene_name = env_cfg.get("name", "scene")

    print(f"RHP-PINN Decision Plot Generator")
    print(f"  Config:  {args.config}")
    print(f"  Scene:   {scene_name}")
    print(f"  Device:  {device}")
    print(f"  Output:  {output_dir}")

    # Build environment
    env, start_xy, goal_xy = _make_env(cfg, device)
    print(f"  Env:     {type(env).__name__}")

    # RSA solve (both resolutions)
    xs_low = np.linspace(bounds.x_min, bounds.x_max, gs[1], dtype=np.float32)
    ys_low = np.linspace(bounds.y_min, bounds.y_max, gs[0], dtype=np.float32)
    speed_low = env.speed_grid(gs)
    rsa_low = RSAEngine(xs=xs_low, ys=ys_low, connectivity=8).solve(
        speed=speed_low, start_xy=start_xy, goal_xy=goal_xy,
    )

    xs_ref = np.linspace(bounds.x_min, bounds.x_max, 200, dtype=np.float32)
    ys_ref = np.linspace(bounds.y_min, bounds.y_max, 200, dtype=np.float32)
    speed_ref = env.speed_grid((200, 200))
    rsa_ref = RSAEngine(xs=xs_ref, ys=ys_ref, connectivity=8).solve(
        speed=speed_ref, start_xy=start_xy, goal_xy=goal_xy,
    )
    ref_path = rsa_ref.backtrack_path_xy()
    print(f"  RSA ref path: {ref_path.shape[0]} points")

    # Train RHP-PINN model (1 seed, reduced steps for speed)
    from RHP_Project.main_bench import _train_rhp

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    np.random.seed(42)

    weight = np.ones(gs, dtype=np.float32)
    xs_arr = xs_low
    ys_arr = ys_low

    print(f"  Training RHP-PINN...", end=" ", flush=True)
    model, conv_step, train_steps = _train_rhp(
        env, start_xy, goal_xy, rsa_ref, ref_path,
        xs_arr, ys_arr, weight, cfg, rng, device,
    )
    print(f"converged at step {conv_step}")

    # Evaluate with coupling (alpha data comes back in paths dict)
    coupling_model = PhysicsGuidedCoupling(
        w_resid=1.0, w_conf=2.0, w_safety=3.0, b=-1.0, k=8.0,
    )

    models = {"rhp_pinn": model}
    metrics, paths = evaluate_methods(
        env=env, start_xy=start_xy, goal_xy=goal_xy,
        rsa_low=rsa_low, rsa_ref=rsa_ref,
        models=models,
        path_step=float(cfg["eval"]["path_step"]),
        max_path_steps=int(cfg["eval"]["max_path_steps"]),
        goal_tol=float(cfg["eval"]["goal_tol"]),
        grad_probe_radius=float(cfg["eval"]["grad_probe_radius"]),
        grad_probe_points=int(cfg["eval"]["grad_probe_points"]),
        strict_goal_tol=bool(cfg["eval"].get("strict_goal_tol", False)),
        coupling_model=coupling_model,
        coupling_default_alpha=0.3,
    )

    rhp_path = paths.get("rhp_pinn")
    alpha_arr = paths.get("rhp_pinn_alpha")
    rsa_path_low = paths.get("rsa")

    if rhp_path is None:
        print("ERROR: RHP-PINN rollout failed (no path)")
        return 1

    m = metrics["rhp_pinn"]
    print(f"  RHP-PINN: SR={int(m.success)}, "
          f"len={m.length:.2f}, TC={m.time_cost:.4f}")
    if alpha_arr is not None:
        print(f"  Alpha:   mean={float(np.mean(alpha_arr)):.3f}, "
              f"min={float(np.min(alpha_arr)):.3f}, "
              f"max={float(np.max(alpha_arr)):.3f}")

    # Convert world coords to pixel coords for plotting
    h_scene, w_scene = 100, 100  # proxy grid for visualisation
    world_to_px = lambda wx, wy: (
        int(round(wy * (h_scene - 1))),
        int(round(wx * (w_scene - 1))),
    )

    # Build a proxy scene grid from env SDF
    xs_plot = np.linspace(bounds.x_min, bounds.x_max, w_scene, dtype=np.float32)
    ys_plot = np.linspace(bounds.y_min, bounds.y_max, h_scene, dtype=np.float32)
    xx, yy = np.meshgrid(xs_plot, ys_plot)
    xy_plot = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    with torch.no_grad():
        sdf_vals = env.sdf(torch.from_numpy(xy_plot)).numpy().reshape(h_scene, w_scene)
    scene_grid = (sdf_vals <= 0.0).astype(np.uint8)

    # Convert world paths to pixel paths
    def path_to_px(path_w: np.ndarray) -> np.ndarray:
        px = np.zeros_like(path_w)
        px[:, 0] = path_w[:, 1] * float(h_scene - 1)  # row = y
        px[:, 1] = path_w[:, 0] * float(w_scene - 1)  # col = x
        return px

    rhp_path_px = path_to_px(rhp_path)
    rsa_path_px = path_to_px(rsa_path_low) if rsa_path_low is not None else None
    ref_path_px = path_to_px(ref_path) if ref_path is not None else None
    start_px = world_to_px(start_xy[0], start_xy[1])
    goal_px = world_to_px(goal_xy[0], goal_xy[1])

    # Generate figures
    plotter = DecisionPlotter(str(output_dir))

    if alpha_arr is not None and alpha_arr.size > 0:
        fig1_path = plotter.plot_decision_path(
            scene_grid=scene_grid,
            rhp_path=rhp_path_px,
            alpha_seq=alpha_arr,
            start_px=start_px,
            goal_px=goal_px,
            rsa_path=rsa_path_px,
            ref_path=ref_path_px,
            scene_name=scene_name,
            save_name="figure1_decision_path.pdf",
        )
        print(f"  [OK] {fig1_path}")

        fig2_path = plotter.plot_alpha_trajectory(
            alpha_seq=alpha_arr,
            path_length_px=float(m.length),
            scene_name=scene_name,
            save_name="figure2_alpha_trajectory.pdf",
        )
        print(f"  [OK] {fig2_path}")

        print(f"\nDone. Figures saved to: {output_dir.resolve()}")
    else:
        print("WARNING: No alpha data collected (coupling not active)")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())