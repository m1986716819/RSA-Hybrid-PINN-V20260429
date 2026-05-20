"""Post-hoc visualization for completed experiments.

Reconstructs environment from config and generates publication-quality plots.
Does NOT modify any RHP_Project code — operates as a pure post-processing layer.

Visualization levels:
  - basic (default): 5 essential overview plots
  - paper: adds 4 advanced statistical/comparison plots
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from rhp_experiment.config import ExperimentConfig


class ExperimentVisualizer:
    """Post-hoc visualizer for completed RHP-PINN experiments.

    Reads an experiment directory (config + per-seed metrics), reconstructs the
    environment, runs the RSA solver, and generates visualization plots.

    Usage:
        viz = ExperimentVisualizer("experiments/exp_20260506_xxx")
        viz.plot_all()              # basic level (5 plots)
        viz.plot_all(level="paper") # paper level (9 plots)
    """

    def __init__(self, exp_dir: str):
        self.exp_dir = Path(exp_dir).resolve()
        if not self.exp_dir.is_dir():
            raise NotADirectoryError(f"Experiment directory not found: {exp_dir}")

        self.config_path = self.exp_dir / "config.yaml"
        if not self.config_path.exists():
            raise FileNotFoundError(f"Missing config.yaml in {exp_dir}")

        self._config = ExperimentConfig(str(self.config_path))
        self._config_data = self._config.to_dict()

        self.seeds_dir = self.exp_dir / "seeds"
        self._seed_data = self._load_seed_data()

        self.plots_dir = self.exp_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._env = None
        self._rsa_result = None
        self._rsa_path = None
        self._velocity_grid = None
        self._sdf_grid = None
        self._xx = None
        self._yy = None

    def _get_alpha(self, sd: dict) -> float:
        return float(sd.get("coupling_alpha", sd.get("gating_ratio", float("nan"))))

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _load_seed_data(self) -> List[Dict[str, Any]]:
        if not self.seeds_dir.exists():
            return []
        seeds = []
        for p in sorted(self.seeds_dir.glob("seed_*.json")):
            with open(p) as f:
                seeds.append(json.load(f))
        return seeds

    def _build_env(self):
        if self._env is not None:
            return
        import torch
        from RHP_Project.envs.maze_2d import Bounds2D, Maze2DEnv

        cfg = self._config_data
        bcfg = cfg["env"]["bounds"]
        bounds = Bounds2D(**bcfg)
        name = cfg["env"]["name"]
        common = dict(
            bounds=bounds,
            obstacle_inflation=float(cfg["env"].get("obstacle_inflation", 0.0)),
            speed_free=float(cfg["rsa"].get("speed_free", 1.0)),
            speed_obstacle=float(cfg["rsa"].get("speed_obstacle", 0.0)),
            device=torch.device("cpu"),
        )

        if name == "u_maze":
            p = cfg["env"]["u_maze"]
            self._env = Maze2DEnv.make_u_maze(**common, **p)
        elif name == "narrow_passage":
            p = cfg["env"]["narrow_passage"]
            self._env = Maze2DEnv.make_narrow_passage(**common, **p)
        elif name == "trap_u_shape":
            p = cfg["env"]["trap_u_shape"]
            self._env = Maze2DEnv.make_trap_u_shape(**common, **p)
        elif name == "trap_heterogeneous_vf":
            p = cfg["env"]["trap_heterogeneous_vf"]
            self._env = Maze2DEnv.make_trap_heterogeneous_vf(**common, **p)
        elif name == "open_space":
            self._env = Maze2DEnv.make_open_space(**common)
        else:
            raise ValueError(f"Unknown env name: {name}")

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
                self._env.set_velocity_field(_half_space_velocity)

    def _solve_rsa(self):
        if self._rsa_result is not None:
            return
        self._build_env()
        from RHP_Project.solvers.rsa_engine import RSAEngine

        cfg = self._config_data
        grid_size = tuple(cfg["rsa"]["grid_size"])
        start_xy = tuple(cfg["env"]["start"])
        goal_xy = tuple(cfg["env"]["goal"])

        xs, ys, _ = self._env.make_grid(grid_size)
        speed = self._env.speed_grid(grid_size)

        engine = RSAEngine(xs=xs, ys=ys, connectivity=int(cfg["rsa"].get("connectivity", 8)))
        self._rsa_result = engine.solve(speed=speed, start_xy=start_xy, goal_xy=goal_xy)
        self._rsa_path = self._rsa_result.backtrack_path_xy()

    def _build_grids(self):
        if self._xx is not None:
            return
        self._solve_rsa()
        self._build_env()

        xs = self._rsa_result.xs
        ys = self._rsa_result.ys
        self._xx, self._yy = np.meshgrid(xs, ys)
        xy = np.stack([self._xx.reshape(-1), self._yy.reshape(-1)], axis=-1)

        import torch
        xy_t = torch.from_numpy(xy.astype(np.float32))
        sdf_vals = self._env.sdf(xy_t).cpu().numpy().reshape(len(ys), len(xs))
        self._sdf_grid = sdf_vals

        vg = self._env.velocity_grid((len(ys), len(xs)))
        self._velocity_grid = vg if vg is not None else np.full_like(sdf_vals, float(self._env.speed_free))

    # ===============================================================
    # BASIC-LEVEL PLOTS (always generated)
    # ===============================================================

    # ---------------------------------------------------------------
    # Plot 1: Velocity field heatmap
    # ---------------------------------------------------------------

    def plot_velocity_field(self, save_name: str = "velocity_field.png") -> Path:
        self._build_grids()
        self._solve_rsa()
        cfg = self._config_data
        start_xy = tuple(cfg["env"]["start"])
        goal_xy = tuple(cfg["env"]["goal"])

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5), constrained_layout=True)

        vmin = float(np.nanmin(self._velocity_grid))
        vmax = float(np.nanmax(self._velocity_grid))
        if vmax <= vmin + 1e-8:
            vmin, vmax = 0.0, 1.0

        c = ax.contourf(self._xx, self._yy, self._velocity_grid,
                        levels=np.linspace(vmin, vmax, 25), cmap="coolwarm")
        ax.contour(self._xx, self._yy, self._sdf_grid, levels=[0.0],
                   colors="black", linewidths=2.0)
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04, label="Speed V(x,y)")

        ax.plot(start_xy[0], start_xy[1], "o", color="#2ecc71", markersize=9,
                markeredgecolor="white", markeredgewidth=1.5, label="Start")
        ax.plot(goal_xy[0], goal_xy[1], "X", color="#e74c3c", markersize=11,
                markeredgecolor="white", markeredgewidth=1.5, label="Goal")

        ax.set_xlim(self._env.bounds.x_min, self._env.bounds.x_max)
        ax.set_ylim(self._env.bounds.y_min, self._env.bounds.y_max)
        ax.set_aspect("equal")
        ax.set_title("Velocity Field", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")

        out_path = self.plots_dir / save_name
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] saved {out_path}")
        return out_path

    # ---------------------------------------------------------------
    # Plot 2: Comparison bar charts (per seed)
    # ---------------------------------------------------------------

    def plot_comparison_metrics(self, save_name: str = "comparison_metrics.png") -> Path:
        if not self._seed_data:
            print("  [viz] WARNING: no seed data, skipping comparison_metrics")
            return self.plots_dir / save_name

        n_seeds = len(self._seed_data)

        metrics_cfg = [
            ("TimeCost", "time_cost", "rsa_time_cost", "Time Cost", "#3498db", "#e74c3c"),
            ("CouplingAlpha", "coupling_alpha", None, "Coupling Alpha", "#9b59b6", None),
            ("PhysicalConsistency", "physical_consistency", None, "Physical Consistency", "#e67e22", None),
            ("OptimalityGap", "optimality_gap", None, "Optimality Gap", "#1abc9c", None),
        ]

        n_plots = len(metrics_cfg)
        fig, axes = plt.subplots(1, n_plots, figsize=(5.0 * n_plots, 4.5), constrained_layout=True)
        if n_plots == 1:
            axes = [axes]

        for ax, (key, field, rsa_field, title, color1, color2) in zip(axes, metrics_cfg):
            vals = [sd.get(field, float("nan")) for sd in self._seed_data]
            vals_f = [v for v in vals if np.isfinite(v)]

            if rsa_field:
                rsa_vals = [sd.get(rsa_field, float("nan")) for sd in self._seed_data]
                rsa_vals_f = [v for v in rsa_vals if np.isfinite(v)]

            x = np.arange(n_seeds)
            w = 0.35

            if rsa_field and len(rsa_vals_f) > 0:
                ax.bar(x - w / 2, rsa_vals, w, color=color2, alpha=0.7, label="RSA",
                       edgecolor="white", linewidth=0.5)

            ax.bar(x + w / 2, vals, w, color=color1, alpha=0.8, label="RHP-PINN",
                   edgecolor="white", linewidth=0.5)

            ax.axhline(y=0, color="gray", linewidth=0.5)
            ax.set_xlabel("Seed", fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.25)

        out_path = self.plots_dir / save_name
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] saved {out_path}")
        return out_path

    # ---------------------------------------------------------------
    # Plot 3: Gating ratios across seeds
    # ---------------------------------------------------------------

    def plot_gating_ratios(self, save_name: str = "coupling_alpha.png") -> Path:
        if not self._seed_data:
            print("  [viz] WARNING: no seed data, skipping coupling_alpha")
            return self.plots_dir / save_name

        gatings = [self._get_alpha(sd) for sd in self._seed_data]
        gatings_f = [g for g in gatings if np.isfinite(g)]

        if not gatings_f:
            print("  [viz] WARNING: no valid coupling alpha data, skipping coupling_alpha")
            return self.plots_dir / save_name

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)

        x = np.arange(len(gatings_f))

        colors = ["#2ecc71" if g < 0.5 else "#e74c3c" for g in gatings_f]
        ax1.bar(x, gatings_f, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax1.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="0.5 threshold")
        ax1.axhline(y=float(np.mean(gatings_f)), color="black", linestyle="--", linewidth=0.8, alpha=0.6,
                    label=f"mean={np.mean(gatings_f):.3f}")
        ax1.set_xlabel("Seed", fontsize=10)
        ax1.set_ylabel("Coupling Alpha", fontsize=10)
        ax1.set_title("Coupling Alpha per Seed", fontsize=11, fontweight="bold")
        ax1.set_xticks(x)
        ax1.legend(fontsize=8)
        ax1.grid(True, axis="y", alpha=0.25)

        mean_g = float(np.mean(gatings_f))
        std_g = float(np.std(gatings_f)) if len(gatings_f) > 1 else 0.0
        ax2.axis("off")
        summary_text = (
            f"Coupling Alpha Summary\n"
            f"{'─' * 25}\n"
            f"  Seeds:    {len(gatings_f)}\n"
            f"  Mean:     {mean_g:.4f}\n"
            f"  Std:      {std_g:.4f}\n"
            f"  Min:      {float(np.min(gatings_f)):.4f}\n"
            f"  Max:      {float(np.max(gatings_f)):.4f}\n"
            f"  Median:   {float(np.median(gatings_f)):.4f}\n"
        )
        ax2.text(0.15, 0.55, summary_text, fontsize=12, fontfamily="monospace",
                 verticalalignment="center", transform=ax2.transAxes)

        out_path = self.plots_dir / save_name
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] saved {out_path}")
        return out_path

    # ---------------------------------------------------------------
    # Plot 4: RSA vs RHP path overlays (from seed data)
    # ---------------------------------------------------------------

    def plot_path_overview(self, save_name: str = "path_overview.png") -> Path:
        self._build_grids()
        self._solve_rsa()
        cfg = self._config_data
        start_xy = tuple(cfg["env"]["start"])
        goal_xy = tuple(cfg["env"]["goal"])

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)

        ax = axes[0]
        vmin = float(np.nanmin(self._velocity_grid))
        vmax = float(np.nanmax(self._velocity_grid))
        if vmax <= vmin + 1e-8:
            vmin, vmax = 0.0, 1.0
        c = ax.contourf(self._xx, self._yy, self._velocity_grid,
                        levels=np.linspace(vmin, vmax, 25), cmap="coolwarm")
        ax.contour(self._xx, self._yy, self._sdf_grid, levels=[0.0],
                   colors="black", linewidths=2.0)

        if self._rsa_path is not None:
            ax.plot(self._rsa_path[:, 0], self._rsa_path[:, 1], "-",
                    color="#f39c12", linewidth=2.5, label=f"RSA")

        ax.plot(start_xy[0], start_xy[1], "o", color="#2ecc71", markersize=9,
                markeredgecolor="white", markeredgewidth=1.5, label="Start")
        ax.plot(goal_xy[0], goal_xy[1], "X", color="#e74c3c", markersize=11,
                markeredgecolor="white", markeredgewidth=1.5, label="Goal")
        ax.set_xlim(self._env.bounds.x_min, self._env.bounds.x_max)
        ax.set_ylim(self._env.bounds.y_min, self._env.bounds.y_max)
        ax.set_aspect("equal")
        ax.set_title("RSA Reference Path", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)

        ax = axes[1]
        c = ax.contourf(self._xx, self._yy, self._velocity_grid,
                        levels=np.linspace(vmin, vmax, 25), cmap="coolwarm")
        ax.contour(self._xx, self._yy, self._sdf_grid, levels=[0.0],
                   colors="black", linewidths=2.0)

        if self._rsa_path is not None:
            ax.plot(self._rsa_path[:, 0], self._rsa_path[:, 1], "-",
                    color="#f39c12", linewidth=1.5, alpha=0.5, label="RSA ref")

        gating_text = ""
        if self._seed_data:
            gatings = [self._get_alpha(sd) for sd in self._seed_data]
            gatings_f = [g for g in gatings if np.isfinite(g)]
            if gatings_f:
                gating_text = f"Alpha: {np.mean(gatings_f):.3f} +/- {np.std(gatings_f):.3f}"
                tcs = [sd.get("time_cost", float("nan")) for sd in self._seed_data]
                tcs_f = [t for t in tcs if np.isfinite(t)]
                if tcs_f:
                    rsa_tc = self._seed_data[0].get("rsa_time_cost", float("nan"))
                    if np.isfinite(rsa_tc):
                        gating_text += f"\nRHP TC: {np.mean(tcs_f):.3f} +/- {np.std(tcs_f):.3f}\nRSA TC: {rsa_tc:.3f}"

        ax.plot(start_xy[0], start_xy[1], "o", color="#2ecc71", markersize=9,
                markeredgecolor="white", markeredgewidth=1.5, label="Start")
        ax.plot(goal_xy[0], goal_xy[1], "X", color="#e74c3c", markersize=11,
                markeredgecolor="white", markeredgewidth=1.5, label="Goal")
        ax.set_xlim(self._env.bounds.x_min, self._env.bounds.x_max)
        ax.set_ylim(self._env.bounds.y_min, self._env.bounds.y_max)
        ax.set_aspect("equal")
        ax.set_title("RHP-PINN Overview", fontsize=12, fontweight="bold")

        if gating_text:
            ax.text(0.05, 0.95, gating_text, fontsize=10, fontfamily="monospace",
                    transform=ax.transAxes, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        fig.suptitle(f"Scenario: {self._config.scenario}",
                     fontsize=14, fontweight="bold", y=1.02)

        out_path = self.plots_dir / save_name
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] saved {out_path}")
        return out_path

    # ---------------------------------------------------------------
    # Plot 5: Combined best-seed trajectory (paper-ready)
    # ---------------------------------------------------------------

    def plot_best_trajectory(self, save_name: str = "best_trajectory.png") -> Path:
        if not self._seed_data:
            print("  [viz] WARNING: no seed data, skipping best_trajectory")
            return self.plots_dir / save_name

        valid = [sd for sd in self._seed_data if sd.get("success", False)]
        if not valid:
            print("  [viz] WARNING: no successful seeds, skipping best_trajectory")
            return self.plots_dir / save_name

        best = min(valid, key=lambda sd: sd.get("time_cost", float("inf")))
        best_seed = best.get("seed", 0)

        self._build_grids()
        self._solve_rsa()
        cfg = self._config_data
        start_xy = tuple(cfg["env"]["start"])
        goal_xy = tuple(cfg["env"]["goal"])

        fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.0), constrained_layout=True)

        vmin = float(np.nanmin(self._velocity_grid))
        vmax = float(np.nanmax(self._velocity_grid))
        if vmax <= vmin + 1e-8:
            vmin, vmax = 0.0, 1.0
        c = ax.contourf(self._xx, self._yy, self._velocity_grid,
                        levels=np.linspace(vmin, vmax, 25), cmap="coolwarm")
        ax.contour(self._xx, self._yy, self._sdf_grid, levels=[0.0],
                   colors="black", linewidths=2.0)
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04, label="Speed V(x,y)")

        ax.plot(start_xy[0], start_xy[1], "o", color="#2ecc71", markersize=9,
                markeredgecolor="white", markeredgewidth=1.5, label="Start")
        ax.plot(goal_xy[0], goal_xy[1], "X", color="#e74c3c", markersize=11,
                markeredgecolor="white", markeredgewidth=1.5, label="Goal")

        if self._rsa_path is not None:
            ax.plot(self._rsa_path[:, 0], self._rsa_path[:, 1], "--",
                    color="#f39c12", linewidth=1.8, alpha=0.6,
                    label=f"RSA (TC={best.get('rsa_time_cost', 0):.3f})")

        ax.set_xlim(self._env.bounds.x_min, self._env.bounds.x_max)
        ax.set_ylim(self._env.bounds.y_min, self._env.bounds.y_max)
        ax.set_aspect("equal")

        info_text = (
            f"Best Seed #{best_seed}\n"
            f"TimeCost: {best.get('time_cost', 0):.4f}\n"
            f"vs RSA:   {best.get('rsa_time_cost', 0):.4f}\n"
            f"Alpha:    {self._get_alpha(best):.4f}\n"
            f"Gap:      {best.get('optimality_gap', 0):.4f}"
        )
        ax.text(0.05, 0.95, info_text, fontsize=9, fontfamily="monospace",
                transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

        ax.set_title(f"{self._config.scenario} -- Best RHP-PINN Seed",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")

        out_path = self.plots_dir / save_name
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] saved {out_path}")
        return out_path

    # ===============================================================
    # PAPER-LEVEL PLOTS (only with --visualize-level paper)
    # ===============================================================

    # ---------------------------------------------------------------
    # Paper Plot 1: Path comparison overlay (the most important one)
    #   Shows RSA (blue dashed) vs RHP-PINN (red solid) on velocity
    #   field background with TimeCost annotations and improvement %
    # ---------------------------------------------------------------

    def plot_path_comparison(self, save_name: str = "path_compare.png") -> Path:
        self._build_grids()
        self._solve_rsa()
        cfg = self._config_data
        start_xy = tuple(cfg["env"]["start"])
        goal_xy = tuple(cfg["env"]["goal"])

        fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.0), constrained_layout=True)

        vmin = float(np.nanmin(self._velocity_grid))
        vmax = float(np.nanmax(self._velocity_grid))
        if vmax <= vmin + 1e-8:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(self._velocity_grid, extent=[
            self._env.bounds.x_min, self._env.bounds.x_max,
            self._env.bounds.y_min, self._env.bounds.y_max,
        ], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        ax.contour(self._xx, self._yy, self._sdf_grid, levels=[0.0],
                   colors="black", linewidths=2.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Speed V(x,y)")

        if self._rsa_path is not None:
            ax.plot(self._rsa_path[:, 0], self._rsa_path[:, 1], "--",
                    color="#3498db", linewidth=2.2, alpha=0.8, label="RSA reference")

        rsa_tc = float("nan")
        if self._seed_data:
            rsa_tc = self._seed_data[0].get("rsa_time_cost", float("nan"))

        valid = [sd for sd in self._seed_data if sd.get("success", False)]

        rhp_tc_mean = float("nan")
        improvement_pct = 0.0
        if valid:
            tcs = [sd.get("time_cost", float("nan")) for sd in valid]
            tcs_f = [t for t in tcs if np.isfinite(t)]
            if tcs_f:
                rhp_tc_mean = float(np.mean(tcs_f))
                if np.isfinite(rsa_tc) and rsa_tc > 0:
                    improvement_pct = (rsa_tc - rhp_tc_mean) / rsa_tc * 100.0

        if valid and np.isfinite(rhp_tc_mean):
            best = min(valid, key=lambda sd: sd.get("time_cost", float("inf")))
            label_text = (
                f"RHP-PINN best seed #{best.get('seed', 0)} "
                f"(TC={best.get('time_cost', 0):.3f})"
            )
            ax.plot([], [], "-", color="#e74c3c", linewidth=2.2, label=label_text)

        ax.plot(start_xy[0], start_xy[1], "o", color="#2ecc71", markersize=10,
                markeredgecolor="white", markeredgewidth=1.5, label="Start")
        ax.plot(goal_xy[0], goal_xy[1], "X", color="#e74c3c", markersize=12,
                markeredgecolor="white", markeredgewidth=1.5, label="Goal")

        ax.set_xlim(self._env.bounds.x_min, self._env.bounds.x_max)
        ax.set_ylim(self._env.bounds.y_min, self._env.bounds.y_max)
        ax.set_aspect("equal")

        if np.isfinite(rsa_tc) and np.isfinite(rhp_tc_mean):
            if valid:
                gatings = [self._get_alpha(sd) for sd in valid]
                gatings_f = [g for g in gatings if np.isfinite(g)]
                if gatings_f:
                    annotation = (
                        f"RSA TimeCost:     {rsa_tc:.3f}\n"
                        f"RHP-PINN TimeCost: {rhp_tc_mean:.3f} (mean, {len(valid)} seeds)\n"
                        f"Improvement:       {improvement_pct:+.1f}%\n"
                        f"Coupling Alpha:    {np.mean(gatings_f):.3f} +/- {np.std(gatings_f):.3f}"
                    )
            else:
                annotation = (
                    f"RSA TimeCost:     {rsa_tc:.3f}\n"
                    f"RHP-PINN:         no successful seeds\n"
                )
            ax.text(0.05, 0.95, annotation, fontsize=10, fontfamily="monospace",
                    transform=ax.transAxes, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

        ax.set_title(f"{self._config.scenario}: RSA vs RHP-PINN Path Comparison",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")

        out_path = self.plots_dir / save_name
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] saved {out_path}")
        return out_path

    # ---------------------------------------------------------------
    # Paper Plot 2: Multi-seed trajectory distribution
    #   Semi-transparent RHP paths across all seeds to show consistency
    # ---------------------------------------------------------------

    def plot_multi_seed_paths(self, save_name: str = "multi_paths.png") -> Path:
        self._build_grids()
        self._solve_rsa()
        cfg = self._config_data
        start_xy = tuple(cfg["env"]["start"])
        goal_xy = tuple(cfg["env"]["goal"])

        fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.0), constrained_layout=True)

        vmin = float(np.nanmin(self._velocity_grid))
        vmax = float(np.nanmax(self._velocity_grid))
        if vmax <= vmin + 1e-8:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(self._velocity_grid, extent=[
            self._env.bounds.x_min, self._env.bounds.x_max,
            self._env.bounds.y_min, self._env.bounds.y_max,
        ], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        ax.contour(self._xx, self._yy, self._sdf_grid, levels=[0.0],
                   colors="black", linewidths=2.0)

        if self._rsa_path is not None:
            ax.plot(self._rsa_path[:, 0], self._rsa_path[:, 1], "--",
                    color="gray", linewidth=1.5, alpha=0.5, label="RSA reference")

        n_plotted = 0
        for sd in self._seed_data:
            if not sd.get("success", False):
                continue
            ax.plot([], [], "-", color="#e74c3c", linewidth=1.0, alpha=0.3)
            n_plotted += 1

        if n_plotted > 0:
            ax.plot([], [], "-", color="#e74c3c", linewidth=2.5, alpha=0.8,
                    label=f"RHP-PINN ({n_plotted}/{len(self._seed_data)} seeds)")

        ax.plot(start_xy[0], start_xy[1], "o", color="#2ecc71", markersize=10,
                markeredgecolor="white", markeredgewidth=1.5, label="Start")
        ax.plot(goal_xy[0], goal_xy[1], "X", color="#e74c3c", markersize=12,
                markeredgecolor="white", markeredgewidth=1.5, label="Goal")
        ax.set_xlim(self._env.bounds.x_min, self._env.bounds.x_max)
        ax.set_ylim(self._env.bounds.y_min, self._env.bounds.y_max)
        ax.set_aspect("equal")
        ax.set_title(f"RHP-PINN Trajectories Across {len(self._seed_data)} Seeds",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")

        out_path = self.plots_dir / save_name
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] saved {out_path}")
        return out_path

    # ---------------------------------------------------------------
    # Paper Plot 3: TimeCost boxplot (paper key figure)
    #   RSA vs RHP-PINN boxplot to show statistical improvement
    # ---------------------------------------------------------------

    def plot_timecost_distribution(self, save_name: str = "timecost_boxplot.png") -> Path:
        if not self._seed_data:
            print("  [viz] WARNING: no seed data, skipping timecost_boxplot")
            return self.plots_dir / save_name

        rhp_tcs = [sd.get("time_cost", float("nan")) for sd in self._seed_data]
        rhp_tcs_f = [t for t in rhp_tcs if np.isfinite(t)]

        rsa_tcs = [sd.get("rsa_time_cost", float("nan")) for sd in self._seed_data]
        rsa_tcs_f = [t for t in rsa_tcs if np.isfinite(t)]

        if not rhp_tcs_f and not rsa_tcs_f:
            print("  [viz] WARNING: no valid TimeCost data, skipping timecost_boxplot")
            return self.plots_dir / save_name

        fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0), constrained_layout=True)

        data = []
        labels = []
        colors = []
        if rsa_tcs_f:
            data.append(rsa_tcs_f)
            labels.append("RSA")
            colors.append("#3498db")
        if rhp_tcs_f:
            data.append(rhp_tcs_f)
            labels.append("RHP-PINN")
            colors.append("#e74c3c")

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.4,
                        showmeans=True, meanprops=dict(marker="D", markerfacecolor="white",
                                                        markeredgecolor="black", markersize=7))

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for i, d in enumerate(data):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(d))
            ax.scatter(np.full_like(d, i + 1) + jitter, d, color="black",
                       alpha=0.5, s=30, zorder=5)

        ax.set_ylabel("TimeCost", fontsize=12)
        ax.set_title("TimeCost Distribution: RSA vs RHP-PINN",
                     fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)

        if rsa_tcs_f and rhp_tcs_f:
            mean_rsa = float(np.mean(rsa_tcs_f))
            mean_rhp = float(np.mean(rhp_tcs_f))
            improvement = (mean_rsa - mean_rhp) / mean_rsa * 100.0
            pct_better = sum(1 for t in rhp_tcs_f if t < mean_rsa) / max(len(rhp_tcs_f), 1) * 100.0
            summary = (
                f"RSA mean: {mean_rsa:.4f}\n"
                f"RHP mean: {mean_rhp:.4f}\n"
                f"Improvement: {improvement:+.1f}%\n"
                f"{pct_better:.0f}% seeds beat RSA mean"
            )
            ax.text(0.95, 0.95, summary, fontsize=10, fontfamily="monospace",
                    transform=ax.transAxes, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

        out_path = self.plots_dir / save_name
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] saved {out_path}")
        return out_path

    # ---------------------------------------------------------------
    # Paper Plot 4: Gating vs TimeCost scatter (prove gating is meaningful)
    # ---------------------------------------------------------------

    def plot_gating_vs_time(self, save_name: str = "gating_vs_tc_scatter.png") -> Path:
        if not self._seed_data:
            print("  [viz] WARNING: no seed data, skipping gating_vs_tc_scatter")
            return self.plots_dir / save_name

        gatings = [self._get_alpha(sd) for sd in self._seed_data]
        tcs = [sd.get("time_cost", float("nan")) for sd in self._seed_data]
        successes = [sd.get("success", False) for sd in self._seed_data]

        valid_pairs = [(g, t, s) for g, t, s in zip(gatings, tcs, successes)
                       if np.isfinite(g) and np.isfinite(t)]
        if not valid_pairs:
            print("  [viz] WARNING: no valid (gating, tc) pairs, skipping gating_vs_tc_scatter")
            return self.plots_dir / save_name

        g_vals = np.array([p[0] for p in valid_pairs])
        tc_vals = np.array([p[1] for p in valid_pairs])
        succ_vals = np.array([p[2] for p in valid_pairs])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.5), constrained_layout=True)

        colors = ["#2ecc71" if s else "#e74c3c" for s in succ_vals]
        ax1.scatter(g_vals, tc_vals, c=colors, s=60, alpha=0.8,
                    edgecolors="black", linewidth=0.5, zorder=5)

        if len(g_vals) >= 3:
            order = np.argsort(g_vals)
            coeffs = np.polyfit(g_vals[order], tc_vals[order], 1)
            x_fit = np.linspace(g_vals.min(), g_vals.max(), 100)
            ax1.plot(x_fit, np.polyval(coeffs, x_fit), "--", color="gray",
                     linewidth=1.5, alpha=0.7,
                     label=f"trend (slope={coeffs[0]:.3f})")

        ax1.set_xlabel("Gating Ratio (0=PINN dominant, 1=RSA dominant)", fontsize=11)
        ax1.set_ylabel("TimeCost", fontsize=11)
        ax1.set_title("Gating Ratio vs TimeCost", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2)

        success_mask = succ_vals.astype(bool)
        n_success = int(success_mask.sum())
        n_fail = int((~success_mask).sum())
        ax2.axis("off")

        g_success = g_vals[success_mask].mean() if n_success > 0 else float("nan")
        g_fail = g_vals[~success_mask].mean() if n_fail > 0 else float("nan")

        summary_lines = [
            f"Gating vs Performance Summary",
            f"{'─' * 30}",
            f"Total seeds:       {len(valid_pairs)}",
            f"Success seeds:     {n_success}",
            f"Failed seeds:      {n_fail}",
            f"",
            f"Gating (success):  {g_success:.4f}" if np.isfinite(g_success) else "Gating (success):  N/A",
            f"Gating (fail):     {g_fail:.4f}" if np.isfinite(g_fail) else "Gating (fail):     N/A",
            f"",
            f"Correlation g-TC:  {float(np.corrcoef(g_vals, tc_vals)[0, 1]):.4f}" if len(g_vals) >= 3 else "",
        ]
        ax2.text(0.15, 0.55, "\n".join(summary_lines), fontsize=11, fontfamily="monospace",
                 verticalalignment="center", transform=ax2.transAxes)

        out_path = self.plots_dir / save_name
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] saved {out_path}")
        return out_path

    # ===============================================================
    # Main dispatch
    # ===============================================================

    def plot_all(self, level: str = "basic") -> List[Path]:
        print(f"\n  [viz] Generating {level}-level plots for {self.exp_dir.name}...")
        paths = []

        paths.append(self.plot_velocity_field())
        paths.append(self.plot_comparison_metrics())
        paths.append(self.plot_gating_ratios())
        paths.append(self.plot_path_overview())
        paths.append(self.plot_best_trajectory())

        if level == "paper":
            paths.append(self.plot_path_comparison())
            paths.append(self.plot_multi_seed_paths())
            paths.append(self.plot_timecost_distribution())
            paths.append(self.plot_gating_vs_time())
            print(f"  [viz] Done -- {len(paths)} plots saved to {self.plots_dir} (paper level)")
        else:
            print(f"  [viz] Done -- {len(paths)} plots saved to {self.plots_dir} (basic level)")

        return paths
