from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from RHP_Project.envs.maze_2d import Bounds2D, Maze2DEnv
from RHP_Project.solvers.rsa_engine import RSAEngine
from RHP_Project.solvers.coupling import PhysicsGuidedCoupling, ConstantCoupling
from RHP_Project.solvers.factored_nn import FactoredTimeNN, VanillaTimeNN
from RHP_Project.solvers.pntfield_2d import PNTField2D, PNTField2DConfig
from RHP_Project.evaluator.metrics import evaluate_methods, EvalMetrics

from ..datasets.scene import Scene, PlanningResult


def _grid_to_env(
    scene: Scene,
    world_bounds: Optional[Bounds2D] = None,
    device: torch.device = torch.device("cpu"),
) -> Maze2DEnv:
    """Convert a Scene occupancy grid to a Maze2DEnv.

    Each obstacle pixel becomes a small box obstacle (0.02 x 0.02)
    at the corresponding world coordinate.
    """
    if world_bounds is None:
        world_bounds = Bounds2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)

    h, w = scene.grid.shape
    obstacles: List[Dict] = []
    obs_rows, obs_cols = np.where(scene.grid == 1)
    cell_w = (world_bounds.x_max - world_bounds.x_min) / float(max(w - 1, 1))
    cell_h = (world_bounds.y_max - world_bounds.y_min) / float(max(h - 1, 1))
    half_w = max(cell_w * 0.45, 0.005)
    half_h = max(cell_h * 0.45, 0.005)

    for r, c in zip(obs_rows, obs_cols):
        wx = world_bounds.x_min + float(c) * cell_w
        wy = world_bounds.y_min + float(r) * cell_h
        obstacles.append(dict(
            kind="box",
            center=(wx, wy),
            half_size=(half_w, half_h),
        ))

    env = Maze2DEnv(
        bounds=world_bounds,
        obstacles=obstacles,
        obstacle_inflation=0.0,
        speed_free=1.0,
        speed_obstacle=0.0,
        device=device,
    )
    return env


def _make_default_config(
    scene: Scene,
    grid_size: Tuple[int, int] = (80, 80),
) -> Dict[str, Any]:
    """Create a minimal config dict compatible with main_bench training functions."""
    start_w = scene.start_world
    goal_w = scene.goal_world
    return {
        "seed": 0,
        "device": "cpu",
        "env": {
            "name": "benchmark",
            "bounds": {"x_min": 0.0, "x_max": 1.0,
                       "y_min": 0.0, "y_max": 1.0},
            "start": [float(start_w[0]), float(start_w[1])],
            "goal": [float(goal_w[0]), float(goal_w[1])],
        },
        "rsa": {
            "grid_size": list(grid_size),
            "connectivity": 8,
            "speed_free": 1.0,
            "speed_obstacle": 0.0,
        },
        "model": {
            "hidden_dim": 128,
            "num_layers": 4,
            "activation": "tanh",
            "dist_eps": 1e-6,
        },
        "train": {
            "lr": 1e-3,
            "batch_size": 8192,
            "max_steps": 3000,
            "warmup_steps": 200,
            "lambda_phys": 1.0,
            "lambda_obs": 1.0,
            "lambda_bc": 0.0,
            "lambda_goal": 5.0,
            "convergence": {"tol": 5e-3, "patience": 30, "check_every": 5},
            "physics": {"mode": "autograd", "fd_step": 0.01},
            "curriculum": {"phys_ramp_start": 0.3, "phys_ramp_end": 0.7,
                           "phys_hold": 1.5},
            "obstacle": {"sdf_band": 0.05, "lambda_int": 100.0,
                         "lambda_grad": 10.0, "lambda_dir": 50.0,
                         "lambda_vort": 20.0, "sdf_scale": 10.0},
            "adaptive_sampling": {"base_weight": 0.2, "wavefront_power": 1.0,
                                  "obstacle_band": 0.0},
            "sdf_surface_band": 0.05,
            "sdf_surface_boost": 0.5,
            "path_anchor": {
                "lambda_path": 20.0, "lambda_distill": 50.0,
                "lambda_mono_dir": 30.0, "radius": 0.05,
                "batch_size": 512, "lambda_mono": 20.0,
                "lambda_dir": 10.0, "tube_radius": 0.1,
                "tube_samples": 256, "tube_batch_frac": 0.85,
            },
        },
        "eval": {
            "path_step": 0.01,
            "max_path_steps": 3000,
            "goal_tol": 0.10,
            "grad_probe_radius": 0.05,
            "grad_probe_points": 200,
            "strict_goal_tol": False,
            "reference_grid_size": list(grid_size),
        },
        "coupling": {
            "type": "physics",
            "w_resid": 1.0, "w_conf": 2.0,
            "w_safety": 3.0, "b": -1.0, "k": 8.0,
        },
    }


class RolloutAdapter:
    """Adapter that runs the existing RHP-PINN training + rollout.

    Wraps the training functions from main_bench.py and the evaluate_methods
    from evaluator/metrics.py without modifying them.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        grid_size: Tuple[int, int] = (80, 80),
        verbose: bool = False,
        checkpoint_dir: Optional[str] = None,
    ):
        self.device = device or torch.device("cpu")
        self.grid_size = grid_size
        self.verbose = bool(verbose)
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def _load_or_train(
        self,
        env: Maze2DEnv,
        cfg: Dict,
        rng: np.random.Generator,
        model_key: str,
        train_fn,
        *args,
        **kwargs,
    ):
        """Load checkpoint if available, otherwise train."""
        if self.checkpoint_dir:
            ckpt_path = os.path.join(self.checkpoint_dir, f"{model_key}.pth")
            if os.path.exists(ckpt_path):
                if self.verbose:
                    print(f"  [checkpoint] loading {ckpt_path}")
                model = train_fn.__wrapped__(*args, **kwargs) if hasattr(train_fn, "__wrapped__") else None
                if model is not None:
                    state = torch.load(ckpt_path, map_location=self.device)
                    model.load_state_dict(state)
                    return model, 0, 0

        model, conv_step, train_steps = train_fn(env, *args, **kwargs)

        if self.checkpoint_dir:
            ckpt_path = os.path.join(self.checkpoint_dir, f"{model_key}.pth")
            torch.save(model.state_dict(), ckpt_path)
            if self.verbose:
                print(f"  [checkpoint] saved {ckpt_path}")

        return model, conv_step, train_steps

    def train_and_evaluate(
        self,
        scene: Scene,
        world_bounds: Optional[Bounds2D] = None,
        seed: int = 0,
    ) -> Tuple[Dict[str, EvalMetrics], Dict[str, np.ndarray], Maze2DEnv]:
        """Full pipeline: build env → train all 3 models → evaluate.

        Returns:
            metrics: dict of method_name -> EvalMetrics
            paths: dict of method_name -> (N,2) path array
            env: the Maze2DEnv (for visualization)
        """
        cfg = _make_default_config(scene, grid_size=self.grid_size)
        cfg["seed"] = seed
        cfg["device"] = str(self.device)

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = _grid_to_env(scene, world_bounds=world_bounds, device=self.device)
        start_xy = scene.start_world
        goal_xy = scene.goal_world

        xs_ref, ys_ref = _make_grid(env.bounds, self.grid_size)
        speed_ref = env.speed_grid(self.grid_size)
        rsa_ref = RSAEngine(xs=xs_ref, ys=ys_ref, connectivity=8).solve(
            speed=speed_ref, start_xy=start_xy, goal_xy=goal_xy,
        )
        ref_path_xy = rsa_ref.backtrack_path_xy()

        xs, ys = _make_grid(env.bounds, self.grid_size)
        speed = env.speed_grid(self.grid_size)
        rsa_low = RSAEngine(xs=xs, ys=ys, connectivity=8).solve(
            speed=speed, start_xy=start_xy, goal_xy=goal_xy,
        )

        from RHP_Project.main_bench import (
            _train_rhp, _train_vanilla_pinn, _train_pntfield_2d,
        )

        weight = np.ones(self.grid_size, dtype=np.float32)

        vanilla, vc, vs = _train_vanilla_pinn(
            env, start_xy, goal_xy, cfg, rng, self.device,
        )
        pntfield, pc, ps = _train_pntfield_2d(
            env, start_xy, goal_xy, cfg, rng, self.device,
        )
        rhp, rc, rs = _train_rhp(
            env, start_xy, goal_xy, rsa_ref, ref_path_xy,
            xs, ys, weight, cfg, rng, self.device,
        )

        coupling_model = PhysicsGuidedCoupling(
            w_resid=1.0, w_conf=2.0, w_safety=3.0, b=-1.0, k=8.0,
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
            coupling_default_alpha=0.3,
        )

        return metrics, paths, env


def _make_grid(
    bounds: Bounds2D,
    grid_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = int(grid_size[0]), int(grid_size[1])
    xs = np.linspace(bounds.x_min, bounds.x_max, w, dtype=np.float32)
    ys = np.linspace(bounds.y_min, bounds.y_max, h, dtype=np.float32)
    return xs, ys
