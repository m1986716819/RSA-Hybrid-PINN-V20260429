from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..datasets.scene import PlanningResult, Scene
from ..evaluation.rollout_adapter import RolloutAdapter
from .base_planner import BasePlanner


class RHPPINNPlanner(BasePlanner):
    """RHP-PINN planner wrapper.

    Internally uses the existing main_bench training pipeline and
    evaluate_methods rollout. Training caches model weights when
    checkpoint_dir is provided.
    """

    def __init__(
        self,
        device: str = "cpu",
        grid_size: Tuple[int, int] = (80, 80),
        verbose: bool = False,
        checkpoint_dir: Optional[str] = None,
    ):
        self.device_str = device
        self.grid_size = grid_size
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self._adapter: Optional[RolloutAdapter] = None
        self._trained = False

    @property
    def name(self) -> str:
        return "rhp_pinn"

    def requires_training(self) -> bool:
        return True

    def train(self, scenes) -> None:
        pass

    def plan(self, scene: Scene) -> PlanningResult:
        import torch
        device = torch.device(self.device_str)

        adapter = RolloutAdapter(
            device=device,
            grid_size=self.grid_size,
            verbose=self.verbose,
            checkpoint_dir=self.checkpoint_dir,
        )

        t0 = time.time()
        metrics_dict, paths_dict, env = adapter.train_and_evaluate(scene, seed=0)
        total_time = time.time() - t0

        rhp_metrics = metrics_dict.get("rhp_pinn")
        rhp_path = paths_dict.get("rhp_pinn")

        if rhp_metrics is None or rhp_path is None:
            return PlanningResult(success=False, inference_time=total_time)

        velocity_field_np_fn = env.velocity_field_np if env.has_velocity_field() else None

        tc = float(rhp_metrics.time_cost)
        if not np.isfinite(tc):
            tc = float("inf")

        path_pixel = None
        if rhp_path is not None and rhp_path.shape[0] >= 2:
            h, w = scene.grid.shape
            path_pixel = np.zeros_like(rhp_path)
            path_pixel[:, 0] = rhp_path[:, 1] * float(h - 1)
            path_pixel[:, 1] = rhp_path[:, 0] * float(w - 1)

        return PlanningResult(
            success=bool(rhp_metrics.success),
            path=rhp_path,
            path_length=float(rhp_metrics.length),
            time_cost=tc,
            smoothness=float(rhp_metrics.smoothness),
            inference_time=total_time,
            coupling_alpha=float(rhp_metrics.coupling_alpha)
            if rhp_metrics.coupling_alpha is not None else None,
            metadata={
                "converge_steps": int(rhp_metrics.converge_steps) if rhp_metrics.converge_steps else -1,
                "eikonal_residual": float(rhp_metrics.physical_consistency),
            },
        )

    def config_snapshot(self) -> Dict[str, Any]:
        return {
            "planner": "rhp_pinn",
            "grid_size": list(self.grid_size),
            "device": self.device_str,
        }
