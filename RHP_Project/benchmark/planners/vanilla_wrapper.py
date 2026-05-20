from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..datasets.scene import PlanningResult, Scene
from ..evaluation.rollout_adapter import RolloutAdapter
from .base_planner import BasePlanner


class VanillaPINNPlanner(BasePlanner):
    """Vanilla PINN planner wrapper.

    Pure neural network baseline (no RSA guidance).
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

    @property
    def name(self) -> str:
        return "vanilla_pinn"

    def requires_training(self) -> bool:
        return True

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

        v_metrics = metrics_dict.get("vanilla_pinn")
        v_path = paths_dict.get("vanilla_pinn")

        if v_metrics is None or v_path is None:
            return PlanningResult(success=False, inference_time=total_time)

        tc = float(v_metrics.time_cost)
        if not np.isfinite(tc):
            tc = float("inf")

        return PlanningResult(
            success=bool(v_metrics.success),
            path=v_path,
            path_length=float(v_metrics.length),
            time_cost=tc,
            smoothness=float(v_metrics.smoothness),
            inference_time=total_time,
            metadata={
                "converge_steps": int(v_metrics.converge_steps) if v_metrics.converge_steps else -1,
            },
        )

    def config_snapshot(self) -> Dict[str, Any]:
        return {
            "planner": "vanilla_pinn",
            "grid_size": list(self.grid_size),
            "device": self.device_str,
        }
