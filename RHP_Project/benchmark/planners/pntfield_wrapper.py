from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..datasets.scene import PlanningResult, Scene
from ..evaluation.rollout_adapter import RolloutAdapter
from .base_planner import BasePlanner


class PNTFieldPlanner(BasePlanner):
    """P-NTFields baseline planner wrapper.

    Pure neural network baseline with Fourier features.
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
        return "pntfield_2d"

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

        p_metrics = metrics_dict.get("pntfield_2d")
        p_path = paths_dict.get("pntfield_2d")

        if p_metrics is None or p_path is None:
            return PlanningResult(success=False, inference_time=total_time)

        tc = float(p_metrics.time_cost)
        if not np.isfinite(tc):
            tc = float("inf")

        return PlanningResult(
            success=bool(p_metrics.success),
            path=p_path,
            path_length=float(p_metrics.length),
            time_cost=tc,
            smoothness=float(p_metrics.smoothness),
            inference_time=total_time,
            metadata={
                "converge_steps": int(p_metrics.converge_steps) if p_metrics.converge_steps else -1,
            },
        )

    def config_snapshot(self) -> Dict[str, Any]:
        return {
            "planner": "pntfield_2d",
            "grid_size": list(self.grid_size),
            "device": self.device_str,
        }
