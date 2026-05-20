from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from RHP_Project.solvers.rsa_engine import RSAEngine

from ..datasets.scene import PlanningResult, Scene
from .base_planner import BasePlanner


class FMMPlanner(BasePlanner):
    """Fast Marching Method planner using RSAEngine.

    Works directly on the occupancy grid by building a speed array
    (0 inside obstacles, 1 in free space).
    """

    def __init__(self, connectivity: int = 8):
        self.connectivity = int(connectivity)

    @property
    def name(self) -> str:
        return "fmm"

    def plan(self, scene: Scene) -> PlanningResult:
        grid = scene.grid
        h, w = grid.shape

        xs = np.linspace(0.0, float(w - 1), w, dtype=np.float32)
        ys = np.linspace(0.0, float(h - 1), h, dtype=np.float32)

        speed = np.where(grid == 0, 1.0, 0.0).astype(np.float32)

        start_w = (float(scene.start_px[1]), float(scene.start_px[0]))
        goal_w = (float(scene.goal_px[1]), float(scene.goal_px[0]))

        t0 = time.time()
        try:
            engine = RSAEngine(xs=xs, ys=ys, connectivity=self.connectivity)
            result = engine.solve(speed=speed, start_xy=start_w, goal_xy=goal_w)
        except (ValueError, RuntimeError) as e:
            elapsed = time.time() - t0
            return PlanningResult(
                success=False,
                path=None,
                path_length=float("inf"),
                inference_time=elapsed,
                deadlock=True,
                metadata={"fmm_error": str(e)},
            )
        elapsed = time.time() - t0

        path = result.backtrack_path_xy()
        if path.shape[0] < 2:
            return PlanningResult(
                success=False,
                path=path,
                path_length=float("inf"),
                inference_time=elapsed,
                deadlock=True,
            )

        diffs = path[1:] - path[:-1]
        path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
        n_expanded = int(np.sum(np.isfinite(result.t)))

        return PlanningResult(
            success=True,
            path=path,
            path_length=path_len,
            inference_time=elapsed,
            n_expanded=n_expanded,
            metadata={"fmm_t_min": float(np.min(result.t[np.isfinite(result.t)]))
                      if np.any(np.isfinite(result.t)) else -1.0},
        )

    def config_snapshot(self) -> dict:
        return {"planner": "fmm", "connectivity": self.connectivity,
                "method": "rsa_engine_fast_marching"}
