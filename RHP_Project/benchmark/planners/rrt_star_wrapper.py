"""RRT* planner wrapper using the existing rrt_star_plan implementation.

Converts Scene occupancy grid to continuous-space API:
  - is_free(point): checks occupancy via bilinear interpolation on grid
  - bounds: [0, 1]^2 world coordinates
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from RHP_Project.solvers.rrt_star import rrt_star_plan

from ..datasets.scene import PlanningResult, Scene
from .base_planner import BasePlanner


def _make_is_free(grid: np.ndarray) -> callable:
    """Create is_free function from occupancy grid.

    Expects world coordinates in [0,1]^2, converts to pixel and checks.
    """
    h, w = grid.shape

    def is_free(p: np.ndarray) -> bool:
        x, y = float(p[0]), float(p[1])
        col = int(round(x * float(w - 1)))
        row = int(round(y * float(h - 1)))
        if col < 0 or col >= w or row < 0 or row >= h:
            return False
        return bool(grid[row, col] == 0)

    return is_free


class RRTStarPlanner(BasePlanner):
    """RRT* planner wrapping rrt_star_plan.

    Operates in [0,1]^2 world coordinates derived from the scene grid.
    """

    def __init__(
        self,
        max_iterations: int = 5000,
        step_size: float = 0.04,
        goal_tol: float = 0.05,
        goal_bias: float = 0.08,
        seed: int = 0,
    ):
        self.max_iterations = int(max_iterations)
        self.step_size = float(step_size)
        self.goal_tol = float(goal_tol)
        self.goal_bias = float(goal_bias)
        self.seed = int(seed)

    @property
    def name(self) -> str:
        return "rrt_star"

    @staticmethod
    def _world_to_pixel_len(path_w: np.ndarray, h: int, w: int) -> float:
        """Convert world-coordinate path length to pixel-coordinate length."""
        if path_w.shape[0] < 2:
            return float("inf")
        path_px = np.zeros_like(path_w)
        path_px[:, 0] = path_w[:, 1] * float(h - 1)
        path_px[:, 1] = path_w[:, 0] * float(w - 1)
        diffs = path_px[1:] - path_px[:-1]
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def plan(self, scene: Scene) -> PlanningResult:
        h, w = scene.grid.shape

        start_w = scene.start_world
        goal_w = scene.goal_world
        is_free = _make_is_free(scene.grid)

        t0 = time.time()
        try:
            result = rrt_star_plan(
                start_xy=start_w,
                goal_xy=goal_w,
                bounds=(0.0, 1.0, 0.0, 1.0),
                is_free=is_free,
                velocity_fn=None,
                max_iterations=self.max_iterations,
                step_size=self.step_size,
                goal_tol=self.goal_tol,
                goal_bias=self.goal_bias,
                seed=self.seed,
            )
        except Exception as e:
            elapsed = time.time() - t0
            return PlanningResult(
                success=False,
                path=None,
                path_length=float("inf"),
                inference_time=elapsed,
                deadlock=True,
                metadata={"rrt_star_error": str(e)},
            )
        elapsed = time.time() - t0

        if not result.success or result.path.shape[0] < 2:
            return PlanningResult(
                success=False,
                path=result.path if result.path.shape[0] >= 2 else None,
                path_length=self._world_to_pixel_len(result.path, h, w) if result.path.shape[0] >= 2 else float("inf"),
                inference_time=elapsed,
                deadlock=not result.success,
                metadata={"rrt_nodes": result.nodes_explored,
                          "rrt_iters": result.iterations},
            )

        pixel_len = self._world_to_pixel_len(result.path, h, w)
        return PlanningResult(
            success=True,
            path=result.path,
            path_length=pixel_len,
            time_cost=result.time_cost,
            inference_time=elapsed,
            metadata={"rrt_nodes": result.nodes_explored,
                      "rrt_iters": result.iterations},
        )

    def config_snapshot(self) -> dict:
        return {
            "planner": "rrt_star",
            "max_iterations": self.max_iterations,
            "step_size": self.step_size,
            "goal_tol": self.goal_tol,
            "goal_bias": self.goal_bias,
        }