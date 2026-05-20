from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PlanningResult:
    """Standardised result returned by every planner."""

    success: bool
    path: Optional[np.ndarray] = None
    path_length: float = float("inf")
    collision: bool = False
    inference_time: float = 0.0
    time_cost: float = float("inf")
    smoothness: float = float("nan")
    n_expanded: int = 0
    deadlock: bool = False
    timeout: bool = False
    coupling_alpha: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        if not self.success or self.path is None:
            return False
        if self.path.shape[0] < 2:
            return False
        return True

    def summary(self) -> Dict[str, Any]:
        return {
            "success": int(self.success),
            "path_length": float(self.path_length) if np.isfinite(self.path_length) else -1.0,
            "time_cost": float(self.time_cost) if np.isfinite(self.time_cost) else -1.0,
            "collision": int(self.collision),
            "deadlock": int(self.deadlock),
            "timeout": int(self.timeout),
            "inference_time_ms": float(self.inference_time * 1000.0),
            "smoothness": float(self.smoothness) if np.isfinite(self.smoothness) else -1.0,
            "coupling_alpha": float(self.coupling_alpha) if self.coupling_alpha is not None else -1.0,
        }


@dataclass
class Scene:
    """A single planning scene with occupancy grid and start/goal.

    Coordinates are in pixel space (row, col).
    Grid: 0 = free, 1 = obstacle.
    """

    grid: np.ndarray
    start_px: Tuple[int, int]
    goal_px: Tuple[int, int]
    optimal_length: Optional[float] = None
    name: str = ""
    map_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._world_scale: Optional[float] = None

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    def set_world_scale(self, scale: float) -> None:
        self._world_scale = float(scale)

    def pixel_to_world(self, px: Tuple[int, int]) -> Tuple[float, float]:
        h, w = self.grid.shape
        return (float(px[1]) / float(max(w - 1, 1)),
                float(px[0]) / float(max(h - 1, 1)))

    def world_to_pixel(self, wx: float, wy: float) -> Tuple[int, int]:
        h, w = self.grid.shape
        return (int(round(wy * float(h - 1))),
                int(round(wx * float(w - 1))))

    @property
    def start_world(self) -> Tuple[float, float]:
        return self.pixel_to_world(self.start_px)

    @property
    def goal_world(self) -> Tuple[float, float]:
        return self.pixel_to_world(self.goal_px)

    def sample_free_pixel(self, rng: np.random.Generator) -> Tuple[int, int]:
        free = np.argwhere(self.grid == 0)
        if free.shape[0] == 0:
            raise RuntimeError("No free cells in grid")
        idx = int(rng.integers(0, free.shape[0]))
        return (int(free[idx, 0]), int(free[idx, 1]))

    @staticmethod
    def from_obstacle_list(
        height: int,
        width: int,
        obstacle_cells: List[Tuple[int, int]],
        start_px: Tuple[int, int],
        goal_px: Tuple[int, int],
        name: str = "",
    ) -> Scene:
        grid = np.zeros((height, width), dtype=np.uint8)
        for r, c in obstacle_cells:
            if 0 <= r < height and 0 <= c < width:
                grid[r, c] = 1
        return Scene(grid=grid, start_px=start_px, goal_px=goal_px, name=name)
