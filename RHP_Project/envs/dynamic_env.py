from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from .maze_2d import Bounds2D, sdf_circle, sdf_union


@dataclass(frozen=True)
class MovingCircle:
    radius: float
    center0: Tuple[float, float]
    velocity: Tuple[float, float]

    def center_at(self, t: float) -> Tuple[float, float]:
        return (self.center0[0] + self.velocity[0] * t, self.center0[1] + self.velocity[1] * t)


class DynamicObstacles2DEnv:
    def __init__(
        self,
        bounds: Bounds2D,
        circles: List[MovingCircle],
        obstacle_inflation: float = 0.0,
        speed_free: float = 1.0,
        speed_obstacle: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.bounds = bounds
        self.circles = list(circles)
        self.obstacle_inflation = float(obstacle_inflation)
        self.speed_free = float(speed_free)
        self.speed_obstacle = float(speed_obstacle)
        self.device = torch.device(device)
        self._t = 0.0

    def set_time(self, t: float) -> None:
        self._t = float(t)

    def sdf(self, xy: torch.Tensor) -> torch.Tensor:
        xy = xy.to(device=self.device, dtype=torch.float32)
        sdfs = []
        for c in self.circles:
            center = torch.tensor(c.center_at(self._t), dtype=torch.float32, device=self.device)
            radius = torch.tensor(c.radius, dtype=torch.float32, device=self.device)
            sdfs.append(sdf_circle(xy, center=center, radius=radius))
        obstacle_sdf = sdf_union(sdfs) - self.obstacle_inflation

        x_min, x_max, y_min, y_max = (
            self.bounds.x_min,
            self.bounds.x_max,
            self.bounds.y_min,
            self.bounds.y_max,
        )
        bx1 = xy[:, 0] - x_min
        bx2 = x_max - xy[:, 0]
        by1 = xy[:, 1] - y_min
        by2 = y_max - xy[:, 1]
        boundary_sdf = torch.minimum(torch.minimum(bx1, bx2), torch.minimum(by1, by2))
        return torch.minimum(obstacle_sdf, boundary_sdf)

    def is_free(self, xy: torch.Tensor) -> torch.Tensor:
        return self.sdf(xy) > 0.0

    def speed(self, xy: torch.Tensor) -> torch.Tensor:
        free = self.is_free(xy)
        f = torch.full((xy.shape[0],), self.speed_obstacle, device=self.device, dtype=torch.float32)
        f = torch.where(free, torch.tensor(self.speed_free, device=self.device), f)
        return f

    def speed_grid(self, grid_size: Tuple[int, int]) -> np.ndarray:
        h, w = int(grid_size[0]), int(grid_size[1])
        xs = np.linspace(self.bounds.x_min, self.bounds.x_max, w, dtype=np.float32)
        ys = np.linspace(self.bounds.y_min, self.bounds.y_max, h, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
        with torch.no_grad():
            sdf_vals = self.sdf(torch.from_numpy(xy)).cpu().numpy().reshape(h, w)
        free = sdf_vals > 0.0
        f = np.full_like(sdf_vals, self.speed_obstacle, dtype=np.float32)
        f[free] = self.speed_free
        return f

