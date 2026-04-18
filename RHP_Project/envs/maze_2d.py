from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class Bounds2D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def as_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [self.x_min, self.x_max, self.y_min, self.y_max],
            dtype=torch.float32,
            device=device,
        )


def _rotate_2d(xy: torch.Tensor, theta_rad: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta_rad)
    s = torch.sin(theta_rad)
    rot = torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2)
    return (rot @ xy.unsqueeze(-1)).squeeze(-1)


def sdf_box(
    xy: torch.Tensor,
    center: torch.Tensor,
    half_size: torch.Tensor,
    rotation_rad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    p = xy - center
    if rotation_rad is not None:
        p = _rotate_2d(p, -rotation_rad)
    q = torch.abs(p) - half_size
    outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.clamp(torch.max(q, dim=-1).values, max=0.0)
    return outside + inside


def sdf_circle(xy: torch.Tensor, center: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(xy - center, dim=-1) - radius


def sdf_union(sdfs: Iterable[torch.Tensor]) -> torch.Tensor:
    sdfs = list(sdfs)
    if len(sdfs) == 0:
        raise ValueError("sdf_union requires at least one sdf tensor.")
    out = sdfs[0]
    for s in sdfs[1:]:
        out = torch.minimum(out, s)
    return out


ObstacleKind = Literal["box", "circle"]


class Maze2DEnv:
    def __init__(
        self,
        bounds: Bounds2D,
        obstacles: List[Dict],
        obstacle_inflation: float = 0.0,
        speed_free: float = 1.0,
        speed_obstacle: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.bounds = bounds
        self.obstacles = obstacles
        self.obstacle_inflation = float(obstacle_inflation)
        self.speed_free = float(speed_free)
        self.speed_obstacle = float(speed_obstacle)
        self.device = torch.device(device)

    @staticmethod
    def make_u_maze(
        bounds: Bounds2D,
        wall_thickness: float = 0.12,
        inner_gap: float = 0.55,
        depth: float = 0.85,
        center: Tuple[float, float] = (0.0, 0.0),
        rotation_deg: float = 0.0,
        obstacle_inflation: float = 0.0,
        speed_free: float = 1.0,
        speed_obstacle: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> "Maze2DEnv":
        cx, cy = center
        rot = np.deg2rad(rotation_deg)

        half_w = inner_gap * 0.5 + wall_thickness
        half_h = depth * 0.5 + wall_thickness
        bottom_y = cy - depth * 0.5
        top_y = cy + depth * 0.5

        left_wall = dict(
            kind="box",
            center=(cx - inner_gap * 0.5 - wall_thickness * 0.5, cy),
            half_size=(wall_thickness * 0.5, depth * 0.5 + wall_thickness),
            rotation_rad=rot,
        )
        right_wall = dict(
            kind="box",
            center=(cx + inner_gap * 0.5 + wall_thickness * 0.5, cy),
            half_size=(wall_thickness * 0.5, depth * 0.5 + wall_thickness),
            rotation_rad=rot,
        )
        bottom_wall = dict(
            kind="box",
            center=(cx, bottom_y - wall_thickness * 0.5),
            half_size=(half_w, wall_thickness * 0.5),
            rotation_rad=rot,
        )

        return Maze2DEnv(
            bounds=bounds,
            obstacles=[left_wall, right_wall, bottom_wall],
            obstacle_inflation=obstacle_inflation,
            speed_free=speed_free,
            speed_obstacle=speed_obstacle,
            device=device,
        )

    @staticmethod
    def make_narrow_passage(
        bounds: Bounds2D,
        passage_width: float = 0.12,
        block_width: float = 0.9,
        block_height: float = 0.35,
        center: Tuple[float, float] = (0.0, 0.0),
        rotation_deg: float = 0.0,
        obstacle_inflation: float = 0.0,
        speed_free: float = 1.0,
        speed_obstacle: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> "Maze2DEnv":
        cx, cy = center
        rot = np.deg2rad(rotation_deg)

        top_block = dict(
            kind="box",
            center=(cx, cy + (passage_width * 0.5 + block_height * 0.5)),
            half_size=(block_width * 0.5, block_height * 0.5),
            rotation_rad=rot,
        )
        bottom_block = dict(
            kind="box",
            center=(cx, cy - (passage_width * 0.5 + block_height * 0.5)),
            half_size=(block_width * 0.5, block_height * 0.5),
            rotation_rad=rot,
        )

        return Maze2DEnv(
            bounds=bounds,
            obstacles=[top_block, bottom_block],
            obstacle_inflation=obstacle_inflation,
            speed_free=speed_free,
            speed_obstacle=speed_obstacle,
            device=device,
        )

    @staticmethod
    def make_obstacle_field(
        bounds: Bounds2D,
        obstacles: List[Dict],
        obstacle_inflation: float = 0.0,
        speed_free: float = 1.0,
        speed_obstacle: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> "Maze2DEnv":
        return Maze2DEnv(
            bounds=bounds,
            obstacles=list(obstacles),
            obstacle_inflation=obstacle_inflation,
            speed_free=speed_free,
            speed_obstacle=speed_obstacle,
            device=device,
        )

    @staticmethod
    def _point_box_clearance(xy: np.ndarray, obs: Dict) -> float:
        p = xy.astype(np.float32) - np.asarray(obs["center"], dtype=np.float32)
        rot = float(obs.get("rotation_rad", 0.0))
        if abs(rot) > 1e-12:
            c = float(np.cos(-rot))
            s = float(np.sin(-rot))
            p = np.asarray([c * p[0] - s * p[1], s * p[0] + c * p[1]], dtype=np.float32)
        half = np.asarray(obs["half_size"], dtype=np.float32)
        q = np.abs(p) - half
        outside = float(np.linalg.norm(np.maximum(q, 0.0)))
        inside = float(min(0.0, np.max(q)))
        return outside + inside

    @staticmethod
    def _point_obstacle_clearance(xy: np.ndarray, obs: Dict) -> float:
        kind = str(obs["kind"])
        if kind == "circle":
            center = np.asarray(obs["center"], dtype=np.float32)
            radius = float(obs["radius"])
            return float(np.linalg.norm(xy.astype(np.float32) - center) - radius)
        if kind == "box":
            return Maze2DEnv._point_box_clearance(xy, obs)
        raise ValueError(f"Unknown obstacle kind: {kind}")

    @staticmethod
    def _path_has_clearance(
        bounds: Bounds2D,
        obstacles: List[Dict],
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        margin: float,
        samples: int = 64,
    ) -> bool:
        start = np.asarray(start_xy, dtype=np.float32)
        goal = np.asarray(goal_xy, dtype=np.float32)
        ts = np.linspace(0.0, 1.0, int(max(2, samples)), dtype=np.float32)
        for t in ts:
            xy = (1.0 - t) * start + t * goal
            if not (bounds.x_min + margin < float(xy[0]) < bounds.x_max - margin):
                return False
            if not (bounds.y_min + margin < float(xy[1]) < bounds.y_max - margin):
                return False
            for obs in obstacles:
                if Maze2DEnv._point_obstacle_clearance(xy, obs) <= margin:
                    return False
        return True

    @staticmethod
    def _random_circle_obstacles(
        bounds: Bounds2D,
        rng: np.random.Generator,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        num_circles: int,
        radius_range: Tuple[float, float],
        clearance: float,
        max_tries: int = 4000,
    ) -> List[Dict]:
        obstacles: List[Dict] = []
        start = np.asarray(start_xy, dtype=np.float32)
        goal = np.asarray(goal_xy, dtype=np.float32)
        for _ in range(int(max_tries)):
            if len(obstacles) >= int(num_circles):
                break
            radius = float(rng.uniform(radius_range[0], radius_range[1]))
            center = np.asarray(
                [
                    rng.uniform(bounds.x_min + radius + clearance, bounds.x_max - radius - clearance),
                    rng.uniform(bounds.y_min + radius + clearance, bounds.y_max - radius - clearance),
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(center - start) <= radius + clearance:
                continue
            if np.linalg.norm(center - goal) <= radius + clearance:
                continue
            valid = True
            for obs in obstacles:
                other = np.asarray(obs["center"], dtype=np.float32)
                other_r = float(obs["radius"])
                if np.linalg.norm(center - other) <= radius + other_r + 0.5 * clearance:
                    valid = False
                    break
            if not valid:
                continue
            candidate = dict(kind="circle", center=(float(center[0]), float(center[1])), radius=radius)
            tmp = obstacles + [candidate]
            if not Maze2DEnv._path_has_clearance(bounds, tmp, start_xy, goal_xy, margin=clearance, samples=80):
                continue
            obstacles.append(candidate)
        return obstacles

    @staticmethod
    def make_random_circle_maze(
        bounds: Bounds2D,
        rng: np.random.Generator,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        num_circles: int = 8,
        radius_range: Tuple[float, float] = (0.06, 0.18),
        clearance: float = 0.06,
        obstacle_inflation: float = 0.0,
        speed_free: float = 1.0,
        speed_obstacle: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> "Maze2DEnv":
        obstacles = Maze2DEnv._random_circle_obstacles(
            bounds=bounds,
            rng=rng,
            start_xy=start_xy,
            goal_xy=goal_xy,
            num_circles=int(num_circles),
            radius_range=radius_range,
            clearance=float(clearance),
        )
        return Maze2DEnv.make_obstacle_field(
            bounds=bounds,
            obstacles=obstacles,
            obstacle_inflation=obstacle_inflation,
            speed_free=speed_free,
            speed_obstacle=speed_obstacle,
            device=device,
        )

    @staticmethod
    def make_rubble_field(
        bounds: Bounds2D,
        rng: np.random.Generator,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        num_clusters: int = 5,
        circles_per_cluster: Tuple[int, int] = (5, 10),
        radius_range: Tuple[float, float] = (0.025, 0.075),
        cluster_spread: float = 0.12,
        clearance: float = 0.05,
        obstacle_inflation: float = 0.0,
        speed_free: float = 1.0,
        speed_obstacle: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> "Maze2DEnv":
        obstacles: List[Dict] = []
        start = np.asarray(start_xy, dtype=np.float32)
        goal = np.asarray(goal_xy, dtype=np.float32)
        max_cluster_tries = 1000
        for _ in range(int(max_cluster_tries)):
            if len(obstacles) >= int(num_clusters) * int(circles_per_cluster[0]):
                break
            center = np.asarray(
                [
                    rng.uniform(bounds.x_min + 0.15, bounds.x_max - 0.15),
                    rng.uniform(bounds.y_min + 0.15, bounds.y_max - 0.15),
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(center - start) <= 0.22 or np.linalg.norm(center - goal) <= 0.22:
                continue
            cluster_n = int(rng.integers(int(circles_per_cluster[0]), int(circles_per_cluster[1]) + 1))
            cluster_obs: List[Dict] = []
            local_tries = 0
            while len(cluster_obs) < cluster_n and local_tries < 500:
                local_tries += 1
                radius = float(rng.uniform(radius_range[0], radius_range[1]))
                offset = rng.normal(loc=0.0, scale=float(cluster_spread), size=(2,)).astype(np.float32)
                cxy = center + offset
                if not (bounds.x_min + radius + clearance < float(cxy[0]) < bounds.x_max - radius - clearance):
                    continue
                if not (bounds.y_min + radius + clearance < float(cxy[1]) < bounds.y_max - radius - clearance):
                    continue
                if np.linalg.norm(cxy - start) <= radius + clearance:
                    continue
                if np.linalg.norm(cxy - goal) <= radius + clearance:
                    continue
                candidate = dict(kind="circle", center=(float(cxy[0]), float(cxy[1])), radius=radius)
                overlap = False
                for obs in obstacles + cluster_obs:
                    other = np.asarray(obs["center"], dtype=np.float32)
                    other_r = float(obs["radius"])
                    if np.linalg.norm(cxy - other) <= radius + other_r + 0.25 * clearance:
                        overlap = True
                        break
                if overlap:
                    continue
                cluster_obs.append(candidate)
            candidate_obs = obstacles + cluster_obs
            if len(cluster_obs) < int(circles_per_cluster[0]):
                continue
            if not Maze2DEnv._path_has_clearance(bounds, candidate_obs, start_xy, goal_xy, margin=clearance, samples=100):
                continue
            obstacles = candidate_obs
            if len(obstacles) >= int(num_clusters) * int(circles_per_cluster[0]):
                break
        return Maze2DEnv.make_obstacle_field(
            bounds=bounds,
            obstacles=obstacles,
            obstacle_inflation=obstacle_inflation,
            speed_free=speed_free,
            speed_obstacle=speed_obstacle,
            device=device,
        )

    def sdf(self, xy: torch.Tensor) -> torch.Tensor:
        xy = xy.to(device=self.device, dtype=torch.float32)
        sdfs: List[torch.Tensor] = []
        for obs in self.obstacles:
            kind: ObstacleKind = obs["kind"]
            if kind == "box":
                center = torch.tensor(obs["center"], dtype=torch.float32, device=self.device)
                half_size = torch.tensor(obs["half_size"], dtype=torch.float32, device=self.device)
                rot = obs.get("rotation_rad", None)
                rotation = (
                    torch.tensor(rot, dtype=torch.float32, device=self.device) if rot is not None else None
                )
                sdfs.append(sdf_box(xy, center=center, half_size=half_size, rotation_rad=rotation))
            elif kind == "circle":
                center = torch.tensor(obs["center"], dtype=torch.float32, device=self.device)
                radius = torch.tensor(obs["radius"], dtype=torch.float32, device=self.device)
                sdfs.append(sdf_circle(xy, center=center, radius=radius))
            else:
                raise ValueError(f"Unknown obstacle kind: {kind}")

        if len(sdfs) == 0:
            obstacle_sdf = torch.full((xy.shape[0],), float("inf"), dtype=torch.float32, device=self.device)
        else:
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

    def make_grid(self, grid_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = int(grid_size[0]), int(grid_size[1])
        xs = np.linspace(self.bounds.x_min, self.bounds.x_max, w, dtype=np.float32)
        ys = np.linspace(self.bounds.y_min, self.bounds.y_max, h, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
        with torch.no_grad():
            sdf_vals = self.sdf(torch.from_numpy(xy)).cpu().numpy().reshape(h, w)
        return xs, ys, sdf_vals

    def speed_grid(self, grid_size: Tuple[int, int]) -> np.ndarray:
        xs, ys, sdf_vals = self.make_grid(grid_size)
        free = sdf_vals > 0.0
        f = np.full_like(sdf_vals, self.speed_obstacle, dtype=np.float32)
        f[free] = self.speed_free
        return f

    def sample_uniform(
        self,
        n: int,
        rng: np.random.Generator,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        device = self.device if device is None else torch.device(device)
        x = rng.uniform(self.bounds.x_min, self.bounds.x_max, size=(n, 1)).astype(np.float32)
        y = rng.uniform(self.bounds.y_min, self.bounds.y_max, size=(n, 1)).astype(np.float32)
        xy = np.concatenate([x, y], axis=1)
        return torch.from_numpy(xy).to(device=device, dtype=torch.float32)
