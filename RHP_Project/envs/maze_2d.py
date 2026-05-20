from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

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
        self._velocity_field_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

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
    def make_trap_u_shape(
        bounds: Bounds2D,
        wall_thickness: float = 0.03,
        left_x: float = 0.35,
        u_depth: float = 0.28,
        u_height: float = 0.55,
        center_y: float = 0.5,
        obstacle_inflation: float = 0.0,
        speed_free: float = 1.0,
        speed_obstacle: float = 0.01,
        device: torch.device | str = "cpu",
    ) -> "Maze2DEnv":
        return Maze2DEnv._build_u_trap(
            bounds=bounds,
            wall_thickness=wall_thickness,
            left_x=left_x,
            u_depth=u_depth,
            u_height=u_height,
            center_y=center_y,
            obstacle_inflation=obstacle_inflation,
            speed_free=speed_free,
            speed_obstacle=speed_obstacle,
            device=device,
        )

    @staticmethod
    def _build_u_trap(
        bounds: Bounds2D,
        wall_thickness: float,
        left_x: float,
        u_depth: float,
        u_height: float,
        center_y: float,
        obstacle_inflation: float = 0.0,
        speed_free: float = 1.0,
        speed_obstacle: float = 0.01,
        device: torch.device | str = "cpu",
    ) -> "Maze2DEnv":
        lx = float(left_x)
        hw = float(wall_thickness) * 0.5
        hh = float(u_height) * 0.5
        cy = float(center_y)
        top_y = cy - hh
        bot_y = cy + hh
        arm_center_x = lx + float(u_depth) * 0.5
        arm_hw = float(u_depth) * 0.5 + hw

        back_wall = dict(kind="box", center=(lx, cy), half_size=(hw, hh))
        top_arm = dict(kind="box", center=(arm_center_x, top_y), half_size=(arm_hw, hw))
        bottom_arm = dict(kind="box", center=(arm_center_x, bot_y), half_size=(arm_hw, hw))

        return Maze2DEnv(
            bounds=bounds,
            obstacles=[back_wall, top_arm, bottom_arm],
            obstacle_inflation=obstacle_inflation,
            speed_free=speed_free,
            speed_obstacle=speed_obstacle,
            device=device,
        )

    @staticmethod
    def make_trap_heterogeneous_vf(
        bounds: Bounds2D,
        wall_thickness: float = 0.03,
        left_x: float = 0.35,
        u_depth: float = 0.28,
        u_height: float = 0.55,
        center_y: float = 0.5,
        v_slow: float = 0.3,
        v_fast: float = 1.0,
        sigmoid_beta: float = 8.0,
        obstacle_inflation: float = 0.0,
        speed_free: float = 1.0,
        speed_obstacle: float = 0.01,
        device: torch.device | str = "cpu",
    ) -> "Maze2DEnv":
        env = Maze2DEnv._build_u_trap(
            bounds=bounds,
            wall_thickness=wall_thickness,
            left_x=left_x,
            u_depth=u_depth,
            u_height=u_height,
            center_y=center_y,
            obstacle_inflation=obstacle_inflation,
            speed_free=speed_free,
            speed_obstacle=speed_obstacle,
            device=device,
        )
        lx = float(left_x)
        hw = float(wall_thickness) * 0.5
        ud = float(u_depth)
        hh = float(u_height) * 0.5
        cy = float(center_y)
        interior_cx = lx + ud * 0.5
        interior_cy = cy
        interior_hx = ud * 0.5
        interior_hy = hh - hw
        beta = float(sigmoid_beta)

        def _hetero_field(xy: torch.Tensor) -> torch.Tensor:
            p = xy - torch.tensor([interior_cx, interior_cy], dtype=xy.dtype, device=xy.device).view(1, 2)
            q = torch.abs(p) - torch.tensor([interior_hx, interior_hy], dtype=xy.dtype, device=xy.device).view(1, 2)
            outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
            inside = torch.clamp(torch.max(q, dim=-1).values, max=0.0)
            sdf_interior = outside + inside
            alpha = torch.sigmoid(-sdf_interior * beta)
            return float(v_slow) * alpha + float(v_fast) * (1.0 - alpha)

        env.set_velocity_field(_hetero_field)
        return env

    @staticmethod
    def make_cost_pit(
        bounds: Bounds2D,
        pit_center: Tuple[float, float] = (0.5, 0.5),
        pit_radius: float = 0.35,
        v_slow: float = 0.05,
        v_fast: float = 1.0,
        sigmoid_beta: float = 10.0,
        chokepoint_width: float = 0.10,
        chokepoint_x: float = 0.38,
        chokepoint_y_low: float = 0.28,
        chokepoint_y_high: float = 0.72,
        obstacle_inflation: float = 0.02,
        speed_free: float = 1.0,
        speed_obstacle: float = 0.01,
        device: torch.device | str = "cpu",
    ) -> "Maze2DEnv":
        """Cost Pit with a Gaussian slow zone and a chokepoint corridor.

        The velocity field has a steep Gaussian dip (v_min=0.05) at pit_center,
        creating a "cost crater" that RSA's straight-line path will blindly
        plunge through.  A narrow chokepoint formed by two box obstacles on
        the left forces paths to pass close to obstacles, testing the coupling's
        ability to balance safety and speed awareness.

        Start: (0.1, 0.1), Goal: (0.9, 0.9)
        """
        cpx = float(chokepoint_x)
        cp_hw = 0.5 * float(chokepoint_width)
        cp_top_y = float(chokepoint_y_high)
        cp_bot_y = float(chokepoint_y_low)
        block_h = (cp_top_y - cp_bot_y - 2 * cp_hw) * 0.5
        block_w = 0.03

        top_block = dict(
            kind="box",
            center=(cpx, cp_top_y - block_h * 0.5),
            half_size=(block_w, block_h),
        )
        bot_block = dict(
            kind="box",
            center=(cpx, cp_bot_y + block_h * 0.5),
            half_size=(block_w, block_h),
        )

        env = Maze2DEnv(
            bounds=bounds,
            obstacles=[top_block, bot_block],
            obstacle_inflation=obstacle_inflation,
            speed_free=speed_free,
            speed_obstacle=speed_obstacle,
            device=device,
        )

        pc = torch.tensor([pit_center[0], pit_center[1]], dtype=torch.float32)
        pr = float(pit_radius)
        vs = float(v_slow)
        vf = float(v_fast)
        beta = float(sigmoid_beta)

        def _cost_pit_field(xy: torch.Tensor) -> torch.Tensor:
            center = pc.to(device=xy.device, dtype=xy.dtype).view(1, 2)
            dist = torch.sqrt(torch.sum((xy - center) ** 2, dim=-1) + 1e-12)
            # Smooth radial transition: slow near center, fast at periphery
            alpha = torch.sigmoid((dist - pr * 0.5) * beta / max(pr, 1e-6))
            return vs + (vf - vs) * alpha

        env.set_velocity_field(_cost_pit_field)
        return env

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
    def make_open_space(
        bounds: Bounds2D,
        obstacle_inflation: float = 0.0,
        speed_free: float = 1.0,
        speed_obstacle: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> "Maze2DEnv":
        return Maze2DEnv(
            bounds=bounds,
            obstacles=[],
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
        if kind == "polygon":
            verts = np.asarray(obs["vertices"], dtype=np.float32)
            # Minimum distance to polygon boundary
            n = len(verts)
            min_d = float("inf")
            for i in range(n):
                a = verts[i]
                b = verts[(i + 1) % n]
                ab = b - a
                t = np.dot(xy - a, ab) / max(float(np.dot(ab, ab)), 1e-12)
                t = max(0.0, min(1.0, t))
                closest = a + t * ab
                d = float(np.linalg.norm(xy.astype(np.float32) - closest))
                if d < min_d:
                    min_d = d
            # Inside check (negative distance if inside)
            inside = True
            for i in range(n):
                a = verts[i]
                b = verts[(i + 1) % n]
                edge_normal = np.array([-(b[1]-a[1]), b[0]-a[0]], dtype=np.float32)
                dot = float(np.dot(xy - a, edge_normal))
                if dot > 0:
                    inside = False
                    break
            return -min_d if inside else min_d
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
            elif kind == "polygon":
                verts = torch.tensor(obs["vertices"], dtype=torch.float32, device=self.device)
                from .random_generator import sdf_polygon
                sdfs.append(sdf_polygon(xy, verts))
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

    def sample_near_obstacles(
        self,
        n: int,
        sdf_band: float = 0.08,
        rng: Optional[np.random.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if rng is None:
            rng = np.random.default_rng(0)
        dev = device or self.device
        candidates = []
        max_tries = int(n * 40)
        for _ in range(max_tries):
            if len(candidates) >= n:
                break
            x = float(rng.uniform(self.bounds.x_min, self.bounds.x_max))
            y = float(rng.uniform(self.bounds.y_min, self.bounds.y_max))
            xy = torch.tensor([[x, y]], dtype=torch.float32, device=dev)
            sdf_val = float(self.sdf(xy).detach().cpu().item())
            if 0.0 <= sdf_val <= float(sdf_band):
                candidates.append(xy)
        if len(candidates) == 0:
            return torch.empty((0, 2), dtype=torch.float32, device=dev)
        return torch.cat(candidates, dim=0)

    def is_free(self, xy: torch.Tensor) -> torch.Tensor:
        return self.sdf(xy) > 0.0

    def speed(self, xy: torch.Tensor) -> torch.Tensor:
        free = self.is_free(xy)
        f = torch.full((xy.shape[0],), self.speed_obstacle, device=xy.device, dtype=torch.float32)
        f = torch.where(free, torch.tensor(self.speed_free, device=xy.device), f)
        return f

    def set_velocity_field(self, fn: Optional[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self._velocity_field_fn = fn

    def has_velocity_field(self) -> bool:
        return self._velocity_field_fn is not None

    def velocity_field_at(self, xy: torch.Tensor) -> Optional[torch.Tensor]:
        if self._velocity_field_fn is None:
            return None
        return self._velocity_field_fn(xy)

    def wave_speed(self, xy: torch.Tensor) -> torch.Tensor:
        free = self.is_free(xy)
        if self._velocity_field_fn is None:
            f = torch.full((xy.shape[0],), self.speed_obstacle, device=xy.device, dtype=torch.float32)
            f = torch.where(free, torch.tensor(self.speed_free, device=xy.device), f)
            return f
        v = self._velocity_field_fn(xy)
        f = torch.full((xy.shape[0],), self.speed_obstacle, device=v.device, dtype=torch.float32)
        f = torch.where(free.to(device=v.device), v, f)
        return f

    def velocity_field_np(self, xy: np.ndarray) -> float:
        if self._velocity_field_fn is None:
            return float(self.speed_free)
        xy_t = torch.from_numpy(xy.astype(np.float32).reshape(1, 2)).to(device=self.device)
        with torch.no_grad():
            v = self._velocity_field_fn(xy_t)
        return float(v.detach().cpu().numpy()[0])

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

    def velocity_grid(self, grid_size: Tuple[int, int]) -> Optional[np.ndarray]:
        if self._velocity_field_fn is None:
            return None
        xs, ys, sdf_vals = self.make_grid(grid_size)
        free = sdf_vals > 0.0
        xx, yy = np.meshgrid(xs, ys)
        xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)
        with torch.no_grad():
            v = self._velocity_field_fn(torch.from_numpy(xy).to(device=self.device))
        v_grid = v.detach().cpu().numpy().reshape(sdf_vals.shape)
        v_grid = np.where(free, v_grid, 0.0)
        return v_grid.astype(np.float32)

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


# ===============================================================
# Perlin noise velocity field
# ===============================================================


def perlin_noise_field(
    grid_size: Tuple[int, int],
    scale: float = 3.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate a smooth 2D Perlin-like noise field in [-1, 1].

    Uses interpolated random gradient dot products at lattice points.
    Independent of PyTorch — returns a plain numpy array.

    Args:
        grid_size: (H, W) size of the output noise grid
        scale: spatial frequency (higher = more detail)
        seed: random seed for reproducibility
    Returns:
        noise: (H, W) array in [-1, 1]
    """
    rng = np.random.default_rng(seed)
    h, w = int(grid_size[0]), int(grid_size[1])
    angles = rng.uniform(0, 2 * np.pi, size=(h + 1, w + 1))
    grad_x = np.cos(angles)
    grad_y = np.sin(angles)

    xs = np.linspace(0, scale, w, endpoint=False)
    ys = np.linspace(0, scale, h, endpoint=False)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    xi = np.floor(xx).astype(np.int64) % (w + 1)
    yi = np.floor(yy).astype(np.int64) % (h + 1)
    fx = xx - np.floor(xx)
    fy = yy - np.floor(yy)

    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6 - 15) + 10)

    u = _fade(fx)
    v = _fade(fy)

    n00 = grad_x[yi, xi] * fx + grad_y[yi, xi] * fy
    n10 = grad_x[yi, (xi + 1) % (w + 1)] * (fx - 1) + grad_y[yi, (xi + 1) % (w + 1)] * fy
    n01 = grad_x[(yi + 1) % (h + 1), xi] * fx + grad_y[(yi + 1) % (h + 1), xi] * (fy - 1)
    n11 = grad_x[(yi + 1) % (h + 1), (xi + 1) % (w + 1)] * (fx - 1) + grad_y[(yi + 1) % (h + 1), (xi + 1) % (w + 1)] * (fy - 1)

    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u
    noise = nx0 * (1 - v) + nx1 * v
    return np.clip(noise / (np.sqrt(2.0) + 1e-12), -1.0, 1.0).T


def make_perlin_velocity_field(
    bounds: Bounds2D,
    grid_size: Tuple[int, int],
    v_min: float = 0.3,
    v_max: float = 1.0,
    noise_scale: float = 3.0,
    seed: int = 0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a smooth, continuous velocity field from Perlin noise.

    The field covers [v_min, v_max] and is evaluated via bilinear
    interpolation of a pre-computed noise grid.

    Args:
        bounds: environment bounds
        grid_size: internal noise grid resolution (H, W)
        v_min: minimum velocity
        v_max: maximum velocity
        noise_scale: spatial frequency of the noise
        seed: random seed

    Returns:
        callable(xy) -> Tensor of per-point speeds
    """
    noise = perlin_noise_field(grid_size, scale=noise_scale, seed=seed)
    h_grid, w_grid = noise.shape
    xs = np.linspace(bounds.x_min, bounds.x_max, w_grid, dtype=np.float64)
    ys = np.linspace(bounds.y_min, bounds.y_max, h_grid, dtype=np.float64)
    noise_norm = 0.5 * (noise + 1.0)  # map [-1, 1] -> [0, 1]
    v_grid_np = float(v_min) + noise_norm * (float(v_max) - float(v_min))
    v_grid = torch.from_numpy(v_grid_np.astype(np.float32))

    def _velocity_fn(xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0].detach().cpu().to(dtype=torch.float64)
        y = xy[:, 1].detach().cpu().to(dtype=torch.float64)
        x_pts = torch.from_numpy(xs)
        y_pts = torch.from_numpy(ys)
        j = torch.searchsorted(x_pts, x) - 1
        i = torch.searchsorted(y_pts, y) - 1
        j = torch.clamp(j, 0, w_grid - 2)
        i = torch.clamp(i, 0, h_grid - 2)
        tx = ((x - xs[j]) / (xs[j + 1] - xs[j] + 1e-12)).float()
        ty = ((y - ys[i]) / (ys[i + 1] - ys[i] + 1e-12)).float()
        v00 = v_grid[i, j]
        v10 = v_grid[i, j + 1]
        v01 = v_grid[i + 1, j]
        v11 = v_grid[i + 1, j + 1]
        v0 = v00 * (1 - tx) + v10 * tx
        v1 = v01 * (1 - tx) + v11 * tx
        result = v0 * (1 - ty) + v1 * ty
        return result.to(device=xy.device)

    return _velocity_fn


# ===============================================================
# Random obstacle generation helpers
# ===============================================================


def _obstacles_overlap(
    obstacles: List[Dict],
    new_center: np.ndarray,
    new_size: np.ndarray,
    kind: str,
    clearance: float,
) -> bool:
    """Check if a new obstacle overlaps with existing ones.

    Handles box-box, circle-circle, and box-circle pairs using
    approximate clearance checks.

    Returns True if overlap detected.
    """
    if kind == "circle":
        new_r = float(new_size)
        for obs in obstacles:
            other = np.asarray(obs["center"], dtype=np.float32)
            if obs["kind"] == "circle":
                other_r = float(obs["radius"])
                if np.linalg.norm(new_center - other) <= new_r + other_r + 0.5 * clearance:
                    return True
            elif obs["kind"] == "box":
                oh = np.asarray(obs["half_size"], dtype=np.float32)
                dx = max(0.0, abs(new_center[0] - other[0]) - oh[0])
                dy = max(0.0, abs(new_center[1] - other[1]) - oh[1])
                if np.sqrt(dx**2 + dy**2) < new_r + 0.5 * clearance:
                    return True
    elif kind == "box":
        nh = np.asarray(new_size, dtype=np.float32)
        for obs in obstacles:
            other = np.asarray(obs["center"], dtype=np.float32)
            if obs["kind"] == "box":
                oh = np.asarray(obs["half_size"], dtype=np.float32)
                dx = max(0.0, abs(new_center[0] - other[0]) - nh[0] - oh[0])
                dy = max(0.0, abs(new_center[1] - other[1]) - nh[1] - oh[1])
                if dx < 0.5 * clearance and dy < 0.5 * clearance:
                    return True
            elif obs["kind"] == "circle":
                other_r = float(obs["radius"])
                dx = max(0.0, abs(new_center[0] - other[0]) - nh[0])
                dy = max(0.0, abs(new_center[1] - other[1]) - nh[1])
                if np.sqrt(dx**2 + dy**2) < other_r + 0.5 * clearance:
                    return True
    return False


def make_random_maze(
    bounds: Bounds2D,
    rng: np.random.Generator,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    num_obstacles: int = 10,
    box_ratio: float = 0.5,
    size_range: Tuple[float, float] = (0.05, 0.18),
    clearance: float = 0.06,
    speed_free: float = 1.0,
    speed_obstacle: float = 0.01,
    obstacle_inflation: float = 0.0,
    device: torch.device | str = "cpu",
) -> Maze2DEnv:
    """Generate a random maze with mixed box and circle obstacles.

    Obstacles are placed via rejection sampling:
      - start and goal regions are kept clear
      - a feasible start-to-goal path is guaranteed
      - obstacles do not overlap with each other

    Args:
        bounds: environment bounds
        rng: numpy random generator
        start_xy: start position
        goal_xy: goal position
        num_obstacles: target obstacle count
        box_ratio: fraction of box obstacles (rest are circles)
        size_range: (min, max) for obstacle size
        clearance: min distance between obstacles and from path
        speed_free: wave speed in free space
        speed_obstacle: wave speed inside obstacles
        obstacle_inflation: SDF inflation for obstacle boundaries
        device: torch device

    Returns:
        Maze2DEnv with randomly placed obstacles
    """
    obstacles: List[Dict] = []
    start = np.asarray(start_xy, dtype=np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)
    size_min, size_max = float(size_range[0]), float(size_range[1])
    max_tries = int(max(2000, num_obstacles * 100))

    for _ in range(max_tries):
        if len(obstacles) >= int(num_obstacles):
            break

        is_box = rng.uniform() < float(box_ratio)
        if is_box:
            half_w = rng.uniform(size_min, size_max)
            half_h = rng.uniform(size_min, size_max)
            cx = rng.uniform(bounds.x_min + half_w + clearance, bounds.x_max - half_w - clearance)
            cy = rng.uniform(bounds.y_min + half_h + clearance, bounds.y_max - half_h - clearance)
            center = np.array([cx, cy], dtype=np.float32)
            candidate = dict(kind="box", center=(float(cx), float(cy)), half_size=(float(half_w), float(half_h)))
            size_vec = np.array([half_w, half_h], dtype=np.float32)
            effective_radius = float(np.linalg.norm(size_vec))
        else:
            radius = rng.uniform(size_min, size_max)
            cx = rng.uniform(bounds.x_min + radius + clearance, bounds.x_max - radius - clearance)
            cy = rng.uniform(bounds.y_min + radius + clearance, bounds.y_max - radius - clearance)
            center = np.array([cx, cy], dtype=np.float32)
            candidate = dict(kind="circle", center=(float(cx), float(cy)), radius=float(radius))
            size_vec = np.array([radius], dtype=np.float32)
            effective_radius = float(radius)

        # Keep start and goal clear
        if np.linalg.norm(center - start) <= effective_radius + clearance:
            continue
        if np.linalg.norm(center - goal) <= effective_radius + clearance:
            continue
        # Check overlap with existing obstacles
        if _obstacles_overlap(obstacles, center, size_vec, candidate["kind"], clearance):
            continue
        # Verify path feasibility
        tmp = obstacles + [candidate]
        if not Maze2DEnv._path_has_clearance(bounds, tmp, start_xy, goal_xy, margin=clearance, samples=80):
            continue

        obstacles.append(candidate)

    return Maze2DEnv.make_obstacle_field(
        bounds=bounds,
        obstacles=obstacles,
        obstacle_inflation=obstacle_inflation,
        speed_free=speed_free,
        speed_obstacle=speed_obstacle,
        device=device,
    )
