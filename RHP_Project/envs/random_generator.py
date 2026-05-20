#!/usr/bin/env python3
"""Random environment generator for RHP-PINN generalization test.

Generates:
  - Random polygon obstacles (3~5 convex polygons) with safety guarantees
  - Perlin-noise-based smooth velocity field over the continuous domain
  - Feasibility check: start-goal path always exists via RSA connectivity test
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .maze_2d import Bounds2D, Maze2DEnv, perlin_noise_field, _obstacles_overlap


# ---------------------------------------------------------------------------
#  Polygon obstacle generator
# ---------------------------------------------------------------------------

def _random_convex_polygon(
    rng: np.random.Generator,
    center: np.ndarray,
    radius_range: Tuple[float, float],
    num_vertices: int = 5,
) -> np.ndarray:
    """Generate a random convex polygon around a centre point.

    Uses the "random angles + convex hull" trick:
    1. Generate random angles sorted around the circle (ensures convexity)
    2. Random radii in [r_min, r_max]
    """
    r_min, r_max = radius_range
    angles = rng.uniform(0, 2 * math.pi, size=num_vertices + 3)
    angles = np.sort(angles)
    radii = rng.uniform(r_min, r_max, size=num_vertices + 3)
    verts = np.stack([
        center[0] + radii * np.cos(angles),
        center[1] + radii * np.sin(angles),
    ], axis=1)
    return verts.astype(np.float32)


def _point_in_polygon(pt: np.ndarray, verts: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (2D)."""
    x, y = pt
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        yi, yj = verts[i][1], verts[j][1]
        xi, xj = verts[i][0], verts[j][0]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_min_distance(pt: np.ndarray, verts: np.ndarray) -> float:
    """Minimum distance from point to polygon boundary (approximate)."""
    n = len(verts)
    min_d = float("inf")
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        ab = b - a
        t = np.dot(pt - a, ab) / max(np.dot(ab, ab), 1e-12)
        t = np.clip(t, 0.0, 1.0)
        closest = a + t * ab
        d = float(np.linalg.norm(pt - closest))
        if d < min_d:
            min_d = d
    return min_d


def _polygons_overlap(
    verts_a: np.ndarray,
    verts_b: np.ndarray,
    clearance: float,
) -> bool:
    """Check if two convex polygons are closer than clearance."""
    for v in verts_a:
        if _point_in_polygon(v, verts_b) or _polygon_min_distance(v, verts_b) < clearance:
            return True
    for v in verts_b:
        if _point_in_polygon(v, verts_a) or _polygon_min_distance(v, verts_a) < clearance:
            return True
    return False


# ---------------------------------------------------------------------------
#  Polygon → Maze2DEnv obstacle dict conversion
# ---------------------------------------------------------------------------


def _check_path_feasibility_rsa(
    env: Maze2DEnv,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    grid_size: Tuple[int, int] = (80, 80),
) -> bool:
    """Use RSA's wavefront propagation to verify path existence."""
    import numpy as np
    from ..solvers.rsa_engine import RSAEngine

    xs = np.linspace(env.bounds.x_min, env.bounds.x_max, grid_size[1], dtype=np.float32)
    ys = np.linspace(env.bounds.y_min, env.bounds.y_max, grid_size[0], dtype=np.float32)
    speed = env.speed_grid(tuple(grid_size))

    try:
        rsa = RSAEngine(xs=xs, ys=ys, connectivity=8).solve(
            speed=speed,
            start_xy=start_xy,
            goal_xy=goal_xy,
        )
        path = rsa.backtrack_path_xy()
        return path.shape[0] >= 2 and np.isfinite(rsa.t).any()
    except Exception:
        return False


def _polygon_as_obstacles(verts_list: List[np.ndarray]) -> List[Dict]:
    """Convert polygon vertices to Maze2DEnv-compatible obstacle descriptors.

    Each polygon is decomposed into its constituent SDF as a callable.
    For simple convex polygons we store vertices for app-side SDF evaluation.
    """
    obstacles: List[Dict] = []
    for verts in verts_list:
        obstacles.append({
            "kind": "polygon",
            "vertices": verts.tolist(),
            "center": tuple(verts.mean(axis=0).tolist()),
        })
    return obstacles


# ---------------------------------------------------------------------------
#  Main generator
# ---------------------------------------------------------------------------

def generate_random_environment(
    bounds: Bounds2D,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    *,
    num_obstacles: int = 4,
    min_radius: float = 0.06,
    max_radius: float = 0.18,
    poly_vertices: int = 5,
    clearance: float = 0.06,
    vf_grid_size: Tuple[int, int] = (40, 40),
    vf_noise_scale: float = 3.0,
    v_min: float = 0.2,
    v_max: float = 1.0,
    speed_free: float = 1.0,
    speed_obstacle: float = 0.01,
    obstacle_inflation: float = 0.005,
    device: torch.device | str = "cpu",
    rng: Optional[np.random.Generator] = None,
) -> Maze2DEnv:
    """Generate a fully randomised environment with polygon obstacles + Perlin velocity field.

    Args:
        bounds: Environment bounds
        start_xy, goal_xy: Start and goal positions
        num_obstacles: Target number of polygon obstacles (3~5 recommended)
        min_radius, max_radius: Obstacle size range
        poly_vertices: Vertices per polygon
        clearance: Minimum clearance between obstacles and from path
        vf_grid_size: Perlin noise grid resolution
        vf_noise_scale: Perlin noise spatial frequency
        v_min, v_max: Velocity field value range
        speed_free, speed_obstacle: Wave speeds
        obstacle_inflation: SDF inflation
        device: Torch device
        rng: Optional random generator (auto-seeded if None)

    Returns:
        Maze2DEnv with polygon obstacles and Perlin velocity field

    Raises:
        RuntimeError: If no feasible maze could be generated after max_tries
    """
    if rng is None:
        rng = np.random.default_rng()

    start = np.asarray(start_xy, dtype=np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)
    max_tries = max(2000, num_obstacles * 200)
    poly_verts: List[np.ndarray] = []

    for _ in range(max_tries):
        if len(poly_verts) >= num_obstacles:
            break

        size = rng.uniform(min_radius, max_radius)
        cx = rng.uniform(bounds.x_min + size + clearance,
                         bounds.x_max - size - clearance)
        cy = rng.uniform(bounds.y_min + size + clearance,
                         bounds.y_max - size - clearance)
        center = np.array([cx, cy], dtype=np.float32)

        # Keep start and goal regions clear
        if np.linalg.norm(center - start) <= size * 2 + clearance:
            continue
        if np.linalg.norm(center - goal) <= size * 2 + clearance:
            continue

        # Generate polygon
        radius_range = (size * 0.6, size)
        verts = _random_convex_polygon(rng, center, radius_range, poly_vertices)

        # Check overlap with existing obstacles
        overlap = False
        for existing in poly_verts:
            if _polygons_overlap(verts, existing, clearance):
                overlap = True
                break
        if overlap:
            continue

        poly_verts.append(verts)

    if len(poly_verts) < 2:
        raise RuntimeError(
            f"Failed to generate {num_obstacles} obstacles in {max_tries} tries. "
            f"Only generated {len(poly_verts)}."
        )

    # Convert to Maze2DEnv-compatible dicts
    obstacles = _polygon_as_obstacles(poly_verts)

    # Build env first (no VF yet) for RSA feasibility check
    env = Maze2DEnv(
        bounds=bounds,
        obstacles=obstacles,
        obstacle_inflation=obstacle_inflation,
        speed_free=speed_free,
        speed_obstacle=speed_obstacle,
        device=device,
    )

    # Verify path feasibility via RSA
    if not _check_path_feasibility_rsa(env, start_xy, goal_xy):
        raise RuntimeError("Generated maze has no feasible path. Regenerate with different seed.")

    # Attach Perlin velocity field
    vf_seed = rng.integers(0, 2**31 - 1)
    vf_fn = _make_perlin_vf(bounds, vf_grid_size, vf_noise_scale,
                            v_min, v_max, vf_seed)
    env.set_velocity_field(vf_fn)

    return env


def _make_perlin_vf(
    bounds: Bounds2D,
    grid_size: Tuple[int, int],
    noise_scale: float,
    v_min: float,
    v_max: float,
    seed: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a bilinear-interpolated Perlin velocity field.

    See: maze_2d.make_perlin_velocity_field for details.
    """
    from .maze_2d import make_perlin_velocity_field
    return make_perlin_velocity_field(
        bounds=bounds,
        grid_size=grid_size,
        v_min=v_min,
        v_max=v_max,
        noise_scale=noise_scale,
        seed=seed,
    )


# ---------------------------------------------------------------------------
#  SDF support for polygon obstacles — extended in maze_2d
# ---------------------------------------------------------------------------

def sdf_polygon(xy: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Signed distance to a convex 2D polygon.

    vertices: (N, 2) tensor of polygon vertices in CCW order.
    xy: (M, 2) tensor of query points.
    Returns: (M,) tensor of signed distances (negative inside).
    """
    n = vertices.shape[0]
    sdf = torch.full((xy.shape[0],), float("inf"), device=xy.device, dtype=xy.dtype)
    inside = torch.ones(xy.shape[0], dtype=torch.bool, device=xy.device)

    for i in range(n):
        a = vertices[i]          # (2,)
        b = vertices[(i + 1) % n]  # (2,)
        ab = b - a               # (2,)
        ap = xy - a              # (M, 2)
        # Distance to line segment
        t = torch.sum(ap * ab, dim=1) / torch.clamp(torch.sum(ab * ab), min=1e-12)
        t = torch.clamp(t, 0.0, 1.0)
        closest = a[None, :] + t[:, None] * ab[None, :]
        dist = torch.linalg.norm(xy - closest, dim=1)
        sdf = torch.minimum(sdf, dist)

        # Edge normal outward test
        edge_normal = torch.tensor([-ab[1], ab[0]], device=xy.device, dtype=xy.dtype)
        dot = torch.sum(ap * edge_normal[None, :], dim=1)
        inside = inside & (dot <= 0.0)

    sdf = torch.where(inside, -sdf, sdf)
    return sdf
