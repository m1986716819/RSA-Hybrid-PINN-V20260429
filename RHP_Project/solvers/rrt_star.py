from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class RRTStarResult:
    path: np.ndarray
    success: bool
    path_length: float
    time_cost: float
    nodes_explored: int
    iterations: int


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _segment_time_cost(
    a: np.ndarray, b: np.ndarray, velocity_fn: Optional[Callable[[np.ndarray], float]] = None
) -> float:
    ds = _distance(a, b)
    if velocity_fn is None:
        return ds
    mid = (a + b) * 0.5
    v = float(velocity_fn(mid))
    if v > 1e-8:
        return ds / v
    return float("inf")


def rrt_star_plan(
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    bounds: Tuple[float, float, float, float],
    is_free: Callable[[np.ndarray], bool],
    velocity_fn: Optional[Callable[[np.ndarray], float]] = None,
    max_iterations: int = 5000,
    step_size: float = 0.04,
    goal_tol: float = 0.05,
    goal_bias: float = 0.08,
    rewire_radius_factor: float = 2.5,
    seed: int = 0,
) -> RRTStarResult:
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = bounds

    start = np.asarray(start_xy, dtype=np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)

    nodes: List[np.ndarray] = [start]
    parents: List[Optional[int]] = [None]
    cost: List[float] = [0.0]

    rewire_radius = step_size * rewire_radius_factor
    goal_reached = False
    goal_candidate: Optional[int] = None
    path: Optional[np.ndarray] = None
    path_time_cost = float("inf")

    for it in range(max_iterations):
        if rng.uniform() < goal_bias:
            sample = goal
        else:
            sample = np.asarray(
                [
                    rng.uniform(x_min, x_max),
                    rng.uniform(y_min, y_max),
                ],
                dtype=np.float32,
            )

        if not is_free(sample):
            continue

        nearest_idx = int(np.argmin([_distance(n, sample) for n in nodes]))
        nearest = nodes[nearest_idx]

        d = _distance(nearest, sample)
        if d > step_size:
            direction = (sample - nearest) / d
            new_node = nearest + direction * step_size
        else:
            new_node = sample

        if not is_free(new_node):
            continue

        new_cost = cost[nearest_idx] + _segment_time_cost(nearest, new_node, velocity_fn)
        if not np.isfinite(new_cost):
            continue

        # Find near neighbors for RRT* rewiring
        near_indices = [
            i
            for i in range(len(nodes))
            if _distance(nodes[i], new_node) < rewire_radius
        ]

        # Choose best parent
        best_parent = nearest_idx
        best_cost = new_cost
        for ni in near_indices:
            if ni == nearest_idx:
                continue
            seg_cost = _segment_time_cost(nodes[ni], new_node, velocity_fn)
            cand_cost = cost[ni] + seg_cost
            if cand_cost < best_cost and np.isfinite(seg_cost):
                n_mid = (nodes[ni] + new_node) * 0.5
                if is_free(n_mid):
                    best_cost = cand_cost
                    best_parent = ni

        new_idx = len(nodes)
        nodes.append(new_node)
        parents.append(best_parent)
        cost.append(best_cost)

        # Rewire neighbors
        for ni in near_indices:
            if ni == best_parent:
                continue
            seg_cost = _segment_time_cost(new_node, nodes[ni], velocity_fn)
            if cost[new_idx] + seg_cost < cost[ni] and np.isfinite(seg_cost):
                mid = (nodes[ni] + new_node) * 0.5
                if is_free(mid):
                    cost[ni] = cost[new_idx] + seg_cost
                    parents[ni] = new_idx

        # Check goal
        if _distance(new_node, goal) <= goal_tol:
            if not goal_reached or cost[new_idx] < path_time_cost:
                goal_reached = True
                goal_candidate = new_idx
                path_time_cost = cost[new_idx]

    # Extract best path
    if goal_reached and goal_candidate is not None:
        path_nodes: List[np.ndarray] = []
        idx = goal_candidate
        while idx is not None:
            path_nodes.append(nodes[idx])
            idx = parents[idx]
        path_nodes.reverse()
        path = np.stack(path_nodes, axis=0).astype(np.float32)
        path_len = float(np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)))

        # Recompute time cost along path
        tc = 0.0
        for i in range(len(path) - 1):
            tc += _segment_time_cost(path[i], path[i + 1], velocity_fn)
        path_time_cost = tc

        return RRTStarResult(
            path=path,
            success=True,
            path_length=path_len,
            time_cost=path_time_cost,
            nodes_explored=len(nodes),
            iterations=it + 1,
        )

    return RRTStarResult(
        path=np.array([start, goal], dtype=np.float32),
        success=False,
        path_length=float("inf"),
        time_cost=float("inf"),
        nodes_explored=len(nodes),
        iterations=max_iterations,
    )


__all__ = ["rrt_star_plan", "RRTStarResult"]
