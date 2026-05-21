"""Dijkstra planner for occupancy grid.

Dijkstra = A* with zero heuristic. Provides a non-informed baseline.
"""

from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..datasets.scene import PlanningResult, Scene
from .base_planner import BasePlanner


_NBR8: List[Tuple[int, int]] = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]


def _dijkstra_search(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_expand: int = 250000,
    goal_tol: float = 0.0,
) -> Tuple[Optional[np.ndarray], int, bool]:
    """Dijkstra search on 8-connected grid (no heuristic).

    Returns:
        path: (N, 2) array of (row, col) waypoints, or None if no path
        n_expanded: number of nodes expanded
        timeout: True if search was terminated by max_expand
    """
    h, w = grid.shape
    if grid[start[0], start[1]] == 1:
        return None, 0, False
    if grid[goal[0], goal[1]] == 1:
        return None, 0, False

    g_score = {start: 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    open_set = [(0.0, start)]
    closed_set: set = set()
    n_expanded = 0

    while open_set and n_expanded < max_expand:
        _, current = heapq.heappop(open_set)
        if current in closed_set:
            continue
        closed_set.add(current)
        n_expanded += 1

        r, c = current
        if abs(r - goal[0]) + abs(c - goal[1]) <= goal_tol + 0.5:
            goal = current

        if current == goal:
            path = []
            node = current
            while node in came_from:
                path.append((float(node[0]), float(node[1])))
                node = came_from[node]
            path.append((float(start[0]), float(start[1])))
            path.reverse()
            return np.asarray(path, dtype=np.float32), n_expanded, False

        for dr, dc in _NBR8:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if grid[nr, nc] == 1:
                continue

            step_cost = np.sqrt(2.0) if dr != 0 and dc != 0 else 1.0
            tentative = g_score[current] + step_cost

            if (nr, nc) not in g_score or tentative < g_score[(nr, nc)]:
                g_score[(nr, nc)] = tentative
                came_from[(nr, nc)] = current
                heapq.heappush(open_set, (tentative, (nr, nc)))

    if n_expanded >= max_expand:
        return None, n_expanded, True
    return None, n_expanded, False


class DijkstraPlanner(BasePlanner):
    """Dijkstra planner on occupancy grid. Uniform-cost search baseline."""

    def __init__(self, max_expand: int = 250000):
        self.max_expand = int(max_expand)

    @property
    def name(self) -> str:
        return "dijkstra"

    def plan(self, scene: Scene) -> PlanningResult:
        grid = scene.grid
        start = scene.start_px
        goal = scene.goal_px

        t0 = time.time()
        path_np, n_expanded, timeout = _dijkstra_search(
            grid, start, goal, max_expand=self.max_expand,
        )
        elapsed = time.time() - t0

        if path_np is None:
            return PlanningResult(
                success=False,
                path=None,
                path_length=float("inf"),
                inference_time=elapsed,
                n_expanded=n_expanded,
                timeout=timeout,
                deadlock=not timeout,
            )

        if path_np.shape[0] < 2:
            return PlanningResult(
                success=False,
                path=path_np,
                path_length=float("inf"),
                inference_time=elapsed,
                n_expanded=n_expanded,
                deadlock=True,
            )

        diffs = path_np[1:] - path_np[:-1]
        path_len_px = float(np.sum(np.linalg.norm(diffs, axis=1)))

        return PlanningResult(
            success=True,
            path=path_np,
            path_length=path_len_px,
            inference_time=elapsed,
            n_expanded=n_expanded,
            metadata={"dijkstra_expanded": n_expanded,
                      "dijkstra_timeout": timeout},
        )

    def config_snapshot(self) -> dict:
        return {"planner": "dijkstra", "max_expand": self.max_expand}