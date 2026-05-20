from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics across multiple scenes for a single planner."""

    planner_name: str = ""

    success_rate: float = 0.0
    collision_rate: float = 0.0
    deadlock_rate: float = 0.0
    timeout_rate: float = 0.0

    avg_path_length: float = 0.0
    std_path_length: float = 0.0
    avg_time_cost: float = 0.0
    std_time_cost: float = 0.0
    avg_smoothness: float = 0.0
    avg_inference_time_ms: float = 0.0
    avg_optimality_gap: float = 0.0
    avg_coupling_alpha: float = 0.0
    trap_escape_rate: float = 0.0

    n_scenes: int = 0
    n_success: int = 0
    per_scene: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = asdict(self)
        base["per_scene"] = self.per_scene
        return base


def compute_optimality_gap(path_length: float, optimal_length: Optional[float]) -> float:
    """Compute (path - optimal) / optimal ratio."""
    if optimal_length is None or optimal_length <= 0:
        return float("nan")
    if not np.isfinite(path_length) or path_length <= 0:
        return float("nan")
    return (path_length - optimal_length) / optimal_length


def compute_benchmark_metrics(
    results: List[tuple],
    planner_name: str = "",
) -> BenchmarkMetrics:
    """Aggregate per-scene PlanningResult list into BenchmarkMetrics.

    Args:
        results: list of (scene, PlanningResult) tuples

    Returns:
        BenchmarkMetrics with aggregated statistics
    """
    if not results:
        return BenchmarkMetrics(planner_name=planner_name)

    n = len(results)
    successes = [r[1].success for r in results]
    collisions = [r[1].collision for r in results]
    deadlocks = [r[1].deadlock for r in results]
    timeouts = [r[1].timeout for r in results]

    path_lengths = []
    time_costs = []
    smoothness_vals = []
    inference_times = []
    coupling_alphas = []
    optimality_gaps = []

    for scene, pr in results:
        if pr.success and pr.path_length > 0 and np.isfinite(pr.path_length):
            path_lengths.append(pr.path_length)
        if pr.success and pr.time_cost > 0 and np.isfinite(pr.time_cost):
            time_costs.append(pr.time_cost)
        if pr.success and np.isfinite(pr.smoothness):
            smoothness_vals.append(pr.smoothness)
        if np.isfinite(pr.inference_time) and pr.inference_time > 0:
            inference_times.append(pr.inference_time)
        if pr.coupling_alpha is not None and np.isfinite(pr.coupling_alpha):
            coupling_alphas.append(pr.coupling_alpha)
        if pr.success:
            og = compute_optimality_gap(pr.path_length, scene.optimal_length)
            if np.isfinite(og):
                optimality_gaps.append(og)

    sr = float(np.mean(successes)) if successes else 0.0
    cr = float(np.mean(collisions)) if collisions else 0.0
    dr = float(np.mean(deadlocks)) if deadlocks else 0.0
    tr = float(np.mean(timeouts)) if timeouts else 0.0

    avg_pl = float(np.mean(path_lengths)) if path_lengths else 0.0
    std_pl = float(np.std(path_lengths)) if len(path_lengths) > 1 else 0.0
    avg_tc = float(np.mean(time_costs)) if time_costs else 0.0
    std_tc = float(np.std(time_costs)) if len(time_costs) > 1 else 0.0
    avg_sm = float(np.mean(smoothness_vals)) if smoothness_vals else float("nan")
    avg_it = float(np.mean(inference_times)) * 1000.0 if inference_times else 0.0
    avg_ca = float(np.mean(coupling_alphas)) if coupling_alphas else 0.0
    avg_og = float(np.mean(optimality_gaps)) if optimality_gaps else float("nan")

    per_scene_data = []
    for scene, pr in results:
        entry = {"scene": scene.name, "success": int(pr.success)}
        entry.update(pr.summary())
        per_scene_data.append(entry)

    return BenchmarkMetrics(
        planner_name=planner_name,
        success_rate=sr,
        collision_rate=cr,
        deadlock_rate=dr,
        timeout_rate=tr,
        avg_path_length=avg_pl,
        std_path_length=std_pl,
        avg_time_cost=avg_tc,
        std_time_cost=std_tc,
        avg_smoothness=avg_sm,
        avg_inference_time_ms=avg_it,
        avg_optimality_gap=avg_og,
        avg_coupling_alpha=avg_ca,
        n_scenes=n,
        n_success=int(sum(successes)),
        per_scene=per_scene_data,
    )
