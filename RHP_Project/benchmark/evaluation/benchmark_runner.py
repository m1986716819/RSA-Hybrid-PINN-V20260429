from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..datasets.scene import PlanningResult, Scene
from ..planners.base_planner import BasePlanner
from .benchmark_metrics import BenchmarkMetrics, compute_benchmark_metrics


def _try_get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


class BenchmarkRunner:
    """Main benchmark loop.

    Orchestrates: scenes → planners → metrics → save.
    """

    def __init__(
        self,
        output_dir: str = "benchmark_results",
        tag: str = "",
        seed: int = 42,
        timeout_per_scene: float = 300.0,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.tag = str(tag) if tag else ""
        self.seed = int(seed)
        self.timeout_per_scene = float(timeout_per_scene)
        self.git_hash = _try_get_git_hash()
        self._run_dir: Optional[Path] = None
        self._rng = np.random.default_rng(self.seed)

    @property
    def run_dir(self) -> Path:
        if self._run_dir is None:
            raise RuntimeError("run() not called yet")
        return self._run_dir

    def _create_run_dir(self, scenes, planners) -> Path:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_part = f"_{self.tag}" if self.tag else ""
        n_scenes = len(scenes)
        n_planners = len(planners)
        dir_name = f"bench_{date_str}_{n_scenes}scenes_{n_planners}planners{tag_part}"
        run_dir = self.output_dir / dir_name
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _save_snapshot(self, scenes, planners) -> None:
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "git_hash": self.git_hash,
            "seed": self.seed,
            "n_scenes": len(scenes),
            "n_planners": len(planners),
            "planners": [p.config_snapshot() for p in planners],
            "scenes": [
                {"name": s.name, "map_name": s.map_name,
                 "height": s.height, "width": s.width,
                 "start": s.start_px, "goal": s.goal_px,
                 "optimal_length": s.optimal_length}
                for s in scenes
            ],
        }
        with open(self.run_dir / "snapshot.json", "w") as f:
            json.dump(snapshot, f, indent=2)

    def run(
        self,
        scenes: List[Scene],
        planners: List[BasePlanner],
        save_results: bool = True,
        verbose: bool = True,
    ) -> Dict[str, BenchmarkMetrics]:
        """Run benchmark: all planners on all scenes.

        Args:
            scenes: list of Scene objects
            planners: list of BasePlanner instances
            save_results: if True, save results to run_dir
            verbose: if True, print per-scene progress

        Returns:
            dict of planner_name -> BenchmarkMetrics
        """
        self._run_dir = self._create_run_dir(scenes, planners)
        self._save_snapshot(scenes, planners)

        results: Dict[str, List[Tuple[Scene, PlanningResult]]] = {
            p.name: [] for p in planners
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Benchmark: {self.run_dir.name}")
            print(f"  Scenes:   {len(scenes)}")
            print(f"  Planners: {[p.name for p in planners]}")
            print(f"  Seed:     {self.seed}")
            print(f"  Git:      {self.git_hash}")
            print(f"{'=' * 60}\n")

        t_start = time.time()

        for si, scene in enumerate(scenes):
            if verbose:
                print(f"[{si + 1}/{len(scenes)}] {scene.name} "
                      f"({scene.height}x{scene.width})")

            for planner in planners:
                if verbose:
                    print(f"  -> {planner.name}...", end=" ", flush=True)

                t_p = time.time()
                try:
                    result = planner.plan(scene)
                except Exception as e:
                    result = PlanningResult(
                        success=False, deadlock=True,
                        inference_time=time.time() - t_p,
                        metadata={"error": str(e)},
                    )
                    if verbose:
                        print(f"ERROR: {e}")
                        continue

                elapsed = time.time() - t_p
                results[planner.name].append((scene, result))

                if verbose:
                    status = "OK" if result.success else "FAIL"
                    print(f"{status} len={result.path_length:.2f} "
                          f"tc={result.time_cost:.3f} "
                          f"t={elapsed:.1f}s", flush=True)

            if verbose:
                print()

        elapsed = time.time() - t_start
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
            print(f"{'=' * 60}\n")

        metrics_dict: Dict[str, BenchmarkMetrics] = {}
        for planner in planners:
            planner_results = results[planner.name]
            metrics = compute_benchmark_metrics(planner_results, planner_name=planner.name)
            metrics_dict[planner.name] = metrics

            if verbose:
                print(f"  {planner.name:<15} SR={metrics.success_rate:.2f}  "
                      f"CR={metrics.collision_rate:.2f}  "
                      f"PL={metrics.avg_path_length:.2f}  "
                      f"TC={metrics.avg_time_cost:.4f}  "
                      f"IT={metrics.avg_inference_time_ms:.1f}ms")

        if save_results:
            self._save_metrics(metrics_dict)
            self._save_per_planner_csv(metrics_dict, scenes)

        return metrics_dict

    def _save_metrics(self, metrics_dict: Dict[str, BenchmarkMetrics]) -> None:
        data = {
            "timestamp": datetime.now().isoformat(),
            "git_hash": self.git_hash,
            "results": {
                name: m.to_dict() for name, m in metrics_dict.items()
            },
        }
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[benchmark] saved: {self.run_dir / 'metrics.json'}")

    def _save_per_planner_csv(
        self,
        metrics_dict: Dict[str, BenchmarkMetrics],
        scenes: List[Scene],
    ) -> None:
        for pname, metrics in metrics_dict.items():
            csv_path = self.run_dir / f"{pname}_per_scene.csv"
            with open(csv_path, "w") as f:
                cols = [
                    "scene", "success", "path_length", "time_cost",
                    "collision", "deadlock", "timeout",
                    "inference_time_ms", "smoothness", "coupling_alpha",
                ]
                f.write(",".join(cols) + "\n")
                for entry in metrics.per_scene:
                    row = [
                        entry.get("scene", ""),
                        str(entry.get("success", 0)),
                        f'{entry.get("path_length", -1):.4f}',
                        f'{entry.get("time_cost", -1):.4f}',
                        str(entry.get("collision", 0)),
                        str(entry.get("deadlock", 0)),
                        str(entry.get("timeout", 0)),
                        f'{entry.get("inference_time_ms", 0):.2f}',
                        f'{entry.get("smoothness", -1):.4f}',
                        f'{entry.get("coupling_alpha", -1):.4f}',
                    ]
                    f.write(",".join(row) + "\n")
            print(f"[benchmark] saved: {csv_path}")

    def summary_table(self, metrics_dict: Dict[str, BenchmarkMetrics]) -> str:
        lines = [
            f"{'Planner':<18} {'SR':>6} {'CR':>6} {'DR':>6} {'PL':>8} "
            f"{'TC':>10} {'IT(ms)':>8} {'OG':>8}",
            "-" * 75,
        ]
        for name, m in metrics_dict.items():
            og_str = f"{m.avg_optimality_gap:.3f}" if np.isfinite(m.avg_optimality_gap) else "N/A"
            lines.append(
                f"{name:<18} {m.success_rate:>6.2%} {m.collision_rate:>6.2%} "
                f"{m.deadlock_rate:>6.2%} {m.avg_path_length:>8.2f} "
                f"{m.avg_time_cost:>10.4f} {m.avg_inference_time_ms:>8.1f} {og_str:>8}"
            )
        return "\n".join(lines)
