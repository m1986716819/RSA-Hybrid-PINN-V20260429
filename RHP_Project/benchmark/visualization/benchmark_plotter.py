from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from matplotlib import pyplot as plt

from ..datasets.scene import PlanningResult, Scene
from ..evaluation.benchmark_metrics import BenchmarkMetrics


class BenchmarkPlotter:
    """Generate publication-quality comparison plots from benchmark results."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name: str, dpi: int = 180) -> Path:
        path = self.output_dir / name
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_metrics_bars(
        self,
        metrics_dict: Dict[str, BenchmarkMetrics],
        save_name: str = "benchmark_bars.png",
    ) -> Path:
        names = list(metrics_dict.keys())
        srs = [metrics_dict[n].success_rate for n in names]
        crs = [metrics_dict[n].collision_rate for n in names]
        pls = [metrics_dict[n].avg_path_length for n in names]
        tcs = [metrics_dict[n].avg_time_cost for n in names]
        its = [metrics_dict[n].avg_inference_time_ms for n in names]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Benchmark Comparison", fontsize=14, fontweight="bold")

        x = np.arange(len(names))
        w = 0.5
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

        ax = axes[0, 0]
        bars = ax.bar(x, srs, w, color=colors[:len(names)], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Success Rate")
        ax.set_title("Success Rate")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, srs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        ax = axes[0, 1]
        ax.bar(x, crs, w, color=colors[:len(names)], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, max(max(crs), 0.05) * 1.2)
        ax.set_ylabel("Collision Rate")
        ax.set_title("Collision Rate")
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[0, 2]
        ax.bar(x, pls, w, color=colors[:len(names)], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Avg Path Length")
        ax.set_title("Path Length")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars if False else [], pls):
            pass
        for i, (bar, v) in enumerate(zip(ax.patches if hasattr(ax, 'patches') else [], pls)):
            pass

        ax = axes[1, 0]
        ax.bar(x, tcs, w, color=colors[:len(names)], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Avg Time Cost")
        ax.set_title("Time Cost")
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[1, 1]
        ax.bar(x, its, w, color=colors[:len(names)], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Avg Inference Time (ms)")
        ax.set_title("Inference Time")
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[1, 2]
        oqs = []
        for n in names:
            og = metrics_dict[n].avg_optimality_gap
            oqs.append(og if np.isfinite(og) else 0.0)
        ax.bar(x, oqs, w, color=colors[:len(names)], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Optimality Gap")
        ax.set_title("Optimality Gap")
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        return self._save(fig, save_name)

    def plot_scatter_sr_vs_tc(
        self,
        metrics_dict: Dict[str, BenchmarkMetrics],
        save_name: str = "scatter_sr_vs_tc.png",
    ) -> Path:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
        for i, (name, m) in enumerate(metrics_dict.items()):
            ax.scatter(m.success_rate, m.avg_time_cost,
                      c=colors[i % len(colors)], s=120, label=name,
                      edgecolors="black", linewidths=0.5, zorder=5)
            ax.annotate(name, (m.success_rate, m.avg_time_cost),
                       xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Success Rate", fontsize=12)
        ax.set_ylabel("Avg Time Cost", fontsize=12)
        ax.set_title("Success Rate vs Time Cost", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        return self._save(fig, save_name)

    def plot_path_comparison(
        self,
        scene: Scene,
        paths: Dict[str, PlanningResult],
        save_name: str = "path_comparison.png",
    ) -> Path:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        grid_display = np.where(scene.grid == 1, 1.0, 0.0)
        ax.imshow(grid_display, cmap="gray_r", origin="upper",
                  extent=[0, scene.width, scene.height, 0],
                  alpha=0.4)

        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
        for i, (pname, pr) in enumerate(paths.items()):
            if pr.success and pr.path is not None and pr.path.shape[0] >= 2:
                path = pr.path
                ax.plot(path[:, 1], path[:, 0],
                       linewidth=1.8, color=colors[i % len(colors)],
                       label=f"{pname} (len={pr.path_length:.2f})")

        sr, sc = scene.start_px
        gr, gc = scene.goal_px
        ax.plot(sc, sr, "o", color="#2ecc71", markersize=10,
                markeredgecolor="white", markeredgewidth=1.5, label="Start")
        ax.plot(gc, gr, "X", color="#e74c3c", markersize=12,
                markeredgecolor="white", markeredgewidth=1.5, label="Goal")

        ax.set_xlim(-1, scene.width + 1)
        ax.set_ylim(scene.height + 1, -1)
        ax.set_aspect("equal")
        ax.set_title(f"Path Comparison: {scene.name}", fontsize=12)
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        return self._save(fig, save_name)

    def plot_all(
        self,
        metrics_dict: Dict[str, BenchmarkMetrics],
    ) -> List[Path]:
        saved = []
        saved.append(self.plot_metrics_bars(metrics_dict))
        saved.append(self.plot_scatter_sr_vs_tc(metrics_dict))
        return saved
