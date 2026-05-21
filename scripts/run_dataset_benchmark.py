#!/usr/bin/env python3
"""RHP-PINN Dataset Benchmark Runner.

Usage:
  # Topology benchmark (no external data needed)
  python scripts/run_dataset_benchmark.py --suite topology --scenes 5

  # MovingAI benchmark (requires downloaded data)
  python scripts/run_dataset_benchmark.py --suite movingai --map maze512 --scenes 20

  # Compare specific planners
  python scripts/run_dataset_benchmark.py --suite topology --planners astar,fmm,rhp_pinn

  # Download MovingAI data first
  python scripts/run_dataset_benchmark.py --download maze512 --data-dir ./movingai_data
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RHP_Project.benchmark import BenchmarkRunner
from RHP_Project.benchmark.datasets.topology_suite import TopologySuite
from RHP_Project.benchmark.datasets.movingai_loader import MovingAILoader
from RHP_Project.benchmark.planners import (
    AStarPlanner, DijkstraPlanner, FMMPlanner, RRTStarPlanner,
    RHPPINNPlanner, VanillaPINNPlanner, PNTFieldPlanner,
)
from RHP_Project.benchmark.visualization import BenchmarkPlotter


def _build_planners(names: list, device: str, grid_size: tuple) -> list:
    registry = {
        "astar": lambda: AStarPlanner(),
        "dijkstra": lambda: DijkstraPlanner(),
        "fmm": lambda: FMMPlanner(),
        "rrt_star": lambda: RRTStarPlanner(),
        "rhp_pinn": lambda: RHPPINNPlanner(
            device=device, grid_size=grid_size, verbose=True,
        ),
        "vanilla_pinn": lambda: VanillaPINNPlanner(
            device=device, grid_size=grid_size, verbose=True,
        ),
        "pntfield": lambda: PNTFieldPlanner(
            device=device, grid_size=grid_size, verbose=True,
        ),
    }
    planners = []
    for n in names:
        n = n.strip().lower().replace("-", "_")
        if n in registry:
            planners.append(registry[n]())
        else:
            print(f"  Warning: unknown planner '{n}', skipping")
    return planners


def main():
    ap = argparse.ArgumentParser(description="RHP-PINN Dataset Benchmark")
    ap.add_argument("--suite", choices=["topology", "movingai"], default="topology")
    ap.add_argument("--map", default="maze512", help="MovingAI map name")
    ap.add_argument("--scenes", type=int, default=10, help="Number of scenes")
    ap.add_argument("--planners",
                    default="astar,fmm,rhp_pinn,vanilla_pinn,pntfield",
                    help="Comma-separated planner list")
    ap.add_argument("--device", default="cpu", help="Device for PINN methods")
    ap.add_argument("--grid-size", type=int, nargs=2, default=[80, 80],
                    help="Grid size (height width) for PINN methods")
    ap.add_argument("--output", default="benchmark_results",
                    help="Output directory")
    ap.add_argument("--tag", default="", help="Experiment tag")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--data-dir", default="./movingai_data",
                    help="MovingAI data root directory")
    ap.add_argument("--download", default="",
                    help="Download a MovingAI map by name")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip visualization")
    args = ap.parse_args()

    if args.download:
        print(f"Downloading MovingAI map: {args.download}")
        path = MovingAILoader.download_map(args.download, args.data_dir)
        print(f"  Downloaded to: {path}")
        return 0

    grid_size = tuple(args.grid_size)
    planners = _build_planners(
        args.planners.split(","), args.device, grid_size,
    )
    if not planners:
        print("Error: no valid planners specified.")
        return 1

    print(f"\nRHP-PINN Benchmark Suite")
    print(f"  Suite:    {args.suite}")
    print(f"  Planners: {[p.name for p in planners]}")
    print(f"  Device:   {args.device}")
    print(f"  Seed:     {args.seed}")

    if args.suite == "topology":
        suite = TopologySuite(grid_size=grid_size)
        all_scenes = suite.generate_all(variants=max(1, args.scenes // 6))
        scenes = []
        for key, scene_list in all_scenes.items():
            scenes.extend(scene_list[:max(1, args.scenes // len(all_scenes))])
        scenes = scenes[:args.scenes]
        print(f"  Topology: {[f'{k}:{len(v)}' for k, v in all_scenes.items()]}")

    else:
        loader = MovingAILoader(args.data_dir)
        scenes = loader.load(args.map, max_scenes=args.scenes)
        print(f"  MovingAI: {args.map} -> {len(scenes)} scenes")

    if not scenes:
        print("Error: no scenes loaded.")
        return 1

    runner = BenchmarkRunner(
        output_dir=args.output,
        tag=args.tag or args.suite,
        seed=args.seed,
    )

    t0 = time.time()
    metrics_dict = runner.run(scenes, planners, verbose=True)
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(runner.summary_table(metrics_dict))
    print(f"{'=' * 60}")
    print(f"Total: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Results: {runner.run_dir}")

    if not args.no_plot:
        plotter = BenchmarkPlotter(str(runner.run_dir))
        saved = plotter.plot_all(metrics_dict)
        for p in saved:
            print(f"  Plot: {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
