#!/usr/bin/env python3
"""Run a single benchmark experiment with full automatic recording.

Usage:
  python scripts/run_benchmark.py --config configs/trap_heterogeneous_vf.yaml
  python scripts/run_benchmark.py --config configs/trap_u_shape.yaml --seeds 10
  python scripts/run_benchmark.py --config configs/ablation_test2_physics.yaml --tag refraction_v2 --notes "with pinn_streak fix"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rhp_experiment import ExperimentRunner, load_config


def main():
    ap = argparse.ArgumentParser(description="RHP-PINN Benchmark Runner")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--seeds", type=int, default=None, help="Override num_seeds")
    ap.add_argument("--tag", default="", help="Optional experiment tag")
    ap.add_argument("--notes", default="", help="Experiment notes")
    ap.add_argument("--visualize-level", choices=["none", "basic", "paper"], default="none",
                    help="Visualization level: none (default), basic (5 plots), paper (+4 advanced plots)")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.seeds:
        config.num_seeds = args.seeds

    runner = ExperimentRunner(config, tag=args.tag, notes=args.notes, visualize_level=args.visualize_level)
    result = runner.run(num_seeds=args.seeds)

    print(f"\nSummary:")
    print(f"  SR:              {result.success_rate:.2%}")
    print(f"  TimeCost:        {result.avg_time_cost:.4f} ± {result.std_time_cost:.4f}")
    print(f"  GatingRatio:     {result.avg_coupling_alpha:.3f} ± {result.std_coupling_alpha:.3f}")
    print(f"  BetterThanRSA:   {result.better_than_rsa_ratio:.0%}")
    print(f"  Results:         experiments/{runner.exp_dir.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
