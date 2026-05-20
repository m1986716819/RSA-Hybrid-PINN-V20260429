#!/usr/bin/env python3
"""Run complete ablation study with automatic recording for all 3 tests.

Usage:
  python scripts/run_ablation.py --seeds 5
  python scripts/run_ablation.py --seeds 5 --tag final_v2 --notes "pinn_streak fix + distill"
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rhp_experiment import ExperimentRunner, load_config

ABLATION_CONFIGS = {
    "test1_topology": "configs/ablation_test1_topology.yaml",
    "test2_physics": "configs/ablation_test2_physics.yaml",
    "test3_unified": "configs/trap_heterogeneous_vf.yaml",
}


def main():
    ap = argparse.ArgumentParser(description="RHP-PINN Ablation Study")
    ap.add_argument("--seeds", type=int, default=5, help="Number of seeds per test")
    ap.add_argument("--skip-pinn", action="store_true", help="Skip PINN tests (RRT* only)")
    ap.add_argument("--skip-rrt", action="store_true", help="Skip RRT* baseline")
    ap.add_argument("--tag", default="", help="Experiment tag")
    ap.add_argument("--notes", default="", help="Experiment notes")
    ap.add_argument("--visualize-level", choices=["none", "basic", "paper"], default="none",
                    help="Visualization level: none (default), basic (5 plots), paper (+4 advanced plots)")
    args = ap.parse_args()

    results = []
    t_total = time.time()

    for test_name, config_path in ABLATION_CONFIGS.items():
        print(f"\n{'=' * 65}")
        print(f"  {test_name}: {config_path}")
        print(f"{'=' * 65}")

        tag = f"{args.tag}_{test_name}" if args.tag else test_name
        config = load_config(config_path)
        config.num_seeds = args.seeds

        runner = ExperimentRunner(config, tag=tag, notes=args.notes, visualize_level=args.visualize_level)
        result = runner.run(num_seeds=args.seeds)
        results.append((test_name, result))

    elapsed = time.time() - t_total
    print(f"\n{'=' * 65}")
    print(f"  ABLATION COMPLETE ({elapsed:.0f}s / {elapsed/60:.1f} min)")
    print(f"{'=' * 65}")
    print(f"{'Test':<20} {'SR':>6} {'TC':>10} {'Alpha':>8} {'BetterRSA':>10}")
    print("-" * 55)
    for name, r in results:
        print(f"{name:<20} {r.success_rate:>6.2%} {r.avg_time_cost:>10.4f} {r.avg_coupling_alpha:>8.3f} {r.better_than_rsa_ratio:>10.0%}")
    print(f"\nResults in: experiments/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
