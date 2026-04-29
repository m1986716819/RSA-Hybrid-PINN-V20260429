#!/usr/bin/env python3
"""第三阶段：自动化消融实验 (Ablation Study)

Test 1 (Topology Only):   U-shape + uniform V=1.0
Test 2 (Physics Only):    Open space + refraction V field
Test 3 (Unified):         Heterogeneous U-trap (障碍物 + 折射场)

各测试对比方法: RSA / Vanilla PINN / P-NTFields-2D / RHP-PINN / RRT*
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT = Path(__file__).resolve().parent
RHP_DIR = PROJECT / "RHP_Project"
CONFIGS_DIR = RHP_DIR / "configs"

ABLATION_CONFIGS = {
    "test1_topology": CONFIGS_DIR / "ablation_test1_topology.yaml",
    "test2_physics": CONFIGS_DIR / "ablation_test2_physics.yaml",
    "test3_unified": CONFIGS_DIR / "trap_heterogeneous_vf.yaml",
}


def load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _fmt(v: Any, ndigits: int = 4) -> str:
    if isinstance(v, (int, float)) and v == v:
        return f"{v:.{ndigits}f}"
    return str(v)


def run_main_bench(config_path: Path, out_dir: str, timeout: int = 7200) -> Optional[Dict]:
    print(f"  运行 benchmark...", end="", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "RHP_Project.main_bench", "--config", str(config_path)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT),
        timeout=timeout,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f" FAIL ({elapsed:.0f}s)")
        print(f"   stderr: {proc.stderr[:300]}")
        return None
    rj = Path(out_dir) / "results.json"
    if not rj.exists():
        print(f" FAIL ({elapsed:.0f}s) — no results.json")
        return None
    data = json.loads(rj.read_text())
    print(f" OK ({elapsed:.0f}s)")
    return data.get("summary", {})


def run_rrt_star(
    env_cfg: Dict,
    n_iterations: int = 5000,
    n_seeds: int = 5,
) -> Dict[str, Any]:
    """Run RRT* baseline for velocity-aware path planning."""
    sys.path.insert(0, str(PROJECT))
    import torch
    from RHP_Project.envs.maze_2d import Bounds2D
    from RHP_Project.main_bench import _make_env
    from RHP_Project.solvers.rrt_star import rrt_star_plan

    device = torch.device("cpu")
    env = _make_env(env_cfg, device)

    start_xy = tuple(env_cfg["env"]["start"])
    goal_xy = tuple(env_cfg["env"]["goal"])
    b = env.bounds
    bounds = (b.x_min, b.x_max, b.y_min, b.y_max)

    def _is_free(xy_np):
        xy_t = torch.from_numpy(xy_np.astype(np.float32)).view(1, 2).to(device=device)
        return bool(env.is_free(xy_t).item())

    has_vf = env.has_velocity_field()

    def _velocity_fn(xy_np):
        if not has_vf:
            return 1.0
        return env.velocity_field_np(xy_np)

    # Compute RSA reference for optimality gap
    from RHP_Project.solvers.rsa_engine import RSAEngine

    xs, ys, _ = env.make_grid((env_cfg.get("rsa", {}).get("grid_size", [80, 80])))
    speed_g = env.speed_grid((80, 80))
    rsa_ref = RSAEngine(xs=xs, ys=ys).solve(
        speed=speed_g, start_xy=start_xy, goal_xy=goal_xy
    )
    ref_path = rsa_ref.backtrack_path_xy()
    ref_len = float(np.sum(np.linalg.norm(ref_path[1:] - ref_path[:-1], axis=1)))

    successes = []
    lengths = []
    time_costs = []
    nodes_list = []

    for seed in range(n_seeds):
        result = rrt_star_plan(
            start_xy=start_xy,
            goal_xy=goal_xy,
            bounds=bounds,
            is_free=_is_free,
            velocity_fn=_velocity_fn,
            max_iterations=n_iterations,
            step_size=0.04,
            goal_tol=0.05,
            goal_bias=0.08,
            seed=seed,
        )
        successes.append(result.success)
        if result.success:
            lengths.append(result.path_length)
            time_costs.append(result.time_cost)
        nodes_list.append(result.nodes_explored)

    sr = float(np.mean(successes))
    avg_len = float(np.mean(lengths)) if lengths else float("nan")
    avg_tc = float(np.mean(time_costs)) if time_costs else float("nan")
    avg_nodes = float(np.mean(nodes_list))
    opt_gap = (avg_len - ref_len) / (ref_len + 1e-12) if np.isfinite(avg_len) else float("nan")

    return {
        "SR_mean": sr,
        "OptimalityGap_mean": opt_gap,
        "TimeCost_mean": avg_tc,
        "Length_mean": avg_len,
        "NodesExplored_mean": avg_nodes,
        "EfficiencyRatio_mean": avg_len / max(1e-12, avg_tc) if np.isfinite(avg_tc) and np.isfinite(avg_len) else float("nan"),
    }


def print_ablation_table(all_results: Dict[str, Dict[str, Dict]]) -> None:
    print("\n" + "=" * 150)
    print("第三阶段：消融实验 (Ablation Study) — 汇总结果")
    print("=" * 150)

    test_names = {
        "test1_topology": "Test 1: Topology Only (U-shape, V=1.0)",
        "test2_physics": "Test 2: Physics Only (No obstacle, refraction V)",
        "test3_unified": "Test 3: Unified (Heterogeneous U-trap)",
    }

    methods_display = ["RSA", "RHP-PINN", "P-NTFields-2D", "Vanilla PINN", "RRT*"]
    metric_cols = ["SR", "OptGap", "TimeCost", "EffRatio", "Nodes"]

    for test_key, test_label in test_names.items():
        if test_key not in all_results:
            continue
        print(f"\n{'─' * 150}")
        print(f"  {test_label}")
        print(f"{'─' * 150}")

        header = f"{'Method':<16}"
        for mc in metric_cols:
            header += f"  {mc:>10}"
        print(header)
        print("-" * len(header))

        test_data = all_results[test_key]
        for md in methods_display:
            md_key = md.lower().replace("-", "_").replace(" ", "_")
            if md_key not in test_data:
                continue
            d = test_data[md_key]
            row = f"{md:<16}"
            for mc in metric_cols:
                key_map = {
                    "SR": "SR_mean",
                    "OptGap": "OptimalityGap_mean",
                    "TimeCost": "TimeCost_mean",
                    "EffRatio": "EfficiencyRatio_mean",
                    "Nodes": "NodesExplored_mean",
                }
                val = d.get(key_map.get(mc, mc.lower()), float("nan"))
                row += f"  {_fmt(val):>10}"
            print(row)

    print("\n" + "=" * 150)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="每个测试的种子数")
    ap.add_argument("--rrt-iterations", type=int, default=5000, help="RRT* 迭代次数")
    ap.add_argument("--skip-pinn", action="store_true", help="跳过 PINN 训练（只跑 RRT*）")
    ap.add_argument("--skip-rrt", action="store_true", help="跳过 RRT*（只跑 PINN）")
    args = ap.parse_args()

    all_results: Dict[str, Dict[str, Dict]] = {}

    for test_key, config_path in ABLATION_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"  测试: {test_key}")
        print(f"{'=' * 60}")

        if not config_path.exists():
            print(f"  ⚠ 配置不存在: {config_path}")
            continue

        cfg = load_yaml(str(config_path))
        cfg["num_seeds"] = args.seeds
        out_dir = cfg.get("eval", {}).get("out_dir", "outputs")

        import yaml
        tmp_config = PROJECT / f"tmp_ablation_{test_key}.yaml"
        with open(tmp_config, "w") as f:
            yaml.dump(cfg, f)

        test_results: Dict[str, Dict] = {}

        # Run PINN-based methods
        if not args.skip_pinn:
            summary = run_main_bench(tmp_config, out_dir)
            if summary:
                for method_name, metrics in summary.items():
                    test_results[method_name] = dict(metrics)

        # Run RRT*
        if not args.skip_rrt:
            print(f"  运行 RRT* ({args.rrt_iterations} iter, {args.seeds} seeds)...", end="", flush=True)
            t0 = time.time()
            rrt_result = run_rrt_star(cfg, n_iterations=args.rrt_iterations, n_seeds=args.seeds)
            elapsed = time.time() - t0
            test_results["rrt*"] = rrt_result
            print(f" OK ({elapsed:.0f}s)")

        all_results[test_key] = test_results

        # Cleanup
        if tmp_config.exists():
            tmp_config.unlink()

    print_ablation_table(all_results)

    out_json = PROJECT / "ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 完整结果已保存: {out_json}")
    print(f"   用 cat ablation_results.json 查看原始 JSON")


if __name__ == "__main__":
    main()
