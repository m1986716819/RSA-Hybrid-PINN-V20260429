#!/usr/bin/env python3
"""一键回归对比脚本：RHP-PINN vs P-NTFields-2D vs RSA 在不同场景下。

用法:
  python3 run_compare.py --scenario trap_u_shape --seeds 5
  python3 run_compare.py --scenario narrow_passage --seeds 5
  python3 run_compare.py --scenario u_maze --seeds 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

PROJECT = Path(__file__).resolve().parent
RHP_DIR = PROJECT / "RHP_Project"
CONFIGS_DIR = RHP_DIR / "configs"

SCENARIO_CONFIGS = {
    "trap_u_shape": CONFIGS_DIR / "trap_u_shape.yaml",
    "narrow_passage": CONFIGS_DIR / "narrow_passage_centered.yaml",
    "u_maze": CONFIGS_DIR / "default.yaml",
}


@dataclass
class Result:
    scenario: str
    seeds: int
    summary: Dict[str, Dict[str, float]]


def run_benchmark(config_path: Path, seeds: int, out_dir: str) -> Result:
    cfg = load_yaml(str(config_path))
    scenario = cfg["env"]["name"]
    print(f"  [{scenario}] 运行中...", end="", flush=True)

    t0 = time.time()
    cmd = [
        sys.executable, "-u", "-m", "RHP_Project.main_bench",
        "--config", str(config_path),
    ]
    env_copy = os.environ.copy()
    env_copy["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT),
        env=env_copy,
        timeout=7200,
    )

    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f" FAIL ({elapsed:.0f}s)")
        print(f"    stderr: {proc.stderr[:500]}")
        return Result(scenario=scenario, seeds=seeds, summary={})

    results_json = Path(str(cfg.get("eval", {}).get("out_dir", "outputs"))) / "results.json"
    if not results_json.exists():
        print(f" FAIL ({elapsed:.0f}s) — results.json 未生成")
        return Result(scenario=scenario, seeds=seeds, summary={})

    data = json.loads(results_json.read_text())
    summary = data.get("summary", {})
    print(f" OK ({elapsed:.0f}s)")

    return Result(scenario=scenario, seeds=seeds, summary=summary)


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_table(results: List[Result]) -> None:
    print()
    print("=" * 145)
    print("回归对比结果总结")
    print("=" * 145)

    all_methods = sorted(set().union(*[set(r.summary.keys()) for r in results if r.summary]))

    for r in results:
        if not r.summary:
            continue
        print(f"\n📁 场景: {r.scenario} ({r.seeds} seeds)")
        header = f"{'方法':<18} {'SR':>6}  {'OptGap':>10}  {'TimeCost':>10}  {'EffRatio':>10}  {'PhysCons':>10}  {'CurvSharp':>10}  {'ConvSteps':>10}"
        print(header)
        print("-" * len(header))
        for method in all_methods:
            if method not in r.summary:
                continue
            s = r.summary[method]
            sr = s.get("SR_mean", float("nan"))
            og = s.get("OptimalityGap_mean", float("nan"))
            tc = s.get("TimeCost_mean", float("nan"))
            er = s.get("EfficiencyRatio_mean", float("nan"))
            pc = s.get("PhysicalConsistency_mean", float("nan"))
            cs_out = s.get("CurvatureSharpness_mean", float("nan"))
            cvs = s.get("ConvergeSteps_mean", float("nan"))

            def _fmt(v, ndigits=4):
                if isinstance(v, (int, float)) and v == v:
                    return f"{v:.{ndigits}f}"
                return str(v)

            print(f"{method:<18} {_fmt(sr):>6}  {_fmt(og):>10}  {_fmt(tc):>10}  {_fmt(er):>10}  {_fmt(pc):>10}  {_fmt(cs_out):>10}  {_fmt(cvs):>10}")

    print()
    print("=" * 145)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, required=True, choices=list(SCENARIO_CONFIGS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--all", action="store_true", help="运行所有场景")
    args = ap.parse_args()

    scenarios_to_run = list(SCENARIO_CONFIGS) if args.all else [args.scenario]

    results: List[Result] = []
    for scenario in scenarios_to_run:
        config_path = SCENARIO_CONFIGS[scenario]
        if not config_path.exists():
            print(f"⚠ 配置文件不存在: {config_path}")
            continue

        # 临时写入 seeds 到配置
        cfg = load_yaml(str(config_path))
        cfg["num_seeds"] = args.seeds
        import yaml
        tmp_config = PROJECT / f"tmp_{scenario}_compare.yaml"
        with open(tmp_config, "w") as f:
            yaml.dump(cfg, f)

        try:
            r = run_benchmark(tmp_config, args.seeds, f"outputs_compare_{scenario}")
            results.append(r)
        finally:
            if tmp_config.exists():
                tmp_config.unlink()

    print_table(results)

    # 保存 JSON
    report = {
        "args": {"scenario": args.scenario, "seeds": args.seeds},
        "results": [
            {
                "scenario": r.scenario,
                "seeds": r.seeds,
                "summary": r.summary,
            }
            for r in results
        ],
    }
    out_json = PROJECT / "compare_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📄 报告已保存: {out_json}")


if __name__ == "__main__":
    main()
