import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rhp_experiment.config import ExperimentConfig


@dataclass
class ExperimentResult:
    success_rate: float = 0.0
    avg_time_cost: float = 0.0
    std_time_cost: float = 0.0
    avg_coupling_alpha: float = 0.0
    std_coupling_alpha: float = 0.0
    avg_physical_consistency: float = 0.0
    better_than_rsa_ratio: float = 0.0
    avg_optimality_gap: float = 0.0
    num_seeds: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    def __init__(self, config: ExperimentConfig, tag: str = "", notes: str = "", visualize_level: str = "basic"):
        self.config = config
        self.tag = tag
        self.notes = notes
        self.visualize_level = visualize_level
        self.exp_dir: Optional[Path] = None
        self.seeds_dir: Optional[Path] = None
        self.plots_dir: Optional[Path] = None
        self._results: Dict[str, Dict] = {}
        self._per_seed: List[Dict] = []

    def _create_experiment_dir(self):
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario = self.config.scenario
        tag_part = f"_{self.tag}" if self.tag else ""
        exp_name = f"exp_{date_str}_{scenario}{tag_part}"
        self.exp_dir = self.PROJECT_ROOT / "experiments" / exp_name
        self.seeds_dir = self.exp_dir / "seeds"
        self.plots_dir = self.exp_dir / "plots"
        self.exp_dir.mkdir(parents=True, exist_ok=False)
        self.seeds_dir.mkdir(parents=True, exist_ok=False)
        self.plots_dir.mkdir(parents=True, exist_ok=False)

    def _save_config_snapshot(self):
        self.config.save_snapshot(self.exp_dir / "config.yaml")

    def _save_meta(self, num_seeds: int):
        meta = {
            "date": datetime.now().isoformat(),
            "scenario": self.config.scenario,
            "num_seeds": num_seeds,
            "device": self.config.device,
            "tag": self.tag or "",
            "notes": self.notes,
        }
        with open(self.exp_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    def _save_per_seed(self, seed_data: List[Dict]):
        for sd in seed_data:
            sid = sd.get("seed", 0)
            path = self.seeds_dir / f"seed_{sid}.json"
            with open(path, "w") as f:
                json.dump(sd, f, indent=2)

    def _aggregate_metrics(self, seed_data: List[Dict]) -> ExperimentResult:
        if not seed_data:
            return ExperimentResult()

        successes = [s.get("success", False) for s in seed_data]
        tcs = [s.get("time_cost", float("inf")) for s in seed_data]
        rsa_tcs = [s.get("rsa_time_cost", None) for s in seed_data]
        gatings = [s.get("coupling_alpha", None) for s in seed_data]
        physcons = [s.get("physical_consistency", 0.0) for s in seed_data]
        optgaps = [s.get("optimality_gap", 0.0) for s in seed_data]

        sr = float(np.mean(successes))
        valid_tcs = [tc for tc in tcs if np.isfinite(tc) and tc > 0]
        avg_tc = float(np.mean(valid_tcs)) if valid_tcs else 0.0
        std_tc = float(np.std(valid_tcs)) if len(valid_tcs) > 1 else 0.0

        valid_alpha = [g for g in gatings if g is not None and np.isfinite(g)]
        avg_gate = float(np.mean(valid_alpha)) if valid_alpha else 0.0
        std_gate = float(np.std(valid_alpha)) if len(valid_alpha) > 1 else 0.0

        avg_pc = float(np.mean(physcons)) if physcons else 0.0
        avg_og = float(np.mean(optgaps)) if optgaps else 0.0

        better_count = 0
        total_compare = 0
        for tc, rtc in zip(tcs, rsa_tcs):
            if rtc is not None and np.isfinite(rtc) and np.isfinite(tc):
                total_compare += 1
                if tc < rtc:
                    better_count += 1
        better_ratio = better_count / max(total_compare, 1)

        result = ExperimentResult(
            success_rate=sr,
            avg_time_cost=avg_tc,
            std_time_cost=std_tc,
            avg_coupling_alpha=avg_gate,
            std_coupling_alpha=std_gate,
            avg_physical_consistency=avg_pc,
            better_than_rsa_ratio=better_ratio,
            avg_optimality_gap=avg_og,
            num_seeds=len(seed_data),
            meta=self._save_meta(len(seed_data)),
        )
        return result

    def _save_metrics(self, result: ExperimentResult):
        metrics = asdict(result)
        with open(self.exp_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    def _setup_logging(self):
        log_path = self.exp_dir / "log.txt"
        sys.stdout = Tee(sys.stdout, open(log_path, "w"))
        sys.stderr = Tee(sys.stderr, open(log_path, "a"))

    def run(
        self,
        seed_callback=None,
        num_seeds: Optional[int] = None,
        use_rhp_config: bool = True,
    ) -> ExperimentResult:
        self._create_experiment_dir()
        self._save_config_snapshot()
        self._setup_logging()

        ns = num_seeds if num_seeds is not None else self.config.num_seeds
        self._save_meta(ns)

        print(f"Experiment: {self.exp_dir.name}")
        print(f"Config: {self.config.config_path}")
        print(f"Scenario: {self.config.scenario}")
        print(f"Seeds: {ns}")
        print(f"Device: {self.config.device}")
        print("=" * 60)

        config_path = self.config.get_rhp_config_path() if use_rhp_config else self.exp_dir / "config.yaml"
        t_start = time.time()

        seeds_results = seed_callback(config_path, ns) if seed_callback else self._default_run(config_path, ns)

        elapsed = time.time() - t_start
        self._save_per_seed(seeds_results)
        result = self._aggregate_metrics(seeds_results)
        self._save_metrics(result)

        print(f"\n{'=' * 60}")
        print(f"Experiment complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"SR={result.success_rate:.2f}  TC={result.avg_time_cost:.4f}±{result.std_time_cost:.4f}")
        print(f"CouplingAlpha={result.avg_coupling_alpha:.3f}±{result.std_coupling_alpha:.3f}")
        print(f"BetterThanRSA={result.better_than_rsa_ratio:.0%}")
        print(f"Results: {self.exp_dir}")
        print(f"{'=' * 60}")

        if self.visualize_level != "none":
            self._generate_visualizations()

        if hasattr(sys.stdout, "flush"):
            sys.stdout.flush()
        return result

    def _generate_visualizations(self):
        try:
            from rhp_experiment.visualizer import ExperimentVisualizer
            viz = ExperimentVisualizer(str(self.exp_dir))
            viz.plot_all(level=self.visualize_level)
        except Exception as e:
            print(f"\n  [viz] WARNING: visualization failed: {e}")

    def _default_run(self, config_path: Path, num_seeds: int) -> List[Dict]:
        print("Running benchmark...")
        cmd = [
            sys.executable, "-u", "-m", "RHP_Project.main_bench",
            "--config", str(config_path),
            "--exp-dir", str(self.exp_dir),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            cwd=str(self.PROJECT_ROOT),
            timeout=14400,
        )
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)

        results_json = self.exp_dir / "rhp_outputs" / "results.json"
        if not results_json.exists():
            print(f"WARNING: results.json not found at {results_json}")
            return []

        data = json.loads(results_json.read_text())
        per_seed = data.get("per_seed", [])
        parsed = []
        for i, sd in enumerate(per_seed):
            rhp = sd.get("rhp_pinn", {})
            rsa = sd.get("rsa", {})
            parsed.append({
                "seed": i,
                "success": bool(rhp.get("success", False)),
                "time_cost": float(rhp.get("time_cost", float("nan"))),
                "rsa_time_cost": float(rsa.get("time_cost", float("nan"))),
                "coupling_alpha": float(rhp.get("coupling_alpha", float("nan"))),
                "path_length": float(rhp.get("length", float("nan"))),
                "physical_consistency": float(rhp.get("physical_consistency", float("nan"))),
                "optimality_gap": float(rhp.get("optimality_gap", float("nan"))),
            })
        return parsed

    @classmethod
    def list_experiments(cls) -> List[Path]:
        exp_dir = cls.PROJECT_ROOT / "experiments"
        if not exp_dir.exists():
            return []
        return sorted(exp_dir.iterdir())


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
