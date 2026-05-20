import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt


class ExperimentAnalyzer:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    def __init__(self):
        self.experiments_dir = self.PROJECT_ROOT / "experiments"
        self.reports_dir = self.PROJECT_ROOT / "reports"
        self.figures_dir = self.reports_dir / "figures"
        self.tables_dir = self.reports_dir / "tables"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def load_all_experiments(self) -> List[Dict]:
        exps = []
        if not self.experiments_dir.exists():
            return exps
        for exp_path in sorted(self.experiments_dir.iterdir()):
            if not exp_path.is_dir():
                continue
            exp_data = self.load_experiment(exp_path)
            if exp_data:
                exps.append(exp_data)
        return exps

    def load_experiment(self, exp_path: Path) -> Optional[Dict]:
        meta_path = exp_path / "meta.json"
        metrics_path = exp_path / "metrics.json"
        if not meta_path.exists() or not metrics_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
            metrics = json.loads(metrics_path.read_text())
            seeds_dir = exp_path / "seeds"
            seeds = []
            if seeds_dir.exists():
                for sp in sorted(seeds_dir.iterdir()):
                    if sp.suffix == ".json":
                        seeds.append(json.loads(sp.read_text()))
            return {
                "path": str(exp_path),
                "name": exp_path.name,
                "meta": meta,
                "metrics": metrics,
                "seeds": seeds,
            }
        except Exception as e:
            print(f"  Warning: {exp_path.name} failed to load: {e}")
            return None

    def generate_main_table(self) -> str:
        exps = self.load_all_experiments()
        if not exps:
            print("No experiments found.")
            return ""

        header = "Experiment,Scenario,SR,AvgTC,StdTC,AvgGating,BetterRSA,Seeds"
        rows = [header]
        for e in exps:
            m = e["metrics"]
            meta = e["meta"]
            scenario = meta.get("scenario", "?")
            rows.append(
                f"{e['name']},{scenario},"
                f"{m.get('success_rate', 0):.2f},"
                f"{m.get('avg_time_cost', 0):.4f},"
                f"{m.get('std_time_cost', 0):.4f},"
                f"{m.get('avg_gating_ratio', m.get('avg_coupling_alpha', 0)):.3f},"
                f"{m.get('better_than_rsa_ratio', 0):.2%},"
                f"{m.get('num_seeds', 0)}"
            )

        csv_content = "\n".join(rows)
        csv_path = self.tables_dir / "main_results.csv"
        csv_path.write_text(csv_content)
        print(f"Saved: {csv_path}")
        return str(csv_path)

    def plot_gating_vs_timecost(self):
        exps = self.load_all_experiments()
        if not exps:
            print("No experiments to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        all_gatings = []
        all_tcs = []
        all_success = []
        colors = []
        scenario_colors = {}
        color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        ci = 0

        for e in exps:
            scenario = e["meta"].get("scenario", "unknown")
            if scenario not in scenario_colors:
                scenario_colors[scenario] = color_cycle[ci % len(color_cycle)]
                ci += 1
            for s in e["seeds"]:
                g = s.get("gating_ratio", s.get("coupling_alpha", None))
                tc = s.get("time_cost", None)
                succ = s.get("success", False)
                if g is not None and tc is not None and np.isfinite(g) and np.isfinite(tc) and tc > 0:
                    all_gatings.append(g)
                    all_tcs.append(tc)
                    all_success.append(succ)
                    colors.append(scenario_colors[scenario])

        if not all_gatings:
            print("No gating/tc data found.")
            return

        g_arr = np.array(all_gatings)
        tc_arr = np.array(all_tcs)
        succ_arr = np.array(all_success)

        for scenario, color in scenario_colors.items():
            mask = [True] * len(all_gatings)
            ax1.scatter([], [], c=color, label=scenario, s=40, alpha=0.7, edgecolors="k", linewidths=0.3)

        ax1.scatter(g_arr, tc_arr, c=colors, s=40, alpha=0.7, edgecolors="k", linewidths=0.3)
        ax1.set_xlabel("Gating Ratio (RSA usage)", fontsize=12)
        ax1.set_ylabel("Time Cost", fontsize=12)
        ax1.set_title("Gating Ratio vs Time Cost", fontsize=13, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)

        # Annotate success/fail
        for i in range(len(all_gatings)):
            if not succ_arr[i]:
                ax1.annotate("✗", (g_arr[i], tc_arr[i]), fontsize=10,
                            color="red", ha="center", va="center")

        # Subplot 2: success rate by gating bins
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_sr = []
        bin_counts = []
        for k in range(len(bins) - 1):
            mask = (g_arr >= bins[k]) & (g_arr < bins[k + 1])
            count = int(mask.sum())
            if count > 0:
                bin_sr.append(float(succ_arr[mask].mean()))
            else:
                bin_sr.append(0.0)
            bin_counts.append(count)

        ax2.bar(bin_centers, bin_sr, width=0.09, color="steelblue", alpha=0.8, edgecolor="k")
        ax2.set_xlabel("Gating Ratio (RSA usage)", fontsize=12)
        ax2.set_ylabel("Success Rate", fontsize=12)
        ax2.set_title("Success Rate by Gating Bins", fontsize=13, fontweight="bold")
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3, axis="y")

        for i in range(len(bin_centers)):
            if bin_counts[i] > 0:
                ax2.text(bin_centers[i], bin_sr[i] + 0.02, f"n={bin_counts[i]}",
                        ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        save_path = self.figures_dir / "gating_vs_tc.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)

    def plot_summary_bars(self):
        exps = self.load_all_experiments()
        if not exps:
            return

        names = [e["name"] for e in exps]
        scenarios = [e["meta"].get("scenario", "?")[:12] for e in exps]
        srs = [e["metrics"].get("success_rate", 0) for e in exps]
        tcs = [e["metrics"].get("avg_time_cost", 0) for e in exps]
        betters = [e["metrics"].get("better_than_rsa_ratio", 0) for e in exps]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(exps))
        w = 0.3
        ax1.bar(x - w/2, srs, w, label="SR", color="steelblue", alpha=0.85)
        ax1.bar(x + w/2, betters, w, label="BetterRSA", color="orange", alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=9)
        ax1.set_ylabel("Rate", fontsize=12)
        ax1.set_title("Success Rate & Better-than-RSA", fontsize=13, fontweight="bold")
        ax1.set_ylim(0, 1.1)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis="y")

        colors2 = ["green" if t < 1.5 else "orange" if t < 3 else "red" for t in tcs]
        bars = ax2.bar(x, tcs, color=colors2, alpha=0.8, edgecolor="k", linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=9)
        ax2.set_ylabel("Avg Time Cost", fontsize=12)
        ax2.set_title("Average Time Cost by Experiment", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        for bar, tc in zip(bars, tcs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{tc:.2f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        save_path = self.figures_dir / "summary_bars.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)

    def run_all(self):
        print("Generating reports...")
        self.generate_main_table()
        self.plot_gating_vs_timecost()
        self.plot_summary_bars()
        print(f"\nReports generated in {self.reports_dir}")
