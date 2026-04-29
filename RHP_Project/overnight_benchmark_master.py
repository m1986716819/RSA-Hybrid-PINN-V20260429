from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt

from .envs.maze_2d import Bounds2D, Maze2DEnv
from .evaluator.metrics import EvalMetrics, evaluate_methods
from .main_bench import _gate_boost_weight, _grid_coords, _set_seed, _train_pntfield_2d, _train_rhp, _train_vanilla_pinn
from .solvers.rsa_engine import RSAEngine, extract_gateway_segment
from .utils.sampler import apply_sdf_surface_boost

matplotlib.use("Agg")


@dataclass(frozen=True)
class ScenarioSpec:
    group: str
    seed: int
    env_kind: str
    params: Dict[str, Any]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not math.isfinite(val) else val
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return _to_builtin(obj.detach().cpu().numpy())
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def _base_bounds(cfg: Dict[str, Any]) -> Bounds2D:
    return Bounds2D(**cfg["env"]["bounds"])


def _make_classic_scenarios(cfg: Dict[str, Any]) -> List[ScenarioSpec]:
    scenarios: List[ScenarioSpec] = []
    u_base = cfg["env"]["u_maze"]
    n_base = cfg["env"]["narrow_passage"]
    scenarios.append(ScenarioSpec(group="classic", seed=0, env_kind="u_maze", params=copy.deepcopy(u_base)))
    for seed in range(1, 10):
        rng = np.random.default_rng(seed)
        if seed % 2 == 0:
            params = {
                "wall_thickness": float(u_base["wall_thickness"] * rng.uniform(0.92, 1.12)),
                "inner_gap": float(u_base["inner_gap"] * rng.uniform(0.88, 1.08)),
                "depth": float(u_base["depth"] * rng.uniform(0.90, 1.08)),
                "center": tuple(u_base.get("center", [0.0, 0.0])),
                "rotation_deg": float(rng.uniform(-12.0, 12.0)),
            }
            scenarios.append(ScenarioSpec(group="classic", seed=seed, env_kind="u_maze", params=params))
        else:
            params = {
                "passage_width": float(n_base["passage_width"] * rng.uniform(0.85, 1.20)),
                "block_width": float(n_base["block_width"] * rng.uniform(0.90, 1.10)),
                "block_height": float(n_base["block_height"] * rng.uniform(0.85, 1.15)),
                "center": (0.0, 0.0),
                "rotation_deg": float(rng.uniform(-10.0, 10.0)),
            }
            scenarios.append(ScenarioSpec(group="classic", seed=seed, env_kind="narrow_passage", params=params))
    return scenarios


def _make_random_scenarios() -> List[ScenarioSpec]:
    scenarios: List[ScenarioSpec] = []
    for k in range(30):
        seed = 100 + k
        rng = np.random.default_rng(seed)
        params = {
            "num_circles": int(rng.integers(6, 12)),
            "radius_range": (float(rng.uniform(0.05, 0.07)), float(rng.uniform(0.13, 0.19))),
            "clearance": float(rng.uniform(0.04, 0.07)),
        }
        scenarios.append(ScenarioSpec(group="random", seed=seed, env_kind="random_circle_maze", params=params))
    return scenarios


def _make_rubble_scenarios() -> List[ScenarioSpec]:
    scenarios: List[ScenarioSpec] = []
    for k in range(20):
        seed = 200 + k
        rng = np.random.default_rng(seed)
        low = int(rng.integers(4, 7))
        high = int(max(low + 1, rng.integers(8, 12)))
        params = {
            "num_clusters": int(rng.integers(4, 8)),
            "circles_per_cluster": (low, high),
            "radius_range": (float(rng.uniform(0.02, 0.035)), float(rng.uniform(0.055, 0.085))),
            "cluster_spread": float(rng.uniform(0.07, 0.14)),
            "clearance": float(rng.uniform(0.035, 0.06)),
        }
        scenarios.append(ScenarioSpec(group="rubble", seed=seed, env_kind="rubble_field", params=params))
    return scenarios


def _all_scenarios(cfg: Dict[str, Any]) -> List[ScenarioSpec]:
    return _make_classic_scenarios(cfg) + _make_random_scenarios() + _make_rubble_scenarios()


def _make_env_for_scenario(cfg: Dict[str, Any], scenario: ScenarioSpec, device: torch.device) -> Maze2DEnv:
    bounds = _base_bounds(cfg)
    start_xy = tuple(cfg["env"]["start"])
    goal_xy = tuple(cfg["env"]["goal"])
    common = dict(
        bounds=bounds,
        obstacle_inflation=float(cfg["env"].get("obstacle_inflation", 0.0)),
        speed_free=float(cfg["rsa"].get("speed_free", 1.0)),
        speed_obstacle=float(cfg["rsa"].get("speed_obstacle", 0.0)),
        device=device,
    )
    if scenario.env_kind == "u_maze":
        return Maze2DEnv.make_u_maze(**common, **scenario.params)
    if scenario.env_kind == "narrow_passage":
        return Maze2DEnv.make_narrow_passage(**common, **scenario.params)
    rng = np.random.default_rng(int(scenario.seed))
    if scenario.env_kind == "random_circle_maze":
        return Maze2DEnv.make_random_circle_maze(
            **common,
            rng=rng,
            start_xy=start_xy,
            goal_xy=goal_xy,
            **scenario.params,
        )
    if scenario.env_kind == "rubble_field":
        return Maze2DEnv.make_rubble_field(
            **common,
            rng=rng,
            start_xy=start_xy,
            goal_xy=goal_xy,
            **scenario.params,
        )
    raise ValueError(f"Unknown env_kind: {scenario.env_kind}")


def _copy_cfg(cfg: Dict[str, Any], scenario: ScenarioSpec) -> Dict[str, Any]:
    cfg_s = copy.deepcopy(cfg)
    cfg_s["seed"] = int(scenario.seed)
    cfg_s["num_seeds"] = 1
    cfg_s["eval"]["save_plots"] = False
    cfg_s["eval"]["strict_goal_tol"] = True
    cfg_s["eval"]["max_path_steps"] = min(int(cfg_s["eval"]["max_path_steps"]), 250)
    return cfg_s


def _sdf_stats_for_path(env: Maze2DEnv, path: np.ndarray) -> Dict[str, float]:
    if path.shape[0] == 0:
        return {"safety_margin": float("nan"), "safety_mean": float("nan")}
    with torch.no_grad():
        sdf = env.sdf(torch.from_numpy(path).to(device=env.device, dtype=torch.float32)).detach().cpu().numpy()
    sdf = sdf[np.isfinite(sdf)]
    if sdf.size == 0:
        return {"safety_margin": float("nan"), "safety_mean": float("nan")}
    return {"safety_margin": float(np.min(sdf)), "safety_mean": float(np.mean(sdf))}


def _evalmetrics_to_dict(m: EvalMetrics) -> Dict[str, Any]:
    return {k: _to_builtin(v) for k, v in asdict(m).items()}


def _aggregate_rollouts(rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    methods = ["rsa", "vanilla_pinn", "pntfield_2d", "rhp_pinn"]
    out: Dict[str, Any] = {}
    for name in methods:
        out[name] = {}
        keys = set()
        for item in rollouts:
            keys.update(item["metrics"][name].keys())
            keys.update(item["safety"][name].keys())
        for key in sorted(keys):
            vals: List[float] = []
            for item in rollouts:
                if key in item["metrics"][name]:
                    val = item["metrics"][name][key]
                else:
                    val = item["safety"][name].get(key)
                if isinstance(val, bool):
                    vals.append(1.0 if val else 0.0)
                elif val is not None and isinstance(val, (int, float)) and math.isfinite(float(val)):
                    vals.append(float(val))
            if vals:
                out[name][f"{key}_mean"] = float(np.mean(vals))
                out[name][f"{key}_std"] = float(np.std(vals))
            else:
                out[name][f"{key}_mean"] = None
                out[name][f"{key}_std"] = None
        first_metrics = rollouts[0]["metrics"][name]
        out[name]["success_rollouts"] = [bool(item["metrics"][name]["success"]) for item in rollouts]
        out[name]["path_examples"] = _to_builtin(rollouts[0]["paths"][name])
        out[name]["safety_margin_rollouts"] = [
            _to_builtin(item["safety"][name]["safety_margin"]) for item in rollouts
        ]
        out[name]["raw_first_metrics"] = _to_builtin(first_metrics)
    return out


def _failure_reason(seed_result: Dict[str, Any], max_path_len: float) -> str:
    if seed_result.get("status") != "ok":
        return str(seed_result.get("error_code", "runtime_error"))
    rhp = seed_result["aggregate"]["rhp_pinn"]
    success = float(rhp.get("success_mean", 0.0))
    if success >= 0.5:
        return "success"
    gating = rhp.get("gating_ratio_mean")
    safety = rhp.get("safety_margin_mean")
    length = rhp.get("length_mean")
    if gating is not None and float(gating) < 0.25:
        return "Transformer 介入太晚"
    if safety is not None and float(safety) < 0.02:
        return "死胡同太窄或贴障过深"
    if length is not None and float(length) >= 0.95 * float(max_path_len):
        return "局部环流导致走满步数上限"
    if gating is not None and float(gating) > 0.75:
        return "高强度托管下仍未穿出复杂拓扑"
    return "复杂拓扑下未形成稳定下降通道"


def _group_summary(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups = ["classic", "random", "rubble", "overall"]
    methods = ["rsa", "vanilla_pinn", "pntfield_2d", "rhp_pinn"]
    metric_keys = ["success_mean", "optimality_gap_mean", "gating_ratio_mean", "safety_margin_mean"]
    out: Dict[str, Any] = {}
    for group in groups:
        items = seed_results if group == "overall" else [x for x in seed_results if x["group"] == group]
        ok_items = [x for x in items if x.get("status") == "ok"]
        out[group] = {}
        for method in methods:
            out[group][method] = {}
            for key in metric_keys:
                vals = []
                for item in ok_items:
                    val = item["aggregate"][method].get(key)
                    if val is not None and math.isfinite(float(val)):
                        vals.append(float(val))
                out[group][method][key.replace("_mean", "")] = {
                    "mean": float(np.mean(vals)) if vals else None,
                    "std": float(np.std(vals)) if vals else None,
                }
    return out


def _draw_env(ax, env: Maze2DEnv) -> None:
    for obs in env.obstacles:
        kind = str(obs["kind"])
        if kind == "circle":
            circle = plt.Circle(obs["center"], obs["radius"], color="#7f7f7f", alpha=0.8)
            ax.add_patch(circle)
        elif kind == "box":
            half = np.asarray(obs["half_size"], dtype=np.float32)
            center = np.asarray(obs["center"], dtype=np.float32)
            angle = float(np.degrees(obs.get("rotation_rad", 0.0)))
            rect = plt.matplotlib.patches.Rectangle(
                (float(center[0] - half[0]), float(center[1] - half[1])),
                float(2.0 * half[0]),
                float(2.0 * half[1]),
                angle=angle,
                rotation_point="center",
                color="#7f7f7f",
                alpha=0.8,
            )
            ax.add_patch(rect)


def _save_performance_matrix(out_path: Path, summary: Dict[str, Any]) -> None:
    methods = ["vanilla_pinn", "pntfield_2d", "rsa", "rhp_pinn"]
    labels = ["Vanilla PINN", "P-NTFields-2D", "RSA", "RHP-PINN"]
    metrics = [("success", "SR"), ("optimality_gap", "Gap"), ("gating_ratio", "Ratio")]
    colors = ["#C44E52", "#55A868", "#4C72B0"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    overall = summary["overall"]
    for ax, (metric_key, title) in zip(axes, metrics):
        vals = []
        errs = []
        for method in methods:
            stat = overall[method][metric_key]
            vals.append(0.0 if stat["mean"] is None else float(stat["mean"]))
            errs.append(0.0 if stat["std"] is None else float(stat["std"]))
        ax.bar(labels, vals, yerr=errs, color=colors, alpha=0.9, capsize=4)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_rubble_case_study(out_path: Path, case: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    device = torch.device(cfg.get("device", "cpu"))
    env = _make_env_for_scenario(cfg, ScenarioSpec(group=case["group"], seed=case["seed"], env_kind=case["env_kind"], params=case["params"]), device)
    start_xy = tuple(cfg["env"]["start"])
    goal_xy = tuple(cfg["env"]["goal"])
    path = np.asarray(case["aggregate"]["rhp_pinn"]["path_examples"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    _draw_env(ax, env)
    if path.ndim == 2 and path.shape[0] > 1:
        ax.plot(path[:, 0], path[:, 1], "-", color="#4C72B0", linewidth=2.5, label="RHP-PINN")
    ax.scatter([start_xy[0]], [start_xy[1]], c="gold", edgecolors="black", s=90, marker="o", label="Start")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="lime", edgecolors="black", s=120, marker="*", label="Goal")
    ax.set_xlim(env.bounds.x_min, env.bounds.x_max)
    ax.set_ylim(env.bounds.y_min, env.bounds.y_max)
    ax.set_aspect("equal")
    ax.set_title(f"Rubble Case Study | seed={case['seed']}")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_markdown_summary(out_path: Path, summary: Dict[str, Any], seed_results: List[Dict[str, Any]], max_path_len: float) -> None:
    lines: List[str] = []
    lines.append("# Final Summary")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append("| Method | SR mean±std | Gap mean±std | Ratio mean±std | Safety mean±std |")
    lines.append("|---|---:|---:|---:|---:|")
    for method, title in [
        ("vanilla_pinn", "Vanilla PINN"),
        ("pntfield_2d", "P-NTFields-2D"),
        ("rsa", "RSA"),
        ("rhp_pinn", "RHP-PINN"),
    ]:
        sr = summary["overall"][method]["success"]
        gap = summary["overall"][method]["optimality_gap"]
        ratio = summary["overall"][method]["gating_ratio"]
        safety = summary["overall"][method]["safety_margin"]
        lines.append(
            f"| {title} | "
            f"{sr['mean'] if sr['mean'] is not None else 'N/A'} ± {sr['std'] if sr['std'] is not None else 'N/A'} | "
            f"{gap['mean'] if gap['mean'] is not None else 'N/A'} ± {gap['std'] if gap['std'] is not None else 'N/A'} | "
            f"{ratio['mean'] if ratio['mean'] is not None else 'N/A'} ± {ratio['std'] if ratio['std'] is not None else 'N/A'} | "
            f"{safety['mean'] if safety['mean'] is not None else 'N/A'} ± {safety['std'] if safety['std'] is not None else 'N/A'} |"
        )
    lines.append("")
    lines.append("## Failed Seeds")
    lines.append("")
    failed = [x for x in seed_results if _failure_reason(x, max_path_len) != "success"]
    if not failed:
        lines.append("- None")
    else:
        for item in failed:
            reason = _failure_reason(item, max_path_len)
            lines.append(f"- {item['group']} | seed={item['seed']} | env={item['env_kind']} | reason={reason}")
    lines.append("")
    lines.append("## Group Breakdown")
    lines.append("")
    for group in ["classic", "random", "rubble"]:
        lines.append(f"### {group}")
        lines.append("")
        lines.append("| Method | SR | Gap | Ratio | Safety |")
        lines.append("|---|---:|---:|---:|---:|")
        for method, title in [
            ("vanilla_pinn", "Vanilla PINN"),
            ("pntfield_2d", "P-NTFields-2D"),
            ("rsa", "RSA"),
            ("rhp_pinn", "RHP-PINN"),
        ]:
            sr = summary[group][method]["success"]["mean"]
            gap = summary[group][method]["optimality_gap"]["mean"]
            ratio = summary[group][method]["gating_ratio"]["mean"]
            safety = summary[group][method]["safety_margin"]["mean"]
            lines.append(
                f"| {title} | "
                f"{sr if sr is not None else 'N/A'} | "
                f"{gap if gap is not None else 'N/A'} | "
                f"{ratio if ratio is not None else 'N/A'} | "
                f"{safety if safety is not None else 'N/A'} |"
            )
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _persist_temp(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(payload), f, ensure_ascii=False, indent=2)


def _run_single_seed(cfg: Dict[str, Any], scenario: ScenarioSpec, device: torch.device) -> Dict[str, Any]:
    cfg_s = _copy_cfg(cfg, scenario)
    env = _make_env_for_scenario(cfg_s, scenario, device=device)
    start_xy = tuple(cfg_s["env"]["start"])
    goal_xy = tuple(cfg_s["env"]["goal"])
    rng = _set_seed(int(scenario.seed))

    xs, ys = _grid_coords(env.bounds, tuple(cfg_s["rsa"]["grid_size"]))
    speed = env.speed_grid(tuple(cfg_s["rsa"]["grid_size"]))
    free_mask = speed > 0.0
    rsa_low = RSAEngine(xs=xs, ys=ys, connectivity=int(cfg_s["rsa"].get("connectivity", 8))).solve(
        speed=speed,
        start_xy=start_xy,
        goal_xy=goal_xy,
    )
    weight = RSAEngine.wavefront_weight(
        t=rsa_low.t,
        free_mask=free_mask,
        base_weight=float(cfg_s["train"]["adaptive_sampling"].get("base_weight", 0.2)),
        wavefront_power=float(cfg_s["train"]["adaptive_sampling"].get("wavefront_power", 1.0)),
        obstacle_band=float(cfg_s["train"]["adaptive_sampling"].get("obstacle_band", 0.0)),
        xs=xs,
        ys=ys,
    )
    xx, yy = np.meshgrid(xs, ys)
    xy_grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)
    with torch.no_grad():
        sdf_grid = env.sdf(torch.from_numpy(xy_grid).to(device=device)).cpu().numpy().reshape(speed.shape)
    weight = apply_sdf_surface_boost(
        weight=weight,
        sdf_grid=sdf_grid,
        band=float(cfg_s["train"].get("sdf_surface_band", 0.05)),
        boost=float(cfg_s["train"].get("sdf_surface_boost", 0.5)),
    )

    xs_ref, ys_ref = _grid_coords(env.bounds, tuple(cfg_s["eval"]["reference_grid_size"]))
    speed_ref = env.speed_grid(tuple(cfg_s["eval"]["reference_grid_size"]))
    rsa_ref = RSAEngine(xs=xs_ref, ys=ys_ref, connectivity=int(cfg_s["rsa"].get("connectivity", 8))).solve(
        speed=speed_ref,
        start_xy=start_xy,
        goal_xy=goal_xy,
    )
    ref_path_xy = rsa_ref.backtrack_path_xy()
    with torch.no_grad():
        sdf_path_ref = env.sdf(torch.from_numpy(ref_path_xy).to(device=device, dtype=torch.float32)).cpu().numpy()
    gate_info = extract_gateway_segment(
        path_xy=ref_path_xy,
        sdf_values=sdf_path_ref,
        segment_len=int(cfg_s["train"].get("gate", {}).get("segment_len", 16)),
        sdf_weight=float(cfg_s["train"].get("gate", {}).get("sdf_weight", 1.0)),
        diff_weight=float(cfg_s["train"].get("gate", {}).get("diff_weight", 1.0)),
    )
    weight = _gate_boost_weight(
        weight=weight,
        xs=xs,
        ys=ys,
        gate_pts=gate_info.gate_xy,
        radius=float(cfg_s["train"].get("gate_forcing", {}).get("radius", 0.15)),
        boost=float(cfg_s["train"].get("gate", {}).get("weight_boost", 2.0)),
    )

    vanilla, vanilla_conv, vanilla_steps = _train_vanilla_pinn(env, start_xy, goal_xy, cfg_s, rng, device=device)
    pntfield, pntfield_conv, pntfield_steps = _train_pntfield_2d(
        env,
        start_xy,
        goal_xy,
        cfg_s,
        rng,
        device=device,
    )
    rhp, rhp_conv, rhp_steps = _train_rhp(
        env,
        start_xy,
        goal_xy,
        rsa_ref,
        ref_path_xy,
        gate_info.gate_xy,
        gate_info.d_target,
        gate_info.d_targets,
        xs,
        ys,
        weight,
        cfg_s,
        rng,
        device,
    )

    models = {"vanilla_pinn": vanilla, "pntfield_2d": pntfield, "rhp_pinn": rhp}
    rollouts: List[Dict[str, Any]] = []
    for _ in range(3):
        metrics, paths = evaluate_methods(
            env=env,
            start_xy=start_xy,
            goal_xy=goal_xy,
            rsa_low=rsa_low,
            rsa_ref=rsa_ref,
            models=models,
            path_step=float(cfg_s["eval"]["path_step"]),
            max_path_steps=int(cfg_s["eval"]["max_path_steps"]),
            goal_tol=float(cfg_s["eval"]["goal_tol"]),
            grad_probe_radius=float(cfg_s["eval"]["grad_probe_radius"]),
            grad_probe_points=int(cfg_s["eval"]["grad_probe_points"]),
            strict_goal_tol=True,
        )
        metrics["vanilla_pinn"] = EvalMetrics(
            success=metrics["vanilla_pinn"].success,
            length=metrics["vanilla_pinn"].length,
            smoothness=metrics["vanilla_pinn"].smoothness,
            optimality_gap=metrics["vanilla_pinn"].optimality_gap,
            regret=metrics["vanilla_pinn"].regret,
            gating_ratio=metrics["vanilla_pinn"].gating_ratio,
            gating_efficiency=metrics["vanilla_pinn"].gating_efficiency,
            grad_norm_start=metrics["vanilla_pinn"].grad_norm_start,
            converge_steps=int(vanilla_conv),
            train_steps=int(vanilla_steps),
            time_cost=metrics["vanilla_pinn"].time_cost,
            efficiency_ratio=metrics["vanilla_pinn"].efficiency_ratio,
            physical_consistency=metrics["vanilla_pinn"].physical_consistency,
            curvature_sharpness=metrics["vanilla_pinn"].curvature_sharpness,
        )
        metrics["pntfield_2d"] = EvalMetrics(
            success=metrics["pntfield_2d"].success,
            length=metrics["pntfield_2d"].length,
            smoothness=metrics["pntfield_2d"].smoothness,
            optimality_gap=metrics["pntfield_2d"].optimality_gap,
            regret=metrics["pntfield_2d"].regret,
            gating_ratio=metrics["pntfield_2d"].gating_ratio,
            gating_efficiency=metrics["pntfield_2d"].gating_efficiency,
            grad_norm_start=metrics["pntfield_2d"].grad_norm_start,
            converge_steps=int(pntfield_conv),
            train_steps=int(pntfield_steps),
            time_cost=metrics["pntfield_2d"].time_cost,
            efficiency_ratio=metrics["pntfield_2d"].efficiency_ratio,
            physical_consistency=metrics["pntfield_2d"].physical_consistency,
            curvature_sharpness=metrics["pntfield_2d"].curvature_sharpness,
        )
        metrics["rhp_pinn"] = EvalMetrics(
            success=metrics["rhp_pinn"].success,
            length=metrics["rhp_pinn"].length,
            smoothness=metrics["rhp_pinn"].smoothness,
            optimality_gap=metrics["rhp_pinn"].optimality_gap,
            regret=metrics["rhp_pinn"].regret,
            gating_ratio=metrics["rhp_pinn"].gating_ratio,
            gating_efficiency=metrics["rhp_pinn"].gating_efficiency,
            grad_norm_start=metrics["rhp_pinn"].grad_norm_start,
            converge_steps=int(rhp_conv),
            train_steps=int(rhp_steps),
            time_cost=metrics["rhp_pinn"].time_cost,
            efficiency_ratio=metrics["rhp_pinn"].efficiency_ratio,
            physical_consistency=metrics["rhp_pinn"].physical_consistency,
            curvature_sharpness=metrics["rhp_pinn"].curvature_sharpness,
        )
        safety = {
            name: _sdf_stats_for_path(env, path)
            for name, path in paths.items()
            if name in {"rsa", "vanilla_pinn", "pntfield_2d", "rhp_pinn"}
        }
        rollouts.append(
            {
                "metrics": {name: _evalmetrics_to_dict(m) for name, m in metrics.items()},
                "paths": {name: _to_builtin(path) for name, path in paths.items()},
                "safety": _to_builtin(safety),
            }
        )

    return {
        "status": "ok",
        "group": scenario.group,
        "seed": int(scenario.seed),
        "env_kind": scenario.env_kind,
        "params": _to_builtin(scenario.params),
        "obstacle_count": int(len(env.obstacles)),
        "aggregate": _aggregate_rollouts(rollouts),
        "rollouts": rollouts,
    }


def run_master(
    cfg: Dict[str, Any],
    limit: int | None = None,
    device: str | None = None,
    groups: List[str] | None = None,
    seed_filter: List[int] | None = None,
) -> Dict[str, Any]:
    root = _project_root()
    outputs_dir = root / "outputs"
    final_report_dir = outputs_dir / "final_report"
    temp_json = outputs_dir / "temp_benchmark.json"
    final_json = final_report_dir / "final_benchmark.json"
    os.makedirs(final_report_dir, exist_ok=True)

    scenarios = _all_scenarios(cfg)
    if groups:
        group_set = {str(g).strip().lower() for g in groups if str(g).strip()}
        scenarios = [s for s in scenarios if s.group.lower() in group_set]
    if seed_filter:
        seed_set = {int(s) for s in seed_filter}
        scenarios = [s for s in scenarios if int(s.seed) in seed_set]
    if limit is not None:
        scenarios = scenarios[: int(limit)]
    dev = torch.device(device or cfg.get("device", "cpu"))
    seed_results: List[Dict[str, Any]] = []
    max_path_len = float(cfg["eval"]["path_step"]) * min(int(cfg["eval"]["max_path_steps"]), 250)

    for idx, scenario in enumerate(scenarios, start=1):
        try:
            result = _run_single_seed(cfg, scenario, dev)
        except RuntimeError as exc:
            result = {
                "status": "error",
                "group": scenario.group,
                "seed": int(scenario.seed),
                "env_kind": scenario.env_kind,
                "params": _to_builtin(scenario.params),
                "error_code": type(exc).__name__,
                "error_message": str(exc),
            }
        except Exception as exc:
            result = {
                "status": "error",
                "group": scenario.group,
                "seed": int(scenario.seed),
                "env_kind": scenario.env_kind,
                "params": _to_builtin(scenario.params),
                "error_code": type(exc).__name__,
                "error_message": str(exc),
            }
        seed_results.append(result)
        payload = {
            "progress": {"done": idx, "total": len(scenarios)},
            "seed_results": seed_results,
        }
        _persist_temp(temp_json, payload)

    summary = _group_summary(seed_results)
    rubble_ok = [x for x in seed_results if x.get("status") == "ok" and x["group"] == "rubble"]
    rubble_case = max(rubble_ok, key=lambda x: (x.get("obstacle_count", 0), x["aggregate"]["rhp_pinn"].get("length_mean", 0.0))) if rubble_ok else None

    performance_path = final_report_dir / "performance_matrix.png"
    rubble_path = final_report_dir / "rubble_case_study.png"
    summary_md = final_report_dir / "final_summary.md"

    _save_performance_matrix(performance_path, summary)
    if rubble_case is not None:
        _save_rubble_case_study(rubble_path, rubble_case, cfg)
    _save_markdown_summary(summary_md, summary, seed_results, max_path_len=max_path_len)

    payload = {
        "summary": summary,
        "seed_results": seed_results,
        "artifacts": {
            "performance_matrix": str(performance_path),
            "rubble_case_study": str(rubble_path) if rubble_case is not None else None,
            "final_summary_md": str(summary_md),
            "temp_benchmark": str(temp_json),
        },
    }
    _persist_temp(final_json, payload)
    _persist_temp(temp_json, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--groups", type=str, default=None)
    ap.add_argument("--seeds", type=str, default=None)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    groups = None if args.groups is None else [x.strip() for x in str(args.groups).split(",")]
    seed_filter = None if args.seeds is None else [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    payload = run_master(cfg=cfg, limit=args.limit, device=args.device, groups=groups, seed_filter=seed_filter)
    print(json.dumps(_to_builtin(payload["summary"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
