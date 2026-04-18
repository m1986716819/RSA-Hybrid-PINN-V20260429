# RHP-PINN Project Stage Report

## 1. Project Positioning

This project studies a hybrid path planning framework for complex non-convex environments under an Eikonal-equation-guided setting.

The core idea is:

- Use a discrete global planner, RSA, to provide reliable topological guidance.
- Use a factored PINN to learn a smooth continuous travel-time field.
- Use an online hybrid gating strategy to decide when to trust the PINN gradient and when to temporarily fall back to RSA.

In short, this is not a pure neural planner and not a pure graph planner. It is a topology-guided neural PDE solver for path planning.

The current active research question is:

- How to keep the success rate high in U-shaped and narrow-passage environments,
- while reducing RSA takeover ratio,
- without letting the planner fall back into local circulation or post-gate drift.

## 2. Scientific Motivation

The project is motivated by three classic weaknesses of vanilla PINN-based path planning:

1. Non-convex traps:
   Pure PINN often gets stuck in U-shaped walls, narrow gateways, and maze-like local minima.

2. Point-source singularity:
   Near the start point, the travel-time field has unstable gradients, which makes direct learning of the full field difficult.

3. Slow or fragile online replanning:
   Even if a neural field is learned, local rollout may still fail around obstacle bottlenecks unless topological guidance is injected.

To address these issues, the project adopts a factored representation and a hybrid decision architecture:

- Factored travel time:
  `T(x) = ||x - x_start|| * tau(x)`

- Eikonal constraint:
  `|grad T(x)| = 1 / f(x)`

The factorization reduces singular behavior near the source, while RSA supplies global topological information that standard PINN training lacks.

## 3. Overall System Architecture

The full system can be understood as five layers:

1. Environment layer:
   Build 2D SDF-based environments such as deep U-maze, narrow passage, random circle maze, and rubble field.

2. Discrete topology layer:
   Run RSA on a grid to obtain a global travel-time prior and a safe reference path around obstacles.

3. Neural field layer:
   Train a factored PINN with PDE, obstacle, path, and topology-related losses to approximate a continuous travel-time field.

4. Gateway/tunnel shaping layer:
   Around the gateway, sample a gate band and a path tube to impose monotonicity, direction, value anchor, and local drop constraints.

5. Online decision layer:
   During rollout, compare PINN and RSA candidates and decide which one to use based on value, safety, progress, blockage, and topology-protection rules.

## 4. Repository Structure

Project tree:

```text
RSA-Guided Hybrid PINN/
├── RHP_Project/
│   ├── __init__.py
│   ├── main_bench.py
│   ├── overnight_benchmark_master.py
│   ├── collect_gating_data.py
│   ├── train_gating_model.py
│   ├── report_generator.py
│   ├── configs/
│   │   └── default.yaml
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── maze_2d.py
│   │   └── dynamic_env.py
│   ├── evaluator/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── plotter.py
│   ├── models/
│   │   └── gating_transformer.py
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── factored_nn.py
│   │   ├── physics_loss.py
│   │   └── rsa_engine.py
│   └── utils/
│       ├── __init__.py
│       └── sampler.py
├── checkpoints/
│   └── gating_transformer_best.pth
├── outputs/
│   ├── results.json
│   ├── temp_benchmark.json
│   ├── benchmark_metrics.png
│   ├── fields_paths_seed0.png
│   ├── fields_quiver_seed0.png
│   ├── final_report/
│   └── report_plots/
├── test_result/
│   └── date-based archived experiment outputs
└── PROJECT_STAGE_REPORT.md
```

## 5. Detailed Module Explanation

### 5.1 `RHP_Project/main_bench.py`

This is the main single-run experiment entry and the most important training file.

Main responsibilities:

- Load config
- Build environment and training grid
- Run RSA baseline
- Train vanilla PINN
- Train RHP-PINN
- Evaluate all methods
- Save plots and benchmark results

Important internal roles:

- `sample_path_tube(...)`
  Samples points around the RSA path and constructs lookahead directions. This is the geometric basis of path-tube supervision.

- `_train_rhp(...)`
  The core RHP-PINN training routine. It mixes:
  - global samples
  - path-tube samples
  - gateway forcing samples
  - gate-band samples

  It then optimizes a weighted combination of:
  - PDE loss
  - obstacle loss
  - path value anchor
  - path direction alignment
  - goal loss
  - RSA direction alignment
  - monotonicity loss
  - gate forcing loss
  - gate-band monotonicity
  - curl regularization
  - gate value anchor
  - gate drop loss
  - lookahead drop loss

This file currently embodies the stable training-side version of the topology-tunneling pipeline.

### 5.2 `RHP_Project/overnight_benchmark_master.py`

This file is the benchmark orchestrator.

Main responsibilities:

- Generate scenario sets for:
  - classic
  - random
  - rubble
- Build environments for each seed
- Call training and evaluation for RSA, vanilla PINN, and RHP-PINN
- Aggregate rollout metrics
- Save summary JSON and report figures

Why it matters:

- It turns the project from a prototype into a reproducible experiment platform.
- It is the file used for the current focused regression:
  `python3 -m RHP_Project.overnight_benchmark_master --device cpu --seeds 4,8`

### 5.3 `RHP_Project/envs/maze_2d.py`

This is the core environment-definition module.

Main responsibilities:

- Define bounded 2D worlds
- Represent obstacles with SDF
- Provide `sdf()` and `speed()` queries
- Generate several environment families:
  - U-maze
  - narrow passage
  - random circle maze
  - rubble field

Why it matters:

- All training losses, RSA propagation, and rollout safety checks depend on the SDF and speed interface defined here.

### 5.4 `RHP_Project/envs/dynamic_env.py`

This is a secondary environment module intended for more dynamic or extended setups.

Current importance:

- Lower than `maze_2d.py` in the current project phase.
- Useful as a structural placeholder for future environment extensions.

### 5.5 `RHP_Project/solvers/factored_nn.py`

This file defines the factored PINN model, typically named `FactoredTimeNN`.

Key idea:

- Instead of learning the full travel time directly, the model learns a smoother factor on top of source distance.

Why it matters:

- It addresses source singularity.
- It is one of the mathematically meaningful design improvements over a naive PINN.

### 5.6 `RHP_Project/solvers/physics_loss.py`

This is the main mathematical constraint file of the project.

Main components:

- `eikonal_residual(...)`
  Computes the residual of the Eikonal equation.

- `physics_loss(...)`
  Standard autograd-based PDE loss.

- `upwind_physics_loss(...)`
  A more stable upwind finite-difference style variant.

- `start_bc_loss(...)`
  Enforces the source boundary condition.

- `obstacle_loss(...)`
  Applies several obstacle-aware constraints:
  - interior penalty
  - boundary gradient penalty
  - boundary direction penalty
  - local vortex penalty

- `loss_gate_forcing(...)`
  Forces local motion near the gate to align with the desired crossing direction.

- `loss_monotonicity(...)`
  A key topology-tunneling loss. It explicitly enforces a minimum descending speed along the RSA target direction.

- `loss_curl(...)`
  Penalizes local rotation of the learned field near the gate.

This file encodes the physical and topological learning principles of the project.

### 5.7 `RHP_Project/solvers/rsa_engine.py`

This file provides the discrete planning prior.

Main responsibilities:

- Run the RSA propagation field
- Extract a reference path
- Identify the gateway segment
- Sample gate-band points and corresponding local directions

Important current functions:

- `extract_gateway_segment(...)`
  Finds the most topology-sensitive local path segment.

- `sample_gate_band(...)`
  Expands a point-like gate into a short direction-aligned capsule region for tunnel supervision.

This file is central to the project's "topology prior" mechanism.

### 5.8 `RHP_Project/evaluator/metrics.py`

This is the online rollout controller and metric computer.

It does far more than compute numbers.

Main responsibilities:

- Roll out paths using PINN gradients
- Build candidate next steps for RSA and PINN
- Perform collision checks, backtracking, and wall sliding
- Compare candidate values
- Decide whether to use PINN or RSA at each step
- Report metrics such as:
  - success
  - path length
  - optimality gap
  - gating ratio
  - safety margin

Current structure of the gating logic:

- Base value comparison
- RSA bonus adjustment under difficult local geometry
- Topology-protection state machine:
  - `gate_entry_protect`
  - `gate_recenter_protect`
  - `gate_mid_rescue`
  - `gate_late_rescue`
  - `gate_hard_rescue`
- Release logic:
  - `gate_release`
  - `hard_exit`
  - cooldown
  - soft exit

This file is currently the most sensitive file in the active debugging loop.

### 5.9 `RHP_Project/evaluator/plotter.py`

This is the visualization utility module.

Main outputs:

- field and path plots
- quiver plots for gradient behavior
- benchmark metric figures

Why it matters:

- It turns failure modes into visual evidence, which was essential for identifying gateway circulation in early experiments.

### 5.10 `RHP_Project/models/gating_transformer.py`

This file defines the Transformer-based gating model.

Role:

- Learn a value-based or sequence-based decision signal for online switching between PINN and RSA.

Importance:

- This is the core learned gating backbone.
- It supports the transition from rule-based gating to a learned value decision model.

### 5.11 `RHP_Project/collect_gating_data.py`

This file collects rollout data for gating model training.

Main responsibilities:

- run trajectories
- record temporal features
- assign value-style labels
- store failed and hard examples
- write compressed dataset files

It converts online switching behavior into a supervised sequence-learning problem.

### 5.12 `RHP_Project/train_gating_model.py`

This file trains the gating Transformer/VDM.

Main responsibilities:

- load dataset
- configure model
- train and validate
- save best checkpoint to `checkpoints/`

Current checkpoint path:

- `checkpoints/gating_transformer_best.pth`

### 5.13 `RHP_Project/report_generator.py`

This file reads benchmark outputs and generates polished report artifacts.

Main outputs:

- `outputs/report_plots/report_stats.json`
- group and method comparison figures
- gate heatmaps
- trajectory comparison figures

This is the reporting layer of the project.

### 5.14 `RHP_Project/utils/sampler.py`

This utility module contains helper routines for adaptive sampling.

Current role:

- Support sample reweighting and surface emphasis around difficult regions.

Even though it is a helper module, sampling quality has a direct effect on whether the PINN learns the gate region properly.

## 6. Data Flow and Call Flow

The main path for a single experiment is:

1. Load config from `configs/default.yaml`
2. Build a 2D environment from `envs/maze_2d.py`
3. Run RSA with `solvers/rsa_engine.py`
4. Extract gateway segment and gate-band samples
5. Train PINN with `main_bench.py`
6. Use `evaluator/metrics.py` to roll out RSA, vanilla PINN, and RHP-PINN
7. Save benchmark JSON and plots under `outputs/`

The learned gating path is:

1. Run `collect_gating_data.py`
2. Save dataset as compressed `npz`
3. Train model with `train_gating_model.py`
4. Save checkpoint to `checkpoints/`
5. Load the checkpoint inside `metrics.py` for online decision support

The benchmark/report path is:

1. Run `overnight_benchmark_master.py`
2. Save `outputs/temp_benchmark.json`
3. Generate plots and markdown with `report_generator.py`
4. Archive important outputs into `test_result/`

## 7. Current Output Files and Their Meanings

### 7.1 Live working outputs

- `outputs/temp_benchmark.json`
  The most important current working result file. It stores the latest benchmark summary and per-seed results.

- `outputs/results.json`
  Earlier benchmark result file used in previous phases.

- `outputs/benchmark_metrics.png`
  Aggregate benchmark plot.

- `outputs/fields_paths_seed0.png`
  Field and path visualization for seed 0.

- `outputs/fields_quiver_seed0.png`
  Quiver plot showing learned gradient behavior near obstacles and gateways.

### 7.2 Final report outputs

- `outputs/final_report/final_benchmark.json`
- `outputs/final_report/final_summary.md`
- `outputs/final_report/performance_matrix.png`
- `outputs/final_report/rubble_case_study.png`
- `outputs/final_report/overnight_master.log`

These are the "finalized" report-oriented experiment outputs.

### 7.3 Plot/report outputs

- `outputs/report_plots/report_stats.json`
- `outputs/report_plots/benchmark_group_summary.png`
- `outputs/report_plots/benchmark_method_comparison.png`
- `outputs/report_plots/gate_probability_heatmap.png`
- `outputs/report_plots/performance_bars.png`
- `outputs/report_plots/trajectory_comparison_seed0.png`

### 7.4 Archived outputs

- `test_result/`

This directory stores historical runs by date and run index. It is useful for tracking how the project evolved over time.

## 8. Current Stable Training-Side Design

The current stable training-side topology-tunneling pipeline includes:

- path-tube sampling
- gate-band sampling
- obstacle-aware PDE constraints
- gate forcing
- monotonicity on tube and gate band
- curl regularization near the gate
- gate value anchor
- gate drop loss
- path lookahead drop loss

What was tried but not kept:

- overly aggressive late-path extra supervision
- stronger asymmetric gate sampling that destabilized training

The current stable version is designed to keep `dot(-gradT, d_look)` positive and reasonably large while avoiding training collapse.

## 9. Current Stable Evaluation-Side Design

The evaluation logic has evolved into a layered hybrid controller.

The main ideas are:

- compare PINN and RSA candidate values
- bias toward RSA under local blockage, stall, or path deviation
- protect the gateway entry region
- apply short-term recentering after release
- rescue mid-drift states
- rescue late-drift states
- apply hard rescue in near-collision or extreme off-path conditions
- release RSA only when the PINN candidate is sufficiently safe and aligned

This structure has been necessary because a pure one-shot value comparison was not enough to survive difficult gateway transitions.

## 10. Progress Achieved So Far

The project has already achieved several substantial milestones:

1. Complete experiment pipeline:
   Environment generation, training, evaluation, reporting, and archiving are all integrated.

2. Clear baseline differentiation:
   - RSA is globally reliable.
   - Vanilla PINN fails consistently in non-convex scenarios.
   - RHP-PINN is meaningfully better than vanilla PINN.

3. Gateway problem localization:
   The project has moved from "the PINN cannot pass the gate at all" to "the planner mostly passes the gate but may drift in the post-gate region."

4. Topology-tunneling integration:
   Gateway region construction, monotonicity loss, curl loss, gate value anchor, and local drop constraints are all implemented and validated.

5. Learned gating support:
   The project has a complete data-collection and Transformer training pipeline for learned online switching.

6. Batch benchmark capability:
   The project can run grouped scenario benchmarks and produce report-ready outputs.

## 11. Latest Reliable Benchmark Snapshot

The latest reliable focused benchmark is stored in:

- `outputs/temp_benchmark.json`

Current focused regression seeds:

- `classic seed 4`
- `classic seed 8`

Current summary for `rhp_pinn` on this focused benchmark:

- success mean: `0.5`
- optimality gap mean: about `0.4041`
- gating ratio mean: about `0.6626`

Per-seed interpretation:

- `seed 8`
  Success has been restored, but RSA dependence is still high.

- `seed 4`
  Still fails in the post-gate region, even after multiple rescue-rule improvements.

This means the project is no longer blocked at the gateway entrance itself. The dominant remaining problem is the post-gate drift regime.

## 12. Failure Mode Analysis

The project has identified multiple failure modes over time.

### 12.1 Early gateway circulation

In earlier stages, the learned field created local rotational behavior around the gate.

Symptoms:

- path spins near the opening
- gradient field is not decisively descending through the gate

This motivated:

- gateway identification
- gate forcing
- monotonicity constraints
- curl regularization

### 12.2 Premature release

The controller may let go of RSA too early, before the path has truly stabilized after gate crossing.

Symptoms:

- the path appears safe locally
- then slowly drifts off the correct topological corridor

This motivated:

- gate entry protection
- release cooldown
- recenter protection

### 12.3 Mid-drift pseudo-safe regime

This is the current core failure mode.

Characteristics:

- the path has already left the gate
- `d_path` is increasing
- but `sdf` is not yet small enough to trigger obvious danger
- the controller sees a state that is neither clearly safe nor clearly catastrophic

This is where `seed 4` still fails most often.

### 12.4 High-takeover-yet-still-failing behavior

Some rollouts still fail even after raising RSA usage.

This reveals that the issue is not simply "RSA is used too little."
Instead, the release timing and the learned post-gate field shape are both involved.

## 13. Main Bottlenecks Right Now

The current bottlenecks are:

1. Post-gate field shaping is still not deep enough.
   The PINN often has the correct broad direction, but the descent funnel after the gate is not yet robust enough.

2. Evaluation logic is approaching complexity saturation.
   More conditions can rescue some trajectories, but too many rules risk turning the system into a brittle hard-coded controller.

3. Success rate and RSA ratio are tightly coupled.
   Increasing rescue often improves success but raises RSA usage.
   Reducing RSA usage often reintroduces failure on the hard classic seeds.

4. `seed 4` remains the hardest case.
   It is the current regression seed that most clearly exposes the unresolved post-gate drift problem.

## 14. What Has Been Tried and Rolled Back

The following directions were explored and later rolled back or deprioritized:

- globally aggressive PINN preference
- dynamic factors that over-suppressed RSA
- over-strong late-path supervision during training
- overly biased gate-band geometry that destabilized training
- several evaluation-only patches that lowered ratio but destroyed success

These experiments were still useful because they clarified the boundaries of the design space.

## 15. Current Working Conclusion

The project is now in a late-stage optimization phase rather than an early exploratory phase.

What is already established:

- the hybrid architecture works
- topology-tunneling mechanisms are implemented
- benchmark and reporting infrastructure are mature
- the dominant failure mode is clearly identified

What is not yet solved:

- stable post-gate convergence for the hardest classic seed while also keeping RSA usage low

So the project status is:

- architecturally mature
- experimentally reproducible
- scientifically meaningful
- but still missing the final balance between robustness and low takeover ratio

## 16. Recommended Next-Step Strategy

The current recommended strategy is:

1. Freeze the training-side stable baseline.
   Avoid introducing new training losses unless there is very clear evidence that one particular field region is systematically wrong.

2. Continue small, targeted edits in `evaluator/metrics.py`.
   Focus only on the post-gate drift and release timing problem.

3. Optimize in two stages:
   - first restore `seed 4 success = True`
   - then reduce gating ratio from that success-preserving baseline

4. Keep the current validation protocol:
   - smoke test with `python3 -m RHP_Project.main_bench`
   - regression test with `python3 -m RHP_Project.overnight_benchmark_master --device cpu --seeds 4,8`

5. After `seed 4,8` are both stable again, return to larger grouped benchmark validation.

## 17. Suggested Reading Order

If someone new joins the project, the best reading order is:

1. High-level experiment flow:
   - `RHP_Project/overnight_benchmark_master.py`
   - `RHP_Project/main_bench.py`

2. Mathematical core:
   - `RHP_Project/solvers/factored_nn.py`
   - `RHP_Project/solvers/physics_loss.py`
   - `RHP_Project/solvers/rsa_engine.py`

3. Online decision logic:
   - `RHP_Project/evaluator/metrics.py`

4. Learned gating branch:
   - `RHP_Project/collect_gating_data.py`
   - `RHP_Project/train_gating_model.py`
   - `RHP_Project/models/gating_transformer.py`

5. Result interpretation:
   - `outputs/temp_benchmark.json`
   - `outputs/final_report/final_benchmark.json`
   - `outputs/report_plots/report_stats.json`

## 18. Final Summary

This project has evolved into a structured hybrid planning system with:

- clear physical grounding
- explicit topology injection
- a learned continuous field
- an online hybrid controller
- reproducible benchmarking and reporting

The current challenge is no longer whether the architecture works.
It does.

The current challenge is whether the system can:

- preserve high success on the hardest non-convex cases,
- while reducing RSA reliance,
- without reintroducing post-gate drift.

That is the main focus of the current stage.
