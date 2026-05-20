# RHP-PINN：RSA 引导的混合物理信息神经网络路径规划框架

一种**混合路径规划框架**，将**离散 RSA（快速扫描算法）**与**连续 PINN（物理信息神经网络）**相结合，用于在**非凸障碍物 + 非均匀速度场**环境中求解**时间最优路径**。

> **核心矛盾**："几何最短" vs "时间最短" —— RHP-PINN 通过**PhysicsGuidedCoupling** 在推理时动态决定"这一步听 RSA（保证拓扑安全）还是听 PINN（利用物理信息抄近道）"。

---

## 核心思想

### 矛盾所在

- **RSA**：离散网格快速扫描 —— 保证拓扑正确，总能找到几何最短路径，但**无法利用速度场信息**（可能直穿慢速区）。
- **PINN**：学习连续到达时间场 T(x,y)，满足 Eikonal 方程 ||∇T|| = 1/V(x,y)。能自然弯曲走向快速区域，但**在非凸拓扑中完全失败**（梯度 rollout 会在 U 形陷阱中卡死）。

### 解决方案：RHP-PINN

一个**PhysicsGuidedCoupling** 在每步 rollout 中提取 3 个物理特征（PDE 残差、梯度冲突、安全风险），经 sigmoid 映射到 [0,1]，动态决策：

- **gating ≈ 1**：跟随 RSA（安全，拓扑保证）
- **gating ≈ 0**：跟随 PINN 梯度（物理最优）

最终效果：RHP-PINN 在非凸环境中 **SR=100%**，同时**利用速度场信息实现了优于 RSA 的时间代价**。

---

## 实验结果

### 消融实验（各 5 seeds，GPU）

| 测试 | 场景 | 障碍物 | 速度场 | RHP-PINN SR | RSA TimeCost | RHP-PINN TimeCost | 提升 | Gating |
|------|------|--------|--------|:-----------:|:------------:|:-----------------:|:----:|:------:|
| test1 | U 形陷阱 | 非凸 U 形墙 | 均匀 V=1.0 | **1.0** | 1.073 | 1.106 | -3.0% | 0.718 |
| test2 | 开放空间折射 | 无 | 上半 V=1.0 / 下半 V=0.3 | **1.0** | 2.684 | **0.883** | **+67.1%** | 0.049 |
| test3 | 异构 U 形陷阱 | 非凸 U 形墙 | 内部 V=0.3 / 外部 V=1.0 | **1.0** | 1.390 | **1.312** | **+5.6%** | 0.789 |

### 关键结论

1. **RSA 引导是必要的**：纯神经网络方法（P-NTFields-2D、Vanilla PINN）在非凸场景中 SR ≤ 40%
2. **RHP-PINN 自适应门控**：开放空间（简单场景）gating=0.05 → PINN 主导；非凸场景（复杂）gating=0.79 → RSA 兜底
3. **速度场利用有效**：开放空间折射场景中，RHP-PINN 比 RSA 的直线路径快 67%
4. **综合难题被攻克**：异构 U 形陷阱（非凸 + 速度场）中，RHP-PINN 达成 SR=100%，TimeCost 比 RSA 好 5.6%

---

## 项目结构

```
RHP-Hybrid-PINN-V20260429-main/
├── RHP_Project/                    # 核心算法（实验框架不动此目录）
│   ├── main_bench.py               # 主入口：训练 + 评估循环
│   ├── envs/
│   │   └── maze_2d.py              # 2D 环境工厂（SDF、障碍物、速度场）
│   ├── solvers/
│   │   ├── rsa_engine.py           # RSA：离散 Eikonal 求解器
│   │   ├── factored_nn.py          # FactoredTimeNN + VanillaTimeNN
│   │   ├── physics_loss.py         # Eikonal 方程损失函数
│   │   ├── pntfield_2d.py          # P-NTFields-2D 基线（傅里叶特征）
│   │   └── rrt_star.py             # RRT* 采样基线
│   ├── evaluator/
│   │   ├── metrics.py              # Rollout、门控决策、全部指标
│   │   └── plotter.py              # 场图 + 路径可视化
│   └── utils/
│       └── sampler.py              # 自适应采样工具
│
├── rhp_experiment/                 # 实验框架（后处理层）
│   ├── runner.py                   # ExperimentRunner：编排 + 自动记录
│   ├── config.py                   # ExperimentConfig：YAML 配置加载
│   ├── visualizer.py               # ExperimentVisualizer：后处理绘图（basic/paper）
│   ├── analyzer.py                 # ExperimentAnalyzer：跨实验对比分析
│   └── __init__.py
│
├── scripts/                        # CLI 入口
│   ├── run_benchmark.py            # 单实验运行器
│   ├── run_ablation.py             # 三测试消融实验
│   ├── analyze_results.py          # 跨实验分析
│
├── configs/                        # YAML 配置文件
│   ├── ablation_test1_topology.yaml    # U 形陷阱（均匀 V）
│   ├── ablation_test2_physics.yaml     # 开放空间折射
│   ├── trap_heterogeneous_vf.yaml      # 异构 U 形陷阱（综合测试）
│   ├── trap_u_shape.yaml              # U 形陷阱独立版
│   └── ...
│
├── experiments/                   # 自动记录的实验结果
│   ├── exp_20260506_185847_*      # Test 1：U 形陷阱
│   ├── exp_20260506_200720_*      # Test 2：开放空间折射
│   └── exp_20260506_205516_*      # Test 3：异构 U 形陷阱
│
├── requirements.txt
└── README.md
```

---

## 支持的场景

| 场景 | 障碍物 | 速度场 | 配置文件 |
|------|--------|--------|----------|
| **U 形陷阱** | 非凸 U 形墙 | 均匀 V=1.0 | `ablation_test1_topology.yaml` |
| **开放空间折射** | 无 | 上半 V=1.0 / 下半 V=0.3 | `ablation_test2_physics.yaml` |
| **异构 U 形陷阱** | 非凸 U 形墙 | 内部 V=0.3 / 外部 V=1.0（Sigmoid） | `trap_heterogeneous_vf.yaml` |
| 窄通道 | 两堵墙留缝隙 | 均匀 V=1.0 | `narrow_passage_centered.yaml` |

---

## 四种规划方法

| 方法 | 模型 | 网络结构 | RSA 引导 | 说明 |
|------|------|----------|:--------:|------|
| **RSA** | `RSAEngine` | 离散网格 | 自用 | 快速扫描网格求解器，几何参考 |
| **Vanilla PINN** | `VanillaTimeNN` | 4 层 MLP（tanh） | 否 | 纯 Eikonal 方程残差训练 |
| **P-NTFields-2D** | `PNTField2D` | 傅里叶特征 + 残差块 | 否 | NTFields 风格神经场基线 |
| **RHP-PINN** ⭐ | `FactoredTimeNN` | 因式分解 MLP + PhysicsGuidedCoupling | 是 | RSA 引导的混合 + 动态门控 |

### RHP-PINN 架构

```
T(x,y) = distance_to_start(x,y) × τ_network(x,y)
```

- **因式分解**：强制 T(start)=0 且 T>0 处处成立
- **训练损失**：
  - Eikonal 残差：||∇T|| - 1/V(x,y)
  - RSA 路径蒸馏：沿参考路径对齐 ∇T_pred 与 ∇T_rsa
  - 单调性约束：梯度必须指向目标方向
  - 障碍物边界约束：SDF 信息约束
  - 门控强制：在通道入口处对齐梯度方向
- **每步 rollout 的门控决策**：
  - 17 维特征：PINN 方向、RSA 方向、SDF、阻塞率、进度、目标对齐度、退出置信度等
  - Transformer 编码器处理 10 步历史序列
  - Sigmoid 输出：0 = 用 PINN，1 = 用 RSA
  - 多重覆盖门控：入口保护、停滞恢复、冷却机制

---

## 评估指标

| 指标 | 定义 | 含义 |
|------|------|------|
| **SR**（成功率） | 终点离目标在容忍范围内 | 越高越好 |
| **TimeCost**（时间代价） | ∫ ds / V(x,y) 沿路径积分 | 越低越好 |
| **GatingRatio**（门控比率） | 使用 RSA 的步数比例 | 非凸场景中 0.6~0.8 健康 |
| **PhysicalConsistency**（物理一致性） | 沿路径 mean(||∇T|| - 1/V) | 越低越符合物理 |
| **OptimalityGap**（最优性差距） | (L - L_ref) / L_ref | 负值 = 优于 RSA |
| **EfficiencyRatio**（效率比） | Length / TimeCost | 越高越好 = 更好地利用速度 |

---

## 使用说明

### 快速开始

```bash
# 单实验 + 基础可视化
python scripts/run_benchmark.py \
  --config configs/trap_heterogeneous_vf.yaml \
  --visualize-level basic

# ---

# 完整消融实验（3 测试 × 5 seeds，约 3 小时）
python scripts/run_ablation.py --seeds 5

# 带论文级可视化
python scripts/run_ablation.py --seeds 5 --visualize-level paper
```

### 可视化级别

```bash
# 无额外开销（默认）
python scripts/run_benchmark.py --config configs/trap_u_shape.yaml

# 5 张基础图（速度场、指标对比、门控分布、路径概览、最优轨迹）
python scripts/run_benchmark.py --config configs/trap_u_shape.yaml \
  --visualize-level basic

# 9 张论文级图（+ 路径对比、多种子轨迹、箱线图、门控散点图）
python scripts/run_benchmark.py --config configs/trap_u_shape.yaml \
  --visualize-level paper
```

### 对已完成实验后处理（无需重跑）

```python
from rhp_experiment.visualizer import ExperimentVisualizer

viz = ExperimentVisualizer("experiments/exp_20260506_205516_*")
viz.plot_all(level="paper")
```

### 跨实验分析

```bash
python scripts/analyze_results.py --output reports
```

---

## 输出目录结构

每次实验自动记录到 `experiments/exp_{时间戳}_{场景}_{标签}/`：

```
experiments/
└── exp_20260506_185847_trap_u_shape_gpu_v2_test1_topology/
    ├── config.yaml          # 使用的 YAML 配置快照
    ├── meta.json            # 实验元数据（日期、设备、种子数）
    ├── log.txt              # 完整控制台输出
    ├── metrics.json         # 聚合指标（SR、TimeCost、Gating 等）
    ├── seeds/               # 每种子指标
    │   ├── seed_0.json
    │   ├── seed_1.json
    │   └── ...
    └── plots/               # 生成的图片
        ├── velocity_field.png         # 速度场热力图
        ├── comparison_metrics.png     # RSA vs RHP 每种子柱状图
        ├── gating_ratios.png          # 门控比率分布
        ├── path_overview.png          # 双面板路径概览
        ├── best_trajectory.png        # 最优种子轨迹
        ├── path_compare.png           # RSA vs RHP 路径叠加    (paper级)
        ├── multi_paths.png            # 多种子轨迹分布         (paper级)
        ├── timecost_boxplot.png       # TimeCost 箱线图        (paper级)
        └── gating_vs_tc_scatter.png   # 门控与性能散点图       (paper级)
```

---

## 数据流

```
YAML 配置 → _make_env() → Maze2DEnv（障碍物 + SDF + 速度场）
                ↓
        RSAEngine.solve() → 离散 T 场 + 参考路径
                ↓
        训练 3 个模型：
          _train_vanilla_pinn()  -- Eikonal PINN（基线）
          _train_pntfield_2d()   -- NTFields 风格（基线）
          _train_rhp()           -- RHP-PINN 带 RSA 蒸馏
                ↓
        evaluate_methods() → 梯度 rollout + 门控决策
                ↓
        _summarize() → 跨种子聚合指标
                ↓
        自动归档到 experiments/exp_{时间戳}_{场景}/
```

---

## 门控决策流程

```
每步 rollout：
  1. 计算 PINN 方向（模型的 ∇T）
  2. 计算 RSA 方向（参考路径前视）
  3. compute_coupling_features() 提取 3 个物理特征：
     - Δ = ||∇T_pinn|| - 1/V(x)     （Eikonal 残差）
     - C = 1 - cos(v_pinn, v_rsa)    （梯度冲突）
     - S = exp(-k · SDF(x))          （安全风险）
  4. PhysicsGuidedCoupling → sigmoid(w·[Δ,C,S] + b) → alpha ∈ [0,1]
  5. 覆盖裁决链：
     - gate_entry_protect：狭窄通道强制 RSA（d_path < 0.03）
     - rsa_streak >= 15：强制尝试 PINN（冷却机制）
     - progress_stall_count >= 5：强制 RSA 救援
     - vf_autonomy：速度场收益感知，倾向 PINN
  6. blend_directions(v_pinn, v_rsa, alpha) → v_out
```

---

## Benchmark Suite

RHP-PINN 附带一个标准化的 benchmark 框架，用于与 **A***、**FMM** 等基线方法进行比较。

### 快速开始（无需下载数据）

```bash
# A* vs FMM on 5 procedural topology scenes
python scripts/run_dataset_benchmark.py \
  --suite topology \
  --scenes 5 \
  --planners astar,fmm
```

### 全量比较（含 PINN 方法）

```bash
# Requires training per scene (takes longer)
python scripts/run_dataset_benchmark.py \
  --suite topology \
  --scenes 5 \
  --planners astar,fmm,rhp_pinn,vanilla_pinn,pntfield
```

### 使用 MovingAI 公开数据集

```bash
# 1. 下载数据
python scripts/run_dataset_benchmark.py \
  --download maze512 \
  --data-dir ./movingai_data

# 2. 运行 benchmark
python scripts/run_dataset_benchmark.py \
  --suite movingai \
  --map maze512 \
  --scenes 50 \
  --planners astar,fmm,rhp_pinn
```

### 支持的 Planner

| Planner | 方法 | 是否需要训练 |
|---------|------|:----------:|
| `astar` | A* 搜索（8 连通，octile 启发式） | 否 |
| `fmm` | Fast Marching Method（RSAEngine） | 否 |
| `rhp_pinn` | RHP-PINN（FactoredTimeNN + 耦合） | 是 |
| `vanilla_pinn` | 纯 MLP Eikonal 求解 | 是 |
| `pntfield` | P-NTFields（Fourier 特征 + 残差块） | 是 |

### 输出格式

每次运行自动创建目录 `benchmark_results/bench_{时间戳}_{场景数}planners_{标签}/`：
- `snapshot.json` — 实验配置快照（git hash、种子、planner 参数）
- `metrics.json` — 聚合指标（SR、PL、TC、IT、OG 等）
- `{planner}_per_scene.csv` — 每个 planner 每场景的详细结果
- `benchmark_bars.png` — 指标对比柱状图
- `scatter_sr_vs_tc.png` — SR vs TimeCost 散点图

---

## 依赖

- Python 3.9+
- PyTorch ≥ 2.0
- NumPy
- PyYAML
- Matplotlib
- SciPy

```bash
pip install -r requirements.txt
```

---

## 参考文献

- **Eikonal 方程**：`||∇T(x)|| = 1/V(x)` —— 波前传播的到达时间场
- **RSA（快速扫描算法）**：Zhao, "A fast sweeping method for Eikonal equations", Math. Comp. 2005
- **PINN**：Raissi, Perdikaris, Karniadakis, "Physics-informed neural networks", J. Comp. Phys. 2019
- **NTFields**：Zang et al., "Neural Fields for Robotic Motion Planning", IROS 2023
- **因式分解时间表示**：T(x) = ||x - start|| × τ(x)，用于嵌入边界条件
