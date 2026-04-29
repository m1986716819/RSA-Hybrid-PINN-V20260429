# RSA-Guided Hybrid PINN (RHP) — 路径规划框架

## 项目概述

本项目实现了一种 **RSA 引导的混合物理信息神经网络（RHP-PINN）** 路径规划方法，用于在非凸障碍物 + 非均匀速度场环境中求解**时间最优路径**。

核心思想：**RSA 提供拓扑全局引导 + Eikonal PINN 提供连续物理建模**。

### 支持的测试场景

| 场景 | 障碍物 | 速度场 | 配置文件名 |
|------|--------|--------|-----------|
| **U 形迷宫 (u_maze)** | U 形墙 | 均匀 V=1.0 | `default.yaml` |
| **窄通道 (narrow_passage)** | 上下两堵墙留缝隙 | 均匀 V=1.0 | `narrow_passage_centered.yaml` |
| **U 形陷阱 (trap_u_shape)** | 开口向左的 U 形墙 | 均匀 V=1.0 | `trap_u_shape.yaml` |
| **异构 U 形陷阱 (heterogeneous)** | U 形墙 | 内部 V=0.3(Sigmoid) / 外部 V=1.0 | `trap_heterogeneous_vf.yaml` |
| **开放空间折射 (open_space)** | 无 | 上半 V=1.0 / 下半 V=0.3 | `ablation_test2_physics.yaml` |

---

## 项目结构

```
RHP_Project/
├── main_bench.py              # 主入口：训练 + 评估 + 存档
├── overnight_benchmark_master.py  # 批处理多场景基准测试
├── collect_gating_data.py      # 数据采集：训练 gating transformer
├── train_gating_model.py       # 训练 gating transformer 模型
├── report_generator.py         # 单次报告生成 + 可视化
│
├── envs/
│   └── maze_2d.py              # 2D 环境：SDF、障碍物、速度场
│
├── solvers/
│   ├── rsa_engine.py           # RSA：快速扫描网格传播
│   ├── rrt_star.py             # RRT*：随机采样基线
│   ├── physics_loss.py         # Eikonal 方程损失函数
│   ├── factored_nn.py          # FactoredTimeNN + VanillaTimeNN
│   └── pntfield_2d.py          # P-NTFields-2D 基线网络
│
├── evaluator/
│   ├── metrics.py              # Rollout 集成 + 评估指标
│   └── plotter.py              # 可视化：路径 + 场图 + 箱线图
│
├── models/
│   └── gating_transformer.py   # Transformer 门控决策模型
│
├── utils/
│   └── sampler.py              # 自适应采样
│
└── configs/                    # 配置文件
    ├── default.yaml
    ├── narrow_passage_centered.yaml
    ├── trap_u_shape.yaml
    ├── trap_heterogeneous_vf.yaml
    ├── velocity_field_refraction.yaml
    ├── ablation_test1_topology.yaml
    ├── ablation_test2_physics.yaml
    └── ...

# 顶层工具脚本
run_ablation.py                 # 自动化消融实验 (3 tests × 5 methods)
run_compare.py                  # 一键对比实验

# checkpoints/ 目录
# 包含预训练的 gating_transformer_best.pth (门控决策模型权重)
```

---

## 四种规划方法

### 1️⃣ RSA (快速扫描网格传播)

**文件**：`solvers/rsa_engine.py`

用 8 连通 Dijkstra 在离散网格上求解 Eikonal 方程，属于平稳的高精度参考解。

**特性**：
- 网格分辨率：80×80（粗糙）或 220×220（参考）
- 8-connectivity 保证各向同性近似
- 输出 `RSAResult`：包括 T 场、反向指针、网格坐标
- `backtrack_path_xy()` 从目标回溯到起点

**用途**：RSA 参考路径 ≈ 几何最优路径，也是 RHP-PINN 的**引导信号源**。

---

### 2️⃣ Vanilla PINN（纯物理信息神经网络）

**模型**：`VanillaTimeNN` in `solvers/factored_nn.py`

标准的全连接网络（MLP），4 层 128 维 + tanh 激活 + Softplus 输出。

**训练目标**：
- `Eikonal loss`: `|∇T| = 1/V`（物理约束）
- `Boundary condition`: `T(goal) = 0`
- `Obstacle loss`: 障碍物内部 T=0 + 边界梯度约束

**训练策略** (Curriculum Learning)：
- **Phase 1**（前 30%）：只训练边界条件（`λ_phys=0`），让模型先学会 T(goal)=0
- **Phase 2**（30%-70%）：线性增加物理损失权重
- **Phase 3**（后 30%）：`λ_phys=2.0` 全物理约束

**已知限制**：在非凸拓扑（U 形障碍）中失败率较高。

---

### 3️⃣ P-NTFields-2D（神经时序场基线）

**模型**：`PNTField2D` in `solvers/pntfield_2d.py`

基于傅里叶特征的残差网络：
```
输入(x,y) → Fourier编码 → Linear → ×4 ResidualBlock → Linear → 输出T
```

**特性**：
- 傅里叶特征：32 维，scale=6.0（高频先验）
- 无 RSA 引导，纯 rollout
- 使用 rollout 稳定器（path-tube lookahead）
- 在速度场场景中 TimeCost 表现最优（因自然弯曲走向快速区）

**训练增强**：
- Warmup（欧几里得距离监督）
- 渐进式物理权重
- 梯度剪切、防平坦、几何排序、可逆性约束等

---

### 4️⃣ RHP-PINN（RSA 引导的混合 PINN）⭐

**模型**：`FactoredTimeNN` in `solvers/factored_nn.py`

```
T(x,y) = distance_to_start(x,y) × τ_network(x,y)
```

**训练增强**（在 vanilla PINN 基础上增加）：
- **路径锚定损失**：RSA 路径附近强制梯度方向对齐
- **门控强制损失**：在关键通道口强制梯度指向目标
- **单调性损失**：沿路径方向确保 T 值单调递减
- **Wavefront 自适应采样**：T 变化剧烈区域多采样

**推理阶段**：`evaluator/metrics.py` 中的 `integrate_path_by_grad()`

使用**门控 Transformer** 做实时决策：
1. 每步同时推算 PINN 方向和 RSA 方向
2. Transformer 根据 14 维特征向量打分（SDF、blockage、stall 等）
3. `gate_entry_protect`：在狭窄/贴壁时强制回退 RSA
4. 15 步冷却机制：连续 15 步用 RSA 后强制尝试 PINN
5. 速度场自主感知：检测非均匀速度场时，优先选择指向快速区的 PINN 方向

---

## 评估指标

**文件**：`evaluator/metrics.py`

| 指标 | 定义 | 含义 |
|------|------|------|
| SR (Success Rate) | 路径终点到目标点距离 < 0.05 | 成功率 |
| Path Length | 路径点的累计欧几里得距离 | 几何最短程度 |
| Time Cost | `∫ ds / V(x,y)` | 实际时间代价 |
| Efficiency Ratio | Length / TimeCost | 有效速度，越高越好 |
| Optimality Gap | `(L - L_ref) / L_ref` | 与 RSA 参考路径的偏差 |
| Smoothness | 路径转角绝对值的均值 | 路径平滑度 |
| Physical Consistency | 沿路径 `mean(\|∇T\| - 1/V)` | Eikonal 方程残差 |
| Curvature Sharpness | 加速度 90% 分位数 | 路径尖锐度 |
| Gating Ratio | 使用 RSA 的步数占比 | RHP-PINN 自主性 |
| GradNorm Start | 起点附近梯度模的均值 | 初始收敛质量 |

---

## 运行方式

### 基础用法

```bash
# 运行一次基准测试（30 seeds × 4 种方法）
python3 -m RHP_Project.main_bench --config RHP_Project/configs/trap_heterogeneous_vf.yaml

# 自定义种子数
# 在 config 中加 num_seeds: 5
```

### 一键对比

```bash
# 单场景对比（5 seeds）
python3 run_compare.py --scenario trap_u_shape --seeds 5

# 可选的场景：trap_u_shape / narrow_passage / u_maze
```

### 消融实验

```bash
# 完整消融实验（3 tests × 5 methods，耗时较长）
python3 run_ablation.py --seeds 5

# 只跑 RRT*（跳过热门的 PINN 训练）
python3 run_ablation.py --seeds 5 --skip-pinn

# 快速验证（2 seeds + 2000 RRT 迭代）
python3 run_ablation.py --seeds 2 --rrt-iterations 2000
```

---

## 配置系统

所以配置通过 YAML 文件管理，核心字段：

```yaml
seed: 0                # 随机种子
device: cpu            # 计算设备 (cpu/mps/cuda)
num_seeds: 30          # 独立实验重复次数

env:                   # 环境配置
  name: trap_heterogeneous_vf
  bounds: {x_min: -0.05, x_max: 1.05, ...}
  start: [0.1, 0.5]
  goal: [0.9, 0.5]
  velocity_field:      # 可选：非均匀速度场
    enabled: true
    type: half_space
    v_upper: 1.0
    v_lower: 0.3
    boundary_y: 0.5

rsa:                   # RSA 求解器参数
  grid_size: [80, 80]
  connectivity: 8

train:                 # 训练参数
  max_steps: 3000
  warmup_steps: 500
  lr: 1.0e-3
  lambda_phys: 1.0
  lambda_bc: 5.0
  curriculum:          # 课程学习调度
    phys_ramp_start: 0.3
    phys_ramp_end: 0.7
    phys_hold: 2.0

model:                 # 网络结构
  hidden_dim: 128
  num_layers: 4
  activation: tanh

eval:                  # 评估参数
  path_step: 0.02
  max_path_steps: 600
  goal_tol: 0.05
  out_dir: outputs
```

---

## 全流程数据流

```
配置文件 → _make_env() → 创建 Maze2DEnv（障碍物 + SDF + 速度场）
                ↓
        RSAEngine.solve() → 离散网格 T 场 + 参考路径
                ↓
      ┌─────────┼─────────┐
      ↓         ↓         ↓
  _train_    _train_   _train_
  vanilla_   pntfield  rhp
  pinn       2d        pinn
      ↓         ↓         ↓
  evaluate_methods() → 每步沿梯度 rollout
      ↓         ↓         ↓
      └─────────┼─────────┘
                ↓
        _summarize() → 聚合指标(30 seeds)
                ↓
        保存 results.json + 可视化
                ↓
        _archive_outputs() → 归档到 test_result/
```

---

## 关键文件函数索引

| 函数 | 文件 | 行号 | 作用 |
|------|------|------|------|
| `run()` | `main_bench.py` | ~1267 | 主循环：训练+评估+存档 |
| `_train_vanilla_pinn()` | `main_bench.py` | ~862 | 基础 Eikonal PINN 训练 |
| `_train_pntfield_2d()` | `main_bench.py` | ~1020 | P-NTFields 训练 |
| `_train_rhp()` | `main_bench.py` | ~404 | RHP-PINN 训练 |
| `_make_env()` | `main_bench.py` | ~118 | 场景工厂 |
| `evaluate_methods()` | `metrics.py` | ~969 | 全部方法的 rollout 评估 |
| `integrate_path_by_grad()` | `metrics.py` | ~259 | PINN 梯度 Rollout + Gating |
| `eikonal_residual()` | `physics_loss.py` | ~12 | Eikonal 方程残差计算 |
| `RSAEngine.solve()` | `rsa_engine.py` | ~130 | 离散网格 Eikonal 求解 |
| `rrt_star_plan()` | `rrt_star.py` | ~50 | RRT* 采样规划 |
| `path_time_cost()` | `metrics.py` | ~216 | 时间代价后处理 |
| `path_eikonal_residual()` | `metrics.py` | ~230 | 物理一致性指标 |

---

## 预期的实验结果模式

```
均匀环境 (U-shape, V=1.0):
  RSA:           SR=1.0, TimeCost≈1.07,   几何最优
  RHP-PINN:      SR=1.0, TimeCost≈1.01,   优于 RSA
  P-NTFields-2D: SR=0.0, TimeCost≈4.35,   非凸拓扑失败
  Vanilla PINN:  SR≈0.2, TimeCost≈9.87,   大部分失败

异构速度场 (U-shape, V内=0.3/V外=1.0):
  RSA:           SR=1.0, TimeCost≈3.40    几何最短但穿慢区
  RHP-PINN:      SR≈1.0, TimeCost≈2.50    绕行快速区
  P-NTFields-2D: SR=0.0                   非凸拓扑失败
  Vanilla PINN:  SR=0.0

开放空间折射 (无障碍物, V上=1.0/V下=0.3):
  RRT*:          SR=1.0, TimeCost≈0.78    采样绕行快速区
  RSA:           SR=1.0, TimeCost≈1.01    走直线穿慢区
  RHP-PINN:      SR=1.0, TimeCost≈0.85    比RSA好但不如RRT*
```

---

## 依赖

- Python 3.9+
- PyTorch ≥ 1.13
- NumPy
- PyYAML
- Matplotlib（可视化）

```bash
pip install torch numpy pyyaml matplotlib
```
