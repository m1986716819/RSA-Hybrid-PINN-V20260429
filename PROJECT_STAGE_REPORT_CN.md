# RHP-PINN 项目阶段性说明文档

## 1. 项目定位

本项目聚焦于复杂非凸环境中的路径规划问题，核心方法是构建一个 `RSA-guided Hybrid PINN`，即：

- 用离散拓扑规划器 `RSA` 提供全局可达性和绕障拓扑先验；
- 用 `Factored PINN` 学习连续、可微、物理一致的旅行时间场；
- 在在线 rollout 阶段通过混合门控策略决定当前一步应当跟随 PINN 梯度，还是临时回退到 RSA。

因此，这个项目不是单纯的神经网络路径规划，也不是传统图搜索算法，而是一个“离散拓扑先验 + 连续物理神经场 + 在线混合决策”的混合系统。

当前阶段的研究重点已经非常明确：

- 在 U 型墙、窄通道、非凸障碍等困难场景中保持较高成功率；
- 同时尽可能降低 `RSA` 的接管比例；
- 避免路径在 gateway 处或穿门后落入局部环流、伪安全漂移、极限环等失败模式。

## 2. 科学问题与研究动机

项目的出发点，是传统 PINN 在 Eikonal 路径规划中存在三个典型弱点：

### 2.1 非凸陷阱

在 U 型障碍、窄通道、迷宫式结构中，纯 PINN 学到的连续势场往往会出现局部极小、错误吸引盆地或者环流，导致路径在开口附近“打转”，无法穿过真正的拓扑通道。

### 2.2 点源奇异性

旅行时间场 `T(x)` 在起点附近天然带有奇异性，直接拟合完整时间场会导致梯度不稳定，训练难度明显上升。

### 2.3 在线控制脆弱

即便 PINN 学到了大致正确的全局场，在复杂障碍边界和 gateway 附近，局部 rollout 仍然可能因为微小误差走入错误通道。因此，单靠“神经网络一次性学对所有拓扑结构”通常不够稳。

针对这三个问题，项目采用了两个核心思想：

### 2.4 因子分解

项目使用因子分解形式表示旅行时间场：

`T(x) = ||x - x_start|| * tau(x)`

这样可以显式编码起点附近的几何结构，减弱源点奇异性，让网络主要学习更平滑的因子部分 `tau(x)`。

### 2.5 离散拓扑引导

项目不要求 PINN 从零发现正确拓扑，而是先由 `RSA` 提供全局参考路径和粗时间场，再通过训练损失与评估门控把这种拓扑信息注入到连续场中。

## 3. 总体方法架构

整个系统可以分成五层：

### 3.1 环境层

构建 2D SDF 环境，包括：

- 深 U 型墙
- 窄道场景
- 随机圆形障碍场
- rubble 碎石场

这些环境统一提供：

- `sdf(x)`：表示点到障碍边界的有符号距离
- `speed(x)`：表示局部传播速度

### 3.2 离散拓扑层

在网格上运行 `RSA`，得到：

- 全局粗旅行时间场
- 一条绕障可达的参考路径
- 最关键的 gateway 区域和穿越方向

这一步的目标不是生成最终平滑路径，而是给神经网络和在线控制器一个可靠的全局拓扑骨架。

### 3.3 连续神经场层

通过 `Factored PINN` 学习连续的时间场 `T(x)`，并通过物理损失约束其满足 Eikonal 方程：

`|grad T(x)| = 1 / f(x)`

同时结合障碍惩罚、边界方向惩罚、路径锚定、单调性损失等，迫使网络在关键位置形成合理势场。

### 3.4 拓扑隧道层

在 gateway 附近，不再只用单点约束，而是构造一个 gate-band / gate-tunnel 区域，对该区域施加：

- 方向一致性
- 单调性约束
- curl 约束
- 局部势差下降约束
- 局部值场锚定

这一层的目标是让 gateway 附近真正形成一条“严格下降通道”，而不是只有局部方向大致正确。

### 3.5 在线混合门控层

在路径积分时，每一步同时评估：

- PINN 候选下一步
- RSA 候选下一步

然后根据：

- value
- sdf
- blockage
- goal alignment
- exit confidence
- path progress
- 与参考路径偏差

等信息决定到底采用哪一个动作。

## 4. 项目目录结构

项目根目录如下：

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
│   └── 按日期归档的历史实验结果
├── PROJECT_STAGE_REPORT.md
└── PROJECT_STAGE_REPORT_CN.md
```

## 5. 各目录与文件的详细说明

### 5.1 `RHP_Project/`

这是项目的核心源码目录。几乎所有算法逻辑、训练流程、评估逻辑和报告生成都集中在这里。

它是整个项目真正的主工作区。

### 5.2 `RHP_Project/configs/default.yaml`

这是项目的统一配置入口。

主要配置内容通常包括：

- 环境边界和起终点
- RSA 参数
- 神经网络结构参数
- 训练超参数
- 评估参数
- 输出目录

可以把它理解为整个实验系统的“总控制台”。

### 5.3 `RHP_Project/envs/`

该目录负责环境定义。

#### `maze_2d.py`

这是当前最重要的环境文件，主要负责：

- 定义 2D 边界
- 用 SDF 表达障碍
- 提供 `sdf()` 与 `speed()` 查询接口
- 构建：
  - U 型墙环境
  - 窄道环境
  - 随机圆障碍环境
  - rubble 环境

它是整个训练、评估、RSA 和可视化的基础。

#### `dynamic_env.py`

这是为动态环境或未来扩展准备的模块。

在当前主线中使用频率不如 `maze_2d.py`，但它反映了项目未来向更一般环境扩展的结构预留。

### 5.4 `RHP_Project/solvers/`

这个目录存放真正的求解器和物理损失。

#### `factored_nn.py`

定义因子分解 PINN 模型 `FactoredTimeNN`。

核心作用：

- 不是直接拟合完整 `T(x)`；
- 而是通过因子分解缓解源点奇异性；
- 让网络学习更平滑、数值更稳定的因子部分。

这是项目相较普通 PINN 的关键数学结构之一。

#### `physics_loss.py`

这是项目中最核心的数学约束文件之一。

它主要包含：

- `eikonal_residual(...)`
  计算 Eikonal 残差。

- `physics_loss(...)`
  标准基于 autograd 的 PDE 物理损失。

- `upwind_physics_loss(...)`
  更稳健的上风格式近似。

- `start_bc_loss(...)`
  起点边界条件。

- `obstacle_loss(...)`
  障碍相关损失，包括：
  - 障碍内部惩罚
  - 边界梯度不足惩罚
  - 边界方向惩罚
  - 局部旋涡惩罚

- `loss_gate_forcing(...)`
  gate 区域基础方向强制项。

- `loss_monotonicity(...)`
  当前 topology tunneling 的核心损失之一，要求沿目标方向必须具有足够下降速度。

- `loss_curl(...)`
  对局部旋转结构进行抑制，防止在 gate 附近形成环流。

这个文件实际上承载了“物理一致性”和“拓扑一致性”两类约束。

#### `rsa_engine.py`

这是离散拓扑先验模块。

主要职责：

- 运行 RSA 传播
- 构建参考时间场
- 提取参考路径
- 找到 gateway 段
- 在 gate 周围生成训练采样点

当前关键函数包括：

- `extract_gateway_segment(...)`
  从参考路径中提取最关键的通道段。

- `sample_gate_band(...)`
  将原本点状的 gate 扩展成一个沿穿越方向拉伸的胶囊带区域，并返回对应方向。

这个文件是“离散规划如何指导连续神经场”的核心入口。

### 5.5 `RHP_Project/evaluator/`

这个目录负责评估、路径积分和可视化。

#### `metrics.py`

这是当前最活跃、最敏感、也是最复杂的文件。

它不只是计算指标，而是在线控制器本身。

主要职责包括：

- 根据 PINN 时间场计算梯度动作
- 构建 PINN 和 RSA 两类候选下一步
- 执行碰撞检测
- 执行 backtracking 和贴墙滑行
- 比较两个候选的价值
- 决定最终动作来源
- 统计：
  - success
  - path length
  - optimality gap
  - gating ratio
  - safety margin

当前内部已经形成一套分层状态机，包括：

- `gate_entry_protect`
- `gate_recenter_protect`
- `gate_mid_rescue`
- `gate_late_rescue`
- `gate_hard_rescue`
- `gate_release`
- `hard_exit`
- `cooldown`
- `soft_exit`

这个文件几乎决定了当前 active branch 的最终行为。

#### `plotter.py`

负责实验可视化。

典型输出包括：

- 时间场和路径图
- 梯度场 quiver 图
- benchmark 指标图

这个模块在早期帮助我们定位了 gateway 处的局部环流问题。

### 5.6 `RHP_Project/models/gating_transformer.py`

这个文件定义门控 Transformer 模型。

它的目标不是直接预测路径，而是学习“当前是否应该交给 RSA”的时序决策信号。

这是从启发式门控升级到学习式门控的基础。

### 5.7 `RHP_Project/collect_gating_data.py`

这个文件负责生成门控训练数据。

主要工作：

- 运行 rollout
- 记录时序特征
- 构造 value 或 future-cost 风格标签
- 收集困难样本、失败样本
- 输出压缩数据集

它的本质是把“启发式决策经验”转成可监督学习的数据。

### 5.8 `RHP_Project/train_gating_model.py`

这个文件负责训练门控 Transformer/VDM。

主要职责：

- 读取数据集
- 创建模型配置
- 进行训练和验证
- 将最优模型保存到 `checkpoints/`

当前默认最佳权重位置是：

- `checkpoints/gating_transformer_best.pth`

### 5.9 `RHP_Project/report_generator.py`

这是报告生成层。

主要职责：

- 读取 benchmark JSON
- 生成统计图表
- 输出结构化报告数据

它的存在使项目不仅能“跑实验”，还能系统地整理结果。

### 5.10 `RHP_Project/utils/sampler.py`

这是采样辅助模块。

虽然不是主入口，但非常重要，因为 PINN 在困难通道中的表现高度依赖采样策略。

它主要用于：

- 样本重加权
- 困难区域采样增强
- 边界附近采样强化

## 6. 整个项目的数据流与调用流

### 6.1 单次训练与评估主流程

单次实验的主流程如下：

1. 从 `default.yaml` 读取配置；
2. 在 `maze_2d.py` 中构建 2D 环境；
3. 用 `rsa_engine.py` 运行 RSA，得到参考时间场与参考路径；
4. 从 RSA 路径中提取 gateway，并构建 gate-band；
5. 在 `main_bench.py` 中训练 `Factored PINN`；
6. 在 `metrics.py` 中对 `RSA / vanilla PINN / RHP-PINN` 做 rollout；
7. 输出结果 JSON、可视化图片和 benchmark 图。

### 6.2 学习式门控主流程

如果走 learned gating 路线，则流程如下：

1. 运行 `collect_gating_data.py`；
2. 生成 `npz` 数据集；
3. 在 `train_gating_model.py` 中训练 Transformer；
4. 保存 checkpoint 到 `checkpoints/`；
5. 在 `metrics.py` 中加载模型，参与在线决策。

### 6.3 批量 benchmark 主流程

大规模 benchmark 流程如下：

1. 运行 `overnight_benchmark_master.py`；
2. 为每个 seed 构建场景；
3. 调用训练与评估；
4. 聚合所有方法结果；
5. 写出 `outputs/temp_benchmark.json`；
6. 调用 `report_generator.py` 生成图表和摘要。

## 7. 当前训练主链的详细说明

### 7.1 训练样本构成

`main_bench.py` 中的 `_train_rhp(...)` 会混合几类采样：

- 全局采样点
- 路径 tube 采样点
- gate forcing 点
- gate band 点

这种设计的意义在于：

- 全局点负责满足 PDE；
- tube 点负责学习沿正确路径下降；
- gate 点负责解决最难的拓扑瓶颈。

### 7.2 当前稳定保留的损失项

目前训练侧稳定保留的损失链路包括：

- PDE 物理损失
- 起点边界条件
- 障碍物损失
- 路径值锚定
- 路径方向对齐
- goal 约束
- RSA 方向对齐
- 单调性损失
- gate forcing
- gate-band monotonicity
- curl 约束
- gate value anchor
- gate drop
- lookahead drop

这些损失共同目标是：让模型在满足 Eikonal 方程的同时，在拓扑关键区形成稳定下降通路。

### 7.3 当前已否决的训练侧方向

近期还尝试过：

- 对路径后半段额外施加强监督；
- 更偏置的 gate 带采样；
- 更强的局部后段 drop 约束。

结果表明这些方向会让训练不稳定、`dot(-gradT, d_look)` 变差，已全部回退。

因此当前训练侧已经进入“稳定基线优先”的阶段。

## 8. 当前评估主链的详细说明

`metrics.py` 当前可以看作是一个在线控制器，而不仅是指标计算器。

### 8.1 基础决策逻辑

每一步会同时评估：

- PINN 候选动作
- RSA 候选动作

并计算：

- value
- sdf
- blockage
- exit confidence
- goal alignment
- 与参考路径距离

### 8.2 当前分层控制结构

在 value 比较基础上，又叠加了：

- `rsa_bonus`
  在障碍复杂、进展停滞、偏离路径等情形下提高 RSA 的相对吸引力。

- `gate_entry_protect`
  在入口附近尽量不允许过早释放 RSA。

- `gate_recenter_protect`
  在刚离开 gate 后短时间内，若出现轻微偏离，就允许用 RSA 做重心纠偏。

- `gate_mid_rescue`
  针对当前最典型的“中漂移伪安全区”问题。

- `gate_late_rescue`
  在后段偏航但尚未彻底撞障时做补救。

- `gate_hard_rescue`
  在接近极端失稳或明显错误时强制回退 RSA。

- `gate_release`
  只有当 PINN 已足够安全、足够对齐、足够居中时才允许真正 release。

- `hard_exit / cooldown / soft_exit`
  用于避免频繁来回抖动。

这也解释了为什么这个文件会越来越长，因为它实际承载的是“局部策略系统”。

## 9. 结果输出与文件说明

### 9.1 `outputs/`

这是当前工作结果目录。

主要文件包括：

- `outputs/results.json`
  较早阶段的 benchmark 结果文件。

- `outputs/temp_benchmark.json`
  当前最重要的最新工作结果文件。

- `outputs/benchmark_metrics.png`
  benchmark 指标汇总图。

- `outputs/fields_paths_seed0.png`
  seed0 的场和路径图。

- `outputs/fields_quiver_seed0.png`
  seed0 的梯度场 quiver 图。

### 9.2 `outputs/final_report/`

这是最终汇报风格的结果目录，主要包括：

- `final_benchmark.json`
- `final_summary.md`
- `performance_matrix.png`
- `rubble_case_study.png`
- `overnight_master.log`

### 9.3 `outputs/report_plots/`

这是报告图表目录，主要包括：

- `report_stats.json`
- `benchmark_group_summary.png`
- `benchmark_method_comparison.png`
- `gate_probability_heatmap.png`
- `performance_bars.png`
- `trajectory_comparison_seed0.png`

### 9.4 `test_result/`

这是历史归档目录。

特点：

- 按日期管理
- 每日可有多个 run
- 每个 run 包含 benchmark、field visualization、report 三类输出

这个目录对于复盘实验演化过程非常重要。

## 10. 当前项目取得的主要进展

到目前为止，项目已经取得了比较系统的进展：

### 10.1 工程链路完整

已经具备：

- 环境生成
- 单次训练
- 在线 rollout
- 大规模 benchmark
- 可视化
- 报告生成
- 历史归档

这意味着项目已经从“算法原型”进化为“完整实验平台”。

### 10.2 主方法已被证明有效

我们已经明确看到：

- `RSA` 能稳定成功；
- `Vanilla PINN` 在复杂非凸场景下稳定失败；
- `RHP-PINN` 显著优于 `Vanilla PINN`。

这证明“离散拓扑先验 + PINN”的大方向是成立的。

### 10.3 失败模式已被清晰定位

项目早期的主要失败是：

- gateway 局部环流

而当前阶段的主要失败已经收缩为：

- 穿门后中后段漂移

这说明问题已经被不断缩小，项目正在进入后期优化阶段。

### 10.4 topology tunneling 链路已经成型

当前与 gateway 相关的训练链路已经不是简单方向对齐，而是包括：

- gate-band 构造
- monotonicity
- curl
- gate value anchor
- gate drop
- lookahead drop

这是一套完整的拓扑隧道塑形机制。

### 10.5 学习式门控链路已经打通

项目不只保留启发式规则，还已经完成：

- gating 数据采集
- Transformer/VDM 模型定义
- 模型训练与 checkpoint 保存
- metrics 中的在线价值比较

这为后续把部分规则进一步学习化奠定了基础。

## 11. 当前最新结果

当前最新可信结果保存在：

- `outputs/temp_benchmark.json`

当前最聚焦的回归对象是：

- `classic seed 4`
- `classic seed 8`

### 11.1 当前汇总结果

`rhp_pinn` 当前聚焦回归汇总约为：

- `success mean = 0.5`
- `optimality_gap mean ≈ 0.4041`
- `gating_ratio mean ≈ 0.6626`

### 11.2 当前 seed 级别结果

- `seed 8`
  已经成功，但对 RSA 的依赖仍偏高。

- `seed 4`
  仍然失败，且主要失败发生在穿过 gate 之后的中后段漂移阶段。

这说明项目当前不是“完全不会过门”，而是“还没有把 gate 后收尾阶段学稳”。

## 12. 当前最核心的失败模式

项目到目前为止，已经识别出几类关键失败模式：

### 12.1 gateway 环流

这是早期最典型问题。

表现为：

- 路径在开口处打转；
- 梯度方向局部旋转；
- 虽然看似平滑，但始终无法真正穿过通道。

这推动了：

- gateway identification
- gate forcing
- monotonicity
- curl regularization

### 12.2 过早 release

在 gate 还没完全稳定通过前，系统过早把控制权交回 PINN，随后轨迹逐渐偏离。

这推动了：

- gate_entry_protect
- release cooldown
- recenter protect

### 12.3 中漂移伪安全区

这是当前最核心的问题。

表现为：

- 路径已经离开 gate；
- `d_path` 开始持续增大；
- 但 `sdf` 又不够小，没被识别为真正危险；
- 门控系统因此很难判断是否立即切回 RSA。

当前 `seed 4` 的主要问题就落在这里。

### 12.4 高接管仍失败

有些轨迹即使提高了 RSA 使用比例仍然失败，说明问题不只是“RSA 用太少”，而是：

- release 时机仍然不对；
- 或者 gate 后势场本身仍不够稳定。

## 13. 当前真正的技术卡点

从目前所有实验来看，真正的瓶颈有四个：

### 13.1 gate 后漏斗深度不足

PINN 已经能学到大方向，但穿门后那段吸引盆地还不够深，无法稳定把轨迹拉向正确的后续走廊。

### 13.2 评估层规则接近复杂度上限

`metrics.py` 中已经堆叠了很多保护和救援逻辑。继续加规则虽然可能救某些 seed，但系统会越来越像“硬编码控制器”，泛化风险更高。

### 13.3 成功率与接管率高度耦合

- 强救援可以提高成功率，但会抬高 `RSA ratio`
- 弱救援可以降低 `RSA ratio`，但会让硬案例再次失败

因此现在的难点不是单指标优化，而是双目标平衡。

### 13.4 `seed 4` 仍是最难 case

它当前最能暴露系统剩余问题，因此后续仍应优先围绕它做回归。

## 14. 已尝试但已回退的方向

以下方向已经尝试过，但目前不再作为主方向：

- 全局更激进的 PINN 偏好
- 过度压低 RSA takeover 的动态 factor
- 更重的后半段路径监督
- 更偏置的 gate-band 几何采样
- 只靠评估层补丁继续堆条件

这些尝试虽然没有成为最终方案，但它们帮助我们明确了设计边界：

- 训练侧不能再随意加重约束；
- 评估侧不能无限堆规则；
- 必须围绕真实失败模式做小而准的调整。

## 15. 当前阶段的总体判断

项目现在已经不属于“架构未成型”的早期探索阶段，而是典型的后期优化阶段。

已经明确成立的部分：

- 混合架构是有效的；
- topology tunneling 是必要的；
- benchmark 与报告链是成熟的；
- 失败模式已经被清晰定位。

尚未完全解决的部分：

- 如何在最难 classic case 上同时做到：
  - `success = True`
  - 较低的 `RSA ratio`

因此当前项目状态可以概括为：

- 结构成熟
- 实验可复现
- 方法有效
- 但尚未完成最后的鲁棒性与效率平衡

## 16. 下一步工作计划

当前最合理的策略是：

### 16.1 冻结训练侧稳定基线

除非发现明确证据，不再轻易新增训练损失或改动训练结构。

原因：

- 当前训练侧稳定版本已经能维持较好的 `dot(-gradT, d_look)`；
- 最近追加的后半段强化实验已证明容易破坏稳定性。

### 16.2 继续精修评估层中后段逻辑

后续主要围绕 `metrics.py` 做更窄、更小粒度的修正，专门针对：

- gate 后中漂移
- release 时机不稳
- PINN/RSA 来回抖动

### 16.3 调整优化顺序

下一阶段不再同时大幅推动两个目标，而是分两步：

1. 先把 `seed 4 success=True` 拉回来；
2. 再从这个成功基线继续压 `gating_ratio`。

### 16.4 保持双层验证流程

每次修改后都继续执行：

- `python3 -m RHP_Project.main_bench`
  做烟测，观察局部行为和方向性指标。

- `python3 -m RHP_Project.overnight_benchmark_master --device cpu --seeds 4,8`
  做正式定点回归。

### 16.5 回到大规模 benchmark

一旦 `seed 4/8` 两个目标 seed 都稳定成功，再回到更大规模 benchmark 上验证泛化性。

## 17. 推荐阅读顺序

如果要系统理解这个项目，推荐按以下顺序阅读：

### 第一层：看全局实验主线

- `RHP_Project/overnight_benchmark_master.py`
- `RHP_Project/main_bench.py`

### 第二层：看数学与训练核心

- `RHP_Project/solvers/factored_nn.py`
- `RHP_Project/solvers/physics_loss.py`
- `RHP_Project/solvers/rsa_engine.py`

### 第三层：看在线决策逻辑

- `RHP_Project/evaluator/metrics.py`

### 第四层：看学习式门控链路

- `RHP_Project/collect_gating_data.py`
- `RHP_Project/train_gating_model.py`
- `RHP_Project/models/gating_transformer.py`

### 第五层：看结果与报告

- `outputs/temp_benchmark.json`
- `outputs/final_report/final_benchmark.json`
- `outputs/report_plots/report_stats.json`

## 18. 总结

这个项目已经发展成一个完整、成体系的混合路径规划研究平台。

它的核心特征是：

- 以离散拓扑先验保证全局正确性；
- 以物理约束神经场保证连续性和可微性；
- 以在线混合门控保证 rollout 阶段的鲁棒性。

项目当前最重要的科学结论不是某一个单独数字，而是：

- 我们已经把问题从“纯 PINN 无法穿越复杂 gate”推进到了“只剩穿门后中后段漂移需要解决”；
- 我们已经明确知道剩余瓶颈在哪个阶段、哪种状态、哪类控制逻辑下出现；
- 我们已经具备完整的工程链路去持续、可复现地优化这一问题。

因此，本项目当前的阶段性评价是：

- 架构正确；
- 工程成熟；
- 方法有效；
- 剩余问题明确；
- 适合进入以定点回归和小步收敛为主的后期优化阶段。
