# D3QN 低暴露路径规划 (D3QN Low-Exposure Path Planning)

## 1. 项目概述

本项目解决**敌方视野约束下的双目标路径规划问题**：在 32×32 三维高度场地图上，智能体需要从起点走到终点，同时优化：

- **路径长度**：步数尽量少
- **隐蔽性**：暴露在敌人视野内的步数尽量少

智能体使用 **D3QN（Double DQN + Dueling Network）** 深度强化学习训练，以 Visibility-A\* 和 Pareto A\* 等经典规划器作为基线对比。

---

## 2. 环境与场景

核心环境代码：[env/battlefield_env.py](env/battlefield_env.py)

### 2.1 地图与移动

- 地图尺寸：`32×32`，每个格子具有 `0 ~ height_levels` 的地形高度。
- 智能体在二维网格上做 **8 邻域离散移动**（上下左右 + 四个对角）。
- 上坡约束：仅当 `to_height - from_height <= agent_max_climb_height` 时可进入目标格。
- 敌人所在格子为硬障碍，不可通行。

### 2.2 随机场景生成

- **地形**：高度场 + 稀疏障碍凸起（伯努利采样，`obstacle_probability = 0.06`），形成可爬坡但非平坦地形。
- **敌人瞭望点优化**：在指定区域条带内搜索使可见面积最大的敌人位置与朝向。
  - 两步筛选：粗采样 → 精评估（含完整 3D 遮挡射线）。
  - 朝向离散化为 `heading_bins` 个角度（默认 12，即每 30° 一个方向）。
- **敌人区域**：限制在地图一侧的 `8×32` 条带内，可通过 `enemy_region_width/enemy_region_side` 配置。
- **起点/终点**：与敌人保持最小距离约束，避免开局贴脸或目标过于暴露。
- 支持固定场景（`scenario_mode = "fixed"`）与随机场景（`"random"`，训练默认）。

### 2.3 统计指标

环境统计以下指标：`total_path_length`、`visible_path_length`、`hidden_path_length`、`hidden_ratio`、`visible_ratio`。

---

## 3. 可见性建模（3D 视线遮挡）

`visibility_map[x, y]` ∈ {0, 1} 表示该格是否被敌人看见。

使用**三维射线检测**：

- 敌人站在自身格子的地形高度上（`enemy_eye_height` 偏移），目标格视点设在 `target_visibility_height` 高度。
- 从敌人视点到目标格视点连线，每格采样 `line_of_sight_samples_per_cell` 个点。
- 若连线中间有任何格子的地形高度超过连线高度（含 `visibility_occluder_bias` 偏置），则该目标格**不可见**。

---

## 4. 智能体与网络架构

模型代码：[models/policy_network.py](models/policy_network.py)，Agent 代码：[train/dqn_agent.py](train/dqn_agent.py)

### 4.1 观测空间（Hybrid 输入）

| 输入 | 维度 | 说明 |
|------|------|------|
| `local_map` | `4 × 32 × 32` | 四通道：occupancy（地形高度）、visibility（可见性）、goal（目标位置）、agent（当前位置） |
| `global_features` | `10` | 相对目标向量、相对敌人向量、目标/敌人距离、敌人朝向、当前位置可见性、hidden_ratio |

### 4.2 网络结构（HybridPolicyNetwork）

- **局部编码器**：3 层 Conv2D（16→32→64 通道），ReLU 激活，MaxPool + AdaptiveAvgPool，输出 1024 维特征。
- **全局编码器**：2 层全连接（64 维），编码 10 维全局特征。
- **融合层**：拼接局部与全局特征，经 128 维隐层。
- **Dueling 头**：分离 Value 流（输出 V(s)）和 Advantage 流（输出 A(s,a)），Q = V + A − mean(A)。
- **Double DQN**：Online 网络选动作，Target 网络（每 500 步同步）评估 Q 值。

### 4.3 动作空间

8 个离散动作，维度由环境 `BattlefieldEnv.ACTIONS` 统一管理。

### 4.4 动作掩码

训练和推理阶段均对无效动作（不可通行格子）进行掩码（mask 为 −∞），在全连接 DQN 目标计算中也应用掩码，确保不可行动作不影响 Q 值估计。

---

## 5. 探索策略

除 ε-greedy 外，引入了两类**引导探索**机制（见 `ExplorationConfig`）：

| 机制 | 初始概率 | 结束概率 | 说明 |
|------|----------|----------|------|
| Heuristic Subset（启发式子集） | 0.50 | 0.10 | 偏向选择朝目标方向移动的动作 |
| Teacher（A\* 引导） | 0.15 | 0.01 | 由 Visibility-A\* 规划完整路径并推荐下一步 |

两者概率随训练逐步衰减，最终将控制权交给学到的策略。

---

## 6. 奖励设计

奖励由以下部分组成（见 `EnvConfig`）：

| 组成部分 | 默认值 | 说明 |
|----------|--------|------|
| `step_penalty` | 0.08 | 每步基础惩罚，鼓励更短路径 |
| `visible_penalty` | 0.8 | 处于可见区域的额外惩罚 |
| `progress_weight` | 0.75 | 向目标接近的奖励（按距离变化量） |
| `hidden_ratio_gain_weight` | 0.25 | 隐蔽比例提升奖励（基于 hidden_ratio 增量） |
| `goal_reward` | 80.0 | 到达终点的奖励 |
| `success_hidden_ratio_weight` | 2.0 | 成功后按隐蔽比例追加奖励 |
| `collision_penalty` | 1.0 | 撞墙/无效移动惩罚 |
| `timeout_penalty` | 40.0 | 超时未达终点惩罚 |

---

## 7. 经典规划器（基线）

### 7.1 Visibility-Aware A\*

代码：[planner/visibility_astar.py](planner/visibility_astar.py)

单目标 A\*，最小化代价 `J(p) = L(p) + λ · V(p)`，其中 L 为路径长度，V 为可见步数，λ 默认 6.0。

### 7.2 Weighted / Scalarized A\*

代码：[planner/weighted_astar.py](planner/weighted_astar.py)

Visibility-Aware A\* 的参数化封装，便于调整 λ 权重。

### 7.3 Pareto A\*

代码：[planner/pareto_astar.py](planner/pareto_astar.py)

多目标 A\*，维护每个节点的非支配 `(path_length, visible_path_length)` 标签，通过支配检查和标签剪枝返回 Pareto 前沿上的一组路径（而非单条解）。

---

## 8. 训练与评估

### 8.1 训练流程

入口：[train/train_ddqn.py](train/train_ddqn.py)

- 使用 `train_scene_seeds`（3500 个场景种子：1000~4499）生成随机场景。
- 定期在 `val_scene_seeds`（100 个种子：5000~5099）上评估。
- 按成功率 → 平均奖励的优先级保存最优模型。
- Early Stop：成功率超过阈值（0.8）但奖励连续 3 个评估周期无显著提升时触发。
- 支持 KeyboardInterrupt 优雅中断并保存 checkpoint。

### 8.2 评估

| 脚本 | 用途 |
|------|------|
| [eval/run_policy.py](eval/run_policy.py) | 单场景对比：D3QN vs Visibility-A\* |
| [eval/evaluate_100.py](eval/evaluate_100.py) | 批量评估（默认 1000 场景），导出逐场景统计和汇总到 Excel |

---

## 9. 可视化

| 脚本 | 功能 |
|------|------|
| [visualize/plot_scene.py](visualize/plot_scene.py) | 3D 地形渲染：高度场 bar3d、敌人 FOV 锥体、可见性热力图、起点/终点/敌人标记 |
| [visualize/plot_episode.py](visualize/plot_episode.py) | Episode 路径渲染：D3QN 路径叠加到 3D 地形和俯视图，支持并排对比 A\* 路径 |

---

## 10. 关键默认配置

见 [config.py](config.py) 四个 frozen dataclass：

| 配置类 | 关键参数 |
|--------|----------|
| `EnvConfig` | `grid_size=32`, `height_levels=8`, `agent_max_climb_height=1`, `max_steps=96`, `enemy_horizontal_fov_deg=70`, `enemy_max_range=24` |
| `ModelConfig` | `local_channels=4`, `global_feature_dim=10` |
| `ExplorationConfig` | `heuristic_subset_enabled=True`, `teacher_enabled=True`, 概率随训练线性衰减 |
| `TrainingDefaults` | `episodes=10000`, `batch_size=256`, `replay_capacity=300000`, `lr=1e-4`, `gamma=0.99`, `target_update_interval=500`, `epsilon_decay_steps=100000`, `early_stop_success_rate_threshold=0.8` |

---

## 11. 项目结构

```
project/
  config.py                -- 全部超参数（4 个 frozen dataclass）
  main.py                  -- 快速冒烟测试入口
  env/
    battlefield_env.py     -- 核心 RL 环境（Gym 风格）
  models/
    policy_network.py      -- D3QN 网络（Hybrid CNN + MLP + Dueling 头）
  train/
    dqn_agent.py           -- DoubleDQNAgent（动作选择、训练步、保存/加载）
    replay_buffer.py       -- 经验回放缓冲（deque 实现）
    train_ddqn.py          -- 训练循环（评估、Early Stop、Checkpoint）
  planner/
    visibility_astar.py    -- Visibility-Aware A* 规划器
    weighted_astar.py      -- 标量化/加权 A* 封装
    pareto_astar.py        -- 多目标 Pareto A* 规划器
  eval/
    run_policy.py          -- 单场景 D3QN vs A* 对比
    evaluate_100.py        -- 批量评估（Excel 导出）
  visualize/
    plot_scene.py          -- 3D 场景渲染
    plot_episode.py        -- Episode 路径可视化
  artifacts/               -- 已保存模型、日志、评估 Excel 文件
  docs/                    -- 详细设计文档
```

---

## 12. 快速开始

```bash
# 冒烟测试（验证环境与模型可正常初始化）
python main.py

# 训练 D3QN
python -m train.train_ddqn

# 单场景对比 D3QN vs A*
python -m eval.run_policy

# 批量评估并导出 Excel
python -m eval.evaluate_100
```
