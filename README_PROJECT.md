# D3QN 低暴露路径规划 (D3QN Low-Exposure Path Planning)

## 1. 项目概述

在真实地形高度场（501×499 格，cell=10m）上训练一个 D3QN（Double DQN + Dueling Network）智能体，使其在**敌方视野威胁**下找到从起点到终点的低暴露路径。

**双目标优化**：路径长度尽量短 + 暴露在敌人视野内的步数尽量少。

核心思路：**从全图 25 万格子中用特征代理选出 8 个瞭望点，预计算全图二值可见性底图，训练时随机滑窗切片即可**——无需每 episode 做射线追踪。

---

## 2. 关键设计决策

### 2.1 敌人选点：地形特征代理 + 空间抑制

**问题**：如何在 501×499 的全局地形中选出 8 个视野尽量好、覆盖尽量广的敌人瞭望点？

**方法**（[env/enemy_search.py](env/enemy_search.py)）：

1. **特征代理评分**（全图每个格子，25×25 邻域窗口）：
   - `height_rank` (0.3)：在邻域内的高度排名 `(h - min) / (max - min)`
   - `openness` (0.3)：比邻域均值高多少 `(h - mean) / (max - min)`
   - `dominance` (0.4)：z-score `(h - mean) / std`，衡量局部支配力

2. **空间抑制 (NMS)**：选最高分 → 抑制半径 60 内的得分 ×0.3 → 重复 8 次，确保 8 个点散布全图

3. **验证**：对每个候选点用 3D 射线追踪计算全图可见性格子数

**关键洞察**：平原上一棵 19m 的孤立树（可见 9.4% ≈ 23624 格）比山区 89m 的山峰（可见 3.2% ≈ 8092 格）视野更好——**局部支配力远比绝对高度重要**。

### 2.2 预计算全图可见性 → 训练时切片

**流程**：

- **一次性**：对 8 个敌人池位置各跑一次全图 3D 射线追踪（每个点 501×499 ≈ 25 万次射线），存为 `artifacts/visibility_maps.npz`（68 KB）
- **每 episode（训练时）**：从敌人池随机选一个敌人 → 取对应全图可见性底图 → 随机切出 50×50 窗口（numpy 切片，~40ms）
- **每 episode（评估时）**：对 20 个固定种子取确定性窗口切片

**收益**：训练 reset 从 200ms（现场射线追踪）降至 40ms（numpy 切片），且敌人位置不再局限于窗口内。

### 2.3 敌人全局坐标

敌人位置是**全局坐标系**（在全图 501×499 中），独立于随机切出的 50×50 训练窗口。敌人可以——且经常——在窗口之外。这意味着：

- 敌人高度 = 该格子的地形高度（tag 可以是地面/建筑/树木，不限于可通行）
- 可见性判断使用全局坐标进行 3D 射线追踪
- 如果敌人恰好在 50×50 窗口内，其所在格子是否阻挡智能体移动由 `tag_map` 决定（可通行则可行走）

### 2.4 爬坡约束：Tan 公式

旧版用 `agent_max_climb_height`（固定高度差阈值），新版用基于梯度的判断：

```
tan = dh / horizontal_distance
horizontal_distance = cell_size（直走 10m）或 cell_size * sqrt(2)（对角线 14.14m）
可通行 ⇔ tan <= max_climb_tan (0.3)
```

即直走最多爬 3m，对角线最多爬 4.24m。这个约束通过**动作掩码**强制（Q 值再高也不能选不可通行的动作）。

### 2.5 360° 全向可见性

敌人可见性为**全向 360°**——无 FOV 锥角限制，无最大距离限制。任何格子的可见性完全由 3D 地形遮挡决定。

### 2.6 3D 视线检测

```
敌人视点 = 地形高度 + enemy_eye_height (1.0m)
目标视点 = 地形高度 + target_visibility_height (0.5m)
从敌人视点到目标视点连线采样 → 若中间格子地形高度超过连线高度 → 不可见
```

---

## 3. 网络架构

模型代码：[models/policy_network.py](models/policy_network.py)

**Hybrid CNN + MLP + Dueling Head**（Double DQN）：

| 组件 | 结构 | 说明 |
|------|------|------|
| 局部编码器 | 3-stage CNN: 5→32→64→128, 2×MaxPool, 2×ResBlock(含BN), AdaptiveAvgPool(4×4), Flatten → 2048 | 处理 5×50×50 局部特征图，BatchNorm + 残差连接稳定深层训练 |
| 全局编码器 | Linear(8→64→64) | 处理全局标量特征 |
| 融合层 | Concat(2048+64) → Linear(128) | |
| Value 头 | Linear(128→64→1) | Dueling V(s) |
| Advantage 头 | Linear(128→64→8) | Dueling A(s,a) |
| Q 值 | V + A − mean(A) | |

**输入**（Hybrid）：

| 输入 | 维度 | 说明 |
|------|------|------|
| `local_map` | `5 × 50 × 50` | occupancy、visibility、goal（one-hot距离图）、agent（one-hot当前位置）、enemy（one-hot敌人位置） |
| `global_features` | 8 维 | 相对目标向量(2)、相对敌人向量(2，归一化到全图尺寸)、目标/敌人距离(2)、当前位置可见性(1)、hidden_ratio(1) |

**动作空间**：8 个离散方向（上下左右 + 四对角）

---

## 4. 训练流程

入口：[train/train_ddqn.py](train/train_ddqn.py)

### 4.1 关键训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| episodes | 10000 | 总训练轮数 |
| max_steps | 200 | 每 episode 最大步数 |
| batch_size | 256 | |
| replay_capacity | 100000 | 经验回放缓冲区大小（uint8 存储，~2GB） |
| gamma | 0.99 | 折扣因子 |
| lr | 3e-5 | 学习率（深网络需要更低lr防loss爆炸） |
| target_update | 每 500 步 | Double DQN 目标网络同步 |
| epsilon | 1.0 → 0.05 | 20 万步线性衰减（含 heuristic/teacher/lambda） |
| enemy_switch_interval | 50 | 同一敌人固定 50 个 episode 后切换 |
| warmup | 5000 步 | 预热后才开始训练（深网络需要更多样经验） |

### 4.2 探索策略

| 机制 | 初始概率 → 结束概率 | 说明 |
|------|---------------------|------|
| ε-greedy | 1.0 → 0.05 | 标准随机探索（20 万步衰减） |
| Heuristic Subset | 0.50 → 0.20 | 偏向朝目标方向移动的动作子集 |
| Teacher (A*) | 0.12 → 0.03 | Visibility-A* 规划路径推荐下一步，λ 从 12.0 衰减到 3.0 |

引导探索概率和 Teacher λ 随训练线性衰减（与 epsilon 共用 20 万步衰减表）。前期 λ=12.0 极度保守，后期 λ=3.0 教会 Agent 接受必要暴露。Teacher 概率大幅降低（0.12→0.03），让 Agent 尽早自主学习而非依赖引导。同一敌人固定 50 个 episode 才切换，减少可见性分布震荡。

### 4.3 奖励设计

单步奖励只包含三项核心信号（不含过程性稠密奖励，避免策略畸形）：

| 组成部分 | 默认值 | 说明 |
|----------|--------|------|
| `step_penalty` | 0.05 | 每步基础惩罚，乘以移动代价（直走 1.0，对角 1.414） |
| `visible_penalty` | 0.4 | 暴露在敌人视野内的额外惩罚（0.4 意味着走 2~3 步暴露格子 ≈ 多走 1 步） |
| `collision_penalty` | 1.0 | 尝试无效动作 |
| `max_consecutive_collisions` | 15 | 连续撞墙 N 次后提前终止 episode（防止死循环浪费步数） |

Episode 终止时：

| 组成部分 | 默认值 | 说明 |
|----------|--------|------|
| `goal_reward` | 100.0 | 到达终点 |
| `success_hidden_ratio_weight` | 5.0 | 成功后按整体隐蔽比例追加（终局结算，不干扰过程决策） |
| `timeout_penalty` | 50.0 | 超时未到达 |

**设计原则**：移除了 `progress_weight`（距离差奖励）和 `hidden_ratio_gain_weight`（过程隐蔽比例增益），因为这两个在线稠密奖励会扭曲状态价值估计——前者引发奖励 hacking，后者鼓励 Agent 在隐蔽区反复踱步刷隐蔽率。新奖励格局让"必要时的短暂暴露"可承受，把隐蔽性权衡放在全局层面（终局结算）而非每步恐慌。

### 4.4 动作掩码

每个 step，环境通过 `get_valid_actions()` → `can_move_between()` 返回哪些动作合法（目标格存在、可通行、满足爬坡约束）。非法动作的 Q 值被设为 -inf，确保不会被选中。

### 4.5 评估

- **快速评估**（每 50 episodes）：20 个验证种子，仅记录统计
- **全量评估**（每 200 episodes）：200 个验证种子，用于模型选择 + early stop
- 按成功率 → 平均奖励的优先级保存最优模型
- Early stop：成功率 > 0.8 但奖励连续 3 个周期无显著提升时触发

---

## 5. 经典规划器（基线）

| 规划器 | 文件 | 说明 |
|--------|------|------|
| Visibility-Aware A* | [planner/visibility_astar.py](planner/visibility_astar.py) | 最小化代价 `J = L + λ·V`，λ=6.0 |
| Weighted A* | [planner/weighted_astar.py](planner/weighted_astar.py) | 可调 λ 的 A* 封装 |
| Pareto A* | [planner/pareto_astar.py](planner/pareto_astar.py) | 多目标 A*，返回 Pareto 前沿上的一组路径 |

---

## 6. 场景模式

在 [config.py](config.py) 中通过 `scenario_mode` 切换：

| 模式 | 说明 |
|------|------|
| `"fixed"` | 固定场景（config 中指定的 start/goal/enemy） |
| `"random"` | 程序化生成地形 + 随机障碍物 + 区域敌人搜索 |
| `"full_map"` | **训练默认**：加载真实地形 → 预计算敌人池 → 滑动窗口切片 |

---

## 7. 项目结构

```
project/
  config.py                  -- 全部超参数（4 个 frozen dataclass）
  main.py                    -- 快速冒烟测试入口
  env/
    battlefield_env.py       -- 核心 RL 环境（Gym 风格, full_map/random/fixed 模式）
    terrain_loader.py        -- 地形 txt 解析器（(height,tag) 格式）
    enemy_search.py          -- 三层漏斗敌人搜索 + 全图可见性预计算
  models/
    policy_network.py        -- D3QN 网络（Hybrid CNN + MLP + Dueling 头）
  train/
    dqn_agent.py             -- DoubleDQNAgent（动作选择、训练步、保存/加载）
    replay_buffer.py         -- 经验回放缓冲（deque 实现）
    train_ddqn.py            -- 训练循环（评估、Early Stop、Checkpoint）
  planner/
    visibility_astar.py      -- Visibility-Aware A* 规划器
    weighted_astar.py        -- 标量化 A* 封装
    pareto_astar.py          -- 多目标 Pareto A* 规划器
  eval/
    run_policy.py            -- 单场景 D3QN vs A* 对比
    evaluate_100.py          -- 批量评估（Excel 导出）
    count.py                 -- 地形文件全面统计分析工具
  visualize/
    plot_scene.py            -- 3D 场景渲染
    plot_episode.py          -- Episode 路径可视化
  artifacts/                 -- 产出（模型、vis maps、enemy pool、评估结果）
```

---

## 8. 快速开始

### 8.1 第一次：预计算敌人池和可见性底图

```bash
# 生成 artifacts/enemy_pool.json 和 artifacts/visibility_maps.npz
# 耗时约 1 小时（8 个点 × 25 万次射线/点）
PYTHONPATH="." python -u env/enemy_search.py
```

### 8.2 训练

```bash
# 冒烟测试
python main.py

# 正式训练
python -m train.train_ddqn
```

### 8.3 分析与评估

```bash
# 地形统计分析
python eval/count.py MyPath_Data417.txt

# 单场景对比
python -m eval.run_policy

# 批量评估
python -m eval.evaluate_100
```

### 8.4 远程训练

如果训练在远程机器上，需要先将预计算文件复制过去：
```bash
scp artifacts/enemy_pool.json artifacts/visibility_maps.npz user@remote:project/artifacts/
```

---

## 9. 配置速查

[config.py](config.py) 四个 frozen dataclass：

| 配置类 | 与训练最相关的参数 |
|--------|-------------------|
| `EnvConfig` | `scenario_mode="full_map"`, `grid_size=50`, `max_steps=200`, `max_climb_tan=0.3`, `visible_penalty=0.4`, `enemy_switch_interval=50`, `max_consecutive_collisions=15` |
| `ModelConfig` | `local_channels=5`, `global_feature_dim=8` |
| `ExplorationConfig` | `heuristic_subset_enabled=True`, `teacher_enabled=True`, `teacher_lambda_start=12.0→end=3.0`, `teacher_action_prob=0.12→0.03` |
| `TrainingDefaults` | `episodes=10000`, `batch_size=256`, `replay_capacity=100000`, `lr=3e-5`, `warmup=5000`, `gamma=0.99`, `epsilon_decay_steps=200000` |

---

## 10. 地形数据格式

```
(height,tag);(height,tag);(height,tag);...
(height,tag);(height,tag);...
```

每行对应一行格子，用 `;` 分隔。`tag` 含义：

| tag | 含义 | 通行 |
|-----|------|------|
| 0 | 地面 | 可通行 |
| 1 | 建筑 | 不可通行 |
| 2 | 树木 | 不可通行 |

不规则行（某行格子数少于最大宽度）自动用 `height=0, tag=1` 填充。

当前地形文件 `MyPath_Data417.txt`：501 行 × 499 列，高度范围 0~147，可通行率约 78%，爬坡违反率约 6.8%。
