# 低暴露路径规划 — D3QN 训练项目

## 问题定义

50×50 栅格战场，8 方向移动。存在地形高度（影响可通行性）和二值可见性（敌人在一边有 FOV，部分格子在敌视野内）。目标：从起点走到终点，**最小化暴露在敌人视野中的路程**。

## 项目结构

```
env/battlefield_env.py    — 环境（step/reset/observation/visibility_map/地形）
models/policy_network.py   — HybridPolicyNetwork (CNN+MLP→8 Q值)
train/dqn_agent.py         — DoubleDQNAgent + TrainingConfig
train/replay_buffer.py     — 环形缓冲区，PER + n-step TD
train/train_ddqn.py        — 训练主循环 + 评估
train/bc_pretrain.py       — Behavioral Cloning 预训练（Visibility-A* 生成专家数据）
planner/visibility_astar.py — Visibility-A* 规划器
config.py                  — EnvConfig / TrainingDefaults / WaypointConfig / ExplorationConfig
visualize/plot_episode.py  — 单集路径可视化
```

## 当前方案：分层航点式 D3QN

### 架构

A* 生成航点序列 → 低层 D3QN 逐段走到航点（每段约 15 A*-steps，最多 45 步）。

### 网络输入

- **local_map**: 6 通道 × 50×50 (occupancy, visibility, goal, agent, enemy, **waypoint**)
- **global_features**: 12 维 (goal_rel(2), enemy_rel(2), goal_dist, enemy_dist, visibility, hidden_ratio, **wp_rel(2), wp_dist, wp_active**)
- 输出 8 个 Q 值（8 方向动作）

### 关键参数（WaypointConfig）

| 参数 | 值 | 说明 |
|------|-----|------|
| interval | 15 | A* 路径上每隔 N 步采一个航点 |
| max_segment_multiplier | 3.0 | 每段最多 15×3=45 步 |
| waypoint_visible_weight | 6.0 | 航点生成时避开暴露区的权重 |
| waypoint_reached_reward | 10.0 | 到达中间航点奖励 |
| segment_timeout_penalty | 20.0 | 段超时惩罚 |

### 训练流程

```bash
# 1. BC 预训练（航点模式：subgoal=goal，航点特征镜像目标特征）
python -m train.bc_pretrain --use-waypoints --episodes 2000 --output artifacts/ddqn_bc_wp.pt

# 2. D3QN 训练
python -m train.train_ddqn --use-waypoints --bc-path artifacts/ddqn_bc_wp.pt

# 3. 多 λ 评估（测试 A* 路径候选的上限）
python -m train.train_ddqn --use-waypoints --bc-path artifacts/ddqn_best.pt --multi-lambda --eval-only
```

## 已尝试过的技术

| 技术 | 效果 |
|------|------|
| 单层 D3QN（无航点，BC 预训练） | 38-44% 成功率 |
| PER（优先经验回放） | 轻微改善 |
| HER-future（失败集重标记） | 效果有限 |
| BC regularization decay | 稳定训练 |
| 分层航点（BC 未加载） | 25%（等于没用） |
| 分层航点 + BC 预训练（iv=25, mult=2.0） | 峰值 45%，最终 42% |
| 分层航点 + 密集航点 + 可见性航点（当前) | ~37% @3600ep |

## 当前核心瓶颈

**低层成功率两极分化严重。** 能走通的地图上 agent 干净利落地 15-17 步/段走完全程；走不通的地图上第一段就 timeout，不管航点怎么摆。

多 λ 评估（尝试 8 种不同的 A* 路径）只能到 ~35%，说明**瓶颈不在航点选择，而在低层控制器的泛化能力**——它在训练见过的地形模式上学会了，但约 60% 的地图上是完全失败的。

## 待探索的方向

1. **反向课程学习（Reverse Curriculum）**：从终点附近开始训，逐段往前扩展
2. **Distributional RL（C51/QR-DQN）**：学回报分布而非期望 Q 值，更好处理 bimodal 结果
3. **加记忆模块（LSTM/GRU）**：网络记住前几步的地形/视野信息
4. **高层网络控制器**：用网络替代 A* 做航点选择（需要低层先有足够的泛化能力）

## 常用命令

```bash
# BC 预训练
python -m train.bc_pretrain --use-waypoints --episodes 2000 --output artifacts/ddqn_bc_wp.pt

# 训练
python -m train.train_ddqn --use-waypoints --bc-path artifacts/ddqn_bc_wp.pt

# 仅评估（单 λ）
python -m train.train_ddqn --use-waypoints --bc-path artifacts/ddqn_best.pt --eval-only

# 多 λ 评估（测上限）
python -m train.train_ddqn --use-waypoints --bc-path artifacts/ddqn_best.pt --multi-lambda --eval-only

# 可视化
python -m visualize.plot_episode --use-waypoints --checkpoint ddqn_best.pt --seed 7201
python -m visualize.plot_episode --use-waypoints --checkpoint ddqn_best.pt --comparison --seed 7201
```

## 训练日志位置

本地: `artifacts/log/`
服务器: `/tempdisk2/gwj/xiaofh/d3qn-low-exposure-path-planning/artifacts/logs/`
