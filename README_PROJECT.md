# 项目说明（对齐当前实现）

## 1. 目标与指标

本项目关注在敌方视野约束下的路径规划问题，目标是：
- 路径尽量短
- 路径尽量隐蔽（尽量少暴露在敌方视野内）

环境会统计以下指标：
`total_path_length`、`visible_path_length`、`hidden_path_length`、`hidden_ratio`、`visible_ratio`。

## 2. 环境与场景

- 地图尺寸：`32 x 32`，高度层级仅用于遮挡视线（不参与运动维度）。
- 智能体在二维平面离散移动（8 邻域）。
- 障碍生成：每个格子以 `obstacle_probability` 概率成为障碍；高度只取 1 或 2。
- 敌人具有位置、朝向、视场角与最大可见距离。
- 支持固定场景与随机场景（训练默认随机）。

核心环境代码：`env/battlefield_env.py`

## 3. 可见性建模

`visibility_map[x, y]` 表示该格是否被敌人看见：
1 表示可见，0 表示不可见。可见性由以下条件决定：
- 非障碍
- 在敌人最大可见距离内
- 在敌人水平视场角内
- 视线未被障碍遮挡

## 4. 智能体与输入设计

模型使用 D3QN（Double DQN + Dueling Network）。

### 4.1 局部/全局观测

`local_map`（4 通道）：
- `occupancy`：障碍占据
- `visibility`：可见性
- `goal`：目标位置
- `agent`：智能体当前位置

当 `local_map_size == grid_size` 时，`local_map` 即为全图。

`global_features`（10 维）：
相对目标、相对敌人、目标/敌人距离、敌人朝向、当前位置可见性、`hidden_ratio`。

### 4.2 动作空间

8 个离散动作（上下左右 + 4 个对角方向），动作维度由环境 `BattlefieldEnv.ACTIONS` 决定。

### 4.3 动作掩码

训练与推理均对无效动作进行掩码，避免撞墙动作影响策略与目标估计。

## 5. 奖励设计（核心逻辑）

奖励由以下部分组成：
- 步长惩罚
- 暴露惩罚
- 朝目标前进奖励
- 隐蔽比例提升奖励
- 达到目标奖励
- 撞墙惩罚

## 6. 训练与评估

训练默认使用随机场景：
`train_scene_seeds` / `val_scene_seeds` / `test_scene_seeds`。

训练机制包含：
`best model` 保存、定期评估、early stop、中断保存。

## 7. 关键默认配置（见 `config.py`）

- `episodes = 10000`
- `replay_capacity = 300000`
- `epsilon_decay_steps = 100000`
- `early_stop_eval_episodes = 10`
- `early_stop_success_rate_threshold = 0.6`

## 8. 可视化与对比

支持 3D 与俯视图展示，并与 `Visibility-A*` 进行路径对比。

相关脚本：
- `visualize/plot_scene.py`
- `visualize/plot_episode.py`
- `eval/run_policy.py`
