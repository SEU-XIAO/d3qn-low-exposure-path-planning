# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 问题域

50×50 栅格战场，8 方向移动。存在地形高度（影响可通行性）和二值可见性（敌人瞭望点有 FOV）。用 PPO 训练智能体从起点走到终点，最小化被敌人看到的暴露时间。

## 项目结构

```
config.py               — EnvConfig 环境参数（frozen dataclass，一字不改）
MyPath_Data417.txt      — 501×499 全地形数据，每格 (height,tag)，tag: 0=地面 1=建筑 2=树木

env/
  terrain_loader.py     — 地形文件解析
  occlusion.py          — 3D 光线追踪遮挡判定（独立模块，无状态）
  enemy_search.py       — 特征代理评分 + 空间抑制 → 8 个瞭望点 + 全图可见性底图
  battlefield_env.py    — 场景生成 + 通行检查 + RL 接口（reset/step/get_obs/get_action_mask）
  scene_pool.py         — 预计算随机障碍场景池（阶段1用，无敌人）
  vectorized_env.py     — N 个并行环境包装器，批量前向传播加速

models/
  policy_network.py     — CNN backbone + Actor/Critic 双头 + Action Masking + 数据增强 + 动作逆变换

train/
  ppo_config.py         — PPOConfig dataclass（独立于 EnvConfig）
  ppo_buffer.py         — RolloutBuffer（存储 transitions + GAE 计算）
  ppo_trainer.py        — PPO 训练循环（rollout → GAE → PPO update → eval）
  train_ppo.py          — 全功能训练入口（全图模式 + 敌人）
  smoke_test.py         — 平坦地形冒烟测试（无敌人无建筑，验证算法可行性）
  stage1_obstacle.py    — 阶段1：障碍地形纯导航训练（无敌人）
  stage1_parallel.py    — 阶段1并行版（多环境批量推理）

artifacts/
  enemy_pool.json       — 预计算的 8 个敌人瞭望点
  visibility_maps.npz   — 预计算的 8 张全图可见性底图
  scene_pool.npz        — 预计算的 N 个随机障碍场景（阶段1用）

visualize/
  visualizer.py         — 训练结果可视化
  find_dense_scenes.py  — 查找密集障碍场景
```

## 各模块功能

### `env/battlefield_env.py` — 场景生成与通行检查

**`BattlefieldEnv`** 类，构造时自动调 `generate_scene()`。

RL 接口：
- `reset(seed)` → obs (7,50,50) float32
- `step(action: int)` → (obs, reward, done, info)
- `get_action_mask()` → (8,) bool，True=可执行
- `_get_observation()` → (7,50,50) 7通道：height + ground/building/tree + visibility + agent/goal 高斯斑(σ=2)
- `compute_bfs_path()` → list[tuple] 或 None

场景模式：`"full_map"`（全图滑动窗口）、`"random"`（程序化地形）、`"fixed"`（固定障碍物）。

通行检查：`_is_blocked` 综合判断边界/标签/爬坡(tan≤0.3)/敌人位置。

### `env/occlusion.py` — 遮挡判定（独立模块，无状态）

- `is_occluded(start, end, height_map, config)` → bool: 3D 光线追踪。**start/end 必须与 height_map 同坐标系**
- `compute_cell_visibility(observer, cell, height_map, config)` → float
- `compute_visibility_map(observer, terrain, config)` → (vis_map, visible_count)

### `env/enemy_search.py` — 敌人瞭望点搜索

- `compute_feature_scores(terrain)` → np.ndarray: 特征代理评分（高度排名0.3 + 开阔度0.3 + 支配力0.4）
- `spatial_suppression(scores, k, radius)` → list[tuple]: NMS 空间抑制

### `models/policy_network.py` — CNN 策略-价值网络

**`ActorCriticCNN`**：
- CNN backbone: Conv(7→32,k5,s2) → Conv(32→64,k3,s2) → Conv(64→64,k3,s1) → Conv(64→128,k3,s1) → AdaptiveAvgPool(8,8) → FC(8192→512)
- Actor: Linear(512→8), Critic: Linear(512→1)
- `forward(obs, action_mask, deterministic)` → (action, log_prob, value, entropy, logits)
- `evaluate(obs, action, action_mask)` → (log_probs, values, entropy) — PPO update 用
- `get_value(obs)` → value — GAE bootstrap 用
- Action masking: 无效动作 logit = -1e9（非 -inf，避免 softmax NaN）

**数据增强** (`random_augment`)：
- 随机旋转（0/90/180/270）+ 水平/垂直翻转（各50%）
- 返回 `(aug_obs, aug_mask, (k, flip_h, flip_v))`
- **关键**：增强在 rollout 时施加一次，buffer 存储增强后数据。PPO update 时原样取出，保证 old/new log_prob 可比。

**动作逆变换** (`deaugment_action(aug_action, k, flip_h, flip_v)`)：
- **必须调用！** 增强空间的动作索引必须先逆变换回原始空间，再交给 `env.step()`。
- 逆序：逆 flip_v → 逆 flip_h → 逆旋转(k 次 forward 映射)

### `train/ppo_buffer.py` — Rollout Buffer

预分配所有 tensor，`add()` 逐条存储 transition，`compute_gae()` 倒序计算 GAE advantage，`normalize_advantages()` 做 z-score 标准化，`sample()` 返回随机 mini-batch 索引。

### `train/ppo_trainer.py` — PPO 训练器

**`PPOTrainer`**：完整训练循环。
- 进度奖励衰减：训练进度 50%-90% 期间 `progress_weight` 线性衰减到 0
- 评估：确定性推理（无增强、argmax），每 `eval_interval` 步一次

### `env/vectorized_env.py` — 并行环境

**`VectorizedEnv`**：N 个独立 `BattlefieldEnv` 实例，每个从场景池独立采样。
- `get_observations()` → (N,7,50,50)
- `step(actions)` → 对所有环境各执行一步，自动 reset 已完成的
- 配合批量 CNN 前向传播，加速约 N 倍

## 训练阶段体系

| 阶段 | 脚本 | 场景 | 敌人 | 目的 |
|------|------|------|------|------|
| 冒烟 | `train/smoke_test.py` | 平坦地形(0,0)→(49,49) | 无 | 验证算法可行性 |
| 阶段1 | `train/stage1_parallel.py` | 场景池（建筑+树木+高度） | 无 | 验证 action masking + 绕行 |
| 阶段2 | `train/train_ppo.py` | 全图滑动窗口 | 有 | 完整隐蔽寻路 |

## 常用命令

```bash
# 生成场景池（阶段1用，一次性）
python -m env.scene_pool --num 5000

# 阶段1训练（并行版，推荐）
python -m train.stage1_parallel --steps 500000 --envs 8 --pool artifacts/scene_pool.npz

# 阶段1训练（串行版，调试用）
python -m train.stage1_obstacle --steps 500000 --pool artifacts/scene_pool.npz

# 全功能训练（全图模式 + 敌人）
python -m train.train_ppo --steps 2000000 --save checkpoints

# 冒烟测试（平坦地形快速验证）
python -m train.smoke_test

# 重新生成敌人池 + 可见性底图
python -m env.enemy_search

# 快速导入验证
python -c "from models import ActorCriticCNN, random_augment, deaugment_action; from train import PPOConfig, RolloutBuffer, PPOTrainer; print('OK')"
```

## 坐标系统约定

- 全局坐标: 在全地形 `height_map` (501×499) 上的坐标
- 窗口坐标: 在 50×50 滑动窗口内的坐标
- `occlusion.py` 中所有函数使用同一坐标系（start/end 与 height_map 对应）
- `BattlefieldEnv._is_occluded_global` 负责窗口→全局坐标转换

## 关键实现细节

- **Action Masking**: 用 `-1e9` 而非 `-inf`，避免全 mask 时 softmax NaN
- **数据增强时机**: 仅在 rollout 时施加一次，buffer 存增强后数据，PPO update 不重新增强
- **动作逆变换**: `random_augment` 变换了 obs 和 mask，CNN 输出增强空间动作，必须 `deaugment_action()` 还原后再 `env.step()`
- **GAE bootstrap**: 用 rollout 最后一步的 obs 计算 `last_value`，done=True 时 `not_done` 因子自动归零
- **进度奖励衰减**: `progress_weight` 在训练进度 50%-90% 期间线性衰减到 0（课程学习）
- **EnvConfig 不可修改**: 所有训练超参在 `PPOConfig` 中配置
