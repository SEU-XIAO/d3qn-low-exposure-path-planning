# PPO 寻路实现方案（修订版）

## 目标

用 PPO 算法训练一个在 50×50 战场中从起点走到终点的智能体，最小化被敌人看到的暴露时间。

---

## 一、总体技术路线

**算法**: PPO (Proximal Policy Optimization) + GAE (Generalized Advantage Estimation)
**框架**: PyTorch，从零实现（不依赖 stable-baselines3），便于自定义 action masking 和 observation 构建
**环境**: `BattlefieldEnv`，补充 RL 接口（`reset`/`step`/`get_obs`）
**泛化**: 每 episode 在全图上随机滑动窗口，训练时加数据增强（旋转/翻转）

---

## 二、神经网络设计

### 2.1 输入：多通道全图观测 (7 × 50 × 50)

全图 CNN 方案：50×50 不算大，智能体需要看到完整的可见性底图和目标位置才能规划隐蔽路径。

| 通道 | 内容 | 范围 |
|------|------|------|
| 0 | 归一化高度图 `height / max(1, height_levels)` | [0, 1] |
| 1 | 地面掩码 `tag == 0` | {0, 1} |
| 2 | 建筑掩码 `tag == 1` | {0, 1} |
| 3 | 树木掩码 `tag == 2` | {0, 1} |
| 4 | 可见性底图 `visibility_map` | {0, 1} |
| 5 | 智能体位置 — 高斯斑，σ=2，中心在 agent 坐标 | [0, 1] |
| 6 | 终点位置 — 高斯斑，σ=2，中心在 goal 坐标 | [0, 1] |

**关键改进**：通道 5/6 不使用 one-hot 而用高斯斑（σ=2），天然携带相对距离信息，避免网络记忆绝对坐标，利于跨窗口泛化。

敌方位置隐含在通道 4 可见性底图中，无需单独通道。

### 2.2 数据增强（训练时）

每个 batch 的观测随机施加：
- 90°/180°/270° 旋转（等概率 4 选 1）
- 水平翻转（50% 概率）
- 垂直翻转（50% 概率）

**动作映射**：增强后的观测对应的有效动作也做相应变换（如旋转 90° 后，"上"变成"右"）。这几乎不增加计算成本，但强制网络学习相对方向关系。

### 2.3 网络结构：CNN Backbone + Actor/Critic 双头

```
Input: (7, 50, 50)
  ↓
Conv2d(7→32, k=5, s=2, p=2) + ReLU     → (32, 25, 25)
Conv2d(32→64, k=3, s=2, p=1) + ReLU    → (64, 13, 13)
Conv2d(64→64, k=3, s=1, p=1) + ReLU    → (64, 13, 13)
Conv2d(64→128, k=3, s=1, p=1) + ReLU   → (128, 13, 13)
AdaptiveAvgPool2d((8, 8))               → (128, 8, 8)
Flatten                                 → 8192
Linear(8192→512) + ReLU                 → 512
  ├─ Actor:  Linear(512→8)              → 8 个动作 logits
  └─ Critic: Linear(512→1)              → 状态价值 V(s)
```

**关键改进**：只做两层 stride-2 下采样（50→25→13），保留 13×13 特征图，再 AdaptiveAvgPool 到 8×8。比之前的三层下采样（7×7）保留更多精细空间信息，适合贴掩体绕行等操作。

### 2.4 输出：8 个离散动作

```python
ACTIONS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),       # 上下左右
    (-1, -1), (-1, 1), (1, -1), (1, 1),      # 对角线
)
```

### 2.5 Action Masking

在前向传播时计算 invalid action mask，将不可执行动作的 logit 设为 `-inf`，softmax 后概率为 0。

不可执行判定（复用 `battlefield_env._is_blocked`）：
- 出界
- 目标格 tag ≠ 0（建筑/树木）
- 爬坡 tan > 0.3

**采样和训练时均使用 mask**，保证从不选无效动作，无效动作不参与梯度。

---

## 三、奖励函数设计

全部参数从 `EnvConfig` 中读取（已有定义，不动 `EnvConfig`）：

| 奖励项 | 值 | 说明 |
|--------|-----|------|
| 步数惩罚 | `-step_penalty` = -0.05 | 每步固定成本，鼓励走捷径 |
| 靠近奖励 | `+progress_weight * Δdist` = +0.1 × Δdist | 稠密引导（权重较原 cfg 减半），后期可衰减 |
| 暴露惩罚 | `-visible_penalty * I(visible)` = -0.4 | 处于可见格子的额外惩罚 |
| 碰撞惩罚 | `-collision_penalty` = -1.0 | 尝试无效动作（mask 下极少发生） |
| 到达奖励 | `+goal_reward` = +100 | 到达终点 |
| 超时惩罚 | `-5.0` | 超过 200 步仍未到达（轻惩罚，避免价值网络震荡） |

**关键改进**：
- `progress_weight` 从 0.2 降为 0.1，减少对必要绕行的惩罚；训练后期可进一步衰减至 0
- 超时惩罚从 -50 降为 -5，失败主要通过累积步数惩罚体现，避免单次大负值震荡价值网络

### Episode 终止条件

1. 到达终点 → 成功
2. `steps >= max_steps` (200) → 超时
3. `consecutive_collisions >= max_consecutive_collisions` (15) → 卡死（mask 下极少发生）

---

## 四、PPO 训练细节

### 4.1 超参数

| 参数 | 值 |
|------|-----|
| γ (折扣因子) | 0.99 |
| λ (GAE) | 0.95 |
| ε (PPO clip) | 0.2 |
| 学习率 | 3e-4 |
| 熵系数 | 0.01 |
| Value loss 系数 | 0.5 |
| 最大梯度范数 | 0.5 |
| Rollout 步数/update | 2048 |
| Minibatch size | 64 |
| Epochs/update | 10 |
| 总训练步数 | 2,000,000 |

### 4.2 训练流程

```
for total_steps:
    # 1. Rollout（收集经验）
    for step in range(2048):
        obs = env.get_obs()          # (7, 50, 50)
        action_mask = env.get_action_mask()
        obs_aug, mask_aug = random_augment(obs, action_mask)
        action, log_prob, value = policy(obs_aug, mask_aug)
        next_obs, reward, done, _ = env.step(action)
        buffer.add(obs, action, log_prob, reward, value, done, mask)
        if done: env.reset()         # 新窗口、新起点终点

    # 2. GAE 计算 advantages 和 returns
    advantages, returns = compute_gae(buffer, γ=0.99, λ=0.95)

    # 3. PPO update（多 epoch）
    for epoch in range(10):
        for minibatch in buffer:
            mb_obs, mb_mask = random_augment(minibatch)
            new_log_prob, new_value, entropy = policy(mb_obs, mb_mask)
            ratio = exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = clip(ratio, 1-ε, 1+ε) * advantage
            policy_loss = -min(surr1, surr2).mean()
            value_loss = 0.5 * (new_value - returns)^2
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            loss.backward()
        clip_grad_norm_(0.5)
        optimizer.step()
```

### 4.3 泛化机制

- **每 episode 随机新窗口**：`env.reset()` 调用 `generate_scene()` 在全图（501×499）上随机采样 50×50 窗口，起点/终点在窗口对角区域随机选取
- **评估用固定窗口集**：预选 50 个训练中未出现的窗口坐标 + 种子，每 10,000 步评估一次
- **评估指标**：到达率、平均路径长度、平均暴露率

---

## 五、文件结构

### 新增文件

```
models/
  __init__.py           — 导出 ActorCriticCNN
  policy_network.py     — CNN backbone + Actor/Critic 头 + action masking

train/
  __init__.py
  ppo_config.py         — PPOConfig dataclass（独立于 EnvConfig）
  ppo_buffer.py         — RolloutBuffer（存储 trajectories + GAE 计算）
  ppo_trainer.py        — PPO 训练循环
  train_ppo.py          — 入口脚本
```

### 修改文件

```
env/battlefield_env.py  — 补充 RL 接口（reset/step/get_obs/get_action_mask）
requirements.txt        — 添加 torch
```

### 不改的文件

```
config.py               — EnvConfig 不动
env/occlusion.py        — 不动
env/enemy_search.py     — 不动
env/terrain_loader.py   — 不动
env/__init__.py         — 不动
visualize/              — 不动
```

---

## 六、`BattlefieldEnv` 需要补充的 RL 方法

```python
def reset(self, seed=None) -> np.ndarray:
    self.generate_scene(scene_seed=seed)
    self.agent_position = self.start_position.copy()
    self.steps = 0
    self.consecutive_collisions = 0
    return self._get_observation()

def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
    old_pos = self.agent_position.copy()
    move = np.array(self.ACTIONS[action], dtype=np.int32)
    candidate = old_pos + move
    if self._is_blocked(candidate):
        self.consecutive_collisions += 1
        reward = -self.config.collision_penalty
    else:
        self.agent_position = candidate.copy()
        self.consecutive_collisions = 0
        reward = -self.config.step_penalty
        old_dist = np.linalg.norm(old_pos.astype(np.float32) - self.goal_position.astype(np.float32))
        new_dist = np.linalg.norm(self.agent_position.astype(np.float32) - self.goal_position.astype(np.float32))
        reward += self.config.progress_weight * (old_dist - new_dist)
        if self.visibility_map[tuple(self.agent_position)] > 0.5:
            reward -= self.config.visible_penalty

    self.steps += 1
    done = False
    if np.array_equal(self.agent_position, self.goal_position):
        reward += self.config.goal_reward
        done = True
    elif self.steps >= self.config.max_steps:
        reward -= 5.0  # 轻超时惩罚
        done = True
    elif self.consecutive_collisions >= self.config.max_consecutive_collisions:
        done = True

    return self._get_observation(), reward, done, {}

def _get_observation(self) -> np.ndarray:
    """构建 7×50×50 多通道观测，见 2.1 节"""

def get_action_mask(self) -> np.ndarray:
    """返回长度为 8 的 bool 数组，True=可执行"""
```

---

## 七、验证方案

1. 导入检查: `python -c "from models import ActorCriticCNN; from train.ppo_config import PPOConfig; from train import RolloutBuffer, PPOTrainer"`
2. 网络前向传播: 构造随机 (7,50,50) 输入，验证输出 (8,) logits + (1,) value
3. Action masking: 构造全阻塞场景，验证 softmax 概率只在有效动作上非零
4. 环境 step 闭环: `obs = env.reset(); obs2, r, d, _ = env.step(0); assert obs.shape == (7,50,50)`
5. 数据增强: 验证旋转/翻转后动作映射正确
6. 训练启动: `python -m train.train_ppo --steps 10000` 小规模运行，确认 loss 下降、无 NaN
7. BFS 对比: 训练后提取策略走 100 个场景，和 BFS 最短路径对比暴露率
