# 项目运行说明

## 项目概览

本项目用于在敌方视野约束下学习“短且隐蔽”的路径。环境为二维栅格移动，但障碍高度仅用于视线遮挡计算。当前默认使用随机场景训练，且每个 episode 都会重新采样场景。

核心要点：
- 地图大小：`32 x 32`，高度层级仅用于遮挡计算。
- 障碍生成：每个格子以 `obstacle_probability` 的概率成为障碍（伯努利采样）。
- 观测输入：`local_map` + `global_features`。
- 算法：`D3QN`（Double DQN + Dueling Network）。
- 动作掩码：训练与推理都对无效动作做掩码，避免撞墙动作干扰。

## 训练与泛化机制

- 训练时从 `train_scene_seeds` 中随机采样 `scene_seed`。
- 验证与测试使用固定种子集合。
- 训练过程中会保存 `latest/best/interrupt` 模型。

## 默认训练参数（见 `config.py`）

- `episodes = 10000`
- `batch_size = 256`
- `replay_capacity = 300000`
- `learning_rate = 3e-4`
- `warmup_steps = 2000`
- `epsilon_decay_steps = 100000`
- `eval_interval = 50`
- `save_interval = 100`

## 快速开始

安装依赖：
```bash
pip install -r requirements.txt
```

检查环境与网络输出：
```bash
python main.py
```

启动训练：
```bash
python -m train.train_ddqn
```

## 常用可视化与评估

查看随机场景 3D + 俯视图：
```bash
python -m visualize.plot_scene
```

评估并对比 D3QN 与 Visibility-A*：
```bash
python -m eval.run_policy
```

绘制路径对比图：
```bash
python -m visualize.plot_episode
```
