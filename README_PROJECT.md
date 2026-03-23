# 项目运行说明

## 当前阶段

当前版本已完成以下内容：

- `32 x 32 x 8` 固定战场环境
- 固定起点、终点、敌人位置
- 中心区域障碍物布局
- 球体扇区敌方视野与遮挡判定
- 与状态表达匹配的 `CNN + MLP` 双分支网络
- `Double DQN` 训练链路
- 三维场景可视化与路径回放可视化

## 快速运行

安装依赖：

```bash
pip install -r requirements.txt
```

检查环境与网络输出：

```bash
python main.py
```

查看三维场景：

```bash
python -m visualize.plot_scene
```

启动 `Double DQN` 训练：

```bash
python -m train.train_ddqn
```

当前默认训练参数为服务器版预设，更适合带 CUDA 的显卡环境，例如 `RTX 4090`：

- `device = cuda`
- `episodes = 800`
- `batch_size = 256`
- `replay_capacity = 50000`
- `learning_rate = 3e-4`
- `warmup_steps = 2000`
- `epsilon_decay_steps = 20000`

加载模型并打印一回合决策过程：

```bash
python -m eval.run_policy
```

绘制训练后路径：

```bash
python -m visualize.plot_episode
```
