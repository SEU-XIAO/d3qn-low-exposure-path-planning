# 项目运行说明

## 项目概览

当前版本已经切换到“短路径 + 高隐蔽比例”的目标，核心能力包括：

- `32 x 32 x 8` 三维栅格战场环境
- 二维地面移动 + 障碍物遮挡敌方视线
- 固定场景与随机场景两种模式
- `D3QN` 智能体
- `epsilon-greedy + heuristic action subset` 探索
- 可选 `Visibility-A* teacher action`
- `Visibility-A*` 基线规划
- D3QN 与 `Visibility-A*` 的 3D/俯视图对比
- 评估驱动的 `best` 保存、early stop、中断保存

## 训练与泛化机制

当前训练默认使用“训练随机化、验证固定种子集”的方式：

- 训练时：
  每个 episode 会从 [config.py](/d:/井九/Documents/大三寒假/temp/config.py) 的 `train_scene_seeds` 中随机采样一个 `scene_seed`，并以 `scenario_mode="random"` 生成场景
- 验证时：
  使用 `val_scene_seeds` 中的一组未见过的固定随机场景
- 测试时：
  可使用 `test_scene_seeds` 做泛化测试

这样做的目标是：

- 不让模型只记住单一地图
- 让评估更接近真实泛化能力
- 让训练日志中的 `scene_seed` 可以被本地完整复现

## 默认训练参数

当前默认配置位于 [config.py](/d:/井九/Documents/大三寒假/temp/config.py)：

- `device = "cuda"`，无 CUDA 时自动回退 CPU
- `episodes = 3000`
- `batch_size = 256`
- `replay_capacity = 50000`
- `learning_rate = 3e-4`
- `warmup_steps = 2000`
- `epsilon_decay_steps = 20000`
- `eval_interval = 50`
- `save_interval = 100`

与当前目标直接相关的环境参数包括：

- `step_penalty`
- `visible_penalty`
- `progress_weight`
- `hidden_ratio_gain_weight`
- `success_hidden_ratio_weight`

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

检查环境和网络前向输出：

```bash
python main.py
```

启动训练：

```bash
python -m train.train_ddqn
```

## 模型文件

训练过程会在 `artifacts/` 目录下生成：

- `ddqn_latest.pt`：最近一次保存的模型
- `ddqn_best.pt`：验证集表现最好的模型
- `ddqn_interrupt.pt`：训练被手动中断时保存的模型

实际评估和可视化时，建议优先使用 `ddqn_best.pt`。

## 复现场景

如果训练日志里打印出了某个 `scene_seed`，就可以在本地复现同一个随机场景。

例如日志里出现：

```text
scene_seed=2003
```

那么可以在本地用同样的 `scene_seed` 做评估和可视化。

## 常用运行命令

查看固定场景的 3D 布局：

```bash
python -m visualize.plot_scene
```

在 Python 中查看指定随机场景：

```python
from visualize.plot_scene import plot_scene

plot_scene(scene_seed=2003, scenario_mode="random")
```

打印模型在指定场景下的一回合决策过程：

```python
from eval.run_policy import run_episode

run_episode(checkpoint_name="ddqn_best.pt", scene_seed=2003, scenario_mode="random")
```

绘制模型在指定场景下的路径：

```python
from visualize.plot_episode import plot_episode

plot_episode(checkpoint_name="ddqn_best.pt", scene_seed=2003, scenario_mode="random")
```

绘制 D3QN 与 `Visibility-A*` 的路径对比：

```python
from visualize.plot_episode import plot_comparison

plot_comparison(checkpoint_name="ddqn_best.pt", scene_seed=2003, scenario_mode="random")
```

直接运行评估脚本默认示例：

```bash
python -m eval.run_policy
```

直接运行路径对比可视化默认示例：

```bash
python -m visualize.plot_episode
```

## 推荐工作流

建议后续按下面流程使用：

1. 本地修改代码并提交
2. 训练模型
3. 关注训练日志中的 `Eval` 指标和 `scene_seed`
4. 如果某个 seed 表现异常，在本地直接复现该场景
5. 用 `run_policy` 和可视化检查路径是否真的“短且隐蔽”
6. 优先使用 `ddqn_best.pt` 继续分析

## 评估时建议关注什么

不建议只看 `success_rate`。

当前更应该同时看：

- 是否成功到达终点
- 路径长度 `path_length`
- 隐蔽比例 `hidden_ratio`
- 可见路径长度 `visible_path_length`
- 3D/俯视图可视化效果
- 与 `Visibility-A*` 的差距

## 当前主要入口文件

- [main.py](/d:/井九/Documents/大三寒假/temp/main.py)：环境与网络前向检查
- [train/train_ddqn.py](/d:/井九/Documents/大三寒假/temp/train/train_ddqn.py)：训练入口
- [eval/run_policy.py](/d:/井九/Documents/大三寒假/temp/eval/run_policy.py)：单场景评估与对比
- [visualize/plot_scene.py](/d:/井九/Documents/大三寒假/temp/visualize/plot_scene.py)：场景可视化
- [visualize/plot_episode.py](/d:/井九/Documents/大三寒假/temp/visualize/plot_episode.py)：路径与路径对比可视化

## 关于随机化场景

当前这轮重构**没有改动随机场景生成机制本身**，随机化逻辑仍然沿用原来的设计，主要还是：

- 随机障碍数量
- 随机障碍尺寸
- 随机障碍位置
- 随机障碍高度
- 随机起点
- 随机终点
- 敌人朝向仍然按原有逻辑朝向地图中心方向生成

我这次主要改的是：

- 奖励函数
- 环境反馈字段
- 可见性地图
- 基线规划器
- 训练/评估/可视化指标

没有去改：

- `train_scene_seeds / val_scene_seeds / test_scene_seeds`
- 随机场景生成的采样流程
- 障碍采样范围
- 起终点采样约束

如果你下一步想继续增强随机化场景，比如让敌人位置、敌人朝向、视场角、障碍形状也一起随机，我可以再继续帮你改这一层。
