# 项目运行说明

## 项目概览

当前版本已经完成以下核心能力：

- `32 x 32 x 8` 三维战场栅格环境
- 地面二维移动 + 三维敌方视野与遮挡建模
- 固定场景与随机场景两种模式
- `D3QN` 智能体
- `epsilon-greedy + heuristic action subset` 探索
- 可选 `Risk-A* teacher action`
- `Risk-A*` 基线规划
- D3QN 与 `Risk-A*` 的 3D 可视化对比
- 评估驱动 best 保存、早停、中断保存

## 训练与泛化机制

当前训练默认采用“训练随机化、验证固定未见场景”的方式：

- 训练时：
  每个 episode 会从 `config.py` 的 `train_scene_seeds` 中随机采样一个 `scene_seed`，并以 `scenario_mode="random"` 生成场景。
- 验证时：
  会使用 `val_scene_seeds` 中固定的一组未见过的场景做评估。
- 测试时：
  可以使用 `test_scene_seeds` 做最终泛化测试。

这样做的目标是：

- 让模型不要只记住单一地图
- 让评估结果更接近真实泛化能力
- 让服务器训练日志里的 `scene_seed` 能在本地完全复现

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

早停默认开启，逻辑为：

- 连续多次评估达到目标成功率
- 且 `avg_reward` 相比最佳值不再明显提升
- 则提前停止训练，避免后期策略坍塌

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

## 模型文件

训练过程会在 `artifacts/` 下保存以下文件：

- `ddqn_latest.pt`：最新模型
- `ddqn_best.pt`：验证集表现最好的模型
- `ddqn_interrupt.pt`：训练被手动中断时保存的模型

实际部署或可视化时，建议优先使用 `ddqn_best.pt`，不要默认相信最后一轮的 `ddqn_latest.pt`。

## 复现场景

这是这个项目后续迭代里最重要的一点：服务器训练日志中只要打印出 `scene_seed`，我们就可以在本地复现同一个随机场景。

例如，如果服务器日志里出现：

```text
scene_seed=2003
```

那么本地就可以用同一个 seed 来做评估和可视化。

## 命令示例

查看固定场景：

```bash
python -m visualize.plot_scene
```

查看指定随机场景：

```python
from visualize.plot_scene import plot_scene

plot_scene(scene_seed=2003, scenario_mode="random")
```

打印模型在指定场景下的一回合决策过程：

```python
from eval.run_policy import run_episode

run_episode(checkpoint_name="ddqn_best.pt", scene_seed=2003, scenario_mode="random")
```

绘制模型在指定场景下的 3D 路径：

```python
from visualize.plot_episode import plot_episode

plot_episode(checkpoint_name="ddqn_best.pt", scene_seed=2003, scenario_mode="random")
```

绘制 D3QN 与 Risk-A* 的 3D 对比：

```python
from visualize.plot_episode import plot_comparison

plot_comparison(checkpoint_name="ddqn_best.pt", scene_seed=2003, scenario_mode="random")
```

## 推荐工作流

建议后续都按这个流程走：

1. 本地改代码并提交到 GitHub
2. 服务器拉取最新代码并训练
3. 关注训练日志中的 `Eval` 指标和 `scene_seed`
4. 如果某个 seed 表现异常，在本地直接复现该场景
5. 用 `run_policy` 和 3D 可视化检查“是否真的符合预期”
6. 优先以 `ddqn_best.pt` 作为候选模型继续分析

## 为什么不能只看 success_rate

到达终点不代表策略真的合理，常见问题包括：

- 虽然到终点了，但贴着高风险区域走
- 走法非常别扭，和我们想要的“低暴露路径”不一致
- 训练 episode 成功，但验证集泛化很差
- 最新模型退化，而 best 模型其实更好

所以评估时建议同时看：

- 是否成功到达终点
- 累计风险
- 路径长度
- 3D 可视化效果
- 与 `Risk-A*` 的差距

## 后续建议

下一步比较值得继续补强的是：

- 增加测试集批量评估脚本
- 输出每个 seed 的详细指标
- 保存失败场景截图
- 对比 `D3QN` 与 `Risk-A*` 的风险和路径长度分布
