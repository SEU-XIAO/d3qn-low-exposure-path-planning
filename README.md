# 低暴露路径规划项目

这是一个面向复杂地形和敌方视线约束的路径规划项目。当前主线已经从早期的 D3QN 原型，演进为一套更完整的系统：

- 在 `10x10` 局部地图上训练 PPO 局部执行器
- 在 `50x50` 场景上使用“全局隐蔽路径 + waypoint + 局部执行”的分层闭环
- 具备训练、固定验证、失败分析、轨迹可视化、基础设施审计、配置集中管理等能力

如果只看一句话，可以这样概括当前方案：

**让规则规划器负责全局结构，让 RL 负责局部执行与局部泛化，在保证成功率的前提下持续降低暴露和对兜底规则的依赖。**

---

## 1. 先看什么

第一次接手这个仓库，推荐按下面顺序阅读：

1. `README.md`
2. `docs/CURRENT_PROJECT_DOCUMENTATION.md`
3. `experiments/local_v2.toml`
4. `config.py`
5. `env/`
6. `planner/`
7. `models/`
8. `train/`
9. `visualize/`

其中最重要的当前版本总文档是：

- `docs/CURRENT_PROJECT_DOCUMENTATION.md`

这份文档覆盖：

- 项目目标与当前阶段判断
- 数据语义与坐标约定
- 环境、观测、模型、训练、评测、可视化
- `10x10 -> 50x50` 的整条分层链路
- 当前关键实验结果
- 已知问题、风险点、排查方式和后续建议

注意：

- `docs/Readme.md`
- `docs/README_PROJECT.md`
- `docs/PROJECT_MODULE_DESIGN.md`

这些旧文档仍有参考价值，但更偏向早期阶段，不应直接替代当前主线说明。

---

## 2. 当前主线

当前推荐主线不是“端到端直接学 50x50 全图寻路”，而是：

1. 从全图地形 `MyPath_Data417.txt` 生成敌点池和全图可见性底图
2. 从全图切出 `10x10` 局部窗口池，重点使用 `subtask` 分布
3. 在局部窗口池上训练 PPO 局部执行器
4. 在 `50x50` 上用隐蔽代价 A* 规划全局路径
5. 从全局路径提取 waypoint
6. 把“到下一个 waypoint”的局部子任务交给 RL 执行
7. 用固定离线评测、失败模式分析、分层闭环评测和轨迹图验证系统表现

当前推荐配置以：

- `experiments/local_v2.toml`

为准。

---

## 3. 当前状态

从工程完成度角度看，这个项目已经不是“只有若干原型脚本”的状态，而是一个比较完整的实验系统。当前已经具备：

- 统一的环境与观测协议
- 基于 `toml` 的实验配置管理
- 训练时自动保存最终生效配置
- `10x10` 固定验证集离线评测
- 失败样本导出与失败模式标注
- `50x50` 分层闭环评测
- 轨迹可视化与单局检查
- `txt -> terrain -> visibility -> window_pool` 的基础设施审计

按照当前记录，`local_v2` 主线已经达到：

- `10x10` 固定验证集无兜底评测：`success_rate ≈ 99.8%`
- `50x50` 分层闭环评测：`success_rate = 100%`
- 当前主结果中，`50x50` 不依赖 fallback 才能成功

但也要保留一个工程判断：

- 当前系统已经很强
- 但仍然有少量基础设施语义问题需要说明清楚
- 当前最重要的风险不再是“RL 学不会”，而是“基础设施语义是否完全一致、结果是否足够可信”

---

## 4. 快速开始

以下命令默认在仓库根目录执行。

### 4.1 一次性生成敌点池和全图可见性底图

```bash
python -m env.enemy_search
```

作用：

- 从全图中选出敌方观察点
- 为每个敌点生成一张全图可见性底图
- 输出到 `artifacts/enemy_pool.json` 和 `artifacts/visibility_maps.npz`

### 4.2 生成 `10x10` 子任务池

```bash
python -m env.build_window_pool_from_fullmap --grid 10 --num 30000 --output artifacts/window_pool_10_subtask_30k.npz --task-mode subtask --subtask-min-dist 2.0 --subtask-max-dist-ratio 0.6 --bucket-ratios 0.45,0.35,0.20 --seed 42 --max-attempt-factor 60
```

作用：

- 从全图切出局部窗口
- 按 `subtask` 分布采样起终点
- 保证样本可达
- 输出 `npz` 训练池

### 4.3 训练当前推荐的 PPO 局部执行器

```bash
python -m train.train_local_pool_ppo --config experiments/local_v2.toml
```

作用：

- 按 `local_v2` 配置加载局部池
- 构建带附加观测通道的局部环境
- 使用 PPO 训练 `10x10` 局部执行器
- 自动保存 `policy_best.pt`、`policy_final.pt`、`effective_config.json`、`source_config.toml`

### 4.4 先做一个冒烟评测

```bash
python -m visualize.failure_case_report --config experiments/local_v2.toml --config-section failure_case_report_smoke
```

作用：

- 在较小样本上快速验证模型和评测链路是否正常

### 4.5 做完整的 `10x10` 固定验证集评测

```bash
python -m visualize.failure_case_report --config experiments/local_v2.toml
```

作用：

- 统计成功率、超时率、平均步数、平均暴露
- 导出失败地图、`summary.json`、`cases.jsonl`、`cases.csv`

### 4.6 做失败模式分析

```bash
python -m visualize.failure_pattern_report --input-dir analysis/local_v2_eval10
```

作用：

- 将失败样本打上 `timeout_near_goal`、`oscillation`、`stagnation`、`path_deviation` 等标签

### 4.7 做 `50x50` 分层闭环评测

```bash
python -m visualize.hierarchical_waypoint_eval --config experiments/local_v2.toml
```

作用：

- 在完整 `50x50` 场景上验证“全局规划 + waypoint + 局部执行”整条链路

### 4.8 生成汇报用轨迹图

```bash
python -m visualize.rollout_plot --config experiments/local_v2.toml --model artifacts/checkpoints_local_v2/policy_best.pt --episodes 12 --seed 20260511 --output-dir analysis/report_rollouts_20260511
```

作用：

- 导出带底图、planner path、RL trajectory、waypoint 编号的轨迹图

---

## 5. 仓库结构

```text
.
├─ artifacts/        中间产物、池文件、checkpoint 等
├─ analysis/         评测结果、失败分析、rollout 图等输出
├─ docs/             文档
├─ env/              环境、地形、遮挡、观测、池构建
├─ experiments/      toml 实验配置
├─ models/           策略网络
├─ planner/          全局隐蔽路径规划与 waypoint 提取
├─ train/            PPO 训练与课程逻辑
├─ visualize/        评测、分析、可视化、基础设施检查
├─ config.py         环境参数 dataclass
├─ experiment_config.py
├─ main.py
└─ MyPath_Data417.txt
```

更详细的模块说明请看：

- `docs/CURRENT_PROJECT_DOCUMENTATION.md`

---

## 6. 关键设计思想

### 6.1 为什么不是直接端到端学 50x50

因为当前任务同时包含：

- 全局路径结构选择
- 地形约束
- 敌方可见性约束
- 局部绕障执行
- 长时序信用分配

如果直接端到端在 `50x50` 上学，训练难度、样本效率、奖励稀疏性和调参成本都会非常高。

因此当前方案采用分层设计：

- 规则规划器负责全局几何结构
- RL 负责局部动作选择和局部泛化
- 两者之间通过 waypoint 对接

### 6.2 为什么局部训练改成 `subtask`

一个关键经验是：

**10x10 训练分布如果和 50x50 分层执行时的真实子任务分布不一致，局部策略即使在训练集上表现很好，也不一定能迁移到分层执行。**

因此当前更推荐：

- 用 `subtask` 或 `mixed` 分布构造局部池
- 不再只依赖最早的 `corner` 采样

### 6.3 为什么允许系统层保留规则兜底

当前目标不是追求“绝对纯 RL”，而是追求：

- 最终系统可用
- 可解释
- 成功率高
- 暴露低
- 并且逐步降低 fallback 使用率

所以工程上允许存在规则兜底，但它更像是：

- 系统安全带
- 分析工具
- 能帮助分离“策略能力”和“系统最终成功率”的辅助机制

---

## 7. 当前最重要的已知问题

### 7.1 地形 `txt` 不是严格矩形

`MyPath_Data417.txt` 存在行长度不一致的问题。当前 `terrain_loader.py` 会把短行尾部缺失区域补成：

- `height = 0`
- `tag = 1`

这会制造一批“通行上是建筑、遮挡上却接近透明”的假格子。

这不是当前主线崩掉的原因，但它是基础设施层面最值得继续修正的问题。

### 7.2 敌点选择优化的是代理目标

当前敌点选择算法比随机点明显有效，但它优化的是：

- 高地代理分数
- 开阔性
- 分散性

不是直接优化“真实地面可见覆盖最大化”。

### 7.3 旧版轨迹图曾有可视化错位问题

本轮已经修复底图额外转置导致的错位问题。如果你在旧目录里看到“free 格子像障碍”的图像，需要先确认图是不是修复前生成的。

---

## 8. 推荐查看的输出目录

- `analysis/local_v2_eval10/`
- `analysis/local_v2_eval50/`
- `analysis/report_rollouts_20260511/`
- `analysis/local_v2_rollout_plot_fixed/`

这些目录通常包含：

- `summary.json`
- `cases.jsonl`
- `fail_maps/`
- rollout 图片
- 汇报用样例图

---

## 9. 这份仓库现在最适合怎么用

如果你的目标是继续推进当前主线，最推荐的工作方式是：

1. 固定使用 `experiments/local_v2.toml` 作为主配置入口
2. 所有新实验尽量在 `experiments/` 下集中管理
3. 每轮训练后都跑固定 `10x10` 评测和 `50x50` 分层评测
4. 对异常高分结果，优先用 rollout 图和基础设施审计交叉验证
5. 下一步重点放在语义统一、基础设施修补和可信度加固，而不是盲目加大模型或无止境调参

---

## 10. 文档导航

建议重点阅读下面几份文档：

- `docs/CURRENT_PROJECT_DOCUMENTATION.md`
- `docs/COMMANDS.md`
- `docs/DATA_INPUT_TYPE_SPEC.md`
- `docs/PARETO_ASTAR_DESIGN.md`

如果你只看一份，就看：

- `docs/CURRENT_PROJECT_DOCUMENTATION.md`

