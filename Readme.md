# D3QN 低暴露路径规划项目

本项目实现了一个“在敌方视野约束下进行路径规划”的完整实验链路，包含：

1. 环境建模（随机场景/固定场景、障碍与视野建模）
2. 强化学习智能体（D3QN：Double DQN + Dueling）
3. 规则规划器（Visibility A*、Weighted A*、Pareto A*）
4. 训练、评估、可视化与结果导出

项目核心目标是学习并比较“短且隐蔽”的路径。

---

## 1. 项目结构

```text
.
├─config.py                     # 全局配置（环境/模型/探索/训练默认参数）
├─main.py                       # 快速自检入口
├─env/
│  └─battlefield_env.py         # 核心环境
├─models/
│  └─policy_network.py          # D3QN 网络结构
├─train/
│  ├─dqn_agent.py               # Agent 与训练配置
│  ├─replay_buffer.py           # 经验回放
│  └─train_ddqn.py              # 训练主循环 + 评估/保存/早停
├─planner/
│  ├─visibility_astar.py        # 单目标可见代价 A*
│  ├─weighted_astar.py          # λ 加权版本（语义封装）
│  └─pareto_astar.py            # 多目标 Pareto A*
├─eval/
│  ├─run_policy.py              # 单场景对比评估（D3QN vs A*）
│  └─evaluate_100.py            # 批量评估 + Excel 导出
├─visualize/
│  ├─plot_scene.py              # 场景可视化
│  └─plot_episode.py            # 回合路径可视化/对比
├─artifacts/                    # 训练模型与评估输出
└─docs/
   ├─PARETO_ASTAR_DESIGN.md     # Pareto A* 详细设计
   └─PROJECT_MODULE_DESIGN.md   # 全模块参数级设计文档
```

---

## 2. 环境与依赖

建议 Python 3.10+。

安装依赖：

```bash
pip install -r requirements.txt
pip install pandas openpyxl
```

`requirements.txt` 当前包含：

- `numpy>=1.26`
- `matplotlib>=3.8`
- `torch>=2.2`

`evaluate_100.py` 导出 Excel 还需要 `pandas` 和 `openpyxl`。

---

## 3. 快速开始

### 3.1 环境与网络自检

```bash
python main.py
```

会打印观测张量形状与网络输出形状，确认环境和模型可正常前向。

### 3.2 启动训练

```bash
python -m train.train_ddqn
```

训练过程会周期性打印：

- `reward`
- `mean20`
- `loss`
- `epsilon`
- `path_len`
- `hidden_ratio`
- `success`
- 动作来源统计（`greedy/heuristic/teacher/random`）

并在 `artifacts/` 下保存模型：

- `ddqn_latest.pt`：周期保存/训练结束保存
- `ddqn_best.pt`：验证评估最优保存
- `ddqn_interrupt.pt`：中断保存（Ctrl+C）

---

## 4. 评估

### 4.1 单场景策略评估（命令行）

```bash
python -m eval.run_policy
```

默认会：

1. 加载 `artifacts/ddqn_best.pt`
2. 跑 D3QN 一局
3. 在同场景跑 Visibility-A*
4. 输出两者指标对比

### 4.2 批量评估并导出 Excel

```bash
python -m eval.evaluate_100
```

脚本当前默认参数（以代码为准）：

- `checkpoint_name="ddqn_best.pt"`
- `num_episodes=1000`
- `seed_start=7200`
- `scenario_mode="random"`

输出文件位于 `artifacts/`，格式如 `eval_100_YYYYMMDD_HHMMSS.xlsx`，包含：

1. `episodes` sheet：逐场景结果 + 最后一行 summary
2. `summary` sheet：总体均值与成功率

命令行也会打印总成功率。

> 说明：脚本文件名叫 `evaluate_100.py`，但默认评估数量是 1000（可在函数参数中调整）。

---

## 5. 可视化

### 5.1 场景可视化

```bash
python -m visualize.plot_scene
```

用于查看障碍分布、敌方视野覆盖以及俯视网格风格。

### 5.2 回合路径可视化

```bash
python -m visualize.plot_episode
```

当前默认入口是 `plot_comparison`，用于显示路径对比视图（含 D3QN 与 Pareto 规划结果）。

---

## 6. 算法组成

### 6.1 强化学习（D3QN）

- 动作空间：8 邻域移动
- 输入：局部地图 4 通道 + 全局特征 10 维
- 训练策略：Double DQN 目标、Dueling 头、经验回放、无效动作掩码

### 6.2 规划算法

1. `VisibilityAwareAStarPlanner`  
   最小化：`path_length + visible_weight * visible_path_length`

2. `WeightedVisibilityAStarPlanner`  
   上述方法的加权语义封装（可视为固定 λ 单目标）

3. `ParetoVisibilityAStarPlanner`  
   保留 `(path_length, visible_path_length)` 的非支配解集（多目标）

---

## 7. 关键约束与行为

1. 敌人所在格是硬障碍，不允许经过（环境与规划器已统一约束）。
2. 每 50 轮训练默认在验证种子集上评估（`val_scene_seeds`）。
3. 奖励中同时考虑步长、暴露、前进增益、隐蔽比增益、到达奖励与碰撞惩罚。

---

## 8. 建议阅读顺序

如果你要快速理解项目，建议按这个顺序看：

1. `config.py`
2. `env/battlefield_env.py`
3. `models/policy_network.py`
4. `train/dqn_agent.py`
5. `train/train_ddqn.py`
6. `planner/pareto_astar.py`
7. `visualize/plot_episode.py`

---

## 9. 文档导航

1. Pareto A* 详细设计：`docs/PARETO_ASTAR_DESIGN.md`
2. 全模块详细设计（参数级）：`docs/PROJECT_MODULE_DESIGN.md`

---

## 10. 许可证与用途

当前仓库以研究/实验为主要目标。若用于生产部署，建议补充：

1. 更严格的异常处理
2. 更完整的单元测试与回归测试
3. 模型版本化与评测基准管理
