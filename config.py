from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvConfig:
    # 地图边长（地图为 grid_size x grid_size）。
    grid_size: int = 32
    # 高度层级数量，仅用于视线遮挡计算（不是运动维度）。
    height_levels: int = 3
    # 视野输入边长（等于 grid_size 时即为全图）。
    local_map_size: int = 32
    # 每个 episode 最大步数，上限到达即终止。
    max_steps: int = 96

    # 场景模式："fixed" 固定地图，"random" 随机地图。
    scenario_mode: str = "random"
    # 敌人水平视场角（单位：度）。
    enemy_horizontal_fov_deg: float = 70.0
    # 敌人最大可见距离（单位：格）。
    enemy_max_range: float = 24.0
    # 敌人到目标点的最小距离约束（避免目标过近）。
    enemy_goal_min_distance: float = 10.0

    # 每个格子独立成为障碍的概率（伯努利采样）。
    obstacle_probability: float = 0.06

    # 起点与终点的最小距离约束（避免太近）。
    min_start_goal_distance: float = 18.0
    # 训练用随机场景种子集合。
    train_scene_seeds: tuple[int, ...] = tuple(range(1000, 4500))
    # 验证用随机场景种子集合。
    val_scene_seeds: tuple[int, ...] = tuple(range(5000, 5020))
    # 测试用随机场景种子集合。
    test_scene_seeds: tuple[int, ...] = tuple(range(6000, 6020))
    # 固定场景默认起点（或随机失败时兜底起点）。
    start: tuple[int, int] = (2, 2)
    # 固定场景默认终点（或随机失败时兜底终点）。
    goal: tuple[int, int] = (29, 29)
    # 固定场景敌人位置（或随机场景默认位置）。
    enemy_position: tuple[int, int] = (16, 31)
    # 敌人朝向向量（会归一化）。
    enemy_forward: tuple[float, float] = (0.0, -1)

    # 每一步基础惩罚，鼓励更短路径。
    step_penalty: float = 0.08
    # 处在可见区域的额外惩罚系数。
    visible_penalty: float = 1.25
    # 向目标接近的奖励权重（按距离变化计算）。
    progress_weight: float = 0.40
    # 隐蔽比例提升的奖励权重（基于 hidden_ratio 增量）。
    hidden_ratio_gain_weight: float = 0.75
    # 到达目标的终点奖励。
    goal_reward: float = 50.0
    # 成功后按隐蔽比例追加的奖励权重。
    success_hidden_ratio_weight: float = 4.0
    # 撞到障碍的惩罚。
    collision_penalty: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    # 局部/全局输入中的局部通道数（occupancy/visibility/goal/agent）。
    local_channels: int = 4
    # 全局特征向量的维度。
    global_feature_dim: int = 10


@dataclass(frozen=True)
class ExplorationConfig:
    # 是否启用启发式动作子集（朝目标方向偏置）。
    heuristic_subset_enabled: bool = True
    # 启发式子集的起始使用概率。
    heuristic_subset_prob_start: float = 0.50
    # 启发式子集的结束使用概率（随训练衰减）。
    heuristic_subset_prob_end: float = 0.10
    # 是否启用 teacher 动作（A* 引导）。
    teacher_enabled: bool = False
    # teacher 动作的起始使用概率。
    teacher_action_prob_start: float = 0.30
    # teacher 动作的结束使用概率。
    teacher_action_prob_end: float = 0.00


@dataclass(frozen=True)
class TrainingDefaults:
    # 训练使用的设备（"cuda" 或 "cpu"）。
    device: str = "cuda"
    # 训练总 episode 数。
    episodes: int = 10000
    # 批大小。
    batch_size: int = 256
    # 经验回放的大小一般在10w-100w之间
    replay_capacity: int = 300000
    # 折扣因子。
    gamma: float = 0.99
    # 学习率。
    learning_rate: float = 3e-4
    # 目标网络更新间隔（步数）。
    target_update_interval: int = 500
    # 预热步数（达到后开始训练）。
    warmup_steps: int = 2000
    # 训练频率（每隔多少步更新一次）。
    train_frequency: int = 1
    # epsilon-greedy 起始值。
    epsilon_start: float = 1.0
    # epsilon-greedy 结束值。
    epsilon_end: float = 0.05
    # epsilon 衰减步数。
    epsilon_decay_steps: int = 100000
    # 评估间隔（每隔多少 episode 评估一次）。
    eval_interval: int = 50
    # 保存间隔（每隔多少 episode 保存一次）。
    save_interval: int = 100
    # 梯度裁剪最大范数。
    max_gradient_norm: float = 5.0
    # 随机种子（影响训练可重复性）。
    seed: int = 42
    # 探索相关配置集合。
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    # 是否启用 early stop。
    early_stop_enabled: bool = True
    # early stop 评估时使用的 episode 数。
    early_stop_eval_episodes: int = 10
    # early stop 成功率阈值。
    early_stop_success_rate_threshold: float = 0.6
    # early stop 平台期容忍次数。
    early_stop_plateau_patience: int = 3
    # early stop 判断提升的最小增量。
    early_stop_min_delta: float = 0.05
