from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvConfig:
    # 地图边长（地图为 grid_size x grid_size）。
    grid_size: int = 50
    # 地形高度层级数量（全图模式从实际数据自动推断，随机模式使用此值）。
    height_levels: int = 8
    # 视野输入边长（等于 grid_size 时即为全图）。
    local_map_size: int = 50
    # 每个 episode 最大步数，上限到达即终止。50×50 地图 + 建筑障碍绕路成本高，200 步给足余量。
    max_steps: int = 200

    # 场景模式："fixed" 固定地图，"random" 随机地图，"full_map" 全图滑动窗口。
    scenario_mode: str = "full_map"

    # ---- 通行与爬坡 ----
    # 单个格子的实际尺寸（米），用于计算爬坡梯度。
    cell_size: float = 10.0
    # 爬坡能力正切值：相邻两格 height 差 / 水平距离 > 此值则不可通行。
    # 水平距离：直走 cell_size，对角线 cell_size * sqrt(2)。
    max_climb_tan: float = 0.3

    # ---- 全图模式配置 ----
    # 全图高度文件路径（.txt），为空时使用程序化地形生成。
    full_map_path: str = "MyPath_Data417.txt"
    # 预计算敌人位置池文件路径（JSON），为空时需运行 enemy_search.py 生成。
    enemy_pool_path: str = "artifacts/enemy_pool.json"
    # 敌人位置池大小（训练时从中随机选取，增加场景多样性）。
    enemy_pool_size: int = 8
    # 敌人切换间隔（episode 数）：同一敌人固定 N 个 episode 再切换，稳定学习信号。
    enemy_switch_interval: int = 50

    # ---- 敌人与可见性 ----
    # 敌人水平视场角（单位：度）。
    enemy_horizontal_fov_deg: float = 70.0
    # 敌人最大可见距离（单位：格）。
    enemy_max_range: float = 38.0
    # 敌人到目标点的最小距离约束（避免目标过近）。
    enemy_goal_min_distance: float = 16.0
    # 敌人到起点的最小距离约束（避免开局贴脸）。
    enemy_start_min_distance: float = 12.0
    # 敌人活动区域宽度（随机场景模式下，例如 50x50 中 8 表示 8x50 条带）。
    enemy_region_width: int = 8
    # 敌人活动条带所在边：north/south/east/west。
    enemy_region_side: str = "north"
    # 全图模式中敌人区域在整个全图的占比（北侧 1/4）。
    enemy_full_region_fraction: float = 0.25
    # 搜索最佳瞭望点时最多评估的候选敌人站位数量（随机场景模式）。
    enemy_search_max_candidates: int = 150
    # 两阶段搜索中进入精评估（含遮挡射线）的候选数量（随机场景模式）。
    enemy_search_topk_refine: int = 38
    # 全图敌人搜索：粗筛阶段采样的候选数量。
    enemy_search_coarse_candidates: int = 500
    # 全图敌人搜索：精筛阶段的候选数量（从粗筛 top 中选取）。
    enemy_search_refine_candidates: int = 30
    # 全图敌人搜索：最终验证阶段的候选数量。
    enemy_search_final_candidates: int = 5

    # 每个格子独立成为障碍的概率（伯努利采样，随机场景模式）。
    obstacle_probability: float = 0.06
    # 视线起点（敌人）相对地表高度偏移。
    enemy_eye_height: float = 1.0
    # 视线终点（目标格）相对地表高度偏移。
    target_visibility_height: float = 0.5
    # 遮挡高度偏置（可用于保守遮挡判定）。
    visibility_occluder_bias: float = 0.0
    # 每个网格长度使用多少个采样点做 3D 视线检测。
    line_of_sight_samples_per_cell: int = 2

    # 起点与终点的最小距离约束（避免太近）。
    min_start_goal_distance: float = 30.0
    # 训练用随机场景种子集合。
    train_scene_seeds: tuple[int, ...] = tuple(range(1000, 4500))
    # 验证用随机场景种子集合。
    val_scene_seeds: tuple[int, ...] = tuple(range(5000, 5200))
    # 测试用随机场景种子集合。
    test_scene_seeds: tuple[int, ...] = tuple(range(6000, 6050))
    # 固定场景默认起点（或随机失败时兜底起点）。
    start: tuple[int, int] = (3, 3)
    # 固定场景默认终点（或随机失败时兜底终点）。
    goal: tuple[int, int] = (46, 46)
    # 固定场景敌人位置（或随机场景默认位置）。
    enemy_position: tuple[int, int] = (25, 48)


    # 每一步基础惩罚，鼓励更短路径。
    step_penalty: float = 0.05
    # 向目标靠近的稠密奖励权重。
    progress_weight: float = 0.2
    # 处在可见区域的额外惩罚系数。0.4 意味着走 2~3 步暴露格子 ≈ 多走 1 步，允许必要时的短暂暴露。
    visible_penalty: float = 0.4
    # 到达目标的终点奖励。
    goal_reward: float = 100.0
    # 到达中间航点的奖励（航点模式）。
    waypoint_reached_reward: float = 10.0
    # 成功后按隐蔽比例追加的奖励权重（终局结算，不干扰过程决策）。
    success_hidden_ratio_weight: float = 5.0
    # 撞到障碍的惩罚。
    collision_penalty: float = 1.0
    # 连续撞墙 N 次后终止 episode（避免死循环浪费步数）。
    max_consecutive_collisions: int = 15
    # 超过最大步数仍未到达终点时的惩罚。
    timeout_penalty: float = 50.0


@dataclass(frozen=True)
class ModelConfig:
    # 局部/全局输入中的局部通道数（occupancy/visibility/goal/agent/enemy）。
    local_channels: int = 5
    # 全局特征向量的维度。
    global_feature_dim: int = 8
    # 是否使用 LSTM 记忆模块。
    use_lstm: bool = False
    # LSTM 隐层大小。
    lstm_hidden_size: int = 128


@dataclass(frozen=True)
class ExplorationConfig:
    # 是否启用启发式动作子集（朝目标方向偏置）。
    heuristic_subset_enabled: bool = True
    # 启发式子集的起始使用概率。
    heuristic_subset_prob_start: float = 0.50
    # 启发式子集的结束使用概率（随训练衰减）。
    heuristic_subset_prob_end: float = 0.20
    # 是否启用 teacher 动作（A* 引导）。
    teacher_enabled: bool = True
    # teacher 动作的起始使用概率。
    teacher_action_prob_start: float = 0.12
    # teacher 动作的结束使用概率。
    teacher_action_prob_end: float = 0.03
    # Teacher Visibility-A* 的 λ 起始值（可见性权重）。
    teacher_lambda_start: float = 12.0
    # Teacher Visibility-A* 的 λ 结束值。
    teacher_lambda_end: float = 3.0


@dataclass(frozen=True)
class WaypointConfig:
    """航点式分层决策配置。"""
    # 是否启用航点模式。
    enabled: bool = False
    # 沿 A* 路径每隔 N 步采样一个航点。
    interval: int = 15
    # 每段最多走 interval * multiplier 步，超时则段失败。
    max_segment_multiplier: float = 3.0
    # 航点生成时 Visibility-A* 的可见性权重（0=最短路径，>0=避开展露区）。
    waypoint_visible_weight: float = 6.0
    # 到达中间航点的奖励。
    waypoint_reached_reward: float = 10.0
    # 段超时的惩罚。
    segment_timeout_penalty: float = 20.0


@dataclass(frozen=True)
class TrainingDefaults:
    # 训练使用的设备（"cuda" 或 "cpu"）。
    device: str = "cuda"
    # 训练总 episode 数。
    episodes: int = 10000
    # 批大小。
    batch_size: int = 256
    # 经验回放容量。100k × ~20KB(uint8) ≈ 2GB，远小于原 300k×80KB=24GB。
    replay_capacity: int = 100000
    # 折扣因子。
    gamma: float = 0.99
    # 学习率（BC 预训练后微调，降低两个数量级防遗忘）。
    learning_rate: float = 1e-5
    # 目标网络更新间隔（步数）。
    target_update_interval: int = 500
    # 预热步数（BC 预训练后减少随机探索污染）。
    warmup_steps: int = 1000
    # 训练频率（每隔多少步更新一次）。
    train_frequency: int = 4
    # epsilon-greedy 起始值。
    epsilon_start: float = 1.0
    # epsilon-greedy 结束值。
    epsilon_end: float = 0.05
    # epsilon 衰减步数（同时控制 heuristic/teacher/lambda 衰减节奏，200k 让引导信号更持久）。
    epsilon_decay_steps: int = 200000
    # 评估间隔（每隔多少 episode 评估一次）。
    eval_interval: int = 50
    # 全量评估间隔（每隔多少 episode 使用全部验证场景评估一次，0 表示关闭）。
    full_eval_interval: int = 200
    # 保存间隔（每隔多少 episode 保存一次）。
    save_interval: int = 100
    # 梯度裁剪最大范数。
    max_gradient_norm: float = 1.0
    # n-step TD 步数（加速奖励传播）。
    n_step: int = 5
    # BC 正则化权重起始值（防止策略遗忘 BC 知识，随训练衰减）。
    bc_reg_weight_start: float = 0.5
    # BC 正则化权重结束值（衰减到接近 0，让 DQN 自由超越专家）。
    bc_reg_weight_end: float = 0.05
    # PER 优先级指数（0=均匀采样，1=完全按优先级）。
    per_alpha: float = 0.6
    # PER 重要性采样修正起始值（纠正非均匀采样的偏差）。
    per_beta_start: float = 0.4
    # PER 重要性采样修正结束值（训练后期收敛到 1.0）。
    per_beta_end: float = 1.0
    # PER 优先级小量（确保所有转移都有非零采样概率）。
    per_epsilon: float = 1e-6
    # BC 数据增强：每条路径额外采样的随机起点数。
    bc_augment_samples: int = 5
    # HER 重新标记：失败 episode 中用多少个"已到达位置"作为伪目标。
    her_relabel_count: int = 4
    # 航点式分层决策配置。
    waypoint: WaypointConfig = field(default_factory=WaypointConfig)
    # 随机种子。
    seed: int = 42
    # 探索相关配置集合。
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    # 是否启用 LSTM 记忆模块。
    use_lstm: bool = False
    # LSTM 记忆模块配置。
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1
    lstm_sequence_length: int = 10  # DRQN 训练时采样的序列长度
    # 反向课程学习配置。
    curriculum_enabled: bool = False
    curriculum_success_threshold: float = 0.7
    curriculum_window: int = 50
    curriculum_patience: int = 5  # 连续成功 N 次后进阶
    # 是否启用 early stop。
    early_stop_enabled: bool = True
    # early stop 评估时使用的 episode 数。
    early_stop_eval_episodes: int = 20
    # early stop 成功率阈值。
    early_stop_success_rate_threshold: float = 0.8
    # early stop 平台期容忍次数。
    early_stop_plateau_patience: int = 3
    # early stop 判断提升的最小增量。
    early_stop_min_delta: float = 0.05
