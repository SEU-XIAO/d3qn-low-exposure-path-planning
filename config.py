from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvConfig:
    grid_size: int = 32
    # Heights are only used for line-of-sight occlusion.
    height_levels: int = 3
    local_map_size: int = 11
    max_steps: int = 96

    scenario_mode: str = "random"
    enemy_horizontal_fov_deg: float = 70.0
    enemy_max_range: float = 24.0
    enemy_goal_min_distance: float = 10.0

    # Each grid cell independently becomes an obstacle with this probability.
    obstacle_probability: float = 0.06

    min_start_goal_distance: float = 18.0
    train_scene_seeds: tuple[int, ...] = tuple(range(1000, 4500))
    val_scene_seeds: tuple[int, ...] = tuple(range(5000, 5020))
    test_scene_seeds: tuple[int, ...] = tuple(range(6000, 6020))
    start: tuple[int, int] = (2, 2)
    goal: tuple[int, int] = (29, 29)
    enemy_position: tuple[int, int] = (16, 31)
    enemy_forward: tuple[float, float] = (0.0, -1)

    step_penalty: float = 0.08
    visible_penalty: float = 1.25
    progress_weight: float = 0.20
    hidden_ratio_gain_weight: float = 0.75
    goal_reward: float = 15.0
    success_hidden_ratio_weight: float = 4.0
    collision_penalty: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    local_channels: int = 4
    global_feature_dim: int = 10
    action_dim: int = 8


@dataclass(frozen=True)
class ExplorationConfig:
    heuristic_subset_enabled: bool = True
    heuristic_subset_prob_start: float = 0.85
    heuristic_subset_prob_end: float = 0.10
    teacher_enabled: bool = False
    teacher_action_prob_start: float = 0.30
    teacher_action_prob_end: float = 0.00


@dataclass(frozen=True)
class TrainingDefaults:
    device: str = "cuda"
    episodes: int = 10000
    batch_size: int = 256
    # 经验回放的大小一般在10w-100w之间
    replay_capacity: int = 300000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    target_update_interval: int = 500
    warmup_steps: int = 2000
    train_frequency: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20000
    eval_interval: int = 50
    save_interval: int = 100
    max_gradient_norm: float = 5.0
    seed: int = 42
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    early_stop_enabled: bool = True
    early_stop_eval_episodes: int = 3
    early_stop_success_rate_threshold: float = 1.0
    early_stop_plateau_patience: int = 3
    early_stop_min_delta: float = 0.05
