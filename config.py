from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvConfig:
    grid_size: int = 32
    height_levels: int = 8
    local_map_size: int = 11
    max_steps: int = 96
    scenario_mode: str = "fixed"
    enemy_horizontal_fov_deg: float = 70.0
    enemy_vertical_fov_deg: float = 55.0
    enemy_max_range: float = 24.0
    enemy_height: int = 4
    obstacle_height: int = 5
    obstacle_half_span: int = 3
    random_obstacle_count_min: int = 5
    random_obstacle_count_max: int = 9
    random_obstacle_size_min: int = 2
    random_obstacle_size_max: int = 6
    min_start_goal_distance: float = 18.0
    train_scene_seeds: tuple[int, ...] = tuple(range(1000, 1100))
    val_scene_seeds: tuple[int, ...] = tuple(range(2000, 2020))
    test_scene_seeds: tuple[int, ...] = tuple(range(3000, 3020))
    start: tuple[int, int] = (2, 2)
    goal: tuple[int, int] = (29, 29)
    enemy_position: tuple[int, int] = (0, 16)
    enemy_forward: tuple[float, float, float] = (1.0, 0.0, 0.0)
    step_penalty: float = 0.05
    risk_weight: float = 1.2
    progress_weight: float = 0.15
    goal_reward: float = 15.0
    collision_penalty: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    local_channels: int = 4
    global_feature_dim: int = 11
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
    episodes: int = 3000
    batch_size: int = 256
    replay_capacity: int = 50000
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
