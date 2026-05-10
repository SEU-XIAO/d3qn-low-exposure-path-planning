from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    grid_size: int = 50
    height_levels: int = 8
    local_map_size: int = 50
    max_steps: int = 200

    scenario_mode: str = "full_map"

    cell_size: float = 10.0
    max_climb_tan: float = 0.3

    full_map_path: str = "MyPath_Data417.txt"
    enemy_pool_path: str = "artifacts/enemy_pool.json"
    enemy_pool_size: int = 8
    enemy_switch_interval: int = 50

    enemy_goal_min_distance: float = 16.0
    enemy_start_min_distance: float = 12.0
    enemy_region_width: int = 8
    enemy_region_side: str = "north"
    enemy_full_region_fraction: float = 0.25
    enemy_search_max_candidates: int = 150
    enemy_search_topk_refine: int = 38
    enemy_search_coarse_candidates: int = 500
    enemy_search_refine_candidates: int = 30
    enemy_search_final_candidates: int = 5

    obstacle_probability: float = 0.06
    enemy_eye_height: float = 1.0
    target_visibility_height: float = 0.5
    visibility_occluder_bias: float = 0.0
    line_of_sight_samples_per_cell: int = 2

    min_start_goal_distance: float = 30.0
    planner_guide_channel: bool = False
    planner_guide_sigma: float = 1.4
    planner_w_len: float = 1.0
    planner_w_vis: float = 2.5
    planner_w_slope: float = 0.8
    planner_w_turn: float = 0.15

    train_scene_seeds: tuple[int, ...] = tuple(range(1000, 4500))
    val_scene_seeds: tuple[int, ...] = tuple(range(5000, 5200))
    test_scene_seeds: tuple[int, ...] = tuple(range(6000, 6050))
    start: tuple[int, int] = (3, 3)
    goal: tuple[int, int] = (46, 46)
    enemy_position: tuple[int, int] = (25, 48)

    step_penalty: float = 0.05
    progress_weight: float = 0.2
    visible_penalty: float = 0.4
    goal_reward: float = 100.0
    waypoint_reached_reward: float = 10.0
    success_hidden_ratio_weight: float = 5.0
    collision_penalty: float = 1.0
    max_consecutive_collisions: int = 15
    timeout_penalty: float = 50.0
