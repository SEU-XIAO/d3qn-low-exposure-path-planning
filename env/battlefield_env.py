from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import acos, degrees, sqrt

import numpy as np

from config import EnvConfig


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, np.ndarray]
    reward: float
    done: bool
    info: dict[str, float | bool]


class BattlefieldEnv:
    ACTIONS = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        self.grid_size = self.config.grid_size
        self.height_levels = self.config.height_levels
        self.default_start = np.array(self.config.start, dtype=np.int32)
        self.default_goal = np.array(self.config.goal, dtype=np.int32)
        self.default_enemy_xy = np.array(self.config.enemy_position, dtype=np.int32)
        self.default_enemy_forward = np.array(self.config.enemy_forward, dtype=np.float32)

        self.agent_position = self.default_start.copy()
        self.goal_position = self.default_goal.copy()
        self.start_position = self.default_start.copy()
        self.enemy_position = np.array((self.default_enemy_xy[0], self.default_enemy_xy[1], 0.0), dtype=np.float32)
        self.enemy_forward = self._normalize(self.default_enemy_forward.copy())

        self.height_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.occupancy_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.visibility_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.cover_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32)

        self.steps = 0
        self.total_path_length = 0.0
        self.visible_path_length = 0.0
        self.hidden_path_length = 0.0
        self.current_scene_seed: int | None = None
        self.current_scenario_mode = self.config.scenario_mode

        self.reset()

    def reset(self, scene_seed: int | None = None, scenario_mode: str | None = None) -> dict[str, np.ndarray]:
        self.steps = 0
        self.total_path_length = 0.0
        self.visible_path_length = 0.0
        self.hidden_path_length = 0.0

        if scenario_mode is not None:
            self.current_scenario_mode = scenario_mode
        else:
            self.current_scenario_mode = self.config.scenario_mode

        if self.current_scenario_mode == "random":
            self.current_scene_seed = scene_seed if scene_seed is not None else int(np.random.randint(0, 10_000_000))
            self._generate_random_scene(self.current_scene_seed)
        else:
            self.current_scene_seed = scene_seed
            self._build_fixed_scene()

        self.agent_position = self.start_position.copy()
        return self.get_observation()

    def step(self, action: int) -> StepResult:
        self.steps += 1
        move = np.array(self.ACTIONS[action], dtype=np.int32)
        candidate = self.agent_position + move
        move_cost = self._move_cost(move)
        # 步进惩罚
        reward = -self.config.step_penalty * move_cost
        done = False
        collision = False
        previous_distance = self._goal_distance(self.agent_position)
        previous_hidden_ratio = self.hidden_ratio

        if self._is_blocked(candidate):
            # 碰撞惩罚
            reward -= self.config.collision_penalty
            collision = True
        else:
            # 维护路径长度相关的变量
            self.agent_position = candidate
            current_visibility = float(self.visibility_map[tuple(self.agent_position)])
            self.total_path_length += move_cost
            self.visible_path_length += move_cost * current_visibility
            self.hidden_path_length += move_cost * (1.0 - current_visibility)
            # 暴露惩罚
            reward -= self.config.visible_penalty * move_cost * current_visibility

        new_distance = self._goal_distance(self.agent_position)
        # 总体朝着目标接近的奖励
        reward += (previous_distance - new_distance) * self.config.progress_weight

        current_hidden_ratio = self.hidden_ratio
        # 路线被遮挡比例上升带来的奖励
        reward += self.config.hidden_ratio_gain_weight * (current_hidden_ratio - previous_hidden_ratio)

        success = np.array_equal(self.agent_position, self.goal_position)
        if success:
            reward += self.config.goal_reward
            reward += self.config.success_hidden_ratio_weight * current_hidden_ratio
            done = True

        if self.steps >= self.config.max_steps:
            done = True

        current_visibility = float(self.visibility_map[tuple(self.agent_position)])
        return StepResult(
            observation=self.get_observation(),
            reward=reward,
            done=done,
            info={
                "collision": collision,
                "visibility": current_visibility,
                "hidden_ratio": current_hidden_ratio,
                "path_length": self.total_path_length,
                "success": success,
            },
        )

    @property
    def hidden_ratio(self) -> float:
        if self.total_path_length <= 1e-6:
            return 1.0
        return float(self.hidden_path_length / self.total_path_length)

    @property
    def visible_ratio(self) -> float:
        if self.total_path_length <= 1e-6:
            return 0.0
        return float(self.visible_path_length / self.total_path_length)

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
            "local_map": self._extract_local_map(),
            "global_features": self._build_global_features(),
        }

    def _build_fixed_scene(self) -> None:
        self.height_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        center = self.grid_size // 2

        self.height_map[center - 3 : center + 4, center - 1 : center + 2] = 2
        self.height_map[center - 1 : center + 2, center - 3 : center + 4] = 2
        self.height_map[center - 2 : center + 3, center + 4 : center + 6] = 1
        self.height_map[center - 5 : center - 3, center - 5 : center + 5] = 1

        self.start_position = self.default_start.copy()
        self.goal_position = self.default_goal.copy()
        self.enemy_position = np.array((self.default_enemy_xy[0], self.default_enemy_xy[1], 0.0), dtype=np.float32)
        self.enemy_forward = self._normalize(self.default_enemy_forward.copy())
        self._finalize_scene_maps()

    def _generate_random_scene(self, scene_seed: int) -> None:
        rng = np.random.default_rng(scene_seed)

        for _ in range(128):
            self.height_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            self._place_random_obstacles(rng)
            start, goal = self._sample_start_goal(rng)

            self.start_position = np.array(start, dtype=np.int32)
            self.goal_position = np.array(goal, dtype=np.int32)
            self.enemy_position = np.array((self.default_enemy_xy[0], self.default_enemy_xy[1], 0.0), dtype=np.float32)
            self.enemy_forward = self._normalize(self._sample_enemy_forward(self.default_enemy_xy))
            self._finalize_scene_maps()

            if self._has_feasible_path():
                return

        raise RuntimeError(f"无法为 scene_seed={scene_seed} 生成可达场景")

    def _place_random_obstacles(self, rng: np.random.Generator) -> None:
        obstacle_mask = rng.random((self.grid_size, self.grid_size)) < self.config.obstacle_probability
        obstacle_heights = rng.integers(1, 3, size=(self.grid_size, self.grid_size), dtype=np.int32)
        self.height_map = np.where(obstacle_mask, obstacle_heights, 0).astype(np.int32)

    def _sample_start_goal(self, rng: np.random.Generator) -> tuple[tuple[int, int], tuple[int, int]]:
        corner_span = 3
        start_candidates = self._free_cells_in_region(0, corner_span - 1, 0, corner_span - 1)
        goal_start = max(0, self.grid_size - corner_span)
        goal_candidates = self._free_cells_in_region(goal_start, self.grid_size - 1, goal_start, self.grid_size - 1)
        enemy_xy = self.default_enemy_xy.astype(np.float32)

        if not start_candidates or not goal_candidates:
            return tuple(self.config.start), tuple(self.config.goal)

        for _ in range(512):
            start = start_candidates[int(rng.integers(0, len(start_candidates)))]
            goal = goal_candidates[int(rng.integers(0, len(goal_candidates)))]
            if start == goal:
                continue
            if np.linalg.norm(np.array(start, dtype=np.float32) - np.array(goal, dtype=np.float32)) < self.config.min_start_goal_distance:
                continue
            if np.linalg.norm(np.array(goal, dtype=np.float32) - enemy_xy) < self.config.enemy_goal_min_distance:
                continue
            return start, goal

        # Relax only the start-goal distance if the sampled map is too restrictive.
        for _ in range(256):
            start = start_candidates[int(rng.integers(0, len(start_candidates)))]
            goal = goal_candidates[int(rng.integers(0, len(goal_candidates)))]
            if start == goal:
                continue
            if np.linalg.norm(np.array(goal, dtype=np.float32) - enemy_xy) < self.config.enemy_goal_min_distance:
                continue
            return start, goal

        return tuple(self.config.start), tuple(self.config.goal)

    def _free_cells(self) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.height_map[x, y] == 0:
                    cells.append((x, y))
        return cells

    def _free_cells_in_region(self, x_min: int, x_max: int, y_min: int, y_max: int) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        x0 = max(0, x_min)
        x1 = min(self.grid_size - 1, x_max)
        y0 = max(0, y_min)
        y1 = min(self.grid_size - 1, y_max)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                if self.height_map[x, y] == 0:
                    cells.append((x, y))
        return cells

    def _sample_enemy_forward(self, enemy_xy: np.ndarray) -> np.ndarray:
        center = np.array((self.grid_size / 2.0, self.grid_size / 2.0), dtype=np.float32)
        return center - enemy_xy.astype(np.float32)

    def _finalize_scene_maps(self) -> None:
        self.occupancy_map = (self.height_map > 0).astype(np.float32)
        self._enforce_start_goal_free_cells()
        self.visibility_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.cover_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        self._recompute_visibility_map()

    def _enforce_start_goal_free_cells(self) -> None:
        # Hard constraint: start and goal must always be traversable cells.
        self.height_map[tuple(self.start_position)] = 0
        self.height_map[tuple(self.goal_position)] = 0
        self.occupancy_map[tuple(self.start_position)] = 0.0
        self.occupancy_map[tuple(self.goal_position)] = 0.0

    def _has_feasible_path(self) -> bool:
        start = tuple(self.start_position.tolist())
        goal = tuple(self.goal_position.tolist())
        queue: deque[tuple[int, int]] = deque([start])
        visited = {start}

        while queue:
            current = queue.popleft()
            if current == goal:
                return True
            current_arr = np.array(current, dtype=np.int32)
            for action_idx in self.get_valid_actions(current_arr):
                move = np.array(self.ACTIONS[action_idx], dtype=np.int32)
                neighbor = tuple((current_arr + move).tolist())
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return False

    def _extract_local_map(self) -> np.ndarray:
        radius = self.config.local_map_size // 2
        padded_occ = np.pad(self.occupancy_map, radius, mode="constant", constant_values=1.0)
        padded_visibility = np.pad(self.visibility_map, radius, mode="constant")
        padded_goal = np.pad(np.zeros_like(self.occupancy_map, dtype=np.float32), radius, mode="constant")

        ax, ay = self.agent_position + radius
        gx, gy = self.goal_position
        goal_x = gx - self.agent_position[0] + radius
        goal_y = gy - self.agent_position[1] + radius
        if 0 <= goal_x < self.config.local_map_size and 0 <= goal_y < self.config.local_map_size:
            padded_goal[ax - radius + goal_x, ay - radius + goal_y] = 1.0

        xs = slice(ax - radius, ax + radius + 1)
        ys = slice(ay - radius, ay + radius + 1)

        local_occ = padded_occ[xs, ys]
        local_visibility = padded_visibility[xs, ys]
        local_goal = padded_goal[xs, ys]

        return np.stack((local_occ, local_visibility, local_goal), axis=0).astype(np.float32)

    def _build_global_features(self) -> np.ndarray:
        relative_goal = (self.goal_position - self.agent_position).astype(np.float32) / self.grid_size
        relative_enemy = (self.enemy_position[:2] - self.agent_position.astype(np.float32)) / self.grid_size
        goal_distance = np.array([self._goal_distance(self.agent_position) / self.grid_size], dtype=np.float32)
        enemy_distance = np.array([np.linalg.norm(relative_enemy)], dtype=np.float32)
        enemy_forward = self.enemy_forward.astype(np.float32)
        current_visibility = np.array([self.visibility_map[tuple(self.agent_position)]], dtype=np.float32)
        hidden_ratio = np.array([self.hidden_ratio], dtype=np.float32)

        return np.concatenate(
            (relative_goal, relative_enemy, goal_distance, enemy_distance, enemy_forward, current_visibility, hidden_ratio),
            dtype=np.float32,
        )

    def _recompute_visibility_map(self) -> None:
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                visibility = self._compute_cell_visibility((x, y))
                self.visibility_map[x, y] = visibility
                self.cover_map[x, y] = 1.0 - visibility

    def _compute_cell_visibility(self, cell: tuple[int, int]) -> float:
        if self.occupancy_map[cell] > 0:
            return 0.0

        direction = np.array((cell[0], cell[1]), dtype=np.float32) - self.enemy_position[:2]
        distance = float(np.linalg.norm(direction))
        if distance < 1e-6 or distance > self.config.enemy_max_range:
            return 0.0

        direction_unit = direction / distance
        total_cosine = float(np.clip(np.dot(direction_unit, self.enemy_forward), -1.0, 1.0))
        horizontal_angle = degrees(acos(total_cosine))
        if horizontal_angle > self.config.enemy_horizontal_fov_deg / 2.0:
            return 0.0
        if self._is_occluded(cell):
            return 0.0
        return 1.0

    def _is_occluded(self, cell: tuple[int, int]) -> bool:
        start = self.enemy_position[:2]
        end = np.array((cell[0], cell[1]), dtype=np.float32)
        line_points = self._sample_line(start, end, samples=40)
        for point in line_points[1:-1]:
            gx = int(np.clip(round(point[0]), 0, self.grid_size - 1))
            gy = int(np.clip(round(point[1]), 0, self.grid_size - 1))
            if self.height_map[gx, gy] > 0:
                return True
        return False

    @staticmethod
    def _sample_line(start: np.ndarray, end: np.ndarray, samples: int) -> np.ndarray:
        ratios = np.linspace(0.0, 1.0, num=samples, dtype=np.float32)
        return start[None, :] * (1.0 - ratios[:, None]) + end[None, :] * ratios[:, None]

    def _is_blocked(self, position: np.ndarray) -> bool:
        x, y = int(position[0]), int(position[1])
        if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return True
        return bool(self.occupancy_map[x, y] > 0)

    def get_valid_actions(self, position: np.ndarray | None = None) -> list[int]:
        current = self.agent_position if position is None else position
        valid_actions: list[int] = []
        for action_idx, move in enumerate(self.ACTIONS):
            candidate = current + np.array(move, dtype=np.int32)
            if not self._is_blocked(candidate):
                valid_actions.append(action_idx)
        return valid_actions

    def _goal_distance(self, position: np.ndarray) -> float:
        return float(np.linalg.norm(self.goal_position.astype(np.float32) - position.astype(np.float32)))

    @staticmethod
    def _move_cost(move: np.ndarray) -> float:
        return sqrt(2.0) if abs(int(move[0])) + abs(int(move[1])) == 2 else 1.0

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return vector
        return vector / norm
