from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import acos, atan2, degrees

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
        self.enemy_position = np.array(
            (self.default_enemy_xy[0], self.default_enemy_xy[1], self.config.enemy_height),
            dtype=np.float32,
        )
        self.enemy_forward = self._normalize(self.default_enemy_forward.copy())

        self.height_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.occupancy_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.risk_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.steps = 0
        self.current_scene_seed: int | None = None
        self.current_scenario_mode = self.config.scenario_mode

        self.reset()

    def reset(self, scene_seed: int | None = None, scenario_mode: str | None = None) -> dict[str, np.ndarray]:
        self.steps = 0
        if scenario_mode is not None:
            self.current_scenario_mode = scenario_mode
        else:
            self.current_scenario_mode = self.config.scenario_mode

        if self.current_scenario_mode == "random":
            self.current_scene_seed = scene_seed if scene_seed is not None else np.random.randint(0, 10_000_000)
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
        reward = -self.config.step_penalty
        done = False
        collision = False
        previous_distance = self._goal_distance(self.agent_position)

        if self._is_blocked(candidate):
            reward -= self.config.collision_penalty
            collision = True
        else:
            self.agent_position = candidate

        current_risk = float(self.risk_map[tuple(self.agent_position)])
        reward -= current_risk * self.config.risk_weight

        new_distance = self._goal_distance(self.agent_position)
        reward += (previous_distance - new_distance) * self.config.progress_weight

        if np.array_equal(self.agent_position, self.goal_position):
            reward += self.config.goal_reward
            done = True

        if self.steps >= self.config.max_steps:
            done = True

        return StepResult(
            observation=self.get_observation(),
            reward=reward,
            done=done,
            info={"collision": collision, "risk": current_risk, "success": np.array_equal(self.agent_position, self.goal_position)},
        )

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
            "local_map": self._extract_local_map(),
            "global_features": self._build_global_features(),
        }

    def _build_fixed_scene(self) -> None:
        self.height_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        center = self.grid_size // 2
        span = self.config.obstacle_half_span

        self.height_map[center - span : center + span + 1, center - 1 : center + 2] = self.config.obstacle_height
        self.height_map[center - 1 : center + 2, center - span : center + span + 1] = self.config.obstacle_height
        self.height_map[center - 2 : center + 3, center + 4 : center + 6] = self.config.obstacle_height - 1
        self.height_map[center - 5 : center - 3, center - 5 : center + 5] = self.config.obstacle_height - 2

        self.start_position = self.default_start.copy()
        self.goal_position = self.default_goal.copy()
        self.enemy_position = np.array(
            (self.default_enemy_xy[0], self.default_enemy_xy[1], self.config.enemy_height),
            dtype=np.float32,
        )
        self.enemy_forward = self._normalize(self.default_enemy_forward.copy())
        self._finalize_scene_maps()

    def _generate_random_scene(self, scene_seed: int) -> None:
        rng = np.random.default_rng(scene_seed)

        for _ in range(128):
            self.height_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            self._place_random_obstacles(rng)
            start, goal = self._sample_start_goal(rng)
            enemy_xy = self._sample_enemy_position(rng)
            enemy_forward = self._sample_enemy_forward(enemy_xy)

            self.start_position = np.array(start, dtype=np.int32)
            self.goal_position = np.array(goal, dtype=np.int32)
            self.goal_position = np.array(goal, dtype=np.int32)
            self.enemy_position = np.array((enemy_xy[0], enemy_xy[1], self.config.enemy_height), dtype=np.float32)
            self.enemy_forward = self._normalize(enemy_forward)
            self._finalize_scene_maps()

            if self._has_feasible_path():
                return

        raise RuntimeError(f"无法为 scene_seed={scene_seed} 生成可达场景")

    def _place_random_obstacles(self, rng: np.random.Generator) -> None:
        obstacle_count = int(rng.integers(self.config.random_obstacle_count_min, self.config.random_obstacle_count_max + 1))
        for _ in range(obstacle_count):
            width = int(rng.integers(self.config.random_obstacle_size_min, self.config.random_obstacle_size_max + 1))
            height = int(rng.integers(self.config.random_obstacle_size_min, self.config.random_obstacle_size_max + 1))
            x = int(rng.integers(2, self.grid_size - width - 2))
            y = int(rng.integers(2, self.grid_size - height - 2))
            obstacle_h = int(rng.integers(max(2, self.config.obstacle_height - 2), self.config.obstacle_height + 1))
            self.height_map[x : x + width, y : y + height] = np.maximum(
                self.height_map[x : x + width, y : y + height], obstacle_h
            )

    def _sample_start_goal(self, rng: np.random.Generator) -> tuple[tuple[int, int], tuple[int, int]]:
        boundary_cells = self._boundary_cells()
        for _ in range(256):
            start = boundary_cells[int(rng.integers(0, len(boundary_cells)))]
            goal = boundary_cells[int(rng.integers(0, len(boundary_cells)))]
            if start == goal:
                continue
            if np.linalg.norm(np.array(start) - np.array(goal)) < self.config.min_start_goal_distance:
                continue
            if self.height_map[start] > 0 or self.height_map[goal] > 0:
                continue
            return start, goal
        return tuple(self.config.start), tuple(self.config.goal)

    def _sample_enemy_position(self, rng: np.random.Generator) -> tuple[int, int]:
        boundary_cells = self._boundary_cells()
        return boundary_cells[int(rng.integers(0, len(boundary_cells)))]

    def _sample_enemy_forward(self, enemy_xy: tuple[int, int]) -> np.ndarray:
        center = np.array((self.grid_size / 2.0, self.grid_size / 2.0, 1.5), dtype=np.float32)
        enemy_point = np.array((enemy_xy[0], enemy_xy[1], self.config.enemy_height), dtype=np.float32)
        return center - enemy_point

    def _boundary_cells(self) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        max_idx = self.grid_size - 1
        for i in range(self.grid_size):
            cells.append((0, i))
            cells.append((max_idx, i))
            cells.append((i, 0))
            cells.append((i, max_idx))
        return list(dict.fromkeys(cells))

    def _finalize_scene_maps(self) -> None:
        self.occupancy_map = (self.height_map > 0).astype(np.float32)
        self.height_map[tuple(self.start_position)] = 0
        self.height_map[tuple(self.goal_position)] = 0
        self.occupancy_map[tuple(self.start_position)] = 0.0
        self.occupancy_map[tuple(self.goal_position)] = 0.0
        self.risk_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._recompute_risk_map()

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
        padded_height = np.pad(self.height_map.astype(np.float32), radius, mode="constant")
        padded_risk = np.pad(self.risk_map, radius, mode="constant")
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
        local_height = padded_height[xs, ys] / max(1, self.height_levels - 1)
        local_risk = padded_risk[xs, ys]
        local_goal = padded_goal[xs, ys]

        return np.stack((local_occ, local_height, local_risk, local_goal), axis=0).astype(np.float32)

    def _build_global_features(self) -> np.ndarray:
        relative_goal = (self.goal_position - self.agent_position).astype(np.float32) / self.grid_size
        agent_point = np.array((self.agent_position[0], self.agent_position[1], 1.0), dtype=np.float32)
        relative_enemy = (self.enemy_position - agent_point) / self.grid_size
        goal_distance = np.array([self._goal_distance(self.agent_position) / self.grid_size], dtype=np.float32)
        enemy_distance = np.array([np.linalg.norm(relative_enemy)], dtype=np.float32)
        enemy_forward = self.enemy_forward.astype(np.float32)
        current_risk = np.array([self.risk_map[tuple(self.agent_position)]], dtype=np.float32)

        return np.concatenate(
            (relative_goal, relative_enemy, goal_distance, enemy_distance, enemy_forward, current_risk),
            dtype=np.float32,
        )

    def _recompute_risk_map(self) -> None:
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.risk_map[x, y] = self._compute_cell_risk((x, y))

    def _compute_cell_risk(self, cell: tuple[int, int]) -> float:
        if self.occupancy_map[cell] > 0:
            return 0.0

        target = np.array((cell[0], cell[1], 1.0), dtype=np.float32)
        direction = target - self.enemy_position
        distance = float(np.linalg.norm(direction))
        if distance < 1e-6 or distance > self.config.enemy_max_range:
            return 0.0

        direction_unit = direction / distance
        total_cosine = float(np.clip(np.dot(direction_unit, self.enemy_forward), -1.0, 1.0))
        total_angle = degrees(acos(total_cosine))
        if total_angle >= 89.9:
            return 0.0

        horizontal_angle, vertical_angle = self._compute_view_angles(direction_unit)
        if horizontal_angle > self.config.enemy_horizontal_fov_deg / 2.0:
            return 0.0
        if vertical_angle > self.config.enemy_vertical_fov_deg / 2.0:
            return 0.0
        if self._is_occluded(cell):
            return 0.0

        distance_risk = 1.0 - distance / self.config.enemy_max_range
        horizontal_risk = 1.0 - horizontal_angle / max(1.0, self.config.enemy_horizontal_fov_deg / 2.0)
        vertical_risk = 1.0 - vertical_angle / max(1.0, self.config.enemy_vertical_fov_deg / 2.0)
        return float(np.clip(0.45 * distance_risk + 0.35 * horizontal_risk + 0.20 * vertical_risk, 0.0, 1.0))

    def _is_occluded(self, cell: tuple[int, int]) -> bool:
        target_height = 1.0
        line_points = self._sample_line(self.enemy_position, np.array((cell[0], cell[1], target_height), dtype=np.float32), samples=40)
        for point in line_points[1:-1]:
            gx = int(np.clip(round(point[0]), 0, self.grid_size - 1))
            gy = int(np.clip(round(point[1]), 0, self.grid_size - 1))
            obstacle_height = float(self.height_map[gx, gy])
            if obstacle_height > 0 and obstacle_height >= float(point[2]):
                return True
        return False

    @staticmethod
    def _sample_line(start: np.ndarray, end: np.ndarray, samples: int) -> np.ndarray:
        ratios = np.linspace(0.0, 1.0, num=samples, dtype=np.float32)
        return start[None, :] * (1.0 - ratios[:, None]) + end[None, :] * ratios[:, None]

    def _compute_view_angles(self, direction_unit: np.ndarray) -> tuple[float, float]:
        forward_xy = self.enemy_forward[:2]
        direction_xy = direction_unit[:2]
        forward_xy_norm = float(np.linalg.norm(forward_xy))
        direction_xy_norm = float(np.linalg.norm(direction_xy))

        if forward_xy_norm < 1e-6 or direction_xy_norm < 1e-6:
            horizontal_angle = 0.0
        else:
            horizontal_cosine = float(
                np.clip(np.dot(forward_xy / forward_xy_norm, direction_xy / direction_xy_norm), -1.0, 1.0)
            )
            horizontal_angle = degrees(acos(horizontal_cosine))

        forward_pitch = degrees(atan2(float(self.enemy_forward[2]), max(forward_xy_norm, 1e-6)))
        direction_pitch = degrees(atan2(float(direction_unit[2]), max(direction_xy_norm, 1e-6)))
        vertical_angle = abs(direction_pitch - forward_pitch)
        return horizontal_angle, vertical_angle

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
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return vector
        return vector / norm
