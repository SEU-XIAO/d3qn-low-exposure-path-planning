from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
import json
from pathlib import Path

import numpy as np

from config import EnvConfig
from env.terrain_loader import FullTerrain, load_terrain


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, np.ndarray]
    reward: float
    done: bool
    info: dict[str, float | bool]


class BattlefieldEnv:
    ACTIONS = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        self.grid_size = self.config.grid_size
        self.height_levels = self.config.height_levels
        self.default_start = np.array(self.config.start, dtype=np.int32)
        self.default_goal = np.array(self.config.goal, dtype=np.int32)
        self.default_enemy_xy = np.array(self.config.enemy_position, dtype=np.int32)

        self.agent_position = self.default_start.copy()
        self.goal_position = self.default_goal.copy()
        self.start_position = self.default_start.copy()
        self.enemy_position = np.array((self.default_enemy_xy[0], self.default_enemy_xy[1], 0.0), dtype=np.float32)
        self.enemy_pose_source = "default"
        self.enemy_pose_score = 0.0

        self.height_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.occupancy_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.visibility_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.cover_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32)

        self.steps = 0
        self.consecutive_collisions = 0
        self.total_path_length = 0.0
        self.visible_path_length = 0.0
        self.hidden_path_length = 0.0
        self.current_scene_seed: int | None = None
        self.current_scenario_mode = self.config.scenario_mode

        # ---- 全图模式状态 ----
        self.full_terrain: FullTerrain | None = None
        self.enemy_pool: list[tuple[int, int]] = []  # 全局坐标的敌人候选位置列表
        self.full_visibility_maps: list[np.ndarray] = []  # 预计算的 8 张全图可见性底图
        self.window_offset: tuple[int, int] = (0, 0)  # 窗口在全局图中的左上角偏移
        self.window_tag_map: np.ndarray | None = None  # 当前窗口的 tag 图
        self._enemy_episode_counter: int = 0  # 敌人切换计数器
        self._current_enemy_idx: int = 0  # 当前使用的敌人池索引

        if self.config.full_map_path:
            self._init_full_map_mode()

        self.reset()

    # ============================================================
    #  全图模式初始化
    # ============================================================

    def _init_full_map_mode(self) -> None:
        self.full_terrain = load_terrain(self.config.full_map_path)
        self.height_levels = int(self.full_terrain.height_map.max())

        # 加载敌人池 JSON
        pool_path = self.config.enemy_pool_path
        if pool_path and Path(pool_path).exists():
            with open(pool_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.enemy_pool = [tuple(e) for e in data.get("enemy_pool", [])]
        else:
            self.enemy_pool = []

        # 加载预计算可见性底图 NPZ
        vis_npz = Path(pool_path).parent / "visibility_maps.npz" if pool_path else None
        if vis_npz and vis_npz.exists():
            loaded = np.load(vis_npz)
            self.full_visibility_maps = [loaded[f"vis_{i}"] for i in range(len(self.enemy_pool))]
            print(f"已加载 {len(self.full_visibility_maps)} 张全图可见性底图")
        else:
            self.full_visibility_maps = []

    # ============================================================
    #  reset / step / observation
    # ============================================================

    def reset(self, scene_seed: int | None = None, scenario_mode: str | None = None) -> dict[str, np.ndarray]:
        self.steps = 0
        self.consecutive_collisions = 0
        self.total_path_length = 0.0
        self.visible_path_length = 0.0
        self.hidden_path_length = 0.0

        if scenario_mode is not None:
            self.current_scenario_mode = scenario_mode
        else:
            self.current_scenario_mode = self.config.scenario_mode

        if self.current_scenario_mode == "full_map":
            self.current_scene_seed = scene_seed if scene_seed is not None else int(np.random.randint(0, 10_000_000))
            self._generate_full_map_scene(self.current_scene_seed)
        elif self.current_scenario_mode == "random":
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
        reward = -self.config.step_penalty * move_cost
        done = False
        collision = False

        if self._is_blocked(candidate):
            reward -= self.config.collision_penalty
            collision = True
            self.consecutive_collisions += 1
        else:
            self.agent_position = candidate
            self.consecutive_collisions = 0
            current_visibility = float(self.visibility_map[tuple(self.agent_position)])
            self.total_path_length += move_cost
            self.visible_path_length += move_cost * current_visibility
            self.hidden_path_length += move_cost * (1.0 - current_visibility)
            reward -= self.config.visible_penalty * move_cost * current_visibility

        current_hidden_ratio = self.hidden_ratio

        success = np.array_equal(self.agent_position, self.goal_position)
        if success:
            reward += self.config.goal_reward
            reward += self.config.success_hidden_ratio_weight * current_hidden_ratio
            done = True

        if self.consecutive_collisions >= self.config.max_consecutive_collisions:
            reward -= self.config.timeout_penalty
            done = True

        if self.steps >= self.config.max_steps:
            if not success:
                remaining_dist = self._goal_distance(self.agent_position) / self.grid_size
                reward -= self.config.timeout_penalty * (1.0 + remaining_dist)
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

    # ============================================================
    #  全图窗口场景生成
    # ============================================================

    def _generate_full_map_scene(self, scene_seed: int) -> None:
        rng = np.random.default_rng(scene_seed)
        ft = self.full_terrain
        if ft is None:
            raise RuntimeError("full_terrain 未加载")

        # 从池中选敌人（全局坐标），每 N 个 episode 切换一次
        if not self.enemy_pool:
            raise RuntimeError("敌人池为空，请先运行 enemy_search.py 生成")
        if self._enemy_episode_counter % self.config.enemy_switch_interval == 0:
            self._current_enemy_idx = int(rng.integers(0, len(self.enemy_pool)))
        self._enemy_episode_counter += 1
        enemy_idx = self._current_enemy_idx
        enemy_global = self.enemy_pool[enemy_idx]
        self.enemy_position = np.array(
            (enemy_global[0], enemy_global[1], float(ft.height_map[enemy_global])),
            dtype=np.float32,
        )
        self.enemy_pose_source = "pool"

        # 选定当前敌人对应的全图可见性底图（预计算好的）
        current_full_vis = (
            self.full_visibility_maps[enemy_idx]
            if enemy_idx < len(self.full_visibility_maps)
            else None
        )

        max_ox = ft.full_width - self.grid_size
        max_oy = ft.full_height - self.grid_size

        for _ in range(256):
            ox = int(rng.integers(0, max_ox + 1))
            oy = int(rng.integers(0, max_oy + 1))

            self.window_offset = (ox, oy)
            self.height_map = ft.height_map[oy:oy + self.grid_size, ox:ox + self.grid_size].copy()
            self.window_tag_map = ft.tag_map[oy:oy + self.grid_size, ox:ox + self.grid_size].copy()

            passable_ratio = float((self.window_tag_map == 0).sum()) / (self.grid_size * self.grid_size)
            if passable_ratio < 0.3:
                continue

            start, goal = self._sample_start_goal_in_window(rng)
            self.start_position = np.array(start, dtype=np.int32)
            self.goal_position = np.array(goal, dtype=np.int32)

            if not self._has_feasible_path_window():
                continue

            # 从预计算底图切片得到可见性（无需射线计算）
            if current_full_vis is not None:
                self.visibility_map = current_full_vis[oy:oy + self.grid_size, ox:ox + self.grid_size].astype(np.float32)
                self.cover_map = 1.0 - self.visibility_map
                self.occupancy_map = (self.height_map.astype(np.float32) / max(1.0, float(self.height_levels)))
            else:
                self._finalize_scene_maps()

            return

        raise RuntimeError(f"无法为 scene_seed={scene_seed} 生成可达窗口场景")

    def _sample_start_goal_in_window(self, rng: np.random.Generator) -> tuple[tuple[int, int], tuple[int, int]]:
        corner_span = 5
        start_pool = [(x, y) for x in range(min(corner_span, self.grid_size))
                      for y in range(min(corner_span, self.grid_size))
                      if self.window_tag_map is not None and self.window_tag_map[x, y] == 0]
        g0 = max(0, self.grid_size - corner_span)
        goal_pool = [(x, y) for x in range(g0, self.grid_size)
                     for y in range(g0, self.grid_size)
                     if self.window_tag_map is not None and self.window_tag_map[x, y] == 0]

        if not start_pool:
            start_pool = [(0, 0)]
        if not goal_pool:
            goal_pool = [(self.grid_size - 1, self.grid_size - 1)]

        for _ in range(512):
            s = start_pool[int(rng.integers(0, len(start_pool)))]
            g = goal_pool[int(rng.integers(0, len(goal_pool)))]
            if s == g:
                continue
            dist = np.linalg.norm(np.array(s, dtype=np.float32) - np.array(g, dtype=np.float32))
            if dist < self.config.min_start_goal_distance:
                continue
            return s, g

        return tuple(self.config.start), tuple(self.config.goal)

    def _pick_enemy_in_window(self, rng: np.random.Generator) -> tuple[int, int] | None:
        """从预计算池中选一个落在当前窗口内的敌人位置（窗口坐标）。"""
        ox, oy = self.window_offset
        candidates: list[tuple[int, int]] = []
        for gx, gy in self.enemy_pool:
            wx = gx - ox
            wy = gy - oy
            if 0 <= wx < self.grid_size and 0 <= wy < self.grid_size:
                # 检查敌人所在格是否可通行（tag=0）
                if self.window_tag_map is not None and self.window_tag_map[wx, wy] == 0:
                    candidates.append((wx, wy))
        if not candidates:
            return None
        return candidates[int(rng.integers(0, len(candidates)))]

    def _fallback_enemy_in_window(self, rng: np.random.Generator) -> tuple[int, int]:
        """池中无可用敌人时，在窗口北侧选一个高点。"""
        fallback: list[tuple[int, int]] = []
        for x in range(self.config.enemy_region_width):
            for y in range(self.grid_size):
                if self.window_tag_map is not None and self.window_tag_map[x, y] == 0:
                    fallback.append((x, y))
        if not fallback:
            return (min(self.config.enemy_region_width - 1, self.grid_size - 1), self.grid_size // 2)
        # 按高度排序，取 top 10%
        fallback.sort(key=lambda c: self.height_map[c], reverse=True)
        top_n = max(1, len(fallback) // 10)
        return fallback[int(rng.integers(0, top_n))]

    def _has_feasible_path_window(self) -> bool:
        """窗口内的 BFS 可达性检查，仅走 tag=0 且满足爬坡约束的格子。"""
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

    def _update_enemy_height_from_full(self) -> None:
        """全图模式下从 full_terrain 读取敌人所在格的高度。"""
        ex, ey = int(self.enemy_position[0]), int(self.enemy_position[1])
        ox, oy = self.window_offset
        if self.full_terrain is not None:
            self.enemy_position[2] = float(self.full_terrain.height_map[oy + ex, ox + ey])
        else:
            self.enemy_position[2] = self._cell_height((ex, ey))

    def _compute_visibility_area_score_window(self, enemy_xy: tuple[int, int]) -> float:
        """窗口内以 enemy_xy 为观察者的可见格子总数。"""
        score = 0.0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                score += self._compute_cell_visibility_from(enemy_xy, (x, y))
        return score

    # ============================================================
    #  通行检查（tan 爬坡 + tag 约束）
    # ============================================================

    def can_move_between(self, current: tuple[int, int], candidate: tuple[int, int]) -> bool:
        current_height = self._cell_height(current)
        candidate_height = self._cell_height(candidate)
        dh = candidate_height - current_height
        if dh <= 0:
            return True
        dx = abs(candidate[0] - current[0])
        dy = abs(candidate[1] - current[1])
        if dx + dy == 2:  # 对角线
            dist = self.config.cell_size * sqrt(2.0)
        else:
            dist = self.config.cell_size
        return (dh / dist) <= self.config.max_climb_tan

    def _cell_passable(self, x: int, y: int) -> bool:
        """检查窗口坐标 (x, y) 是否可通行（tag=0）。全图模式检查 tag，随机模式始终可通行。"""
        if self.window_tag_map is not None:
            return bool(self.window_tag_map[x, y] == 0)
        return True

    def _is_blocked(self, position: np.ndarray, current: np.ndarray | None = None) -> bool:
        x, y = int(position[0]), int(position[1])
        if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return True
        if not self._cell_passable(x, y):
            return True
        # 全图模式下敌人可能在窗口外，仅敌人在窗口内时检查碰撞
        if self.current_scenario_mode != "full_map":
            if x == int(self.enemy_position[0]) and y == int(self.enemy_position[1]):
                return True
        if current is None:
            current = self.agent_position
        return not self.can_move_between((int(current[0]), int(current[1])), (x, y))

    def get_valid_actions(self, position: np.ndarray | None = None) -> list[int]:
        current = self.agent_position if position is None else position
        valid_actions: list[int] = []
        for action_idx, move in enumerate(self.ACTIONS):
            candidate = current + np.array(move, dtype=np.int32)
            if not self._is_blocked(candidate, current=current):
                valid_actions.append(action_idx)
        return valid_actions

    # ============================================================
    #  可见性计算（遮挡判定用全图高度）
    # ============================================================

    def _finalize_scene_maps(self) -> None:
        self.occupancy_map = (self.height_map.astype(np.float32) / max(1.0, float(self.height_levels))).astype(np.float32)
        self.visibility_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.cover_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        self._recompute_visibility_map()

    def _recompute_visibility_map(self) -> None:
        enemy_cell = (int(round(float(self.enemy_position[0]))), int(round(float(self.enemy_position[1]))))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                visibility = self._compute_cell_visibility_from(enemy_cell, (x, y))
                self.visibility_map[x, y] = visibility
                self.cover_map[x, y] = 1.0 - visibility

    def _compute_cell_visibility_from(
        self,
        observer_xy: tuple[int, int],
        cell: tuple[int, int],
    ) -> float:
        direction = np.array((cell[0], cell[1]), dtype=np.float32) - np.array(observer_xy, dtype=np.float32)
        distance = float(np.linalg.norm(direction))
        if distance < 1e-6:
            return 0.0
        if self._is_occluded_from(observer_xy, cell):
            return 0.0
        return 1.0

    def _is_occluded_from(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        """start 在 enemy_position 坐标系（全图模式=全局，否则=窗口），end 在窗口坐标。"""
        if start == end:
            return False

        if self.current_scenario_mode == "full_map" and self.full_terrain is not None:
            return self._is_occluded_global(start, end)

        # 随机/固定模式：都在窗口坐标内
        start_x, start_y = float(start[0]) + 0.5, float(start[1]) + 0.5
        end_x, end_y = float(end[0]) + 0.5, float(end[1]) + 0.5
        start_z = self._cell_height(start) + float(self.config.enemy_eye_height)
        end_z = self._cell_height(end) + float(self.config.target_visibility_height)

        length_xy = max(abs(end_x - start_x), abs(end_y - start_y))
        samples = max(2, int(length_xy * max(1, self.config.line_of_sight_samples_per_cell)))
        occluder_bias = float(self.config.visibility_occluder_bias)

        for i in range(1, samples):
            t = i / samples
            px = start_x + (end_x - start_x) * t
            py = start_y + (end_y - start_y) * t
            pz = start_z + (end_z - start_z) * t
            cx = int(np.clip(np.floor(px), 0, self.grid_size - 1))
            cy = int(np.clip(np.floor(py), 0, self.grid_size - 1))
            cell = (cx, cy)
            if cell == start or cell == end:
                continue
            if self._cell_height(cell) + occluder_bias >= pz:
                return True
        return False

    def _is_occluded_global(self, start_global: tuple[int, int], end_window: tuple[int, int]) -> bool:
        """全图模式：start 是全局坐标，end 是窗口坐标，射线在全局空间采样。"""
        ox, oy = self.window_offset
        ft = self.full_terrain
        H, W = ft.height_map.shape
        end_global = (end_window[0] + oy, end_window[1] + ox)

        if start_global == end_global:
            return False

        sx = float(start_global[0]) + 0.5
        sy = float(start_global[1]) + 0.5
        ex = float(end_global[0]) + 0.5
        ey = float(end_global[1]) + 0.5
        sz = float(ft.height_map[start_global]) + float(self.config.enemy_eye_height)
        ez = float(ft.height_map[end_global]) + float(self.config.target_visibility_height)

        length_xy = max(abs(ex - sx), abs(ey - sy))
        samples = max(2, int(length_xy * max(1, self.config.line_of_sight_samples_per_cell)))
        bias = float(self.config.visibility_occluder_bias)

        for i in range(1, samples):
            t = i / samples
            px = sx + (ex - sx) * t
            py = sy + (ey - sy) * t
            pz = sz + (ez - sz) * t
            cx = int(np.clip(np.floor(px), 0, H - 1))
            cy = int(np.clip(np.floor(py), 0, W - 1))
            cell = (cx, cy)
            if cell == start_global or cell == end_global:
                continue
            if float(ft.height_map[cell]) + bias >= pz:
                return True
        return False

    def _global_cell_height(self, cell: tuple[int, int]) -> float:
        """读取格子高度。全图模式下从 full_terrain 读取以支持窗口外遮挡判定。
        坐标惯例：(x, y) 对应 height_map[x, y]，其中 x=行, y=列。"""
        x, y = cell
        if self.full_terrain is not None:
            ox, oy = self.window_offset  # ox=列偏移, oy=行偏移
            global_row = x + oy
            global_col = y + ox
            if 0 <= global_row < self.full_terrain.full_height and 0 <= global_col < self.full_terrain.full_width:
                return float(self.full_terrain.height_map[global_row, global_col])
            return 0.0
        return float(self.height_map[x, y])

    def _cell_height(self, cell: tuple[int, int]) -> float:
        """读取格子高度（窗口视图）。"""
        return float(self.height_map[cell])

    # ============================================================
    #  固定场景 / 随机场景（保留向后兼容）
    # ============================================================

    def _build_fixed_scene(self) -> None:
        self.height_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        center = self.grid_size // 2

        self.height_map[center - 5: center + 6, center - 2: center + 3] = 2
        self.height_map[center - 2: center + 3, center - 5: center + 6] = 2
        self.height_map[center - 3: center + 4, center + 6: center + 9] = 1
        self.height_map[center - 8: center - 5, center - 8: center + 8] = 1

        self.start_position = self.default_start.copy()
        self.goal_position = self.default_goal.copy()
        self.enemy_position = np.array((self.default_enemy_xy[0], self.default_enemy_xy[1], 0.0), dtype=np.float32)
        self.enemy_pose_source = "fixed-default"
        self.window_offset = (0, 0)
        self.window_tag_map = None
        self._finalize_scene_maps()
        enemy_xy = tuple(int(v) for v in self.enemy_position[:2].astype(np.int32).tolist())
        self.enemy_pose_score = self._compute_visibility_area_score_window(enemy_xy)

    def _generate_random_scene(self, scene_seed: int) -> None:
        rng = np.random.default_rng(scene_seed)
        self.window_offset = (0, 0)
        self.window_tag_map = None

        for _ in range(128):
            self.height_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            self._place_random_obstacles(rng)
            start, goal = self._sample_start_goal(rng)

            self.start_position = np.array(start, dtype=np.int32)
            self.goal_position = np.array(goal, dtype=np.int32)

            enemy_xy, enemy_score = self._search_enemy_lookout_pose(rng)
            self.enemy_position = np.array((enemy_xy[0], enemy_xy[1], 0.0), dtype=np.float32)
            self.enemy_pose_source = "random-search"
            self.enemy_pose_score = float(enemy_score)
            self._finalize_scene_maps()

            if self._has_feasible_path():
                return

        raise RuntimeError(f"无法为 scene_seed={scene_seed} 生成可达场景")

    def _place_random_obstacles(self, rng: np.random.Generator) -> None:
        base = rng.integers(0, self.height_levels + 1, size=(self.grid_size, self.grid_size), dtype=np.int32)
        peaks_mask = rng.random((self.grid_size, self.grid_size)) < self.config.obstacle_probability
        peaks = rng.integers(1, self.height_levels + 1, size=(self.grid_size, self.grid_size), dtype=np.int32)
        self.height_map = np.maximum(base, np.where(peaks_mask, peaks, 0)).astype(np.int32)

    def _sample_start_goal(self, rng: np.random.Generator) -> tuple[tuple[int, int], tuple[int, int]]:
        corner_span = 5
        start_candidates = self._free_cells_in_region(0, corner_span - 1, 0, corner_span - 1)
        goal_start = max(0, self.grid_size - corner_span)
        goal_candidates = self._free_cells_in_region(goal_start, self.grid_size - 1, goal_start, self.grid_size - 1)
        enemy_xy = self.default_enemy_xy.astype(np.float32)
        enemy_cell = tuple(int(v) for v in enemy_xy.tolist())

        if not start_candidates or not goal_candidates:
            return tuple(self.config.start), tuple(self.config.goal)

        for _ in range(512):
            start = start_candidates[int(rng.integers(0, len(start_candidates)))]
            goal = goal_candidates[int(rng.integers(0, len(goal_candidates)))]
            if start == goal or start == enemy_cell or goal == enemy_cell:
                continue
            if np.linalg.norm(np.array(start, dtype=np.float32) - np.array(goal, dtype=np.float32)) < self.config.min_start_goal_distance:
                continue
            if np.linalg.norm(np.array(goal, dtype=np.float32) - enemy_xy) < self.config.enemy_goal_min_distance:
                continue
            return start, goal

        for _ in range(256):
            start = start_candidates[int(rng.integers(0, len(start_candidates)))]
            goal = goal_candidates[int(rng.integers(0, len(goal_candidates)))]
            if start == goal or start == enemy_cell or goal == enemy_cell:
                continue
            if np.linalg.norm(np.array(goal, dtype=np.float32) - enemy_xy) < self.config.enemy_goal_min_distance:
                continue
            return start, goal

        return tuple(self.config.start), tuple(self.config.goal)

    def _free_cells(self) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
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
                cells.append((x, y))
        return cells

    def _enemy_region_cells(self) -> list[tuple[int, int]]:
        width = max(1, min(self.grid_size, int(self.config.enemy_region_width)))
        side = self.config.enemy_region_side.lower()

        if side == "north":
            return self._free_cells_in_region(0, self.grid_size - 1, self.grid_size - width, self.grid_size - 1)
        if side == "south":
            return self._free_cells_in_region(0, self.grid_size - 1, 0, width - 1)
        if side == "east":
            return self._free_cells_in_region(self.grid_size - width, self.grid_size - 1, 0, self.grid_size - 1)
        if side == "west":
            return self._free_cells_in_region(0, width - 1, 0, self.grid_size - 1)
        return self._free_cells_in_region(0, self.grid_size - 1, self.grid_size - width, self.grid_size - 1)

    def _search_enemy_lookout_pose(self, rng: np.random.Generator) -> tuple[tuple[int, int], float]:
        free_cells = self._free_cells()
        region_cells = self._enemy_region_cells()
        start = tuple(self.start_position.tolist())
        goal = tuple(self.goal_position.tolist())

        candidates: list[tuple[int, int]] = []
        for cell in region_cells:
            if cell == start or cell == goal:
                continue
            goal_dist = float(np.linalg.norm(np.array(cell, dtype=np.float32) - self.goal_position.astype(np.float32)))
            if goal_dist < self.config.enemy_goal_min_distance:
                continue
            start_dist = float(np.linalg.norm(np.array(cell, dtype=np.float32) - self.start_position.astype(np.float32)))
            if start_dist < self.config.enemy_start_min_distance:
                continue
            candidates.append(cell)

        if not candidates:
            fallback_xy = tuple(int(v) for v in self.default_enemy_xy.tolist())
            fallback_score = self._compute_visibility_area_score_window(fallback_xy)
            return fallback_xy, fallback_score

        max_candidates = max(1, int(self.config.enemy_search_max_candidates))
        if len(candidates) > max_candidates:
            sampled_idx = rng.choice(len(candidates), size=max_candidates, replace=False)
            candidates = [candidates[int(idx)] for idx in sampled_idx]

        refine_topk = max(1, int(self.config.enemy_search_topk_refine))
        ranked = candidates[:]
        rng.shuffle(ranked)
        ranked = ranked[:refine_topk]

        best_score = -1.0
        best_candidate = ranked[0]
        for candidate in ranked:
            score = self._compute_visibility_area_score_window(candidate)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate, float(best_score)

    def _update_enemy_height(self) -> None:
        ex, ey = int(self.enemy_position[0]), int(self.enemy_position[1])
        self.enemy_position[2] = self._cell_height((ex, ey))

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

    # ============================================================
    #  observation 构建
    # ============================================================

    def _extract_local_map(self) -> np.ndarray:
        size = self.config.local_map_size
        grid = self.grid_size

        # 敌人在窗口内的位置（窗口坐标），不在窗口内则 clamp 到最近边界
        enemy_win_x = int(self.enemy_position[0])
        enemy_win_y = int(self.enemy_position[1])
        if self.current_scenario_mode == "full_map" and self.full_terrain is not None:
            ox, oy = self.window_offset
            enemy_win_x = enemy_win_x - ox
            enemy_win_y = enemy_win_y - oy
        enemy_win_x = max(0, min(grid - 1, enemy_win_x))
        enemy_win_y = max(0, min(grid - 1, enemy_win_y))

        if size >= grid:
            occ = self.occupancy_map.astype(np.float32)
            vis = self.visibility_map.astype(np.float32)
            goal = np.zeros_like(occ, dtype=np.float32)
            goal[tuple(self.goal_position)] = 1.0
            agent = np.zeros_like(occ, dtype=np.float32)
            agent[tuple(self.agent_position)] = 1.0
            enemy_ch = np.zeros_like(occ, dtype=np.float32)
            enemy_ch[enemy_win_x, enemy_win_y] = 1.0
            return np.stack((occ, vis, goal, agent, enemy_ch), axis=0).astype(np.float32)

        radius = size // 2
        padded_occ = np.pad(self.occupancy_map, radius, mode="constant", constant_values=1.0)
        padded_visibility = np.pad(self.visibility_map, radius, mode="constant")
        padded_goal = np.pad(np.zeros_like(self.occupancy_map, dtype=np.float32), radius, mode="constant")
        padded_agent = np.pad(np.zeros_like(self.occupancy_map, dtype=np.float32), radius, mode="constant")
        padded_enemy = np.pad(np.zeros_like(self.occupancy_map, dtype=np.float32), radius, mode="constant")

        ax, ay = self.agent_position + radius
        gx, gy = self.goal_position
        goal_x = gx - self.agent_position[0] + radius
        goal_y = gy - self.agent_position[1] + radius
        if 0 <= goal_x < size and 0 <= goal_y < size:
            padded_goal[ax - radius + goal_x, ay - radius + goal_y] = 1.0
        padded_agent[ax, ay] = 1.0
        # 敌人在智能体为中心的局部视野中
        enemy_lx = enemy_win_x - self.agent_position[0] + radius
        enemy_ly = enemy_win_y - self.agent_position[1] + radius
        if 0 <= enemy_lx < size and 0 <= enemy_ly < size:
            padded_enemy[ax - radius + enemy_lx, ay - radius + enemy_ly] = 1.0

        xs = slice(ax - radius, ax + radius + 1)
        ys = slice(ay - radius, ay + radius + 1)
        return np.stack(
            (padded_occ[xs, ys], padded_visibility[xs, ys], padded_goal[xs, ys], padded_agent[xs, ys], padded_enemy[xs, ys]),
            axis=0,
        ).astype(np.float32)

    def _build_global_features(self) -> np.ndarray:
        relative_goal = (self.goal_position - self.agent_position).astype(np.float32) / self.grid_size

        if self.current_scenario_mode == "full_map" and self.full_terrain is not None:
            ox, oy = self.window_offset
            agent_global = self.agent_position.astype(np.float32) + np.array((float(oy), float(ox)), dtype=np.float32)
            norm = float(max(self.full_terrain.full_height, self.full_terrain.full_width))
            relative_enemy = (self.enemy_position[:2] - agent_global) / norm
            enemy_distance = np.array([np.linalg.norm(self.enemy_position[:2] - agent_global) / norm], dtype=np.float32)
        else:
            relative_enemy = (self.enemy_position[:2] - self.agent_position.astype(np.float32)) / self.grid_size
            enemy_distance = np.array([np.linalg.norm(relative_enemy)], dtype=np.float32)

        goal_distance = np.array([self._goal_distance(self.agent_position) / self.grid_size], dtype=np.float32)
        current_visibility = np.array([self.visibility_map[tuple(self.agent_position)]], dtype=np.float32)
        hidden_ratio = np.array([self.hidden_ratio], dtype=np.float32)

        return np.concatenate(
            (relative_goal, relative_enemy, goal_distance, enemy_distance, current_visibility, hidden_ratio),
            dtype=np.float32,
        )

    # ============================================================
    #  工具方法
    # ============================================================

    def _goal_distance(self, position: np.ndarray) -> float:
        return float(np.linalg.norm(self.goal_position.astype(np.float32) - position.astype(np.float32)))

    @staticmethod
    def _move_cost(move: np.ndarray) -> float:
        return sqrt(2.0) if abs(int(move[0])) + abs(int(move[1])) == 2 else 1.0

