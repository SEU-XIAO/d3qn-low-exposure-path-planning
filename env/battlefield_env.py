"""鎴樺満鍦烘櫙鐢熸垚涓庨€氳妫€鏌ユā鍧椼€?
璐熻矗锛氬湴褰㈠姞杞姐€佺獥鍙ｉ噰鏍枫€佽捣鐐?缁堢偣/鏁屼汉閫夊彇銆佸満鏅敓鎴愩€?閫氳鎬ч獙璇侊紙鏀€鐖害鏉?+ 鏍囩绾︽潫锛夈€佸彲瑙佹€ц绠楋紙濮旀墭缁?occlusion 妯″潡锛夈€?"""

from __future__ import annotations

from collections import deque
import json
from math import sqrt
from pathlib import Path

import numpy as np

from config import EnvConfig
from env.obs import build_obs
from env.terrain_loader import FullTerrain, load_terrain
from env.occlusion import is_occluded
from planner import StealthCostConfig, plan_stealth_path


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

        self.current_scene_seed: int | None = None
        self.current_scenario_mode = self.config.scenario_mode
        self.steps = 0
        self.consecutive_collisions = 0
        self.total_collisions = 0
        self.current_progress_weight = self.config.progress_weight
        self._obs_visited = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._obs_prev_move = np.zeros((2,), dtype=np.float32)
        self._obs_goal_cache: tuple[int, int] | None = None
        self._obs_goal_best_dist = 0.0
        self._obs_goal_stagnation = 0

        # ---- 鍏ㄥ浘妯″紡鐘舵€?----
        self.full_terrain: FullTerrain | None = None
        self.enemy_pool: list[tuple[int, int]] = []
        self.full_visibility_maps: list[np.ndarray] = []
        self.window_offset: tuple[int, int] = (0, 0)
        self.window_tag_map: np.ndarray | None = None

        if self.config.full_map_path:
            self._init_full_map_mode()

        self.generate_scene()

    # ============================================================
    #  鍏ㄥ浘妯″紡鍒濆鍖?    # ============================================================

    def _init_full_map_mode(self) -> None:
        self.full_terrain = load_terrain(self.config.full_map_path)
        self.height_levels = int(self.full_terrain.height_map.max())

        pool_path = self.config.enemy_pool_path
        if pool_path and Path(pool_path).exists():
            with open(pool_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.enemy_pool = [tuple(e) for e in data.get("enemy_pool", [])]
        else:
            self.enemy_pool = []

        vis_npz = Path(pool_path).parent / "visibility_maps.npz" if pool_path else None
        if vis_npz and vis_npz.exists():
            loaded = np.load(vis_npz)
            self.full_visibility_maps = [loaded[f"vis_{i}"] for i in range(len(self.enemy_pool))]
            print(f"已加载 {len(self.full_visibility_maps)} 张全图可见性底图")
        else:
            self.full_visibility_maps = []

    # ============================================================
    #  鍦烘櫙鐢熸垚锛堝叕寮€鍏ュ彛锛?    # ============================================================

    def generate_scene(
        self,
        scene_seed: int | None = None,
        scenario_mode: str | None = None,
        window_offset: tuple[int, int] | None = None,
    ) -> None:
        if scenario_mode is not None:
            self.current_scenario_mode = scenario_mode
        else:
            self.current_scenario_mode = self.config.scenario_mode

        if self.current_scenario_mode == "full_map":
            self.current_scene_seed = scene_seed if scene_seed is not None else int(np.random.randint(0, 10_000_000))
            self._generate_full_map_scene(self.current_scene_seed, window_offset=window_offset)
        elif self.current_scenario_mode == "random":
            self.current_scene_seed = scene_seed if scene_seed is not None else int(np.random.randint(0, 10_000_000))
            self._generate_random_scene(self.current_scene_seed)
        else:
            self.current_scene_seed = scene_seed
            self._build_fixed_scene()

        self.agent_position = self.start_position.copy()
        self._reset_obs_state()

    # ============================================================
    #  鍏ㄥ浘绐楀彛鍦烘櫙鐢熸垚
    # ============================================================

    def _generate_full_map_scene(
        self, scene_seed: int, window_offset: tuple[int, int] | None = None,
    ) -> None:
        rng = np.random.default_rng(scene_seed)
        ft = self.full_terrain
        if ft is None:
            raise RuntimeError("full_terrain 未加载")

        if not self.enemy_pool:
            raise RuntimeError("鏁屼汉姹犱负绌猴紝璇峰厛杩愯 enemy_search.py 鐢熸垚")

        enemy_idx = int(rng.integers(0, len(self.enemy_pool)))
        enemy_global = self.enemy_pool[enemy_idx]
        self.enemy_position = np.array(
            (enemy_global[0], enemy_global[1], float(ft.height_map[enemy_global])),
            dtype=np.float32,
        )
        self.enemy_pose_source = "pool"

        current_full_vis = (
            self.full_visibility_maps[enemy_idx]
            if enemy_idx < len(self.full_visibility_maps)
            else None
        )

        max_ox = ft.full_width - self.grid_size
        max_oy = ft.full_height - self.grid_size

        # 鎸囧畾绐楀彛鍋忕Щ閲忔椂璺宠繃闅忔満鎼滅储
        offsets_to_try: list[tuple[int, int]]
        if window_offset is not None:
            ox, oy = window_offset
            if not (0 <= ox <= max_ox and 0 <= oy <= max_oy):
                raise ValueError(f"window_offset {window_offset} 瓒呭嚭鑼冨洿 "
                                 f"[0,{max_ox}] 脳 [0,{max_oy}]")
            offsets_to_try = [(ox, oy)]
        else:
            offsets_to_try = [(int(rng.integers(0, max_ox + 1)),
                               int(rng.integers(0, max_oy + 1)))
                              for _ in range(256)]

        for ox, oy in offsets_to_try:
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

            if current_full_vis is not None:
                self.visibility_map = current_full_vis[oy:oy + self.grid_size, ox:ox + self.grid_size].astype(np.float32)
                self.cover_map = 1.0 - self.visibility_map
                self.occupancy_map = (self.height_map.astype(np.float32) / max(1.0, float(self.height_levels)))
            else:
                self._finalize_scene_maps()

            return

        raise RuntimeError(f"鏃犳硶涓?scene_seed={scene_seed} 鐢熸垚鍙揪绐楀彛鍦烘櫙")

    def _sample_start_goal_in_window(self, rng: np.random.Generator) -> tuple[tuple[int, int], tuple[int, int]]:
        # 杩欓噷鍚庣画鑰冭檻鎶婅繖涓猚orner_span瑙ｈ€﹀埌config褰撲腑杩涜閰嶇疆
        corner_span = 15
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

        # 鍥為€€锛氭壘浠绘剰涓€瀵硅窛绂诲杩滅殑鍙€氳鏍煎瓙
        all_free = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                    if self.window_tag_map is not None and self.window_tag_map[x, y] == 0]
        if len(all_free) >= 2:
            best = (all_free[0], all_free[-1])
            best_dist = float(np.linalg.norm(
                np.array(all_free[0], dtype=np.float32) - np.array(all_free[-1], dtype=np.float32)))
            for _ in range(256):
                i, j = int(rng.integers(0, len(all_free))), int(rng.integers(0, len(all_free)))
                if i == j:
                    continue
                d = float(np.linalg.norm(
                    np.array(all_free[i], dtype=np.float32) - np.array(all_free[j], dtype=np.float32)))
                if d > best_dist:
                    best_dist = d
                    best = (all_free[i], all_free[j])
            return best

        return tuple(self.config.start), tuple(self.config.goal)


    def _has_feasible_path_window(self) -> bool:
        return self.compute_bfs_path() is not None

    # ============================================================
    #  閫氳妫€鏌ワ紙tan 鐖潯 + tag 绾︽潫锛?    # ============================================================

    def can_move_between(self, current: tuple[int, int], candidate: tuple[int, int]) -> bool:
        current_height = self._cell_height(current)
        candidate_height = self._cell_height(candidate)
        dh = candidate_height - current_height
        if dh <= 0:
            return True
        dx = abs(candidate[0] - current[0])
        dy = abs(candidate[1] - current[1])
        # 鍒ゆ柇鏄惁瀵硅绉诲姩
        if dx + dy == 2:
            dist = self.config.cell_size * sqrt(2.0)
        else:
            dist = self.config.cell_size
        return (dh / dist) <= self.config.max_climb_tan

    def _cell_passable(self, x: int, y: int) -> bool:
        if self.window_tag_map is not None:
            return bool(self.window_tag_map[x, y] == 0)
        return True

    def _is_blocked(self, position: np.ndarray, current: np.ndarray | None = None) -> bool:
        x, y = int(position[0]), int(position[1])
        if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return True
        if not self._cell_passable(x, y):
            return True
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
    #  RL 鎺ュ彛
    # ============================================================

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.generate_scene(scene_seed=seed)
        else:
            self.generate_scene()
        self.agent_position = self.start_position.copy()
        self.steps = 0
        self.consecutive_collisions = 0
        self.total_collisions = 0
        return self._get_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        old_pos = self.agent_position.copy()
        move = np.array(self.ACTIONS[action], dtype=np.int32)
        candidate = old_pos + move
        moved = False

        if self._is_blocked(candidate):
            self.consecutive_collisions += 1
            self.total_collisions += 1
            reward = -self.config.collision_penalty
        else:
            self.agent_position = candidate.copy()
            self.consecutive_collisions = 0
            moved = True
            reward = -self.config.step_penalty

            old_dist = float(np.linalg.norm(
                old_pos.astype(np.float32) - self.goal_position.astype(np.float32)))
            new_dist = float(np.linalg.norm(
                self.agent_position.astype(np.float32) - self.goal_position.astype(np.float32)))
            reward += self.current_progress_weight * (old_dist - new_dist)

            if self.visibility_map[tuple(self.agent_position)] > 0.5:
                reward -= self.config.visible_penalty

        self.steps += 1
        self._update_obs_state(old_pos=old_pos, moved=moved)
        done = False
        info: dict = {}

        if np.array_equal(self.agent_position, self.goal_position):
            reward += self.config.goal_reward
            done = True
            info["result"] = "success"
        elif self.steps >= self.config.max_steps:
            reward -= 5.0
            done = True
            info["result"] = "timeout"
        elif self.consecutive_collisions >= self.config.max_consecutive_collisions:
            done = True
            info["result"] = "stuck"

        info["collisions"] = self.total_collisions
        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        return build_obs(self)

    def _planner_cfg(self) -> StealthCostConfig:
        return StealthCostConfig(
            w_len=self.config.planner_w_len,
            w_vis=self.config.planner_w_vis,
            w_slope=self.config.planner_w_slope,
            w_turn=self.config.planner_w_turn,
        )

    def _planner_corridor_channel(self) -> np.ndarray:
        H, W = self.grid_size, self.grid_size
        start = tuple(self.agent_position.tolist())
        goal = tuple(self.goal_position.tolist())
        if start == goal:
            out = np.zeros((H, W), dtype=np.float32)
            out[start] = 1.0
            return out

        path = plan_stealth_path(self, start, goal, self._planner_cfg())
        if not path:
            return np.zeros((H, W), dtype=np.float32)

        gx, gy = np.indices((H, W), dtype=np.float32)
        pts = np.array(path, dtype=np.float32)
        dx = gx[None, :, :] - pts[:, 0][:, None, None]
        dy = gy[None, :, :] - pts[:, 1][:, None, None]
        dist2 = np.min(dx * dx + dy * dy, axis=0)
        sigma = max(0.5, float(self.config.planner_guide_sigma))
        return np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)

    def get_action_mask(self) -> np.ndarray:
        valid = self.get_valid_actions()
        mask = np.zeros(8, dtype=bool)
        mask[valid] = True
        return mask

    def compute_bfs_path(self) -> list[tuple[int, int]] | None:
        """BFS 鏈€鐭矾寰勬悳绱紝浣跨敤涓庡彲杈炬€ф鏌ュ畬鍏ㄧ浉鍚岀殑閫氳瑙勫垯銆?
        Returns:
            浠?start_position 鍒?goal_position 鐨勬牸瀛愬簭鍒楋紙鍚捣缁堢偣锛夛紝鏃犺矾寰勬椂杩斿洖 None銆?        """
        start = tuple(self.start_position.tolist())
        goal = tuple(self.goal_position.tolist())
        queue: deque[tuple[int, int]] = deque([start])
        visited = {start}
        parent: dict[tuple[int, int], tuple[int, int]] = {}

        while queue:
            current = queue.popleft()
            if current == goal:
                path: list[tuple[int, int]] = [current]
                while current in parent:
                    current = parent[current]
                    path.append(current)
                path.reverse()
                return path

            current_arr = np.array(current, dtype=np.int32)
            for action_idx in self.get_valid_actions(current_arr):
                move = np.array(self.ACTIONS[action_idx], dtype=np.int32)
                neighbor = tuple((current_arr + move).tolist())
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        return None

    # ============================================================
    #  鍙鎬т笌閬尅锛堝鎵樼粰 env.occlusion锛?    # ============================================================

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
        if start == end:
            return False
        if self.current_scenario_mode == "full_map" and self.full_terrain is not None:
            return self._is_occluded_global(start, end)
        return is_occluded(start, end, self.height_map, self.config)

    def _is_occluded_global(self, start_global: tuple[int, int], end_window: tuple[int, int]) -> bool:
        ox, oy = self.window_offset
        ft = self.full_terrain
        end_global = (end_window[0] + oy, end_window[1] + ox)
        return is_occluded(start_global, end_global, ft.height_map, self.config)

    def _cell_height(self, cell: tuple[int, int]) -> float:
        return float(self.height_map[cell])

    # ============================================================
    #  鍥哄畾鍦烘櫙 / 闅忔満鍦烘櫙
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

        raise RuntimeError(f"鏃犳硶涓?scene_seed={scene_seed} 鐢熸垚鍙揪鍦烘櫙")

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
        return self.compute_bfs_path() is not None

    def _compute_visibility_area_score_window(self, enemy_xy: tuple[int, int]) -> float:
        score = 0.0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                score += self._compute_cell_visibility_from(enemy_xy, (x, y))
        return score

    # ============================================================
    #  宸ュ叿鏂规硶
    # ============================================================

    def _goal_distance(self, position: np.ndarray) -> float:
        return float(np.linalg.norm(self.goal_position.astype(np.float32) - position.astype(np.float32)))

    def set_goal(self, goal: tuple[int, int] | np.ndarray) -> None:
        self.goal_position = np.array(goal, dtype=np.int32)
        self._sync_obs_goal(force=True)

    def _reset_obs_state(self) -> None:
        self._obs_visited = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._obs_prev_move = np.zeros((2,), dtype=np.float32)
        self._obs_goal_cache = None
        self._obs_goal_best_dist = self._goal_distance(self.agent_position)
        self._obs_goal_stagnation = 0
        self._sync_obs_goal(force=True)
        self._mark_visit()

    def _ensure_obs_state(self) -> None:
        if self._obs_visited.shape != (self.grid_size, self.grid_size):
            self._obs_visited = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._sync_obs_goal(force=False)

    def _sync_obs_goal(self, force: bool) -> None:
        goal = (int(self.goal_position[0]), int(self.goal_position[1]))
        if force or self._obs_goal_cache != goal:
            self._obs_goal_cache = goal
            self._obs_goal_best_dist = self._goal_distance(self.agent_position)
            self._obs_goal_stagnation = 0

    def _mark_visit(self) -> None:
        decay = float(np.clip(self.config.obs_visit_decay, 0.0, 1.0))
        self._obs_visited *= decay
        ax, ay = int(self.agent_position[0]), int(self.agent_position[1])
        if 0 <= ax < self.grid_size and 0 <= ay < self.grid_size:
            self._obs_visited[ax, ay] = 1.0

    def _update_obs_state(self, old_pos: np.ndarray, moved: bool) -> None:
        self._sync_obs_goal(force=False)
        if moved:
            self._obs_prev_move = np.clip(
                self.agent_position.astype(np.float32) - old_pos.astype(np.float32),
                -1.0,
                1.0,
            )
        else:
            self._obs_prev_move = np.zeros((2,), dtype=np.float32)

        new_dist = self._goal_distance(self.agent_position)
        if new_dist + 1e-4 < self._obs_goal_best_dist:
            self._obs_goal_best_dist = new_dist
            self._obs_goal_stagnation = 0
        else:
            self._obs_goal_stagnation += 1
        self._mark_visit()

    @staticmethod
    def _move_cost(move: np.ndarray) -> float:
        return sqrt(2.0) if abs(int(move[0])) + abs(int(move[1])) == 2 else 1.0

