from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import sqrt

import numpy as np


GridPos = tuple[int, int]


@dataclass(frozen=True)
class StealthCostConfig:
    w_len: float = 1.0
    w_vis: float = 3.0
    w_slope: float = 0.8
    w_turn: float = 0.15


@dataclass(frozen=True)
class WaypointConfig:
    spacing: int = 6
    keep_turn_points: bool = True
    min_segment: int = 3


_MOVES: tuple[GridPos, ...] = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)


def _move_cost(move: GridPos) -> float:
    return sqrt(2.0) if abs(move[0]) + abs(move[1]) == 2 else 1.0


def _heuristic(a: GridPos, b: GridPos) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dmin = min(dx, dy)
    dmax = max(dx, dy)
    return sqrt(2.0) * dmin + (dmax - dmin)


def _neighbors(env, node: GridPos) -> list[GridPos]:
    cur = np.array(node, dtype=np.int32)
    out: list[GridPos] = []
    for move in _MOVES:
        nxt = cur + np.array(move, dtype=np.int32)
        x, y = int(nxt[0]), int(nxt[1])
        if x < 0 or y < 0 or x >= env.grid_size or y >= env.grid_size:
            continue
        if env._is_blocked(nxt, current=cur):
            continue
        out.append((x, y))
    return out


def _segment_direction(a: GridPos, b: GridPos) -> GridPos:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return (0 if dx == 0 else dx // abs(dx), 0 if dy == 0 else dy // abs(dy))


def plan_stealth_path(env, start: GridPos, goal: GridPos, cfg: StealthCostConfig) -> list[GridPos] | None:
    """在当前窗口地图上做隐蔽代价 A*，不修改环境状态。"""
    open_heap: list[tuple[float, GridPos]] = []
    heapq.heappush(open_heap, (0.0, start))

    came_from: dict[GridPos, GridPos] = {}
    g_score: dict[GridPos, float] = {start: 0.0}
    dir_from: dict[GridPos, GridPos] = {start: (0, 0)}

    while open_heap:
        _f, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        prev_dir = dir_from.get(current, (0, 0))
        for nxt in _neighbors(env, current):
            step_dir = _segment_direction(current, nxt)
            turn_penalty = 0.0 if prev_dir == (0, 0) or step_dir == prev_dir else 1.0
            edge_len = _move_cost((nxt[0] - current[0], nxt[1] - current[1]))
            vis_penalty = float(env.visibility_map[nxt])
            h0 = float(env.height_map[current])
            h1 = float(env.height_map[nxt])
            slope_penalty = max(0.0, h1 - h0)

            edge_cost = (
                cfg.w_len * edge_len
                + cfg.w_vis * vis_penalty
                + cfg.w_slope * slope_penalty
                + cfg.w_turn * turn_penalty
            )
            tentative = g_score[current] + edge_cost
            if tentative >= g_score.get(nxt, float("inf")):
                continue

            came_from[nxt] = current
            g_score[nxt] = tentative
            dir_from[nxt] = step_dir
            f = tentative + cfg.w_len * _heuristic(nxt, goal)
            heapq.heappush(open_heap, (f, nxt))

    return None


def extract_waypoints(path: list[GridPos], cfg: WaypointConfig) -> list[GridPos]:
    """将全局路径压缩为子目标序列（含终点）。"""
    if len(path) <= 1:
        return path

    waypoints: list[GridPos] = []
    last_pick = 0
    prev_dir: GridPos | None = None

    for i in range(1, len(path)):
        cur_dir = _segment_direction(path[i - 1], path[i])
        is_turn = prev_dir is not None and cur_dir != prev_dir
        enough_gap = (i - last_pick) >= max(1, cfg.spacing)

        if enough_gap or (cfg.keep_turn_points and is_turn and (i - last_pick) >= cfg.min_segment):
            waypoints.append(path[i])
            last_pick = i
        prev_dir = cur_dir

    if waypoints and waypoints[-1] != path[-1]:
        waypoints.append(path[-1])
    elif not waypoints:
        waypoints = [path[-1]]
    return waypoints

