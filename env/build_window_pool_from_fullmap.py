"""Build reachable small-window training pools from full-map terrain and precomputed visibility.

Example:
python -m env.build_window_pool_from_fullmap --grid 15 --num 1000 --output artifacts/window_pool_15.npz
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from math import sqrt
from pathlib import Path

import numpy as np

from config import EnvConfig
from env.terrain_loader import load_terrain

ACTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)


def _can_climb(h1: float, h2: float, dx: int, dy: int, cfg: EnvConfig) -> bool:
    dh = h2 - h1
    if dh <= 0:
        return True
    dist = cfg.cell_size * sqrt(2.0) if dx + dy == 2 else cfg.cell_size
    return (dh / dist) <= cfg.max_climb_tan


def _valid_neighbors(
    cur: tuple[int, int],
    height_map: np.ndarray,
    tag_map: np.ndarray,
    cfg: EnvConfig,
) -> list[tuple[int, int]]:
    g = height_map.shape[0]
    out: list[tuple[int, int]] = []
    for dx, dy in ACTIONS:
        nx, ny = cur[0] + dx, cur[1] + dy
        if nx < 0 or ny < 0 or nx >= g or ny >= g:
            continue
        if tag_map[nx, ny] != 0:
            continue
        if not _can_climb(
            float(height_map[cur[0], cur[1]]),
            float(height_map[nx, ny]),
            abs(dx),
            abs(dy),
            cfg,
        ):
            continue
        out.append((nx, ny))
    return out


def _bfs_length(
    start: tuple[int, int],
    goal: tuple[int, int],
    height_map: np.ndarray,
    tag_map: np.ndarray,
    cfg: EnvConfig,
) -> int | None:
    queue: deque[tuple[int, int]] = deque([start])
    visited = {start}
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    while queue:
        cur = queue.popleft()
        if cur == goal:
            length = 0
            while cur in parent:
                cur = parent[cur]
                length += 1
            return length
        for nxt in _valid_neighbors(cur, height_map, tag_map, cfg):
            if nxt in visited:
                continue
            visited.add(nxt)
            parent[nxt] = cur
            queue.append(nxt)
    return None


def _sample_start_goal(
    rng: np.random.Generator,
    grid_size: int,
    tag_map: np.ndarray,
    min_dist: float,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    span = max(3, int(grid_size * 0.35))
    g0 = max(0, grid_size - span)
    start_pool = [(x, y) for x in range(span) for y in range(span) if tag_map[x, y] == 0]
    goal_pool = [(x, y) for x in range(g0, grid_size) for y in range(g0, grid_size) if tag_map[x, y] == 0]
    if not start_pool or not goal_pool:
        return None

    for _ in range(512):
        s = start_pool[int(rng.integers(0, len(start_pool)))]
        g = goal_pool[int(rng.integers(0, len(goal_pool)))]
        if s == g:
            continue
        dist = float(np.linalg.norm(np.array(s, dtype=np.float32) - np.array(g, dtype=np.float32)))
        if dist < min_dist:
            continue
        return s, g
    return None


def generate_window_pool(
    grid_size: int,
    num_scenes: int,
    output_path: str,
    seed: int = 42,
    max_attempt_factor: int = 40,
) -> None:
    cfg = EnvConfig()
    terrain = load_terrain(cfg.full_map_path)
    pool_path = Path(cfg.enemy_pool_path)
    vis_path = pool_path.parent / "visibility_maps.npz"

    if not pool_path.exists():
        raise FileNotFoundError(f"Missing enemy pool: {pool_path}")
    if not vis_path.exists():
        raise FileNotFoundError(f"Missing visibility maps: {vis_path}")

    with open(pool_path, "r", encoding="utf-8") as f:
        enemy_pool_data = json.load(f)
    enemy_pool: list[tuple[int, int]] = [tuple(e) for e in enemy_pool_data.get("enemy_pool", [])]

    loaded = np.load(vis_path)
    vis_maps = [loaded[f"vis_{i}"] for i in range(len(enemy_pool))]
    if not enemy_pool or not vis_maps:
        raise RuntimeError("enemy_pool or visibility_maps is empty")

    max_ox = terrain.full_width - grid_size
    max_oy = terrain.full_height - grid_size
    if max_ox < 0 or max_oy < 0:
        raise ValueError(f"grid_size={grid_size} exceeds full-map dimensions")

    rng = np.random.default_rng(seed)
    min_dist = max(4.0, grid_size * 0.55)

    heights_list: list[np.ndarray] = []
    tags_list: list[np.ndarray] = []
    vis_list: list[np.ndarray] = []
    starts_list: list[np.ndarray] = []
    goals_list: list[np.ndarray] = []
    bfs_len_list: list[int] = []
    offsets_list: list[np.ndarray] = []
    enemy_idx_list: list[int] = []

    attempts = 0
    max_attempts = num_scenes * max(1, max_attempt_factor)

    while len(starts_list) < num_scenes and attempts < max_attempts:
        attempts += 1
        eidx = int(rng.integers(0, len(enemy_pool)))
        ox = int(rng.integers(0, max_ox + 1))
        oy = int(rng.integers(0, max_oy + 1))

        height = terrain.height_map[oy:oy + grid_size, ox:ox + grid_size].astype(np.int32)
        tags = terrain.tag_map[oy:oy + grid_size, ox:ox + grid_size].astype(np.int32)
        vis = vis_maps[eidx][oy:oy + grid_size, ox:ox + grid_size].astype(np.float32)

        sample = _sample_start_goal(rng, grid_size, tags, min_dist=min_dist)
        if sample is None:
            continue

        start, goal = sample
        bfs_len = _bfs_length(start, goal, height, tags, cfg)
        if bfs_len is None:
            continue

        heights_list.append(height)
        tags_list.append(tags)
        vis_list.append(vis)
        starts_list.append(np.array(start, dtype=np.int32))
        goals_list.append(np.array(goal, dtype=np.int32))
        bfs_len_list.append(int(bfs_len))
        offsets_list.append(np.array((ox, oy), dtype=np.int32))
        enemy_idx_list.append(eidx)

        if len(starts_list) % 200 == 0:
            print(
                f"Accepted {len(starts_list)}/{num_scenes} "
                f"(attempts {attempts}, pass {len(starts_list)/attempts:.1%})"
            )

    if len(starts_list) < num_scenes:
        print(
            f"Warning: reached max attempts {max_attempts}, "
            f"only generated {len(starts_list)} reachable windows"
        )

    heights = np.stack(heights_list, axis=0) if heights_list else np.zeros((0, grid_size, grid_size), dtype=np.int32)
    tags = np.stack(tags_list, axis=0) if tags_list else np.zeros((0, grid_size, grid_size), dtype=np.int32)
    visibility = np.stack(vis_list, axis=0) if vis_list else np.zeros((0, grid_size, grid_size), dtype=np.float32)
    starts = np.stack(starts_list, axis=0) if starts_list else np.zeros((0, 2), dtype=np.int32)
    goals = np.stack(goals_list, axis=0) if goals_list else np.zeros((0, 2), dtype=np.int32)
    bfs_lengths = np.array(bfs_len_list, dtype=np.int32)
    offsets = np.stack(offsets_list, axis=0) if offsets_list else np.zeros((0, 2), dtype=np.int32)
    enemy_indices = np.array(enemy_idx_list, dtype=np.int32)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        grid_size=np.array([grid_size], dtype=np.int32),
        heights=heights,
        tags=tags,
        visibility=visibility,
        starts=starts,
        goals=goals,
        bfs_lengths=bfs_lengths,
        window_offsets=offsets,
        enemy_indices=enemy_indices,
    )

    n = len(starts_list)
    print("\nWindow pool generated")
    print(f"output: {out}")
    print(f"grid_size: {grid_size}")
    print(f"reachable samples: {n}")
    print(f"attempts: {attempts}")
    print(f"pass rate: {n / max(1, attempts):.1%}")
    if n > 0:
        print(f"BFS length: min={bfs_lengths.min()} max={bfs_lengths.max()} avg={bfs_lengths.mean():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build reachable local-window pools from full-map data")
    parser.add_argument("--grid", type=int, default=15, help="window edge size, e.g. 10/15/20")
    parser.add_argument("--num", type=int, default=1000, help="target number of reachable samples")
    parser.add_argument("--output", type=str, default="artifacts/window_pool_15.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempt-factor", type=int, default=40, help="max attempts = num * factor")
    args = parser.parse_args()

    generate_window_pool(
        grid_size=args.grid,
        num_scenes=args.num,
        output_path=args.output,
        seed=args.seed,
        max_attempt_factor=args.max_attempt_factor,
    )
