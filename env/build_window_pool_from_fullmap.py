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
from typing import Literal

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
    path = _bfs_path(start, goal, height_map, tag_map, cfg)
    if not path:
        return None
    return max(0, len(path) - 1)


def _bfs_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    height_map: np.ndarray,
    tag_map: np.ndarray,
    cfg: EnvConfig,
) -> list[tuple[int, int]] | None:
    queue: deque[tuple[int, int]] = deque([start])
    visited = {start}
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    while queue:
        cur = queue.popleft()
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return path
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


def _sample_subtask_start_goal(
    rng: np.random.Generator,
    tag_map: np.ndarray,
    min_dist: float,
    max_dist: float,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    free = np.argwhere(tag_map == 0)
    n = len(free)
    if n < 2:
        return None
    for _ in range(1024):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        s = (int(free[i, 0]), int(free[i, 1]))
        g = (int(free[j, 0]), int(free[j, 1]))
        d = float(np.linalg.norm(np.array(s, dtype=np.float32) - np.array(g, dtype=np.float32)))
        if d < min_dist or d > max_dist:
            continue
        return s, g
    return None


def _bucket_from_bfs_len(bfs_len: int, grid_size: int) -> int:
    # 0=short, 1=mid, 2=long
    t1 = max(4, int(round(grid_size * 0.45)))
    t2 = max(t1 + 1, int(round(grid_size * 0.80)))
    if bfs_len <= t1:
        return 0
    if bfs_len <= t2:
        return 1
    return 2


def _parse_bucket_ratios(text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--bucket-ratios 需要3个逗号分隔的数字，例如 0.4,0.35,0.25")
    vals = [float(p) for p in parts]
    s = sum(vals)
    if s <= 0:
        raise ValueError("--bucket-ratios 总和必须 > 0")
    return vals[0] / s, vals[1] / s, vals[2] / s


def generate_window_pool(
    grid_size: int,
    num_scenes: int,
    output_path: str,
    seed: int = 42,
    max_attempt_factor: int = 40,
    task_mode: Literal["corner", "subtask", "mixed"] = "corner",
    mixed_corner_ratio: float = 0.35,
    subtask_min_dist: float = 2.0,
    subtask_max_dist_ratio: float = 0.6,
    bucket_ratios: tuple[float, float, float] = (0.4, 0.35, 0.25),
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
    corner_min_dist = max(4.0, grid_size * 0.55)
    subtask_max_dist = max(subtask_min_dist + 1.0, grid_size * subtask_max_dist_ratio)

    heights_list: list[np.ndarray] = []
    tags_list: list[np.ndarray] = []
    vis_list: list[np.ndarray] = []
    starts_list: list[np.ndarray] = []
    goals_list: list[np.ndarray] = []
    bfs_len_list: list[int] = []
    start_goal_dist_list: list[float] = []
    vis_along_bfs_list: list[float] = []
    bucket_id_list: list[int] = []
    offsets_list: list[np.ndarray] = []
    enemy_idx_list: list[int] = []

    attempts = 0
    max_attempts = num_scenes * max(1, max_attempt_factor)
    bucket_names = ("short", "mid", "long")
    bucket_targets = [int(num_scenes * bucket_ratios[i]) for i in range(3)]
    while sum(bucket_targets) < num_scenes:
        i = int(np.argmin(bucket_targets))
        bucket_targets[i] += 1
    bucket_counts = [0, 0, 0]

    while len(starts_list) < num_scenes and attempts < max_attempts:
        attempts += 1
        eidx = int(rng.integers(0, len(enemy_pool)))
        ox = int(rng.integers(0, max_ox + 1))
        oy = int(rng.integers(0, max_oy + 1))

        height = terrain.height_map[oy:oy + grid_size, ox:ox + grid_size].astype(np.int32)
        tags = terrain.tag_map[oy:oy + grid_size, ox:ox + grid_size].astype(np.int32)
        vis = vis_maps[eidx][oy:oy + grid_size, ox:ox + grid_size].astype(np.float32)

        use_corner = False
        if task_mode == "corner":
            use_corner = True
        elif task_mode == "subtask":
            use_corner = False
        elif task_mode == "mixed":
            use_corner = bool(rng.random() < mixed_corner_ratio)
        else:
            raise ValueError(f"unknown task_mode: {task_mode}")

        if use_corner:
            sample = _sample_start_goal(rng, grid_size, tags, min_dist=corner_min_dist)
        else:
            sample = _sample_subtask_start_goal(
                rng,
                tags,
                min_dist=subtask_min_dist,
                max_dist=subtask_max_dist,
            )
        if sample is None:
            continue

        start, goal = sample
        bfs_path = _bfs_path(start, goal, height, tags, cfg)
        if bfs_path is None:
            continue
        bfs_len = max(0, len(bfs_path) - 1)
        b_id = _bucket_from_bfs_len(bfs_len, grid_size)
        quota_not_met = any(bucket_counts[i] < bucket_targets[i] for i in range(3))
        quota_phase = attempts < int(max_attempts * 0.85)
        if quota_phase and quota_not_met and bucket_counts[b_id] >= bucket_targets[b_id]:
            continue

        d_sg = float(
            np.linalg.norm(np.array(start, dtype=np.float32) - np.array(goal, dtype=np.float32))
        )
        vis_along = float(np.mean([vis[x, y] for (x, y) in bfs_path]))

        heights_list.append(height)
        tags_list.append(tags)
        vis_list.append(vis)
        starts_list.append(np.array(start, dtype=np.int32))
        goals_list.append(np.array(goal, dtype=np.int32))
        bfs_len_list.append(int(bfs_len))
        start_goal_dist_list.append(d_sg)
        vis_along_bfs_list.append(vis_along)
        bucket_id_list.append(int(b_id))
        bucket_counts[b_id] += 1
        offsets_list.append(np.array((ox, oy), dtype=np.int32))
        enemy_idx_list.append(eidx)

        if len(starts_list) % 200 == 0:
            print(
                f"Accepted {len(starts_list)}/{num_scenes} "
                f"(attempts {attempts}, pass {len(starts_list)/attempts:.1%}) "
                f"bucket={dict(zip(bucket_names, bucket_counts))}"
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
    start_goal_dist = np.array(start_goal_dist_list, dtype=np.float32)
    vis_along_bfs = np.array(vis_along_bfs_list, dtype=np.float32)
    bucket_ids = np.array(bucket_id_list, dtype=np.int32)
    bucket_labels = np.array([bucket_names[i] for i in bucket_id_list], dtype="<U8")
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
        start_goal_dist=start_goal_dist,
        vis_along_bfs=vis_along_bfs,
        bucket_ids=bucket_ids,
        bucket_labels=bucket_labels,
        window_offsets=offsets,
        enemy_indices=enemy_indices,
    )

    n = len(starts_list)
    print("\nWindow pool generated")
    print(f"output: {out}")
    print(f"grid_size: {grid_size}")
    print(f"task_mode: {task_mode}")
    print(f"reachable samples: {n}")
    print(f"attempts: {attempts}")
    print(f"pass rate: {n / max(1, attempts):.1%}")
    print(f"bucket target: {dict(zip(bucket_names, bucket_targets))}")
    print(f"bucket count:  {dict(zip(bucket_names, bucket_counts))}")
    if n > 0:
        print(f"BFS length: min={bfs_lengths.min()} max={bfs_lengths.max()} avg={bfs_lengths.mean():.2f}")
        print(
            f"StartGoalDist: min={start_goal_dist.min():.2f} "
            f"max={start_goal_dist.max():.2f} avg={start_goal_dist.mean():.2f}"
        )
        print(f"VisAlongBFS: min={vis_along_bfs.min():.3f} max={vis_along_bfs.max():.3f} avg={vis_along_bfs.mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build reachable local-window pools from full-map data")
    parser.add_argument("--grid", type=int, default=15, help="window edge size, e.g. 10/15/20")
    parser.add_argument("--num", type=int, default=1000, help="target number of reachable samples")
    parser.add_argument("--output", type=str, default="artifacts/window_pool_15.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempt-factor", type=int, default=40, help="max attempts = num * factor")
    parser.add_argument("--task-mode", type=str, default="corner", choices=["corner", "subtask", "mixed"])
    parser.add_argument("--mixed-corner-ratio", type=float, default=0.35, help="task-mode=mixed 时 corner 采样占比")
    parser.add_argument("--subtask-min-dist", type=float, default=2.0, help="subtask 起终点最小欧氏距离")
    parser.add_argument("--subtask-max-dist-ratio", type=float, default=0.6, help="subtask 起终点最大欧氏距离=ratio*grid")
    parser.add_argument("--bucket-ratios", type=str, default="0.4,0.35,0.25", help="short,mid,long 配额比例")
    args = parser.parse_args()
    ratios = _parse_bucket_ratios(args.bucket_ratios)

    generate_window_pool(
        grid_size=args.grid,
        num_scenes=args.num,
        output_path=args.output,
        seed=args.seed,
        max_attempt_factor=args.max_attempt_factor,
        task_mode=args.task_mode,
        mixed_corner_ratio=args.mixed_corner_ratio,
        subtask_min_dist=args.subtask_min_dist,
        subtask_max_dist_ratio=args.subtask_max_dist_ratio,
        bucket_ratios=ratios,
    )
