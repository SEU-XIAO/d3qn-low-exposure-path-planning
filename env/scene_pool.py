"""预计算随机障碍场景池，供阶段1纯导航训练使用。

生成包含建筑、树木、高度起伏的随机 50×50 场景，
每个场景经 BFS 验证起点→终点可达后存入压缩 NPZ 文件。

用法: python -m env.scene_pool --num 5000 --output artifacts/scene_pool.npz
"""

from __future__ import annotations

import argparse
from collections import deque
from math import sqrt
from pathlib import Path

import numpy as np

# —— 与 BattlefieldEnv 保持一致的常量 ——
ACTIONS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)
CELL_SIZE = 10.0
MAX_CLIMB_TAN = 0.3
GRID_SIZE = 50
HEIGHT_LEVELS = 8


def _can_climb(h1: float, h2: float, dx: int, dy: int) -> bool:
    dh = h2 - h1
    if dh <= 0:
        return True
    dist = CELL_SIZE * sqrt(2.0) if dx + dy == 2 else CELL_SIZE
    return (dh / dist) <= MAX_CLIMB_TAN


def _get_valid_actions(
    pos: tuple[int, int],
    height_map: np.ndarray,
    tag_map: np.ndarray,
) -> list[int]:
    x, y = pos
    valid = []
    for ai, (dx, dy) in enumerate(ACTIONS):
        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx >= GRID_SIZE or ny >= GRID_SIZE:
            continue
        if tag_map[nx, ny] != 0:
            continue
        if not _can_climb(
            float(height_map[x, y]), float(height_map[nx, ny]), abs(dx), abs(dy),
        ):
            continue
        valid.append(ai)
    return valid


def _bfs(
    start: tuple[int, int],
    goal: tuple[int, int],
    height_map: np.ndarray,
    tag_map: np.ndarray,
) -> int | None:
    """BFS 最短路径，返回步数或 None。"""
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

        for ai in _get_valid_actions(cur, height_map, tag_map):
            dx, dy = ACTIONS[ai]
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = cur
                queue.append(nxt)

    return None


def _generate_one_scene(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
    """生成一个随机障碍场景。

    Returns:
        (height_map, tag_map, start, goal, bfs_length) 或 None（生成失败）
    """
    # 1. 基础高度图（低频噪声 + 平滑）
    base = rng.integers(0, 2, size=(GRID_SIZE, GRID_SIZE)).astype(np.float32)
    # 简单 box blur 3×3
    smoothed = np.zeros_like(base)
    for x in range(GRID_SIZE):
        x0, x1 = max(0, x - 1), min(GRID_SIZE, x + 2)
        for y in range(GRID_SIZE):
            y0, y1 = max(0, y - 1), min(GRID_SIZE, y + 2)
            smoothed[x, y] = base[x0:x1, y0:y1].mean()
    height_map = np.round(smoothed).astype(np.int32)

    # 随机高峰
    peak_mask = rng.random((GRID_SIZE, GRID_SIZE)) < 0.08
    peak_h = rng.integers(3, HEIGHT_LEVELS + 1, size=(GRID_SIZE, GRID_SIZE))
    height_map = np.maximum(height_map, np.where(peak_mask, peak_h, 0))

    # 2. 放置建筑 (tag=1)
    tag_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    num_buildings = int(rng.integers(5, 16))
    for _ in range(num_buildings * 3):  # 多尝试几次
        bw = int(rng.integers(3, 9))
        bh = int(rng.integers(3, 9))
        bx = int(rng.integers(0, GRID_SIZE - bw + 1))
        by = int(rng.integers(0, GRID_SIZE - bh + 1))
        if tag_map[bx:bx + bw, by:by + bh].any():
            continue
        tag_map[bx:bx + bw, by:by + bh] = 1
        bldg_h = int(rng.integers(3, 6))
        height_map[bx:bx + bw, by:by + bh] = bldg_h

    # 3. 放置树木 (tag=2)，单格或 2×1 小簇
    num_trees = int(rng.integers(20, 55))
    for _ in range(num_trees):
        tx = int(rng.integers(0, GRID_SIZE))
        ty = int(rng.integers(0, GRID_SIZE))
        if tag_map[tx, ty] != 0:
            continue
        tag_map[tx, ty] = 2
        height_map[tx, ty] = int(rng.integers(1, 3))
        # 偶尔再延伸一个相邻格
        if rng.random() < 0.3:
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                nx2, ny2 = tx + dx, ty + dy
                if 0 <= nx2 < GRID_SIZE and 0 <= ny2 < GRID_SIZE and tag_map[nx2, ny2] == 0:
                    if rng.random() < 0.5:
                        tag_map[nx2, ny2] = 2
                        height_map[nx2, ny2] = int(rng.integers(1, 3))
                        break

    # 4. 采样起点（左上角区域）和终点（右下角区域）
    corner = 10
    start_pool = [(x, y) for x in range(corner) for y in range(corner) if tag_map[x, y] == 0]
    g0 = GRID_SIZE - corner
    goal_pool = [(x, y) for x in range(g0, GRID_SIZE) for y in range(g0, GRID_SIZE) if tag_map[x, y] == 0]

    if not start_pool or not goal_pool:
        return None

    for _ in range(512):
        s = start_pool[int(rng.integers(0, len(start_pool)))]
        g = goal_pool[int(rng.integers(0, len(goal_pool)))]
        if s == g:
            continue
        dist = float(np.linalg.norm(
            np.array(s, dtype=np.float32) - np.array(g, dtype=np.float32)))
        if dist < 30.0:
            continue

        bfs_len = _bfs(s, g, height_map, tag_map)
        if bfs_len is not None and bfs_len >= 30:
            return (
                height_map.copy(),
                tag_map.copy(),
                np.array(s, dtype=np.int32),
                np.array(g, dtype=np.int32),
                bfs_len,
            )

    return None


def generate_scene_pool(
    num_scenes: int = 5000,
    output_path: str = "artifacts/scene_pool.npz",
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)

    heights_list: list[np.ndarray] = []
    tags_list: list[np.ndarray] = []
    starts_list: list[np.ndarray] = []
    goals_list: list[np.ndarray] = []
    bfs_lengths_list: list[int] = []

    attempts = 0
    generated = 0
    print(f"开始生成 {num_scenes} 个随机障碍场景...")

    while generated < num_scenes:
        result = _generate_one_scene(rng)
        attempts += 1
        if result is None:
            continue
        h, t, s, g, b = result
        heights_list.append(h)
        tags_list.append(t)
        starts_list.append(s)
        goals_list.append(g)
        bfs_lengths_list.append(b)
        generated += 1

        if generated % 1000 == 0:
            print(f"  已生成 {generated}/{num_scenes} "
                  f"(成功率 {generated / attempts:.1%})")

    heights = np.stack(heights_list, axis=0)       # (N, 50, 50)
    tags = np.stack(tags_list, axis=0)              # (N, 50, 50)
    starts = np.stack(starts_list, axis=0)          # (N, 2)
    goals = np.stack(goals_list, axis=0)            # (N, 2)
    bfs_lengths = np.array(bfs_lengths_list, dtype=np.int32)  # (N,)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        heights=heights,
        tags=tags,
        starts=starts,
        goals=goals,
        bfs_lengths=bfs_lengths,
    )

    # 统计
    total_cells = num_scenes * GRID_SIZE * GRID_SIZE
    n_building = int((tags == 1).sum())
    n_tree = int((tags == 2).sum())

    print(f"\n场景池已保存至 {output}")
    print(f"  场景数:     {num_scenes}")
    print(f"  heights:    {heights.shape} ({heights.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  tags:       {tags.shape} ({tags.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  生成成功率: {generated / attempts:.1%}")
    print(f"  BFS 长度:   [{bfs_lengths.min()}, {bfs_lengths.max()}], "
          f"均值 {bfs_lengths.mean():.1f}")
    print(f"  建筑格占比: {n_building / total_cells:.1%}")
    print(f"  树木格占比: {n_tree / total_cells:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成随机障碍场景池")
    parser.add_argument("--num", type=int, default=5000)
    parser.add_argument("--output", type=str, default="artifacts/scene_pool.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_scene_pool(args.num, output_path=args.output, seed=args.seed)
