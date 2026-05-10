"""地形特征代理 + 空间抑制 → 选出 8 个瞭望点 → 预计算全图可见性底图。

流程:
1. 特征代理评分（高度排名 + 开阔度 + 支配力），全图所有格子
2. 空间抑制（NMS），选出 8 个散布全图的瞭望点
3. 对每个点计算全图 500×500 二值可见性
4. 存为 JSON（位置）+ NPZ（8 张可见性底图）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

from config import EnvConfig
from env.terrain_loader import FullTerrain, load_terrain
from env.occlusion import is_occluded, compute_visibility_map


def compute_feature_scores(terrain: FullTerrain) -> np.ndarray:
    """全图每个格子的特征代理评分。

    三个邻域特征（25×25 窗口），加权合成:
      - 高度排名  (0.3): 在邻域内算老几
      - 开阔度    (0.3): 比邻域均值高多少
      - 支配力    (0.4): 比平均水平高几个标准差
    """
    h = terrain.height_map.astype(np.float64)
    size = 25

    local_mean = uniform_filter(h, size=size, mode="reflect")
    local_max = maximum_filter(h, size=size, mode="reflect")
    local_min = minimum_filter(h, size=size, mode="reflect")
    sq_mean = uniform_filter(h * h, size=size, mode="reflect")
    local_std = np.sqrt(np.maximum(sq_mean - local_mean * local_mean, 1e-6))

    denom = np.maximum(local_max - local_min, 1.0)
    height_rank = (h - local_min) / denom
    openness = (h - local_mean) / denom
    dominance = (h - local_mean) / local_std

    # 裁剪异常值
    dominance = np.clip(dominance, -5.0, 5.0)

    score = height_rank * 0.3 + openness * 0.3 + dominance * 0.4
    return score.astype(np.float64)


def spatial_suppression(scores: np.ndarray, k: int = 8, radius: int = 60) -> list[tuple[int, int]]:
    """NMS 式空间抑制：选最高分，抑制邻域，重复 k 次。"""
    remaining = scores.copy()
    selected: list[tuple[int, int]] = []
    H, W = scores.shape

    for _ in range(k):
        idx = np.unravel_index(np.argmax(remaining), (H, W))
        selected.append((int(idx[0]), int(idx[1])))

        x0 = max(0, idx[0] - radius)
        x1 = min(H, idx[0] + radius + 1)
        y0 = max(0, idx[1] - radius)
        y1 = min(W, idx[1] + radius + 1)
        remaining[x0:x1, y0:y1] *= 0.3

    return selected


def main() -> None:
    config = EnvConfig()
    full_map_path = config.full_map_path
    if not full_map_path:
        print("错误: config.full_map_path 为空，请先设置地形文件路径")
        sys.exit(1)

    terrain_path = Path(full_map_path)
    if not terrain_path.exists():
        project_root = Path(__file__).resolve().parents[1]
        alt = project_root / full_map_path
        if alt.exists():
            full_map_path = str(alt)
        else:
            print(f"错误: 找不到地形文件 {full_map_path} (也试过 {alt})")
            sys.exit(1)

    print(f"加载地形: {full_map_path}")
    terrain = load_terrain(full_map_path)
    H, W = terrain.full_height, terrain.full_width
    print(f"地形尺寸: {H}×{W}")
    print(f"高度范围: {terrain.height_map.min()}~{terrain.height_map.max()}")

    # ---- Step 1: 特征代理评分 ----
    print("\n=== 特征代理评分 (25×25 邻域) ===")
    t0 = time.perf_counter()
    scores = compute_feature_scores(terrain)
    print(f"评分完成, 耗时 {time.perf_counter() - t0:.1f}s")
    print(f"  得分范围: {scores.min():.2f} ~ {scores.max():.2f}")

    # ---- Step 2: 空间抑制选点 ----
    print(f"\n=== 空间抑制 (k={config.enemy_pool_size}, radius=60) ===")
    pool = spatial_suppression(scores, k=config.enemy_pool_size, radius=60)
    for i, (x, y) in enumerate(pool):
        h = terrain.height_map[x, y]
        tag = terrain.tag_map[x, y]
        tag_name = ["地面", "建筑", "树木"][tag]
        score = scores[x, y]
        print(f"  [{i}] ({x:3d},{y:3d}) h={h:3d} ({tag_name}) score={score:.3f}")

    # ---- Step 3: 预计算全图可见性底图 ----
    print(f"\n=== 预计算 {len(pool)} 张全图可见性底图 ===")
    vis_maps: dict[str, np.ndarray] = {}
    vis_counts: list[int] = []
    total_start = time.perf_counter()

    for i, pos in enumerate(pool):
        print(f"[{i + 1}/{len(pool)}] 敌人 ({pos[0]},{pos[1]}) h={terrain.height_map[pos]}...")
        vis_map, n_visible = compute_visibility_map(pos, terrain, config)
        vis_maps[f"vis_{i}"] = vis_map
        vis_counts.append(int(n_visible))
        pct = n_visible / (H * W) * 100
        print(f"  可见: {n_visible}/{H * W} ({pct:.1f}%)")

    print(f"\n预计算总耗时: {time.perf_counter() - total_start:.0f}s")

    # ---- 保存 ----
    output_dir = Path(__file__).resolve().parents[1] / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # NPZ 可见性底图
    npz_path = output_dir / "visibility_maps.npz"
    np.savez_compressed(npz_path, **{k: v for k, v in vis_maps.items()})
    print(f"\n可见性底图已保存: {npz_path}")
    print(f"  文件大小: {npz_path.stat().st_size / 1024:.0f} KB")

    # JSON 敌人池
    json_path = output_dir / "enemy_pool.json"
    json_data = {
        "enemy_pool": [[int(x), int(y)] for x, y in pool],
        "heights": [int(terrain.height_map[p]) for p in pool],
        "tags": [int(terrain.tag_map[p]) for p in pool],
        "visible_counts": vis_counts,
        "full_map_path": str(full_map_path),
        "terrain_shape": [H, W],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"敌人池已保存: {json_path}")


if __name__ == "__main__":
    main()
