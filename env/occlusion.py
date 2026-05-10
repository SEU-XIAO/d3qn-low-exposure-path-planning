"""3D 光线追踪遮挡判定模块。

提供独立的遮挡检测、单格可见性判断和全图可见性底图计算。
start/end 坐标需与 height_map 在同一坐标系。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config import EnvConfig
    from env.terrain_loader import FullTerrain


def is_occluded(
    start: tuple[int, int],
    end: tuple[int, int],
    height_map: np.ndarray,
    config: EnvConfig,
) -> bool:
    """检查 start -> end 视线是否被地形遮挡。

    Args:
        start: 观察者坐标 (x, y)，与 height_map 同坐标系。
        end: 目标坐标 (x, y)，与 height_map 同坐标系。
        height_map: 地形高度图 (H, W)。
        config: 环境配置（眼高、目标高、采样密度、遮挡偏置）。

    Returns:
        True 如果视线被遮挡。
    """
    if start == end:
        return False

    H, W = height_map.shape
    sx = float(start[0]) + 0.5
    sy = float(start[1]) + 0.5
    ex = float(end[0]) + 0.5
    ey = float(end[1]) + 0.5
    sz = float(height_map[start]) + float(config.enemy_eye_height)
    ez = float(height_map[end]) + float(config.target_visibility_height)

    length_xy = max(abs(ex - sx), abs(ey - sy))
    samples = max(2, int(length_xy * max(1, config.line_of_sight_samples_per_cell)))
    bias = float(config.visibility_occluder_bias)

    for i in range(1, samples):
        t = i / samples
        px = sx + (ex - sx) * t
        py = sy + (ey - sy) * t
        pz = sz + (ez - sz) * t
        cx = int(np.clip(np.floor(px), 0, H - 1))
        cy = int(np.clip(np.floor(py), 0, W - 1))
        cell = (cx, cy)
        if cell == start or cell == end:
            continue
        if float(height_map[cell]) + bias >= pz:
            return True
    return False


def compute_cell_visibility(
    observer: tuple[int, int],
    cell: tuple[int, int],
    height_map: np.ndarray,
    config: EnvConfig,
) -> float:
    """判断单个格子对观察者是否可见。

    Returns:
        1.0 可见，0.0 不可见（被遮挡或为观察者自身）。
    """
    if observer == cell:
        return 0.0
    if is_occluded(observer, cell, height_map, config):
        return 0.0
    return 1.0


def compute_visibility_map(
    observer: tuple[int, int],
    terrain: FullTerrain,
    config: EnvConfig,
) -> tuple[np.ndarray, float]:
    """为单个观察者计算全图二值可见性底图。

    Args:
        observer: 观察者全局坐标 (x, y)。
        terrain: 完整地形数据。
        config: 环境配置。

    Returns:
        (vis_map, visible_count): vis_map 为 (H, W) bool 数组，visible_count 为可见格子数。
    """
    H, W = terrain.height_map.shape
    vis_map = np.zeros((H, W), dtype=np.bool_)
    t0 = time.perf_counter()
    visible = 0

    for x in range(H):
        for y in range(W):
            cell = (x, y)
            if cell == observer:
                continue
            if not is_occluded(observer, cell, terrain.height_map, config):
                vis_map[x, y] = True
                visible += 1

        if x % 50 == 0 and x > 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (x + 1) * (H - x - 1)
            print(f"    row {x}/{H} | visible={visible:7d} | {elapsed:.0f}s eta={eta:.0f}s")

    return vis_map, float(visible)
