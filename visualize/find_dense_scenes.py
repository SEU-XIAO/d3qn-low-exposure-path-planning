"""在全地形(501×499)上滑动50×50窗口，找出障碍物密集的区域。

用法:
    python -m visualize.find_dense_scenes                  # 列出 Top 20
    python -m visualize.find_dense_scenes --show 0         # 完整可视化排名第 0 的窗口
    python -m visualize.find_dense_scenes --try 20         # 从排名最高开始逐个尝试，第一个 BFS 有解的可视化
    python -m visualize.find_dense_scenes --quick 0        # 快速预览排名第 0 的窗口（纯地形，秒开）
    python -m visualize.find_dense_scenes --min 0.3        # 只列障碍占比 ≥ 30% 的
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import EnvConfig
from env.terrain_loader import load_terrain

# 配色
C_GROUND   = np.array([0.90, 0.87, 0.80, 1.0])
C_BUILDING = np.array([0.25, 0.25, 0.27, 1.0])
C_TREE     = np.array([0.10, 0.30, 0.15, 1.0])


def scan_dense_windows(terrain, window_size: int = 50, stride: int = 10):
    """滑动窗口扫描，返回按障碍占比降序排列的窗口列表。"""
    tag = terrain.tag_map
    H, W = terrain.full_height, terrain.full_width
    results = []

    for oy in range(0, H - window_size + 1, stride):
        for ox in range(0, W - window_size + 1, stride):
            window = tag[oy:oy + window_size, ox:ox + window_size]
            total = window_size * window_size
            n_building = int((window == 1).sum())
            n_tree = int((window == 2).sum())
            n_obstacle = n_building + n_tree

            results.append({
                "ox": ox, "oy": oy,
                "ratio": n_obstacle / total,
                "building": n_building,
                "tree": n_tree,
                "ground": total - n_obstacle,
            })

    results.sort(key=lambda r: r["ratio"], reverse=True)
    return results


def print_table(results: list[dict], top_n: int = 20, min_ratio: float = 0.0):
    print(f"\n{'排名':<5} {'ox':>5} {'oy':>5} {'障碍占比':>8} {'建筑':>6} {'树木':>6} {'地面':>6}")
    print("-" * 50)
    shown = 0
    for r in results:
        if r["ratio"] < min_ratio:
            continue
        print(f"{shown:<5} {r['ox']:>5} {r['oy']:>5} "
              f"{r['ratio']:>7.1%} "
              f"{r['building']:>6} {r['tree']:>6} {r['ground']:>6}")
        shown += 1
        if shown >= top_n:
            break


def full_view(window: dict, save_path: str | None = None):
    """完整可视化：场景地形 + 可见性/遮挡 + BFS 路径。需要加载 NPZ（约 8s）。"""
    from env import BattlefieldEnv
    from visualize import visualize_all

    ox, oy = window["ox"], window["oy"]
    print(f"\n创建 BattlefieldEnv (加载 NPZ...)")
    env = BattlefieldEnv()
    try:
        env.generate_scene(scene_seed=42, scenario_mode="full_map", window_offset=(ox, oy))
    except RuntimeError as e:
        print(f"场景生成失败: {e}")
        return

    print(f"窗口 offset=({ox},{oy})  障碍占比={window['ratio']:.1%}  "
          f"起点={tuple(env.start_position.tolist())}  终点={tuple(env.goal_position.tolist())}  "
          f"敌人=({int(env.enemy_position[0])},{int(env.enemy_position[1])})")
    visualize_all(env, scene_seed=42, save_path=save_path)


def try_top_until_feasible(results: list[dict], max_attempts: int = 50,
                           save_path: str | None = None):
    """从排名最高的窗口开始逐个尝试场景生成，返回第一个 BFS 有解的场景。

    加载 BattlefieldEnv（含 NPZ）一次，然后依次在不同窗口上尝试。
    """
    from env import BattlefieldEnv
    from visualize import visualize_all

    print(f"\n加载 BattlefieldEnv (NPZ 约 8s)...")
    env = BattlefieldEnv()

    for rank, w in enumerate(results[:max_attempts]):
        ox, oy = w["ox"], w["oy"]
        print(f"[{rank}] 尝试窗口 ({ox}, {oy})  障碍占比 {w['ratio']:.1%}  ...", end=" ")

        try:
            env.generate_scene(scene_seed=42, scenario_mode="full_map", window_offset=(ox, oy))
        except RuntimeError as e:
            print(f"场景生成失败: {e}")
            continue

        path = env.compute_bfs_path()
        if path is None:
            print(f"BFS 无解 (起点={tuple(env.start_position.tolist())} 终点={tuple(env.goal_position.tolist())})")
            continue

        print(f"成功! 路径步数={len(path) - 1}  "
              f"起点={tuple(env.start_position.tolist())}  终点={tuple(env.goal_position.tolist())}  "
              f"敌人=({int(env.enemy_position[0])},{int(env.enemy_position[1])})")
        visualize_all(env, scene_seed=42, save_path=save_path)
        return

    print(f"\n前 {max_attempts} 个窗口均无可行 BFS 路径。")


def quick_view(terrain, window: dict, save_path: str | None = None):
    """纯地形标签快览——不加载 BattlefieldEnv，秒开。"""
    ox, oy = window["ox"], window["oy"]
    tag = terrain.tag_map[oy:oy + 50, ox:ox + 50]
    rgba = np.zeros((50, 50, 4), dtype=np.float32)
    rgba[tag == 0] = C_GROUND
    rgba[tag == 1] = C_BUILDING
    rgba[tag == 2] = C_TREE

    _, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgba.transpose(1, 0, 2), origin="lower", interpolation="nearest")

    # 网格线
    ax.set_xticks(np.arange(-0.5, 50.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 50.5, 1), minor=True)
    ax.grid(which="minor", color="#AAAAAA", linewidth=0.35, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlim(-0.5, 49.5)
    ax.set_ylim(-0.5, 49.5)
    ax.set_xticks(range(0, 50, 5))
    ax.set_yticks(range(0, 50, 5))

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C_GROUND, label=f"地面 ({window['ground']})"),
        Patch(facecolor=C_BUILDING, label=f"建筑 ({window['building']})"),
        Patch(facecolor=C_TREE, label=f"树木 ({window['tree']})"),
    ], loc="upper right", fontsize=9, framealpha=0.92)

    ax.set_title(f"窗口 (ox={ox}, oy={oy})  障碍占比 {window['ratio']:.1%}",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"图片已保存至: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="查找障碍物密集的 50×50 窗口")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min", type=float, default=0.0, dest="min_ratio",
                        help="最低障碍占比，如 0.3")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--show", type=int, default=None,
                        help="完整可视化排名第 N 的窗口（含 BFS 路径，需加载 NPZ ~8s）")
    parser.add_argument("--quick", type=int, default=None,
                        help="快速预览排名第 N 的窗口（纯地形，秒开）")
    parser.add_argument("--try", type=int, default=None, dest="try_top",
                        metavar="N", help="从排名最高开始逐个尝试，最多 N 个，找到第一个 BFS 有解的可视化")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    config = EnvConfig()
    t0 = time.perf_counter()
    terrain = load_terrain(config.full_map_path)
    print(f"地形 {terrain.full_height}×{terrain.full_width} 加载完成 ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    results = scan_dense_windows(terrain, window_size=50, stride=args.stride)
    print(f"扫描 {len(results)} 个窗口 ({time.perf_counter() - t0:.1f}s)")

    print_table(results, top_n=args.top, min_ratio=args.min_ratio)

    if args.show is not None:
        if args.show < 0 or args.show >= len(results):
            print(f"索引 {args.show} 超出范围 (0~{len(results) - 1})")
            return
        full_view(results[args.show], save_path=args.save)

    if args.try_top is not None:
        try_top_until_feasible(results, max_attempts=args.try_top, save_path=args.save)

    if args.quick is not None:
        if args.quick < 0 or args.quick >= len(results):
            print(f"索引 {args.quick} 超出范围 (0~{len(results) - 1})")
            return
        quick_view(terrain, results[args.quick], save_path=args.save)


if __name__ == "__main__":
    main()
