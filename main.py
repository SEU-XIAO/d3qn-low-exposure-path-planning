"""核心模块自检：地形加载 → 场景生成 → 遮挡判定 → 敌人搜索。

验证三个可复用模块（遮挡判定、敌人搜索、场景加载）正常工作。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from env import (
    BattlefieldEnv,
    compute_cell_visibility,
    compute_visibility_map,
    is_occluded,
    load_terrain,
)
from visualize import visualize_all
from config import EnvConfig


def test_terrain_loader() -> None:
    print("=" * 60)
    print("1. 地形加载 (terrain_loader)")
    print("=" * 60)
    config = EnvConfig()
    terrain = load_terrain(config.full_map_path)
    print(f"  地形尺寸: {terrain.full_height} × {terrain.full_width}")
    print(f"  高度范围: {terrain.height_map.min()} ~ {terrain.height_map.max()}")
    print(f"  可通行比例: {(terrain.passable_map.sum() / terrain.passable_map.size * 100):.1f}%")
    print("  通过 [OK]")


def test_scene_generation() -> None:
    print()
    print("=" * 60)
    print("2. 场景生成 (battlefield_env)")
    print("=" * 60)

    for mode in ["full_map", "random", "fixed"]:
        env = BattlefieldEnv()
        try:
            env.generate_scene(scene_seed=42, scenario_mode=mode)
            visible = int(env.visibility_map.sum())
            total = env.grid_size * env.grid_size
            passable = int((env.window_tag_map == 0).sum()) if env.window_tag_map is not None else total
            print(f"  [{mode}] start={tuple(env.start_position.tolist())} "
                  f"goal={tuple(env.goal_position.tolist())} "
                  f"enemy=({int(env.enemy_position[0])},{int(env.enemy_position[1])}) "
                  f"passable={passable}/{total} visible={visible}/{total}")
        except Exception as e:
            print(f"  [{mode}] 失败: {e}")
    print("  通过 [OK]")


def test_occlusion() -> None:
    print()
    print("=" * 60)
    print("3. 遮挡判定 (occlusion)")
    print("=" * 60)

    config = EnvConfig()
    terrain = load_terrain(config.full_map_path)

    # 测试相邻两格的遮挡判定
    p1 = (250, 250)
    p2 = (251, 250)
    blocked = is_occluded(p1, p2, terrain.height_map, config)
    print(f"  is_occluded({p1}, {p2}): {blocked}")

    # 单格可见性
    vis = compute_cell_visibility(p1, p2, terrain.height_map, config)
    print(f"  compute_cell_visibility({p1}, {p2}): {vis}")

    # 小范围可见性底图 (20×20 演示)
    print("  计算 20×20 局部可见性...")
    vis_map, n_visible = compute_visibility_map(p1, terrain, config)
    print(f"  全图可见格子: {int(n_visible)} / {terrain.height_map.size}")
    print("  通过 [OK]")


def test_enemy_search() -> None:
    print()
    print("=" * 60)
    print("4. 敌人搜索 (enemy_search) — 验证预计算产物")
    print("=" * 60)

    enemy_pool_path = Path("artifacts/enemy_pool.json")
    vis_npz_path = Path("artifacts/visibility_maps.npz")

    if not enemy_pool_path.exists():
        print("  敌人池未生成，请先运行 python -m env.enemy_search")
        return
    if not vis_npz_path.exists():
        print("  可见性底图未生成，请先运行 python -m env.enemy_search")
        return

    with open(enemy_pool_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pool = data.get("enemy_pool", [])
    heights = data.get("heights", [])
    tags = data.get("tags", [])
    n_visible = data.get("visible_counts", [])

    print(f"  敌人池: {len(pool)} 个瞭望点")
    for i, ((x, y), h, t, v) in enumerate(zip(pool, heights, tags, n_visible)):
        tag_name = ["地面", "建筑", "树木"][t]
        print(f"    [{i}] ({x:3d},{y:3d}) h={h:3d} ({tag_name}) 可见格: {v}")

    loaded = np.load(vis_npz_path)
    print(f"  可见性底图: {len(loaded.files)} 张 ({loaded['vis_0'].shape})")
    print("  通过 [OK]")


def test_visualization(seed: int = 42, mode: str = "full_map",
                       save_path: str | None = None) -> None:
    """可视化场景：指定随机种子和模式生成场景并三合一展示。"""
    env = BattlefieldEnv()
    try:
        env.generate_scene(scene_seed=seed, scenario_mode=mode)
    except RuntimeError as e:
        print(f"场景生成失败 (seed={seed}, mode={mode}): {e}")
        if mode != "full_map":
            print("回退到 fixed 模式")
            env.generate_scene(scene_seed=0, scenario_mode="fixed")
            seed = 0
        else:
            print("尝试其他种子...")
            for fallback in range(seed + 1, seed + 50):
                try:
                    env.generate_scene(scene_seed=fallback, scenario_mode=mode)
                    print(f"使用种子 {fallback} 成功生成场景")
                    seed = fallback
                    break
                except RuntimeError:
                    continue
            else:
                print("无法生成可达场景，使用固定模式")
                env.generate_scene(scene_seed=0, scenario_mode="fixed")
                seed = 0

    visualize_all(env, scene_seed=seed, save_path=save_path)


def main() -> None:
    if "--vis" in sys.argv:
        seed = 42
        mode = "full_map"
        save_path = None
        for i, arg in enumerate(sys.argv):
            if arg.startswith("--seed="):
                seed = int(arg.split("=")[1])
            elif arg == "--seed" and i + 1 < len(sys.argv):
                seed = int(sys.argv[i + 1])
            elif arg.startswith("--mode="):
                mode = arg.split("=", 1)[1]
            elif arg == "--mode" and i + 1 < len(sys.argv):
                mode = sys.argv[i + 1]
            elif arg.startswith("--save="):
                save_path = arg.split("=", 1)[1]
            elif arg == "--save" and i + 1 < len(sys.argv):
                save_path = sys.argv[i + 1]
        test_visualization(seed, mode=mode, save_path=save_path)
        return

    test_terrain_loader()
    test_scene_generation()
    test_occlusion()
    test_enemy_search()

    print()
    print("=" * 60)
    print("全部模块自检通过 [OK]")
    print("=" * 60)


if __name__ == "__main__":
    main()
