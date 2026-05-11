"""查找适合展示的 50x50 场景。

相比早期只按“障碍多不多”排，这个版本支持一套更适合当前项目的筛选方式：

1. 用积分图快速扫描整张全图窗口
2. 按可见地面格数量 / 比例给窗口打分
3. 对少量高分候选再做局部 BFS 精筛
4. 进一步看“建议路径本身会不会暴露很多”，挑出更值得看的窗口

这样做的重点是：

- 不靠随机暴搜
- 不需要把 50x50 场景一个个生成出来撞运气
- 可以直接给出一批更适合人工检查的窗口

示例：
    python -m visualize.find_dense_scenes
    python -m visualize.find_dense_scenes --metric visible-ground --stride 5 --top 12
    python -m visualize.find_dense_scenes --metric visible-ground --show 0
    python -m visualize.find_dense_scenes --metric visible-ground --export-top 6 --output-dir analysis/high_vis_50
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from config import EnvConfig
from env.terrain_loader import FullTerrain, load_terrain


C_GROUND = np.array([0.90, 0.87, 0.80, 1.0], dtype=np.float32)
C_BUILDING = np.array([0.22, 0.22, 0.24, 1.0], dtype=np.float32)
C_TREE = np.array([0.12, 0.35, 0.18, 1.0], dtype=np.float32)
C_VISIBLE = np.array([0.98, 0.80, 0.36, 1.0], dtype=np.float32)

ACTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def _integral_image(arr: np.ndarray) -> np.ndarray:
    arr64 = arr.astype(np.float64, copy=False)
    ii = arr64.cumsum(axis=0).cumsum(axis=1)
    return np.pad(ii, ((1, 0), (1, 0)), mode="constant")


def _rect_sum(ii: np.ndarray, oy: int, ox: int, window_size: int) -> float:
    y1 = oy + window_size
    x1 = ox + window_size
    return float(ii[y1, x1] - ii[oy, x1] - ii[y1, ox] + ii[oy, ox])


def _load_visibility_bundle(cfg: EnvConfig) -> tuple[np.ndarray, list[tuple[int, int]]]:
    pool_path = Path(cfg.enemy_pool_path)
    vis_path = pool_path.parent / "visibility_maps.npz"
    if not pool_path.exists():
        raise FileNotFoundError(f"缺少敌点池: {pool_path}")
    if not vis_path.exists():
        raise FileNotFoundError(f"缺少全图可见性底图: {vis_path}")

    with open(pool_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    enemy_pool = [tuple(map(int, p)) for p in data.get("enemy_pool", [])]

    loaded = np.load(vis_path)
    vis_stack = np.stack(
        [loaded[f"vis_{i}"].astype(np.float32) for i in range(len(enemy_pool))],
        axis=0,
    )
    return vis_stack, enemy_pool


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
        if int(tag_map[nx, ny]) != 0:
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


def _bfs_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    height_map: np.ndarray,
    tag_map: np.ndarray,
    cfg: EnvConfig,
) -> list[tuple[int, int]] | None:
    if start == goal:
        return [start]

    queue: list[tuple[int, int]] = [start]
    head = 0
    visited = {start}
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    while head < len(queue):
        cur = queue[head]
        head += 1
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


def _subsample_points(points: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    if len(points) <= limit:
        return points
    idx = np.linspace(0, len(points) - 1, num=limit, dtype=np.int32)
    return [points[int(i)] for i in idx.tolist()]


def _suggest_corner_case(
    height_map: np.ndarray,
    tag_map: np.ndarray,
    vis_map: np.ndarray | None,
    cfg: EnvConfig,
    corner_span: int,
    candidate_limit: int,
) -> dict | None:
    g = int(tag_map.shape[0])
    start_pool = [
        (x, y)
        for x in range(min(corner_span, g))
        for y in range(min(corner_span, g))
        if int(tag_map[x, y]) == 0
    ]
    g0 = max(0, g - corner_span)
    goal_pool = [
        (x, y)
        for x in range(g0, g)
        for y in range(g0, g)
        if int(tag_map[x, y]) == 0
    ]
    if not start_pool or not goal_pool:
        return None

    start_pool.sort(key=lambda p: (p[0] + p[1], p[0], p[1]))
    goal_pool.sort(key=lambda p: (-(p[0] + p[1]), -p[0], -p[1]))

    starts = _subsample_points(start_pool[: max(candidate_limit * 4, candidate_limit)], candidate_limit)
    goals = _subsample_points(goal_pool[: max(candidate_limit * 4, candidate_limit)], candidate_limit)

    pairs: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
    for s in starts:
        for t in goals:
            dist = float(np.linalg.norm(np.array(s, dtype=np.float32) - np.array(t, dtype=np.float32)))
            if dist < float(cfg.min_start_goal_distance):
                continue
            pairs.append((dist, s, t))
    if not pairs:
        return None

    pairs.sort(key=lambda item: item[0], reverse=True)

    best: dict | None = None
    for dist, start, goal in pairs:
        path = _bfs_path(start, goal, height_map, tag_map, cfg)
        if not path:
            continue
        vis_count = 0
        vis_ratio = 0.0
        if vis_map is not None and path:
            vis_count = int(sum(float(vis_map[x, y]) > 0.5 for x, y in path))
            vis_ratio = vis_count / max(1, len(path))

        candidate = {
            "start": [int(start[0]), int(start[1])],
            "goal": [int(goal[0]), int(goal[1])],
            "bfs_len": max(0, len(path) - 1),
            "bfs_visible_count": vis_count,
            "bfs_visible_ratio": vis_ratio,
            "euclid_dist": dist,
            "bfs_path": [[int(x), int(y)] for x, y in path],
        }
        if best is None:
            best = candidate
            continue

        best_key = (best["bfs_visible_count"], best["bfs_visible_ratio"], best["bfs_len"], best["euclid_dist"])
        cur_key = (candidate["bfs_visible_count"], candidate["bfs_visible_ratio"], candidate["bfs_len"], candidate["euclid_dist"])
        if cur_key > best_key:
            best = candidate

    return best


def _apply_diversity_filter(
    results: list[dict],
    diverse_radius: int,
    window_size: int,
) -> list[dict]:
    if diverse_radius <= 0:
        return list(results)

    filtered: list[dict] = []
    radius2 = float(diverse_radius * diverse_radius)
    for item in results:
        cy = item["oy"] + window_size / 2.0
        cx = item["ox"] + window_size / 2.0
        keep = True
        for prev in filtered:
            py = prev["oy"] + window_size / 2.0
            px = prev["ox"] + window_size / 2.0
            if (cy - py) ** 2 + (cx - px) ** 2 < radius2:
                keep = False
                break
        if keep:
            filtered.append(item)
    return filtered


def _scan_obstacle_windows(
    terrain: FullTerrain,
    window_size: int,
    stride: int,
) -> list[dict]:
    tag = terrain.tag_map
    h, w = tag.shape
    obs_ii = _integral_image((tag != 0).astype(np.float32))
    ground_ii = _integral_image((tag == 0).astype(np.float32))
    build_ii = _integral_image((tag == 1).astype(np.float32))
    tree_ii = _integral_image((tag == 2).astype(np.float32))

    results: list[dict] = []
    total = window_size * window_size
    for oy in range(0, h - window_size + 1, stride):
        for ox in range(0, w - window_size + 1, stride):
            obstacle_count = int(round(_rect_sum(obs_ii, oy, ox, window_size)))
            ground_count = int(round(_rect_sum(ground_ii, oy, ox, window_size)))
            building = int(round(_rect_sum(build_ii, oy, ox, window_size)))
            tree = int(round(_rect_sum(tree_ii, oy, ox, window_size)))
            results.append(
                {
                    "ox": ox,
                    "oy": oy,
                    "score": obstacle_count / total,
                    "obstacle_ratio": obstacle_count / total,
                    "obstacle_count": obstacle_count,
                    "ground_count": ground_count,
                    "passable_ratio": ground_count / total,
                    "building": building,
                    "tree": tree,
                    "best_enemy_idx": None,
                    "visible_ground_count": 0,
                    "visible_ground_ratio": 0.0,
                    "visible_total_count": 0,
                }
            )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def _scan_visibility_windows(
    terrain: FullTerrain,
    vis_stack: np.ndarray,
    window_size: int,
    stride: int,
    metric: str,
    aggregate: str,
    enemy_index: int,
    min_passable_ratio: float,
) -> list[dict]:
    tag = terrain.tag_map
    h, w = tag.shape
    total = window_size * window_size

    ground = (tag == 0).astype(np.float32)
    ground_ii = _integral_image(ground)
    obs_ii = _integral_image((tag != 0).astype(np.float32))
    build_ii = _integral_image((tag == 1).astype(np.float32))
    tree_ii = _integral_image((tag == 2).astype(np.float32))
    vis_total_iis = [_integral_image(vis_stack[i]) for i in range(vis_stack.shape[0])]
    vis_ground_iis = [_integral_image(vis_stack[i] * ground) for i in range(vis_stack.shape[0])]

    if enemy_index >= 0:
        enemy_indices = [enemy_index]
    else:
        enemy_indices = list(range(vis_stack.shape[0]))

    results: list[dict] = []
    for oy in range(0, h - window_size + 1, stride):
        for ox in range(0, w - window_size + 1, stride):
            ground_count = int(round(_rect_sum(ground_ii, oy, ox, window_size)))
            passable_ratio = ground_count / total
            if passable_ratio < min_passable_ratio:
                continue

            obstacle_count = int(round(_rect_sum(obs_ii, oy, ox, window_size)))
            building = int(round(_rect_sum(build_ii, oy, ox, window_size)))
            tree = int(round(_rect_sum(tree_ii, oy, ox, window_size)))

            per_enemy: list[dict] = []
            for eidx in enemy_indices:
                visible_ground_count = int(round(_rect_sum(vis_ground_iis[eidx], oy, ox, window_size)))
                visible_total_count = int(round(_rect_sum(vis_total_iis[eidx], oy, ox, window_size)))
                visible_ground_ratio = visible_ground_count / max(1, ground_count)
                visible_total_ratio = visible_total_count / total

                if metric == "visible-ground":
                    score = float(visible_ground_count)
                elif metric == "visible-ground-ratio":
                    score = float(visible_ground_ratio)
                elif metric == "visible-total":
                    score = float(visible_total_count)
                else:
                    raise ValueError(f"未知 metric: {metric}")

                per_enemy.append(
                    {
                        "enemy_idx": int(eidx),
                        "score": score,
                        "visible_ground_count": visible_ground_count,
                        "visible_ground_ratio": visible_ground_ratio,
                        "visible_total_count": visible_total_count,
                        "visible_total_ratio": visible_total_ratio,
                    }
                )

            if not per_enemy:
                continue

            best_enemy = max(per_enemy, key=lambda item: item["score"])
            if aggregate == "mean" and len(per_enemy) > 1:
                score = float(np.mean([item["score"] for item in per_enemy]))
            else:
                score = float(best_enemy["score"])

            results.append(
                {
                    "ox": ox,
                    "oy": oy,
                    "score": score,
                    "obstacle_ratio": obstacle_count / total,
                    "obstacle_count": obstacle_count,
                    "ground_count": ground_count,
                    "passable_ratio": passable_ratio,
                    "building": building,
                    "tree": tree,
                    "best_enemy_idx": int(best_enemy["enemy_idx"]),
                    "visible_ground_count": int(best_enemy["visible_ground_count"]),
                    "visible_ground_ratio": float(best_enemy["visible_ground_ratio"]),
                    "visible_total_count": int(best_enemy["visible_total_count"]),
                    "visible_total_ratio": float(best_enemy["visible_total_ratio"]),
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def _refine_candidates(
    results: list[dict],
    terrain: FullTerrain,
    vis_stack: np.ndarray | None,
    cfg: EnvConfig,
    window_size: int,
    refine_top: int,
    corner_span: int,
    pair_candidates: int,
    metric: str,
) -> list[dict]:
    refined: list[dict] = []
    for item in results[: max(0, refine_top)]:
        ox, oy = int(item["ox"]), int(item["oy"])
        height = terrain.height_map[oy : oy + window_size, ox : ox + window_size].astype(np.int32)
        tag = terrain.tag_map[oy : oy + window_size, ox : ox + window_size].astype(np.int32)

        vis_map = None
        enemy_idx = item.get("best_enemy_idx")
        if vis_stack is not None and enemy_idx is not None:
            vis_map = vis_stack[int(enemy_idx), oy : oy + window_size, ox : ox + window_size].astype(np.float32)

        case = _suggest_corner_case(
            height_map=height,
            tag_map=tag,
            vis_map=vis_map,
            cfg=cfg,
            corner_span=corner_span,
            candidate_limit=pair_candidates,
        )
        if case is None:
            continue

        merged = dict(item)
        merged.update(case)
        refined.append(merged)

    if metric.startswith("visible-"):
        refined.sort(
            key=lambda item: (
                int(item.get("bfs_visible_count", 0)),
                float(item.get("bfs_visible_ratio", 0.0)),
                int(item.get("visible_ground_count", 0)),
                int(item.get("bfs_len", 0)),
                float(item.get("score", 0.0)),
            ),
            reverse=True,
        )
    else:
        refined.sort(
            key=lambda item: (
                float(item.get("score", 0.0)),
                int(item.get("bfs_len", 0)),
                int(item.get("ground_count", 0)),
            ),
            reverse=True,
        )
    return refined


def _print_table(results: list[dict], top_n: int, metric: str) -> None:
    if metric.startswith("visible-"):
        print(
            f"\n{'排名':<4} {'ox':>4} {'oy':>4} {'敌点':>4} {'可见地面':>8} {'占地比':>7} "
            f"{'BFS暴露':>7} {'BFS长':>6} {'地面':>6} {'障碍比':>7}"
        )
        print("-" * 78)
        for rank, r in enumerate(results[:top_n]):
            print(
                f"{rank:<4} {r['ox']:>4} {r['oy']:>4} {int(r['best_enemy_idx']):>4} "
                f"{int(r.get('visible_ground_count', 0)):>8} {float(r.get('visible_ground_ratio', 0.0)):>6.1%} "
                f"{int(r.get('bfs_visible_count', 0)):>7} {int(r.get('bfs_len', 0)):>6} "
                f"{int(r.get('ground_count', 0)):>6} {float(r.get('obstacle_ratio', 0.0)):>6.1%}"
            )
    else:
        print(f"\n{'排名':<4} {'ox':>4} {'oy':>4} {'障碍比':>7} {'地面':>6} {'建筑':>6} {'树木':>6} {'BFS长':>6}")
        print("-" * 58)
        for rank, r in enumerate(results[:top_n]):
            print(
                f"{rank:<4} {r['ox']:>4} {r['oy']:>4} {float(r.get('obstacle_ratio', 0.0)):>6.1%} "
                f"{int(r.get('ground_count', 0)):>6} {int(r.get('building', 0)):>6} "
                f"{int(r.get('tree', 0)):>6} {int(r.get('bfs_len', 0)):>6}"
            )


def _render_rgba(tag_map: np.ndarray, vis_map: np.ndarray | None) -> np.ndarray:
    rgba = np.zeros((tag_map.shape[0], tag_map.shape[1], 4), dtype=np.float32)
    rgba[tag_map == 0] = C_GROUND
    rgba[tag_map == 1] = C_BUILDING
    rgba[tag_map == 2] = C_TREE

    if vis_map is not None:
        mask = (tag_map == 0) & (vis_map > 0.5)
        rgba[mask, :3] = rgba[mask, :3] * 0.35 + C_VISIBLE[:3] * 0.65
        rgba[mask, 3] = 1.0
    return rgba


def _plot_candidate(
    terrain: FullTerrain,
    vis_stack: np.ndarray | None,
    enemy_pool: list[tuple[int, int]] | None,
    item: dict,
    window_size: int,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    ox, oy = int(item["ox"]), int(item["oy"])
    tag = terrain.tag_map[oy : oy + window_size, ox : ox + window_size]
    vis_map = None
    enemy_idx = item.get("best_enemy_idx")
    enemy_text = "无"
    enemy_local = None
    if vis_stack is not None and enemy_idx is not None:
        vis_map = vis_stack[int(enemy_idx), oy : oy + window_size, ox : ox + window_size]
        if enemy_pool is not None and 0 <= int(enemy_idx) < len(enemy_pool):
            ey, ex = enemy_pool[int(enemy_idx)]
            enemy_text = f"{int(enemy_idx)}@({ey},{ex})"
            if oy <= ey < oy + window_size and ox <= ex < ox + window_size:
                enemy_local = (ey - oy, ex - ox)

    rgba = _render_rgba(tag, vis_map)
    fig, ax = plt.subplots(figsize=(8.8, 8.8))
    ax.imshow(rgba, origin="lower", interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, window_size + 0.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, window_size + 0.5, 1), minor=True)
    ax.grid(which="minor", color="#888888", linewidth=0.30, alpha=0.40)
    ax.tick_params(which="minor", bottom=False, left=False)

    bfs_path = np.array(item.get("bfs_path", []), dtype=np.float32)
    if len(bfs_path) > 1:
        ax.plot(bfs_path[:, 1], bfs_path[:, 0], color="#1565c0", linewidth=2.2, alpha=0.95, label="suggested bfs")

    start = item.get("start")
    goal = item.get("goal")
    if start:
        ax.scatter([start[1]], [start[0]], c="#00acc1", s=110, marker="o", edgecolors="white", linewidths=1.4, zorder=5, label="start")
    if goal:
        ax.scatter([goal[1]], [goal[0]], c="#f44336", s=150, marker="*", edgecolors="white", linewidths=1.4, zorder=5, label="goal")
    if enemy_local is not None:
        ax.scatter([enemy_local[1]], [enemy_local[0]], c="#6a1b9a", s=90, marker="^", edgecolors="white", linewidths=1.2, zorder=5, label="enemy")

    title = (
        f"50x50 窗口 (ox={ox}, oy={oy}) | enemy={enemy_text} | "
        f"可见地面={int(item.get('visible_ground_count', 0))} ({float(item.get('visible_ground_ratio', 0.0)):.1%})"
    )
    subtitle = (
        f"BFS长={int(item.get('bfs_len', 0))} | BFS暴露={int(item.get('bfs_visible_count', 0))} "
        f"({float(item.get('bfs_visible_ratio', 0.0)):.1%}) | 地面={int(item.get('ground_count', 0))} "
        f"| 障碍比={float(item.get('obstacle_ratio', 0.0)):.1%}"
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(subtitle)
    ax.set_xlim(-0.5, window_size - 0.5)
    ax.set_ylim(-0.5, window_size - 0.5)
    ax.set_xticks(range(0, window_size, 5))
    ax.set_yticks(range(0, window_size, 5))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
        print(f"图片已保存: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _quick_plot(
    terrain: FullTerrain,
    item: dict,
    window_size: int,
    save_path: str | None = None,
) -> None:
    ox, oy = int(item["ox"]), int(item["oy"])
    tag = terrain.tag_map[oy : oy + window_size, ox : ox + window_size]
    rgba = _render_rgba(tag, vis_map=None)

    fig, ax = plt.subplots(figsize=(8.4, 8.4))
    ax.imshow(rgba, origin="lower", interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, window_size + 0.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, window_size + 0.5, 1), minor=True)
    ax.grid(which="minor", color="#888888", linewidth=0.30, alpha=0.40)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-0.5, window_size - 0.5)
    ax.set_ylim(-0.5, window_size - 0.5)
    ax.set_xticks(range(0, window_size, 5))
    ax.set_yticks(range(0, window_size, 5))
    ax.set_title(f"纯地形预览 (ox={ox}, oy={oy})")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
        print(f"图片已保存: {save_path}")
    plt.show()


def _export_top(
    results: list[dict],
    terrain: FullTerrain,
    vis_stack: np.ndarray | None,
    enemy_pool: list[tuple[int, int]] | None,
    window_size: int,
    export_top: int,
    output_dir: str,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for rank, item in enumerate(results[:export_top]):
        png_path = out_dir / f"rank_{rank:02d}_ox_{item['ox']}_oy_{item['oy']}.png"
        _plot_candidate(
            terrain=terrain,
            vis_stack=vis_stack,
            enemy_pool=enemy_pool,
            item=item,
            window_size=window_size,
            save_path=str(png_path),
            show=False,
        )
        copied = dict(item)
        copied["image"] = png_path.name
        manifest.append(copied)

    with open(out_dir / "candidates.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"已导出 {min(export_top, len(results))} 个候选到: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="查找适合人工检查的 50x50 场景")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5, help="滑窗步长；靠积分图扫描，5 一般已经很快")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--metric",
        type=str,
        default="visible-ground",
        choices=["obstacle", "visible-ground", "visible-ground-ratio", "visible-total"],
        help="obstacle=障碍占比；visible-ground=可见地面格数量；visible-ground-ratio=可见地面占比；visible-total=窗口内总可见格数量",
    )
    parser.add_argument("--aggregate", type=str, default="max", choices=["max", "mean"], help="可见性打分时对 8 个敌点的聚合方式")
    parser.add_argument("--enemy-index", type=int, default=-1, help="-1 表示自动为每个窗口选择最强敌点；>=0 表示固定敌点")
    parser.add_argument("--min-passable-ratio", type=float, default=0.30, help="只保留可通行地面占比足够的窗口")
    parser.add_argument("--diverse-radius", type=int, default=25, help="去掉位置过近的近重复窗口，单位是全图格子")
    parser.add_argument("--refine-top", type=int, default=80, help="只对前 N 个高分窗口做 BFS 精筛")
    parser.add_argument("--corner-span", type=int, default=15, help="从左上/右下角区域挑建议起终点")
    parser.add_argument("--pair-candidates", type=int, default=8, help="每个角区域抽多少个候选点做配对搜索")
    parser.add_argument("--show", type=int, default=None, help="可视化排名第 N 的候选窗口")
    parser.add_argument("--quick", type=int, default=None, help="快速看纯地形的排名第 N 窗口")
    parser.add_argument("--try", type=int, default=None, dest="try_top", metavar="N", help="从前 N 个候选里展示第一个精筛成功的窗口")
    parser.add_argument("--export-top", type=int, default=0, help="批量导出前 N 个候选窗口")
    parser.add_argument("--output-dir", type=str, default="analysis/high_vis_50")
    parser.add_argument("--save", type=str, default=None, help="给单张 show/quick 输出图片路径")
    args = parser.parse_args()

    cfg = EnvConfig()
    terrain = load_terrain(cfg.full_map_path)
    print(f"地形加载完成: {terrain.full_height}x{terrain.full_width}")

    vis_stack: np.ndarray | None = None
    enemy_pool: list[tuple[int, int]] | None = None
    if args.metric != "obstacle":
        vis_stack, enemy_pool = _load_visibility_bundle(cfg)
        print(f"已加载 {vis_stack.shape[0]} 张全图可见性底图")
        if args.enemy_index >= vis_stack.shape[0]:
            raise ValueError(f"--enemy-index={args.enemy_index} 超出范围 0~{vis_stack.shape[0] - 1}")

    if args.metric == "obstacle":
        results = _scan_obstacle_windows(
            terrain=terrain,
            window_size=args.window_size,
            stride=args.stride,
        )
    else:
        results = _scan_visibility_windows(
            terrain=terrain,
            vis_stack=vis_stack,
            window_size=args.window_size,
            stride=args.stride,
            metric=args.metric,
            aggregate=args.aggregate,
            enemy_index=args.enemy_index,
            min_passable_ratio=args.min_passable_ratio,
        )

    print(f"原始候选窗口数: {len(results)}")
    results = _apply_diversity_filter(results, diverse_radius=args.diverse_radius, window_size=args.window_size)
    print(f"去重后候选窗口数: {len(results)}")

    refined = _refine_candidates(
        results=results,
        terrain=terrain,
        vis_stack=vis_stack,
        cfg=cfg,
        window_size=args.window_size,
        refine_top=args.refine_top,
        corner_span=args.corner_span,
        pair_candidates=args.pair_candidates,
        metric=args.metric,
    )
    print(f"精筛成功窗口数: {len(refined)}")
    if not refined:
        print("没有找到满足条件的窗口，请尝试降低 --min-passable-ratio 或增大 --refine-top。")
        return

    _print_table(refined, top_n=args.top, metric=args.metric)

    if args.quick is not None:
        if args.quick < 0 or args.quick >= len(refined):
            raise IndexError(f"--quick={args.quick} 超出范围 0~{len(refined) - 1}")
        _quick_plot(terrain=terrain, item=refined[args.quick], window_size=args.window_size, save_path=args.save)

    if args.show is not None:
        if args.show < 0 or args.show >= len(refined):
            raise IndexError(f"--show={args.show} 超出范围 0~{len(refined) - 1}")
        _plot_candidate(
            terrain=terrain,
            vis_stack=vis_stack,
            enemy_pool=enemy_pool,
            item=refined[args.show],
            window_size=args.window_size,
            save_path=args.save,
            show=True,
        )

    if args.try_top is not None:
        limit = min(max(1, args.try_top), len(refined))
        _plot_candidate(
            terrain=terrain,
            vis_stack=vis_stack,
            enemy_pool=enemy_pool,
            item=refined[0 if limit <= 1 else 0],
            window_size=args.window_size,
            save_path=args.save,
            show=True,
        )

    if args.export_top > 0:
        _export_top(
            results=refined,
            terrain=terrain,
            vis_stack=vis_stack,
            enemy_pool=enemy_pool,
            window_size=args.window_size,
            export_top=args.export_top,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
