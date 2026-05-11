"""战场场景可视化模块。

可视化：地形标签图、二值可见性/遮挡区域、BFS 最短路径。
每个 50×50 格子都有清晰边框，配色对比明显。
"""

from __future__ import annotations

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 配置中文字体
_cn_fonts = [f.name for f in fm.fontManager.ttflist if "YaHei" in f.name]
if _cn_fonts:
    matplotlib.rcParams["font.family"] = _cn_fonts[0]
elif any("Hei" in f.name for f in fm.fontManager.ttflist):
    matplotlib.rcParams["font.family"] = "SimHei"

# ── 配色方案 ─────────────────────────────────────────
# 场景地形 (比之前稍深，让白色网格线更显眼)
C_GROUND   = np.array([0.90, 0.87, 0.80, 1.0])   # 暖沙色
C_BUILDING = np.array([0.25, 0.25, 0.27, 1.0])   # 深炭灰
C_TREE     = np.array([0.10, 0.30, 0.15, 1.0])   # 深林绿

# 可见性（用于 visualize_path 底图）
C_VIS_GROUND = np.array([1.00, 0.95, 0.10, 1.0])   # 黄色 — 敌人可见
C_OCC_GROUND = np.array([1.00, 0.55, 0.05, 1.0])   # 橙色 — 敌人不可见

# 可见性（用于 visualize_visibility 面板）
C_VISIBLE  = np.array([0.93, 0.68, 0.15, 1.0])   # 琥珀金
C_OCCLUDED = np.array([0.20, 0.30, 0.58, 1.0])   # 靛蓝

# 关键点
C_START  = "#00BCD4"   # 青
C_GOAL   = "#F44336"   # 红
C_ENEMY  = "#FF6D00"   # 橙

# 网格线 (中等灰度，在所有底色上都可见)
GRID_COLOR = "#AAAAAA"
GRID_LW    = 0.35


def _draw_cell_grid(ax, grid_size: int) -> None:
    """在每个格子边界画细线，让每个格子清晰可见。

    用 minor ticks 驱动 grid 渲染，比 102 条 axhline/axvline 更快更可靠。
    """
    ax.set_xticks(np.arange(-0.5, grid_size + 0.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size + 0.5, 1), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linewidth=GRID_LW, alpha=0.55, zorder=10)
    ax.tick_params(which="minor", bottom=False, left=False)


def _compute_bfs_path(env):
    """委托给 BattlefieldEnv.compute_bfs_path，使用完全相同的通行规则。"""
    return env.compute_bfs_path()


def _draw_markers(ax, env) -> None:
    """绘制起点(●)、终点(★)、敌人(▲)。"""
    sx, sy = int(env.start_position[0]), int(env.start_position[1])
    gx, gy = int(env.goal_position[0]), int(env.goal_position[1])
    ex, ey = int(env.enemy_position[0]), int(env.enemy_position[1])

    ax.scatter(sy, sx, c=C_START, s=200, marker="o",
               edgecolors="white", linewidths=2.0, zorder=20)
    ax.scatter(gy, gx, c=C_GOAL, s=260, marker="*",
               edgecolors="white", linewidths=2.0, zorder=20)
    ax.scatter(ey, ex, c=C_ENEMY, s=200, marker="^",
               edgecolors="white", linewidths=2.0, zorder=20)


def _build_marker_legend() -> list:
    return [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_START,
                   markersize=11, label="起点"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=C_GOAL,
                   markersize=13, label="终点"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=C_ENEMY,
                   markersize=11, label="敌人"),
    ]


def visualize_scene(env, title: str = "场景地形标签", ax=None):
    """可视化场景地形标签：地面/建筑/树木 + 起点/终点/敌人。

    地面: 暖白  |  建筑: 炭灰  |  树木: 深绿
    """
    tag_map = env.window_tag_map
    H, W = tag_map.shape if tag_map is not None else (env.grid_size, env.grid_size)
    rgba = np.zeros((H, W, 4), dtype=np.float32)

    if tag_map is not None:
        rgba[tag_map == 0] = C_GROUND
        rgba[tag_map == 1] = C_BUILDING
        rgba[tag_map == 2] = C_TREE
    else:
        rgba[:, :] = C_GROUND

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 8.5))

    ax.imshow(rgba, origin="lower", interpolation="nearest")
    _draw_cell_grid(ax, env.grid_size)
    _draw_markers(ax, env)

    legend_elements = [
        mpatches.Patch(facecolor=C_GROUND,   label="地面"),
        mpatches.Patch(facecolor=C_BUILDING, label="建筑"),
        mpatches.Patch(facecolor=C_TREE,     label="树木"),
        *_build_marker_legend(),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.92, edgecolor="#CCCCCC")

    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_xticks(range(0, env.grid_size, 5))
    ax.set_yticks(range(0, env.grid_size, 5))
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=14, fontweight="bold")


def visualize_visibility(env, title: str = "二值可见性 / 遮挡", ax=None):
    """可视化二值可见性地图。

    可见: 琥珀金  |  遮挡: 靛蓝  |  障碍物: 深色 (画斜线)
    """
    vis_map = env.visibility_map
    tag_map = env.window_tag_map
    H, W = vis_map.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)

    for x in range(H):
        for y in range(W):
            is_obstacle = tag_map is not None and tag_map[x, y] != 0
            if is_obstacle:
                rgba[x, y] = (0.18, 0.18, 0.20, 1.0)
            elif vis_map[x, y] > 0.5:
                rgba[x, y] = C_VISIBLE
            else:
                rgba[x, y] = C_OCCLUDED

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 8.5))

    ax.imshow(rgba, origin="lower", interpolation="nearest")

    # 障碍物格子画斜线纹理，与可见/遮挡区分
    if tag_map is not None:
        for x in range(H):
            for y in range(W):
                if tag_map[x, y] != 0:
                    ax.plot([y - 0.4, y + 0.4], [x - 0.4, x + 0.4],
                            color="#555555", linewidth=0.4, zorder=9)
                    ax.plot([y - 0.4, y + 0.4], [x + 0.4, x - 0.4],
                            color="#555555", linewidth=0.4, zorder=9)

    _draw_cell_grid(ax, env.grid_size)
    _draw_markers(ax, env)

    n_visible = int((vis_map > 0.5).sum())
    n_occluded = int((vis_map <= 0.5).sum()) - (
        int((tag_map != 0).sum()) if tag_map is not None else 0)
    n_blocked = int((tag_map != 0).sum()) if tag_map is not None else 0

    legend_elements = [
        mpatches.Patch(facecolor=C_VISIBLE,  label=f"可见 ({n_visible})"),
        mpatches.Patch(facecolor=C_OCCLUDED, label=f"遮挡 ({n_occluded})"),
        mpatches.Patch(facecolor=(0.18, 0.18, 0.20), label=f"障碍 ({n_blocked})"),
        *_build_marker_legend(),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.92, edgecolor="#CCCCCC")

    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_xticks(range(0, env.grid_size, 5))
    ax.set_yticks(range(0, env.grid_size, 5))
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=14, fontweight="bold")

    pct = n_visible / (n_visible + n_occluded) * 100 if (n_visible + n_occluded) > 0 else 0
    ax.set_xlabel(f"可见 {n_visible}  /  遮挡 {n_occluded}  ({pct:.1f}%)", fontsize=10)


def visualize_path(env, title: str = "BFS 最短路径", ax=None):
    """可视化 BFS 最短路径。

    底图：可通行格子按二值可见性涂黄/橙，建筑物画 X，树木画竖线。
    路径用青→品红渐变表示方向 (起点青 → 终点品红)。
    """
    path = _compute_bfs_path(env)
    vis_map = env.visibility_map
    tag_map = env.window_tag_map
    H, W = vis_map.shape
    C_BG = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    rgba = np.zeros((H, W, 4), dtype=np.float32)

    for x in range(H):
        for y in range(W):
            if tag_map is not None and tag_map[x, y] != 0:
                rgba[x, y] = C_BG
            elif vis_map[x, y] > 0.5:
                rgba[x, y] = C_VIS_GROUND
            else:
                rgba[x, y] = C_OCC_GROUND

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 8.5))

    ax.imshow(rgba, origin="lower", interpolation="nearest")
    _draw_cell_grid(ax, env.grid_size)

    # 障碍物图案：建筑 = X，树木 = 竖线
    if tag_map is not None:
        for x in range(H):
            for y in range(W):
                tag = tag_map[x, y]
                if tag == 1:
                    ax.plot([y - 0.35, y + 0.35], [x - 0.35, x + 0.35],
                            color="#333333", linewidth=1.0, zorder=9)
                    ax.plot([y - 0.35, y + 0.35], [x + 0.35, x - 0.35],
                            color="#333333", linewidth=1.0, zorder=9)
                elif tag == 2:
                    ax.plot([y, y], [x - 0.35, x + 0.35],
                            color="#333333", linewidth=1.2, zorder=9)

    # 绘制 BFS 路径
    if path:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        n = len(path)

        ax.plot(py, px, color="white", linewidth=5.0, zorder=11, solid_capstyle="round")
        for i in range(n - 1):
            t = i / max(n - 2, 1)
            r = 0.0 + 0.9 * t
            g = 0.75 * (1 - t) + 0.1 * t
            b = 0.85 * (1 - t) + 0.55 * t
            ax.plot([py[i], py[i + 1]], [px[i], px[i + 1]],
                    color=(r, g, b), linewidth=3.0, zorder=12, solid_capstyle="round")

        ax.scatter(py, px, c="white", s=12, zorder=13, edgecolors="none")

        n_visible = sum(1 for p in path if vis_map[p[0], p[1]] > 0.5)
        ax.set_xlabel(f"路径步数: {n - 1}  |  暴露步数: {n_visible}  |  "
                      f"暴露率: {n_visible / max(n - 1, 1) * 100:.0f}%", fontsize=10)
    else:
        ax.set_xlabel("无可达路径", fontsize=10, color="red")

    _draw_markers(ax, env)

    legend_elements = [
        mpatches.Patch(facecolor=C_VIS_GROUND, label="可见"),
        mpatches.Patch(facecolor=C_OCC_GROUND, label="遮挡"),
        plt.Line2D([0], [0], marker="x", color="#333333", linestyle="None",
                   markersize=10, markeredgewidth=1.5, label="建筑"),
        plt.Line2D([0], [0], marker="|", color="#333333", linestyle="None",
                   markersize=12, markeredgewidth=2.0, label="树木"),
        plt.Line2D([0], [0], color="#00BCD4", linewidth=3, label="路径"),
        *_build_marker_legend(),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=8, fontsize=7.5,
              framealpha=0.92, edgecolor="#CCCCCC")

    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_xticks(range(0, env.grid_size, 5))
    ax.set_yticks(range(0, env.grid_size, 5))
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=14, fontweight="bold")


def visualize_all(env, scene_seed: int | None = None, save_path: str | None = None):
    """三合一综合视图：场景总览 | 可见性/遮挡 | BFS 路径。"""
    fig, axes = plt.subplots(1, 3, figsize=(26, 9))
    fig.suptitle(f"战场场景可视化  (种子: {scene_seed}, 模式: {env.current_scenario_mode})",
                 fontsize=15, fontweight="bold", y=0.99)

    visualize_scene(env, title="场景地形标签", ax=axes[0])
    visualize_visibility(env, title="二值可见性 / 遮挡", ax=axes[1])
    visualize_path(env, title="BFS 最短路径", ax=axes[2])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"图片已保存至: {save_path}")

    plt.show()
    return fig
