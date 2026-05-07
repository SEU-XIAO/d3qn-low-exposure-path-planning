from __future__ import annotations
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.battlefield_env import BattlefieldEnv


def _compute_fov_masks(env: BattlefieldEnv) -> tuple[np.ndarray, np.ndarray]:
    visible_mask = env.visibility_map > 0.5
    occluded_mask = ~visible_mask
    ex = int(env.enemy_position[0]) - env.window_offset[1]  # row - row_offset
    ey = int(env.enemy_position[1]) - env.window_offset[0]  # col - col_offset
    if 0 <= ex < env.grid_size and 0 <= ey < env.grid_size:
        occluded_mask[ex, ey] = False
    return visible_mask, occluded_mask


def plot_scene(
    save_path: str | None = None,
    scene_seed: int | None = None,
    scenario_mode: str | None = None,
) -> None:
    env = BattlefieldEnv()
    env.reset(scene_seed=scene_seed, scenario_mode=scenario_mode)

    start = env.agent_position.astype(np.float32)
    goal = env.goal_position.astype(np.float32)
    fig = plt.figure(figsize=(20, 8.2))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.35, 1.0, 1.0))
    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_visibility = fig.add_subplot(gs[0, 2])

    draw_3d_scene(ax_3d, env)
    _plot_reference_path(ax_3d, start, goal)

    title = "50x50x8 Battlefield Layout"
    if env.current_scenario_mode == "random":
        title += f" | random seed={env.current_scene_seed}"
    else:
        title += " | fixed scene"
    title += f" | enemy=({int(env.enemy_position[0])},{int(env.enemy_position[1])})"

    ax_3d.set_title(title)

    _plot_topdown_scene(ax_top, env, title="Top-Down Visibility View")
    _plot_reference_path_topdown(ax_top, start, goal)
    _plot_binary_visibility_scene(ax_visibility, env, title="Direct Binary Visibility")

    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.05, top=0.93, wspace=0.08)
    _add_shared_legend(fig, [ax_3d, ax_top, ax_visibility])

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    else:
        plt.show()


def draw_3d_scene(ax: plt.Axes, env: BattlefieldEnv) -> None:
    config = env.config
    height_map = env.height_map

    x, y = np.meshgrid(np.arange(config.grid_size), np.arange(config.grid_size), indexing="ij")
    dx = np.full_like(x, 0.8, dtype=np.float32)
    dy = np.full_like(y, 0.8, dtype=np.float32)
    base = np.zeros_like(x, dtype=np.float32)

    mask = height_map > 0
    x_bar = x[mask].astype(np.float32)
    y_bar = y[mask].astype(np.float32)
    z_bar = base[mask]
    dx_bar = dx[mask]
    dy_bar = dy[mask]
    dz_bar = height_map[mask].astype(np.float32)
    ax.bar3d(x_bar, y_bar, z_bar, dx_bar, dy_bar, dz_bar, color="#8c6d46", alpha=0.75, shade=True)

    start = env.agent_position.astype(np.float32)
    goal = env.goal_position.astype(np.float32)
    enemy_global = env.enemy_position.astype(np.float32)
    ex = enemy_global[0] - float(env.window_offset[1])  # row - row_offset
    ey = enemy_global[1] - float(env.window_offset[0])  # col - col_offset
    ez = enemy_global[2]
    start_h = float(env.height_map[int(start[0]), int(start[1])])
    goal_h = float(env.height_map[int(goal[0]), int(goal[1])])

    ax.scatter(start[0] + 0.4, start[1] + 0.4, start_h + 0.45, color="green", s=80, label="Start")
    ax.scatter(goal[0] + 0.4, goal[1] + 0.4, goal_h + 0.45, color="blue", s=80, label="Goal")
    if 0 <= ex < config.grid_size and 0 <= ey < config.grid_size:
        enemy_h = float(height_map[int(ex), int(ey)])
        ax.scatter(ex + 0.4, ey + 0.4, enemy_h + 0.3, color="red", s=90, label="Enemy Lookout")

    _plot_3d_floor_grid(ax, env)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    ax.set_xlim(0, config.grid_size)
    ax.set_ylim(0, config.grid_size)
    z_visual_scale = 2.2
    ax.set_zlim(0, config.height_levels * z_visual_scale)
    major_xy_step = 10
    ax.set_xticks(np.arange(0, config.grid_size + 0.01, major_xy_step))
    ax.set_yticks(np.arange(0, config.grid_size + 0.01, major_xy_step))
    ax.set_zticks(np.arange(0, config.height_levels * z_visual_scale + 0.01, 2.0))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(True, alpha=0.35)
    ax.set_box_aspect((1.0, 1.0, 0.75))
    ax.view_init(elev=28, azim=-60)


def _plot_3d_floor_grid(ax: plt.Axes, env: BattlefieldEnv) -> None:
    edge_values = np.arange(-0.5, env.grid_size + 0.5, 1.0, dtype=np.float32)
    for x in edge_values:
        ax.plot(
            [x, x],
            [-0.5, env.grid_size - 0.5],
            [0.0, 0.0],
            color="#c8c8c8",
            alpha=0.38,
            linewidth=0.45,
            zorder=0,
        )
    for y in edge_values:
        ax.plot(
            [-0.5, env.grid_size - 0.5],
            [y, y],
            [0.0, 0.0],
            color="#c8c8c8",
            alpha=0.38,
            linewidth=0.45,
            zorder=0,
        )


def _plot_reference_path(ax: plt.Axes, start: np.ndarray, goal: np.ndarray) -> None:
    path_x = np.linspace(start[0] + 0.4, goal[0] + 0.4, 25)
    path_y = np.linspace(start[1] + 0.4, goal[1] + 0.4, 25)
    path_z = np.full_like(path_x, 0.4)
    ax.plot(path_x, path_y, path_z, linestyle="--", color="#1f77b4", alpha=0.5, label="Reference Line")


def _plot_topdown_scene(ax: plt.Axes, env: BattlefieldEnv, title: str) -> None:
    vis_binary = (env.visibility_map > 0.5).astype(np.float32)
    edge_values = np.arange(env.grid_size + 1, dtype=np.float32) - 0.5
    cmap = ListedColormap(["#6b8a9e", "#e85d5d"])
    ax.pcolormesh(
        edge_values, edge_values, vis_binary.T,
        cmap=cmap, shading="flat", vmin=0, vmax=1, alpha=0.82, zorder=1,
    )

    obstacle_cells = np.argwhere(env.height_map > 0)
    for x, y in obstacle_cells:
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5), 1.0, 1.0,
                facecolor="#373737", edgecolor="#252525",
                linewidth=0.6, alpha=0.90, zorder=3,
            )
        )

    start = env.start_position
    goal = env.goal_position
    ex = int(env.enemy_position[0]) - env.window_offset[1]  # row - row_offset
    ey = int(env.enemy_position[1]) - env.window_offset[0]  # col - col_offset

    ax.scatter(start[0], start[1], color="green", s=110, label="Start", zorder=5)
    ax.scatter(goal[0], goal[1], color="blue", s=110, label="Goal", zorder=5)
    if 0 <= ex < env.grid_size and 0 <= ey < env.grid_size:
        ax.scatter(ex, ey, color="red", s=120, label="Enemy Lookout", zorder=5)

    ax.scatter([], [], marker="s", s=80, color="#e85d5d", alpha=0.88, label="FOV Visible")
    ax.scatter([], [], marker="s", s=80, color="#6b8a9e", alpha=0.88, label="FOV Occluded")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_aspect("equal")
    grid_edges = np.arange(-0.5, env.grid_size + 0.5, 1.0)
    ax.set_xticks(grid_edges, minor=True)
    ax.set_yticks(grid_edges, minor=True)
    ax.grid(which="minor", color="#000000", alpha=0.6, linewidth=0.45, zorder=4)
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)


def _plot_reference_path_topdown(ax: plt.Axes, start: np.ndarray, goal: np.ndarray) -> None:
    path_x = np.linspace(start[0], goal[0], 25)
    path_y = np.linspace(start[1], goal[1], 25)
    ax.plot(path_x, path_y, linestyle="--", color="#1f77b4", alpha=0.5, label="Reference Line", zorder=6)


def _compute_direct_visibility_binary(env: BattlefieldEnv) -> np.ndarray:
    enemy_cell = (int(round(float(env.enemy_position[0]))), int(round(float(env.enemy_position[1]))))
    binary_map = np.zeros((env.grid_size, env.grid_size), dtype=np.int32)
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            visible = env._compute_cell_visibility_from(enemy_cell, (x, y))
            binary_map[x, y] = 1 if visible > 0.5 else 0
    return binary_map


def _plot_binary_visibility_scene(ax: plt.Axes, env: BattlefieldEnv, title: str) -> None:
    binary_map = _compute_direct_visibility_binary(env)
    edge_values = np.arange(env.grid_size + 1, dtype=np.float32) - 0.5
    cmap = ListedColormap(["#6b8a9e", "#e85d5d"])

    ax.pcolormesh(
        edge_values,
        edge_values,
        binary_map.T,
        cmap=cmap,
        shading="flat",
        vmin=0,
        vmax=1,
        alpha=0.95,
        zorder=1,
    )

    start = env.start_position
    goal = env.goal_position
    enemy = env.enemy_position
    ax.scatter(start[0], start[1], color="green", s=100, label="Start", zorder=4)
    ax.scatter(goal[0], goal[1], color="blue", s=100, label="Goal", zorder=4)
    ax.scatter(enemy[0], enemy[1], color="red", s=110, label="Enemy Lookout", zorder=4)

    visible_ratio = float(np.mean(binary_map))
    ax.text(
        0.02,
        0.98,
        f"visible={visible_ratio:.2%}\nnon-visible={1.0 - visible_ratio:.2%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        zorder=5,
    )

    ax.scatter([], [], marker="s", s=80, color="#e85d5d", alpha=0.95, label="Direct Visible")
    ax.scatter([], [], marker="s", s=80, color="#6b8a9e", alpha=0.95, label="Direct Non-Visible")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_aspect("equal")
    grid_edges = np.arange(-0.5, env.grid_size + 0.5, 1.0)
    ax.set_xticks(grid_edges, minor=True)
    ax.set_yticks(grid_edges, minor=True)
    ax.grid(which="minor", color="#000000", alpha=0.45, linewidth=0.35, zorder=3)
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)


def _add_shared_legend(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    handles: list[plt.Artist] = []
    labels: list[str] = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in labels:
                labels.append(label)
                handles.append(handle)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.87, 0.5),
            frameon=True,
        )


if __name__ == "__main__":
    plot_scene(scene_seed=7736)
