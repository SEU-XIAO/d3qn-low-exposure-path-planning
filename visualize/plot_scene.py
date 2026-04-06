from __future__ import annotations
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.battlefield_env import BattlefieldEnv


def _compute_fov_masks(env: BattlefieldEnv) -> tuple[np.ndarray, np.ndarray]:
    enemy_xy = env.enemy_position[:2].astype(np.float32)
    forward = env.enemy_forward.astype(np.float32)
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-6:
        return np.zeros_like(env.visibility_map, dtype=bool), np.zeros_like(env.visibility_map, dtype=bool)
    forward = forward / forward_norm

    xx, yy = np.meshgrid(np.arange(env.grid_size, dtype=np.float32), np.arange(env.grid_size, dtype=np.float32), indexing="ij")
    vectors = np.stack((xx - enemy_xy[0], yy - enemy_xy[1]), axis=-1)
    distances = np.linalg.norm(vectors, axis=-1)

    valid_distance = (distances > 1e-6) & (distances <= env.config.enemy_max_range)
    vectors_unit = np.zeros_like(vectors, dtype=np.float32)
    vectors_unit[valid_distance] = vectors[valid_distance] / distances[valid_distance, None]

    cos_threshold = float(np.cos(np.deg2rad(env.config.enemy_horizontal_fov_deg / 2.0)))
    cos_values = np.einsum("ijk,k->ij", vectors_unit, forward)
    in_fov = valid_distance & (cos_values >= cos_threshold)
    free_cells = env.occupancy_map <= 0

    # Reuse env precomputed maps so visualization stays consistent with training/evaluation logic.
    visible_mask = in_fov & free_cells & (env.visibility_map > 0)
    occluded_mask = in_fov & free_cells & (env.cover_map > 0)
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
    fig = plt.figure(figsize=(16, 8.5))
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_top = fig.add_subplot(1, 2, 2)

    draw_3d_scene(ax_3d, env)
    _plot_reference_path(ax_3d, start, goal)

    title = "32x32x8 Battlefield Layout"
    if env.current_scenario_mode == "random":
        title += f" | random seed={env.current_scene_seed}"
    else:
        title += " | fixed scene"

    ax_3d.set_title(title)
    ax_3d.legend(loc="upper right")

    _plot_topdown_scene(ax_top, env, title="Top-Down Visibility View")
    _plot_reference_path_topdown(ax_top, start, goal)
    ax_top.legend(loc="upper right")

    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.05, top=0.93, wspace=0.06)

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
    enemy = env.enemy_position.astype(np.float32)

    ax.scatter(start[0] + 0.4, start[1] + 0.4, 0.45, color="green", s=80, label="Start")
    ax.scatter(goal[0] + 0.4, goal[1] + 0.4, 0.45, color="blue", s=80, label="Goal")
    ax.scatter(enemy[0] + 0.4, enemy[1] + 0.4, 0.05, color="red", s=90, label="Enemy")

    _plot_3d_floor_grid(ax, env)
    _plot_enemy_fov(ax, env)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    ax.set_xlim(0, config.grid_size)
    ax.set_ylim(0, config.grid_size)
    z_visual_scale = 2.2
    ax.set_zlim(0, config.height_levels * z_visual_scale)
    major_xy_step = 8
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


def _plot_enemy_fov(ax: plt.Axes, env: BattlefieldEnv) -> None:
    config = env.config
    enemy = env.enemy_position
    forward = env.enemy_forward
    horizontal_half = np.deg2rad(config.enemy_horizontal_fov_deg / 2.0)
    yaw_center = np.arctan2(forward[1], forward[0])
    radius = config.enemy_max_range
    yaw_values = np.linspace(yaw_center - horizontal_half, yaw_center + horizontal_half, 72, dtype=np.float32)
    fan_x = enemy[0] + radius * np.cos(yaw_values)
    fan_y = enemy[1] + radius * np.sin(yaw_values)
    fan_z = np.full_like(fan_x, 0.05)

    poly_x = np.concatenate(([enemy[0]], fan_x, [enemy[0]]))
    poly_y = np.concatenate(([enemy[1]], fan_y, [enemy[1]]))
    poly_z = np.full_like(poly_x, 0.05)
    ax.plot_trisurf(poly_x, poly_y, poly_z, color="#9aa0a6", alpha=0.12, linewidth=0.0, shade=False)

    visible_mask, occluded_mask = _compute_fov_masks(env)
    if np.any(occluded_mask):
        occluded_cells = np.argwhere(occluded_mask)
        ax.scatter(
            occluded_cells[:, 0] + 0.5,
            occluded_cells[:, 1] + 0.5,
            np.full(len(occluded_cells), 0.08, dtype=np.float32),
            marker="s",
            s=24,
            color="#9aa0a6",
            alpha=0.35,
            depthshade=False,
            label="FOV Occluded",
        )
    if np.any(visible_mask):
        visible_cells = np.argwhere(visible_mask)
        ax.scatter(
            visible_cells[:, 0] + 0.5,
            visible_cells[:, 1] + 0.5,
            np.full(len(visible_cells), 0.09, dtype=np.float32),
            marker="s",
            s=24,
            color="#ff8a80",
            alpha=0.55,
            depthshade=False,
            label="FOV Visible",
        )

    left_x = [enemy[0], enemy[0] + radius * np.cos(yaw_center - horizontal_half)]
    left_y = [enemy[1], enemy[1] + radius * np.sin(yaw_center - horizontal_half)]
    right_x = [enemy[0], enemy[0] + radius * np.cos(yaw_center + horizontal_half)]
    right_y = [enemy[1], enemy[1] + radius * np.sin(yaw_center + horizontal_half)]
    ax.plot(left_x, left_y, [0.06, 0.06], color="#e85d5d", alpha=0.75, linewidth=1.2)
    ax.plot(right_x, right_y, [0.06, 0.06], color="#e85d5d", alpha=0.75, linewidth=1.2)
    ax.plot(fan_x, fan_y, fan_z + 0.01, color="#e85d5d", alpha=0.65, linewidth=1.2)


def _plot_reference_path(ax: plt.Axes, start: np.ndarray, goal: np.ndarray) -> None:
    path_x = np.linspace(start[0] + 0.4, goal[0] + 0.4, 25)
    path_y = np.linspace(start[1] + 0.4, goal[1] + 0.4, 25)
    path_z = np.full_like(path_x, 0.4)
    ax.plot(path_x, path_y, path_z, linestyle="--", color="#1f77b4", alpha=0.5, label="Reference Line")


def _plot_topdown_scene(ax: plt.Axes, env: BattlefieldEnv, title: str) -> None:
    visible_mask, occluded_mask = _compute_fov_masks(env)
    visible = np.ma.masked_where(~visible_mask, env.visibility_map)
    occluded = np.ma.masked_where(~occluded_mask, np.ones_like(env.visibility_map, dtype=np.float32))
    edge_values = np.arange(env.grid_size + 1, dtype=np.float32) - 0.5
    ax.pcolormesh(
        edge_values,
        edge_values,
        occluded.T,
        cmap="Greys",
        shading="flat",
        alpha=0.22,
        vmin=0.0,
        vmax=1.0,
        zorder=1,
    )
    ax.pcolormesh(
        edge_values,
        edge_values,
        visible.T,
        cmap="Reds",
        shading="flat",
        alpha=0.32,
        vmin=0.0,
        vmax=1.0,
        zorder=2,
    )

    obstacle_cells = np.argwhere(env.height_map > 0)
    for x, y in obstacle_cells:
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                1.0,
                1.0,
                facecolor="#6f6f6f",
                edgecolor="#4f4f4f",
                linewidth=0.6,
                alpha=0.85,
                zorder=3,
            )
        )

    start = env.start_position
    goal = env.goal_position
    enemy = env.enemy_position

    ax.scatter(start[0], start[1], color="green", s=110, label="Start", zorder=5)
    ax.scatter(goal[0], goal[1], color="blue", s=110, label="Goal", zorder=5)
    ax.scatter(enemy[0], enemy[1], color="red", s=120, label="Enemy", zorder=5)

    ax.scatter([], [], marker="s", s=80, color="#ff8a80", alpha=0.55, label="FOV Visible")
    ax.scatter([], [], marker="s", s=80, color="#9aa0a6", alpha=0.5, label="FOV Occluded")

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


if __name__ == "__main__":
    plot_scene(scene_seed=1)
