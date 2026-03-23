from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from env.battlefield_env import BattlefieldEnv


def plot_scene(save_path: str | None = None) -> None:
    env = BattlefieldEnv()
    start = np.array(env.config.start, dtype=np.float32)
    goal = np.array(env.config.goal, dtype=np.float32)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    draw_3d_scene(ax, env)
    _plot_reference_path(ax, start, goal)

    ax.set_title("32x32x8 Battlefield Layout")

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

    start = np.array(config.start, dtype=np.float32)
    goal = np.array(config.goal, dtype=np.float32)
    enemy = np.array((config.enemy_position[0], config.enemy_position[1], config.enemy_height), dtype=np.float32)

    ax.scatter(start[0] + 0.4, start[1] + 0.4, 0.45, color="green", s=80, label="Start")
    ax.scatter(goal[0] + 0.4, goal[1] + 0.4, 0.45, color="blue", s=80, label="Goal")
    ax.scatter(enemy[0] + 0.4, enemy[1] + 0.4, enemy[2], color="red", s=90, label="Enemy")

    _plot_enemy_fov(ax, env)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    ax.set_xlim(0, config.grid_size)
    ax.set_ylim(0, config.grid_size)
    ax.set_zlim(0, config.height_levels)
    ax.view_init(elev=28, azim=-60)
    ax.legend(loc="upper right")


def _plot_enemy_fov(ax: plt.Axes, env: BattlefieldEnv) -> None:
    config = env.config
    enemy = env.enemy_position
    forward = env.enemy_forward
    horizontal_half = np.deg2rad(config.enemy_horizontal_fov_deg / 2.0)
    vertical_half = np.deg2rad(config.enemy_vertical_fov_deg / 2.0)
    yaw_center = np.arctan2(forward[1], forward[0])
    pitch_center = np.arctan2(forward[2], np.linalg.norm(forward[:2]))

    yaw_values = np.linspace(yaw_center - horizontal_half, yaw_center + horizontal_half, 36, dtype=np.float32)
    pitch_values = np.linspace(pitch_center - vertical_half, pitch_center + vertical_half, 24, dtype=np.float32)
    yaw_grid, pitch_grid = np.meshgrid(yaw_values, pitch_values, indexing="xy")

    radius = config.enemy_max_range
    x = enemy[0] + radius * np.cos(pitch_grid) * np.cos(yaw_grid)
    y = enemy[1] + radius * np.cos(pitch_grid) * np.sin(yaw_grid)
    z = enemy[2] + radius * np.sin(pitch_grid)

    z = np.clip(z, 0.0, config.height_levels)
    ax.plot_surface(x, y, z, color="#ff6b6b", alpha=0.12, linewidth=0.0, shade=False)

    edge_pitch = np.linspace(pitch_center - vertical_half, pitch_center + vertical_half, 36, dtype=np.float32)
    for yaw in (yaw_center - horizontal_half, yaw_center + horizontal_half):
        edge_x = enemy[0] + radius * np.cos(edge_pitch) * np.cos(yaw)
        edge_y = enemy[1] + radius * np.cos(edge_pitch) * np.sin(yaw)
        edge_z = np.clip(enemy[2] + radius * np.sin(edge_pitch), 0.0, config.height_levels)
        ax.plot(edge_x, edge_y, edge_z, color="#e85d5d", alpha=0.55, linewidth=1.0)

    edge_yaw = np.linspace(yaw_center - horizontal_half, yaw_center + horizontal_half, 48, dtype=np.float32)
    for pitch in (pitch_center - vertical_half, pitch_center + vertical_half):
        edge_x = enemy[0] + radius * np.cos(pitch) * np.cos(edge_yaw)
        edge_y = enemy[1] + radius * np.cos(pitch) * np.sin(edge_yaw)
        edge_z = np.clip(np.full_like(edge_x, enemy[2] + radius * np.sin(pitch)), 0.0, config.height_levels)
        ax.plot(edge_x, edge_y, edge_z, color="#e85d5d", alpha=0.55, linewidth=1.0)


def _plot_reference_path(ax: plt.Axes, start: np.ndarray, goal: np.ndarray) -> None:
    path_x = np.linspace(start[0] + 0.4, goal[0] + 0.4, 25)
    path_y = np.linspace(start[1] + 0.4, goal[1] + 0.4, 25)
    path_z = np.full_like(path_x, 0.4)
    ax.plot(path_x, path_y, path_z, linestyle="--", color="#1f77b4", alpha=0.5, label="Reference Line")


if __name__ == "__main__":
    plot_scene()
