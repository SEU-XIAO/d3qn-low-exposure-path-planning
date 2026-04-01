from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from env.battlefield_env import BattlefieldEnv
from planner.visibility_astar import PathResult, VisibilityAwareAStarPlanner
from train.dqn_agent import DoubleDQNAgent, TrainingConfig
from visualize.plot_scene import draw_3d_scene


def plot_episode(
    checkpoint_name: str = "ddqn_latest.pt",
    save_path: str | None = None,
    scene_seed: int | None = None,
    scenario_mode: str = "fixed",
) -> None:
    dqn_summary, env = _collect_dqn_path(checkpoint_name, scene_seed=scene_seed, scenario_mode=scenario_mode)
    title = _build_single_title(env, dqn_summary)
    _render_single_path_3d(env, dqn_summary, title, save_path)


def plot_comparison(
    checkpoint_name: str = "ddqn_latest.pt",
    save_path: str | None = None,
    scene_seed: int | None = None,
    scenario_mode: str = "fixed",
) -> None:
    dqn_summary, env = _collect_dqn_path(checkpoint_name, scene_seed=scene_seed, scenario_mode=scenario_mode)
    astar_result = VisibilityAwareAStarPlanner(env).plan(
        start=tuple(env.start_position.tolist()),
        goal=tuple(env.goal_position.tolist()),
    )
    title = _build_comparison_title(env, dqn_summary, astar_result)
    _render_comparison_3d(env, dqn_summary, astar_result, title, save_path)


def _collect_dqn_path(
    checkpoint_name: str,
    scene_seed: int | None = None,
    scenario_mode: str = "fixed",
) -> tuple[dict[str, Any], BattlefieldEnv]:
    root_dir = Path(__file__).resolve().parents[1]
    checkpoint_path = root_dir / "artifacts" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {checkpoint_path}")

    env = BattlefieldEnv()
    agent = DoubleDQNAgent(action_dim=len(BattlefieldEnv.ACTIONS), config=TrainingConfig())
    agent.load(str(checkpoint_path))

    env.reset(scene_seed=scene_seed, scenario_mode=scenario_mode)
    path_cells = [tuple(env.agent_position.tolist())]
    done = False
    success = False
    final_visibility = float(env.visibility_map[tuple(env.agent_position)])

    while not done:
        observation = env.get_observation()
        action = agent.select_action(observation, epsilon=0.0, env=env)
        result = env.step(action)
        path_cells.append(tuple(env.agent_position.tolist()))
        done = result.done
        success = bool(result.info["success"])
        final_visibility = float(result.info["visibility"])

    summary = {
        "path": path_cells,
        "success": success,
        "steps": max(0, len(path_cells) - 1),
        "final_position": tuple(env.agent_position.tolist()),
        "goal_position": tuple(env.goal_position.tolist()),
        "final_visibility": final_visibility,
        "path_length": env.total_path_length,
        "hidden_ratio": env.hidden_ratio,
        "visible_path_length": env.visible_path_length,
        "hidden_path_length": env.hidden_path_length,
    }
    print(
        f"[Plot] D3QN success={summary['success']} | steps={summary['steps']:03d} | "
        f"path_length={summary['path_length']:.3f} | hidden_ratio={summary['hidden_ratio']:.3f}"
    )
    return summary, env


def _render_single_path_3d(
    env: BattlefieldEnv,
    dqn_summary: dict[str, Any],
    title: str,
    save_path: str | None,
) -> None:
    fig = plt.figure(figsize=(16, 8.5))
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_top = fig.add_subplot(1, 2, 2)

    draw_3d_scene(ax_3d, env)
    _plot_path_3d(ax_3d, dqn_summary["path"], color="#1f77b4", label="D3QN Path", z_offset=0.22, linestyle="-")
    _annotate_status(ax_3d, dqn_summary, None)
    ax_3d.set_title(title)
    ax_3d.legend(loc="upper right")

    _plot_topdown_scene(ax_top, env, title="Top-Down Visibility View")
    _plot_path_topdown(ax_top, dqn_summary["path"], color="#1f77b4", label="D3QN Path", linestyle="-")
    ax_top.legend(loc="upper right")

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    else:
        plt.show()


def _render_comparison_3d(
    env: BattlefieldEnv,
    dqn_summary: dict[str, Any],
    astar_result: PathResult,
    title: str,
    save_path: str | None,
) -> None:
    fig = plt.figure(figsize=(17, 8.8))
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_top = fig.add_subplot(1, 2, 2)

    draw_3d_scene(ax_3d, env)
    _plot_path_3d(ax_3d, dqn_summary["path"], color="#1f77b4", label="D3QN", z_offset=0.18, linestyle="-")
    _plot_path_3d(ax_3d, astar_result.path, color="#ff9f1c", label="Visibility-A*", z_offset=0.88, linestyle="-")
    _annotate_status(ax_3d, dqn_summary, astar_result)
    ax_3d.set_title(title)
    ax_3d.legend(loc="upper right")

    _plot_topdown_scene(ax_top, env, title="Top-Down Path Comparison")
    _plot_path_topdown(ax_top, dqn_summary["path"], color="#1f77b4", label="D3QN", linestyle="-")
    _plot_path_topdown(ax_top, astar_result.path, color="#ff9f1c", label="Visibility-A*", linestyle="-")
    ax_top.legend(loc="upper right")

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    else:
        plt.show()


def _plot_path_3d(
    ax: plt.Axes,
    path_cells: list[tuple[int, int]],
    color: str,
    label: str,
    z_offset: float,
    linestyle: str,
) -> None:
    xs = [cell[0] + 0.4 for cell in path_cells]
    ys = [cell[1] + 0.4 for cell in path_cells]
    zs = [z_offset for _ in path_cells]

    ax.plot(xs, ys, zs, color=color, linewidth=3.2, linestyle=linestyle, label=label)
    ax.scatter(xs[0], ys[0], zs[0], color=color, s=42, alpha=0.95)
    ax.scatter(xs[-1], ys[-1], zs[-1], color=color, s=52, alpha=0.95)


def _plot_topdown_scene(ax: plt.Axes, env: BattlefieldEnv, title: str) -> None:
    visible = np.ma.masked_where(env.visibility_map <= 0, env.visibility_map)
    edge_values = np.arange(env.grid_size + 1, dtype=np.float32) - 0.5
    ax.pcolormesh(
        edge_values,
        edge_values,
        visible.T,
        cmap="Reds",
        shading="flat",
        alpha=0.18,
        vmin=0.0,
        vmax=1.0,
        zorder=1,
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

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linewidth=0.6)


def _plot_path_topdown(
    ax: plt.Axes,
    path_cells: list[tuple[int, int]],
    color: str,
    label: str,
    linestyle: str,
) -> None:
    xs = [cell[0] for cell in path_cells]
    ys = [cell[1] for cell in path_cells]
    ax.plot(xs, ys, color=color, linewidth=2.6, linestyle=linestyle, label=label, zorder=6)
    ax.scatter(xs[0], ys[0], color=color, s=36, alpha=0.95, zorder=7)
    ax.scatter(xs[-1], ys[-1], color=color, s=50, alpha=0.95, zorder=7)


def _build_title(prefix: str, env: BattlefieldEnv) -> str:
    if env.current_scenario_mode == "random":
        return f"{prefix} | random seed={env.current_scene_seed}"
    return f"{prefix} | fixed scene"


def _build_single_title(env: BattlefieldEnv, dqn_summary: dict[str, Any]) -> str:
    title = _build_title("3D D3QN Episode Path", env)
    return f"{title} | success={dqn_summary['success']} | hidden_ratio={dqn_summary['hidden_ratio']:.3f}"


def _build_comparison_title(env: BattlefieldEnv, dqn_summary: dict[str, Any], astar_result: PathResult) -> str:
    title = _build_title("3D D3QN vs Visibility-A* Path Comparison", env)
    return f"{title} | D3QN={dqn_summary['success']} | Visibility-A*={astar_result.success}"


def _annotate_status(ax: plt.Axes, dqn_summary: dict[str, Any], astar_result: PathResult | None) -> None:
    lines = [
        f"D3QN success={dqn_summary['success']}",
        f"D3QN final={dqn_summary['final_position']}",
        f"Goal={dqn_summary['goal_position']}",
        f"D3QN path_length={dqn_summary['path_length']:.3f}",
        f"D3QN hidden_ratio={dqn_summary['hidden_ratio']:.3f}",
        f"D3QN final_visible={dqn_summary['final_visibility']:.0f}",
    ]

    if astar_result is not None:
        lines.append(f"Visibility-A* success={astar_result.success}")
        lines.append(f"Visibility-A* steps={astar_result.steps}")
        lines.append(f"Visibility-A* hidden_ratio={astar_result.hidden_ratio:.3f}")
        lines.append(f"Visibility-A* final={astar_result.path[-1]}")
        if not astar_result.success:
            lines.append("Visibility-A* failed to find a full path")

    ax.text2D(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
    )


if __name__ == "__main__":
    plot_comparison(
        checkpoint_name="ddqn_best.pt",
        scene_seed=1078,
        scenario_mode="random",
    )
