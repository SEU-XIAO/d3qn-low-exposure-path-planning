from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from env.battlefield_env import BattlefieldEnv
from planner.visibility_astar import PathResult, VisibilityAwareAStarPlanner
from planner.weighted_astar import ScalarizedVisibilityAStarPlanner
from train.dqn_agent import DoubleDQNAgent, TrainingConfig
from visualize.plot_scene import draw_3d_scene, _compute_fov_masks


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
    t0 = time.perf_counter()
    print("[Plot] Stage 1/4: collecting D3QN path...")
    dqn_summary, env = _collect_dqn_path(checkpoint_name, scene_seed=scene_seed, scenario_mode=scenario_mode)
    print(f"[Plot] Stage 1/4 done in {time.perf_counter() - t0:.2f}s")

    print("[Plot] Stage 2/4: running scalarized planner J=L+lambda*V...")
    t_plan = time.perf_counter()
    try:
        scalar_result = ScalarizedVisibilityAStarPlanner(env, lambda_visibility=6.0).plan(
            start=tuple(env.start_position.tolist()),
            goal=tuple(env.goal_position.tolist()),
        )
        print(
            f"[Plot] Stage 2/4 done in {time.perf_counter() - t_plan:.2f}s | "
            f"success={scalar_result.success} steps={scalar_result.steps}"
        )
    except Exception as exc:
        start = tuple(env.start_position.tolist())
        print(f"[Plot Warning] Scalarized planning failed, fallback to D3QN-only rendering: {exc}")
        scalar_result = PathResult(
            path=[start],
            total_cost=float("inf"),
            path_length=0.0,
            visible_path_length=0.0,
            hidden_path_length=0.0,
            hidden_ratio=1.0,
            steps=0,
            success=False,
        )
    title = _build_comparison_title(env, dqn_summary, scalar_result)
    print("[Plot] Stage 3/4: rendering figure...")
    _render_comparison_3d(env, dqn_summary, scalar_result, title, save_path)
    print(f"[Plot] Stage 4/4 done | total elapsed={time.perf_counter() - t0:.2f}s")


def _collect_dqn_path(
    checkpoint_name: str,
    scene_seed: int | None = None,
    scenario_mode: str = "fixed",
) -> tuple[dict[str, Any], BattlefieldEnv]:
    root_dir = Path(__file__).resolve().parents[1]
    checkpoint_path = root_dir / "artifacts/v2" / checkpoint_name
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
        action = agent.select_action_masked(observation, env=env)
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
    _annotate_status(ax_3d, env, dqn_summary, None)
    ax_3d.set_title(title)

    _plot_topdown_scene(ax_top, env, title="Top-Down Visibility View")
    _plot_path_topdown(ax_top, dqn_summary["path"], color="#1f77b4", label="D3QN Path", linestyle="-")

    fig.subplots_adjust(right=0.82)
    _add_shared_legend(fig, [ax_3d, ax_top])

    _finalize_figure(fig, save_path, default_name="episode_plot.png")


def _render_comparison_3d(
    env: BattlefieldEnv,
    dqn_summary: dict[str, Any],
    scalar_result: PathResult,
    title: str,
    save_path: str | None,
) -> None:
    fig = plt.figure(figsize=(14.8, 7.6))
    ax_left = fig.add_subplot(1, 2, 1)
    ax_top = fig.add_subplot(1, 2, 2)

    _plot_topdown_scene(ax_left, env, title="")
    _plot_path_topdown(ax_left, scalar_result.path, color="#ff9f1c", label="J(p) A*", linestyle="-")

    _plot_topdown_scene(ax_top, env, title="Top-Down Path Comparison")
    _plot_path_topdown(ax_top, dqn_summary["path"], color="#1f77b4", label="D3QN", linestyle="-")

    fig.suptitle(title, x=0.5, y=0.98)
    fig.subplots_adjust(left=0.04, right=0.80, bottom=0.06, top=0.92, wspace=0.06)
    status_lines = _build_status_lines(env, dqn_summary, scalar_result)
    fig.text(
        0.82,
        0.89,
        "\n".join(status_lines),
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc"},
    )
    _add_shared_legend(fig, [ax_left, ax_top], bbox_to_anchor=(0.82, 0.43))

    _finalize_figure(fig, save_path, default_name="comparison_plot.png")


def _finalize_figure(fig: plt.Figure, save_path: str | None, default_name: str) -> None:
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[Plot] saved to: {save_path}")
        plt.close(fig)
        return

    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_dir = Path(__file__).resolve().parents[1] / "artifacts" / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / default_name
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"[Plot] backend={plt.get_backend()} (non-interactive), saved to: {output_path}")
        plt.close(fig)
        return

    print(f"[Plot] backend={plt.get_backend()} interactive window opening (close window to continue)...")
    plt.show(block=True)
    plt.close(fig)


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
    ax.scatter(enemy[0], enemy[1], color="red", s=120, label="Enemy Lookout", zorder=5)
    heading_scale = 2.2
    ax.arrow(
        float(enemy[0]),
        float(enemy[1]),
        float(env.enemy_forward[0]) * heading_scale,
        float(env.enemy_forward[1]) * heading_scale,
        width=0.08,
        head_width=0.55,
        head_length=0.65,
        length_includes_head=True,
        color="#d32f2f",
        alpha=0.95,
        zorder=6,
    )
    ax.text(
        float(enemy[0]) + 0.5,
        float(enemy[1]) + 0.6,
        f"theta={env.enemy_heading_deg:.1f}deg\nscore={env.enemy_pose_score:.0f}\n{env.enemy_pose_source}",
        fontsize=8.5,
        color="#7a1f1f",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": "#ddbbbb"},
        zorder=7,
    )
    ax.scatter([], [], marker="s", s=80, color="#ff8a80", alpha=0.55, label="FOV Visible")
    ax.scatter([], [], marker="s", s=80, color="#9aa0a6", alpha=0.5, label="FOV Occluded")
    ax.scatter([], [], marker="", label="Enemy heading shown by arrow")

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
    return (
        f"{title} | enemy=({int(env.enemy_position[0])},{int(env.enemy_position[1])}) "
        f"theta={env.enemy_heading_deg:.1f}deg | success={dqn_summary['success']} "
        f"| hidden_ratio={dqn_summary['hidden_ratio']:.3f}"
    )


def _build_comparison_title(env: BattlefieldEnv, dqn_summary: dict[str, Any], scalar_result: PathResult) -> str:
    title = _build_title("J(p)=L+lambda*V A* vs D3QN", env)
    return (
        f"{title} | enemy=({int(env.enemy_position[0])},{int(env.enemy_position[1])}) "
        f"theta={env.enemy_heading_deg:.1f}deg | J(p) A*={scalar_result.success} | D3QN={dqn_summary['success']}"
    )


def _annotate_status(
    ax: plt.Axes,
    env: BattlefieldEnv,
    dqn_summary: dict[str, Any],
    scalar_result: PathResult | None,
) -> None:
    lines = _build_status_lines(env, dqn_summary, scalar_result)

    ax.text(
        0.98,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
    )


def _build_status_lines(
    env: BattlefieldEnv,
    dqn_summary: dict[str, Any],
    scalar_result: PathResult | None,
) -> list[str]:
    lines = [
        f"Enemy lookout=({int(env.enemy_position[0])},{int(env.enemy_position[1])})",
        f"Enemy theta={env.enemy_heading_deg:.1f}deg ({env.enemy_pose_source})",
        f"Enemy score={env.enemy_pose_score:.0f}",
        f"D3QN success={dqn_summary['success']}",
        f"D3QN final={dqn_summary['final_position']}",
        f"Goal={dqn_summary['goal_position']}",
        f"D3QN path_length={dqn_summary['path_length']:.3f}",
        f"D3QN hidden_ratio={dqn_summary['hidden_ratio']:.3f}",
        f"D3QN final_visible={dqn_summary['final_visibility']:.0f}",
    ]

    if scalar_result is not None:
        lines.append(f"J(p) A* success={scalar_result.success}")
        lines.append(f"J(p) A* steps={scalar_result.steps}")
        lines.append(f"J(p) A* hidden_ratio={scalar_result.hidden_ratio:.3f}")
        lines.append(f"J(p) A* final={scalar_result.path[-1]}")
        if not scalar_result.success:
            lines.append("J(p) A* failed to find a full path")

    return lines


def _add_shared_legend(fig: plt.Figure, axes: list[plt.Axes], bbox_to_anchor: tuple[float, float] = (0.93, 0.5)) -> None:
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
            bbox_to_anchor=bbox_to_anchor,
            frameon=True,
        )
 

if __name__ == "__main__":
    plot_comparison(
        checkpoint_name="ddqn_best.pt",
        scene_seed=7201,
        scenario_mode="random",
    )
