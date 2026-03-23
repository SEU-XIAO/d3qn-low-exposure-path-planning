from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from env.battlefield_env import BattlefieldEnv
from planner.risk_astar import RiskAStarPlanner
from train.dqn_agent import DoubleDQNAgent, TrainingConfig
from visualize.plot_scene import draw_3d_scene


def plot_episode(checkpoint_name: str = "ddqn_latest.pt", save_path: str | None = None) -> None:
    path_cells, env = _collect_dqn_path(checkpoint_name)
    _render_single_path_3d(env, path_cells, "3D DQN Episode Path", save_path)


def plot_comparison(checkpoint_name: str = "ddqn_latest.pt", save_path: str | None = None) -> None:
    dqn_path, env = _collect_dqn_path(checkpoint_name)
    astar_path = RiskAStarPlanner(env).plan().path
    _render_comparison_3d(env, dqn_path, astar_path, save_path)


def _collect_dqn_path(checkpoint_name: str) -> tuple[list[tuple[int, int]], BattlefieldEnv]:
    root_dir = Path(__file__).resolve().parents[1]
    checkpoint_path = root_dir / "artifacts" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {checkpoint_path}")

    env = BattlefieldEnv()
    agent = DoubleDQNAgent(action_dim=len(BattlefieldEnv.ACTIONS), config=TrainingConfig())
    agent.load(str(checkpoint_path))

    env.reset()
    path_cells = [tuple(env.agent_position.tolist())]
    done = False

    while not done:
        observation = env.get_observation()
        action = agent.select_action(observation, epsilon=0.0)
        result = env.step(action)
        path_cells.append(tuple(env.agent_position.tolist()))
        done = result.done

    return path_cells, env


def _render_single_path_3d(
    env: BattlefieldEnv,
    path_cells: list[tuple[int, int]],
    title: str,
    save_path: str | None,
) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    draw_3d_scene(ax, env)
    _plot_path_3d(ax, path_cells, color="#1f77b4", label="DQN Path", z_offset=0.18, linestyle="-")
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    else:
        plt.show()


def _render_comparison_3d(
    env: BattlefieldEnv,
    dqn_path: list[tuple[int, int]],
    astar_path: list[tuple[int, int]],
    save_path: str | None,
) -> None:
    fig = plt.figure(figsize=(12.5, 9.5))
    ax = fig.add_subplot(111, projection="3d")
    draw_3d_scene(ax, env)

    _plot_path_3d(ax, dqn_path, color="#1f77b4", label="Double DQN", z_offset=0.22, linestyle="-")
    _plot_path_3d(ax, astar_path, color="#ff9f1c", label="Risk-A*", z_offset=0.42, linestyle="--")
    ax.set_title("3D DQN vs Risk-A* Path Comparison")

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

    ax.plot(xs, ys, zs, color=color, linewidth=2.5, linestyle=linestyle, label=label)
    ax.scatter(xs[0], ys[0], zs[0], color=color, s=24, alpha=0.9)
    ax.scatter(xs[-1], ys[-1], zs[-1], color=color, s=30, alpha=0.9)


if __name__ == "__main__":
    plot_comparison()
