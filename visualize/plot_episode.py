from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EnvConfig, WaypointConfig
from env.battlefield_env import BattlefieldEnv
from planner.visibility_astar import PathResult, VisibilityAwareAStarPlanner
from planner.weighted_astar import ScalarizedVisibilityAStarPlanner
from train.dqn_agent import DoubleDQNAgent, TrainingConfig


def _make_env(scenario_mode: str) -> BattlefieldEnv:
    """创建指定模式的 env，非 full_map 模式跳过地形加载以加速启动。"""
    if scenario_mode in ("random", "fixed"):
        config = EnvConfig(scenario_mode=scenario_mode, full_map_path="")
    else:
        config = EnvConfig()
    return BattlefieldEnv(config=config)


def _find_checkpoint(checkpoint_name: str) -> Path:
    root_dir = Path(__file__).resolve().parents[1]
    candidates = [
        root_dir / "artifacts/v1" / checkpoint_name,
        root_dir / "artifacts" / checkpoint_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到模型文件，尝试过: {candidates}")


# ============================================================
#  单条路径可视化
# ============================================================


def plot_episode(
    checkpoint_name: str = "ddqn_best.pt",
    save_path: str | None = None,
    scene_seed: int | None = None,
    scenario_mode: str = "random",
    use_waypoints: bool = False,
) -> None:
    t0 = time.perf_counter()
    dqn_summary, env = _collect_dqn_path(checkpoint_name, scene_seed=scene_seed, scenario_mode=scenario_mode, use_waypoints=use_waypoints)
    print(f"[Plot] D3QN collection done in {time.perf_counter() - t0:.1f}s")

    title = _make_title("D3QN Episode", env, dqn_summary)
    fig, ax = plt.subplots(figsize=(10.5, 9))
    fig.subplots_adjust(left=0.06, right=0.84, bottom=0.05, top=0.93)
    _draw_scene(ax, env)
    _draw_path(ax, dqn_summary["path"], color="#1f77b4", label="D3QN Path")
    ax.set_title(title)
    _add_legend(fig, ax)

    _finalize_figure(fig, save_path, "episode_plot.png")


# ============================================================
#  D3QN vs J(p) A* 对比可视化
# ============================================================


def plot_comparison(
    checkpoint_name: str = "ddqn_best.pt",
    save_path: str | None = None,
    scene_seed: int | None = None,
    scenario_mode: str = "random",
    use_waypoints: bool = False,
) -> None:
    t0 = time.perf_counter()

    print("[Plot] Stage 1/3: collecting D3QN path...")
    dqn_summary, env = _collect_dqn_path(checkpoint_name, scene_seed=scene_seed, scenario_mode=scenario_mode, use_waypoints=use_waypoints)
    print(f"[Plot] Stage 1/3 done in {time.perf_counter() - t0:.1f}s")

    print("[Plot] Stage 2/3: running J(p)=L+lambda*V A* planner...")
    t_plan = time.perf_counter()
    try:
        scalar_result = ScalarizedVisibilityAStarPlanner(env, lambda_visibility=6.0).plan(
            start=tuple(env.start_position.tolist()),
            goal=tuple(env.goal_position.tolist()),
        )
        print(
            f"[Plot] Stage 2/3 done in {time.perf_counter() - t_plan:.1f}s | "
            f"success={scalar_result.success} steps={scalar_result.steps}"
        )
    except Exception as exc:
        start = tuple(env.start_position.tolist())
        print(f"[Plot Warning] A* planning failed: {exc}")
        scalar_result = PathResult(
            path=[start], total_cost=float("inf"), path_length=0.0,
            visible_path_length=0.0, hidden_path_length=0.0,
            hidden_ratio=1.0, steps=0, success=False,
        )

    print("[Plot] Stage 3/3: rendering...")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))

    _draw_scene(ax_left, env)
    _draw_path(ax_left, scalar_result.path, color="#ff9f1c", label="J(p) A*")
    ax_left.set_title("J(p)=L+lambda*V A*")

    _draw_scene(ax_right, env)
    _draw_path(ax_right, dqn_summary["path"], color="#1f77b4", label="D3QN")
    ax_right.set_title("D3QN")

    title = _make_comparison_title(env, dqn_summary, scalar_result)
    fig.suptitle(title, x=0.5, y=0.98)
    fig.subplots_adjust(left=0.04, right=0.78, bottom=0.06, top=0.92, wspace=0.06)

    _add_shared_legend(fig, [ax_left, ax_right], bbox_to_anchor=(0.80, 0.50))

    _finalize_figure(fig, save_path, "comparison_plot.png")
    print(f"[Plot] total elapsed={time.perf_counter() - t0:.1f}s")


# ============================================================
#  D3QN 路径采集
# ============================================================


def _collect_dqn_path(
    checkpoint_name: str,
    scene_seed: int | None = None,
    scenario_mode: str = "random",
    use_waypoints: bool = False,
) -> tuple[dict[str, Any], BattlefieldEnv]:
    checkpoint_path = _find_checkpoint(checkpoint_name)
    print(f"[Plot] Loading checkpoint: {checkpoint_path}")

    env = _make_env(scenario_mode)
    wp_cfg = WaypointConfig(enabled=True) if use_waypoints else WaypointConfig()
    cfg = TrainingConfig(waypoint=wp_cfg)
    agent = DoubleDQNAgent(action_dim=len(BattlefieldEnv.ACTIONS), config=cfg)
    agent.load(str(checkpoint_path))
    print(f"[Plot] Checkpoint loaded, running episode (scenario={scenario_mode}, seed={scene_seed}, waypoints={use_waypoints})...")

    env.reset(scene_seed=scene_seed, scenario_mode=scenario_mode)
    agent.reset_episode_stats()
    path_cells = [tuple(env.agent_position.tolist())]
    success = False
    final_visibility = float(env.visibility_map[tuple(env.agent_position)])

    t_start = time.perf_counter()
    if use_waypoints:
        wp = cfg.waypoint
        waypoints = _generate_waypoints_viz(env, wp.interval)
        max_segment_steps = int(wp.interval * wp.max_segment_multiplier)
        for wp_idx, waypoint in enumerate(waypoints):
            is_final = (wp_idx == len(waypoints) - 1)
            env.set_subgoal(waypoint, is_final=is_final)
            segment_steps = 0
            while segment_steps < max_segment_steps:
                observation = env.get_observation()
                action = agent.select_action_masked(observation, env=env)
                result = env.step(action)
                path_cells.append(tuple(env.agent_position.tolist()))
                success = bool(result.info["success"])
                final_visibility = float(result.info["visibility"])
                segment_steps += 1
                if result.done or result.info["waypoint_reached"]:
                    break
            if result.done:
                break
    else:
        done = False
        while not done:
            observation = env.get_observation()
            action = agent.select_action_masked(observation, env=env)
            result = env.step(action)
            path_cells.append(tuple(env.agent_position.tolist()))
            done = result.done
            success = bool(result.info["success"])
            final_visibility = float(result.info["visibility"])
    elapsed = time.perf_counter() - t_start

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
        f"[Plot] Episode done in {elapsed:.1f}s | success={success} | "
        f"steps={summary['steps']} | hidden_ratio={summary['hidden_ratio']:.3f}"
    )
    return summary, env


def _generate_waypoints_viz(env: BattlefieldEnv, interval: int) -> list[tuple[int, int]]:
    """可视化用：A* 生成路径并采样航点。"""
    start = tuple(env.agent_position.tolist())
    goal = tuple(env.goal_position.tolist())
    try:
        result = VisibilityAwareAStarPlanner(env, visible_weight=6.0).plan(start=start, goal=goal)
        if result.success and len(result.path) >= 2:
            waypoints: list[tuple[int, int]] = []
            for i in range(interval, len(result.path), interval):
                waypoints.append(result.path[i])
            if not waypoints or waypoints[-1] != result.path[-1]:
                waypoints.append(result.path[-1])
            return waypoints
    except Exception:
        pass
    return [goal]


# ============================================================
#  绘制场景（二值可见性 + 障碍物 + 起终点）
# ============================================================


def _draw_scene(ax: plt.Axes, env: BattlefieldEnv) -> None:
    """绘制 50x50 二值可见性底图 + 起终点。"""
    # 二值可见性：可见=珊瑚红(被敌人看到/危险)，不可见=钢蓝灰(安全/遮挡)
    vis_binary = (env.visibility_map > 0.5).astype(np.float32)
    edge = np.arange(env.grid_size + 1, dtype=np.float32) - 0.5
    cmap = ListedColormap(["#6b8a9e", "#e85d5d"])

    ax.pcolormesh(
        edge, edge, vis_binary.T,
        cmap=cmap, shading="flat", vmin=0, vmax=1, alpha=0.82, zorder=1,
    )

    # 起点、终点
    sx, sy = int(env.start_position[0]), int(env.start_position[1])
    gx, gy = int(env.goal_position[0]), int(env.goal_position[1])
    ax.scatter(sx, sy, color="green", s=100, label="Start", zorder=5)
    ax.scatter(gx, gy, color="blue", s=100, label="Goal", zorder=5)

    # 图例占位（颜色需与 pcolormesh cmap 一致）
    ax.scatter([], [], marker="s", s=80, color="#e85d5d", alpha=0.88, label="FOV Visible")
    ax.scatter([], [], marker="s", s=80, color="#6b8a9e", alpha=0.88, label="FOV Occluded")

    # 网格线与坐标轴
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_aspect("equal")
    grid_edges = np.arange(-0.5, env.grid_size + 0.5, 1.0)
    ax.set_xticks(grid_edges, minor=True)
    ax.set_yticks(grid_edges, minor=True)
    ax.grid(which="minor", color="#000000", alpha=0.50, linewidth=0.40, zorder=4)
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)


# ============================================================
#  路径绘制
# ============================================================


def _draw_path(
    ax: plt.Axes,
    path_cells: list[tuple[int, int]],
    color: str,
    label: str,
) -> None:
    xs = [c[0] for c in path_cells]
    ys = [c[1] for c in path_cells]
    ax.plot(xs, ys, color=color, linewidth=2.8, linestyle="-", label=label, zorder=6)
    ax.scatter(xs[0], ys[0], color=color, s=42, alpha=0.95, zorder=7)
    ax.scatter(xs[-1], ys[-1], color=color, s=56, alpha=0.95, zorder=7)


# ============================================================
#  标题与状态信息
# ============================================================


def _make_title(prefix: str, env: BattlefieldEnv, dqn_summary: dict[str, Any]) -> str:
    title = prefix
    if env.current_scenario_mode == "random":
        title += f" | random seed={env.current_scene_seed}"
    else:
        title += f" | fixed scene"
    return (
        f"{title}"
        f" | success={dqn_summary['success']}"
        f" | steps={dqn_summary['steps']}"
        f" | hidden_ratio={dqn_summary['hidden_ratio']:.3f}"
    )


def _make_comparison_title(
    env: BattlefieldEnv, dqn_summary: dict[str, Any], scalar_result: PathResult,
) -> str:
    title = "J(p)=L+lambda*V A* vs D3QN"
    if env.current_scenario_mode == "random":
        title += f" | seed={env.current_scene_seed}"
    return f"{title} | A*={scalar_result.success} | D3QN={dqn_summary['success']}"


# ============================================================
#  图例与输出
# ============================================================


def _add_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique = []
    seen: set[str] = set()
    for h, lab in zip(handles, labels):
        if lab and lab not in seen:
            seen.add(lab)
            unique.append((h, lab))
    if unique:
        fig.legend(
            [h for h, _ in unique], [lab for _, lab in unique],
            loc="center left", bbox_to_anchor=(0.86, 0.5),
            frameon=True, borderaxespad=0,
        )


def _add_shared_legend(
    fig: plt.Figure, axes: list[plt.Axes], bbox_to_anchor: tuple[float, float] = (0.93, 0.5),
) -> None:
    handles: list = []
    labels: list[str] = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in labels:
                labels.append(label)
                handles.append(handle)
    if handles:
        fig.legend(
            handles, labels,
            loc="center left", bbox_to_anchor=bbox_to_anchor, frameon=True,
        )


def _finalize_figure(fig: plt.Figure, save_path: str | None, default_name: str) -> None:
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[Plot] Saved to: {save_path}")
        plt.close(fig)
        return

    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_dir = Path(__file__).resolve().parents[1] / "artifacts" / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / default_name
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"[Plot] Non-interactive backend, saved to: {output_path}")
        plt.close(fig)
        return

    print("[Plot] Opening interactive window (close to continue)...")
    plt.show(block=True)
    plt.close(fig)


# ============================================================
#  入口
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize D3QN episode paths")
    parser.add_argument("--checkpoint", default="ddqn_best.pt", help="Checkpoint file name")
    parser.add_argument("--seed", type=int, default=7201, help="Scene seed")
    parser.add_argument("--mode", default="random", choices=["random", "fixed", "full_map"])
    parser.add_argument("--comparison", action="store_true", help="Run comparison with J(p) A* planner")
    parser.add_argument("--save", type=str, default=None, help="Save path for the figure")
    parser.add_argument("--use-waypoints", action="store_true", help="Enable waypoint mode for visualization")
    args = parser.parse_args()

    if args.comparison:
        plot_comparison(
            checkpoint_name=args.checkpoint,
            scene_seed=args.seed,
            scenario_mode=args.mode,
            save_path=args.save,
            use_waypoints=args.use_waypoints,
        )
    else:
        plot_episode(
            checkpoint_name=args.checkpoint,
            scene_seed=args.seed,
            scenario_mode=args.mode,
            save_path=args.save,
            use_waypoints=args.use_waypoints,
        )
