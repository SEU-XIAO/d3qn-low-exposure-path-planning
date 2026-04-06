from __future__ import annotations

from pathlib import Path
from typing import Any

from env.battlefield_env import BattlefieldEnv
from planner.visibility_astar import VisibilityAwareAStarPlanner
from train.dqn_agent import DoubleDQNAgent, TrainingConfig


def run_episode(
    checkpoint_name: str = "ddqn_latest.pt",
    scene_seed: int | None = None,
    scenario_mode: str = "fixed",
) -> None:
    root_dir = Path(__file__).resolve().parents[1]
    checkpoint_path = root_dir / "artifacts" / checkpoint_name
    env = BattlefieldEnv()

    print(f"场景模式: {scenario_mode} | scene_seed={scene_seed}")

    dqn_summary: dict[str, Any] | None = None
    if checkpoint_path.exists():
        agent = DoubleDQNAgent(action_dim=len(BattlefieldEnv.ACTIONS), config=TrainingConfig())
        agent.load(str(checkpoint_path))
        dqn_summary = _run_dqn_episode(env, agent, checkpoint_path, scene_seed, scenario_mode)
    else:
        print(f"未找到 D3QN 模型文件，跳过 D3QN 对比: {checkpoint_path}")

    astar_summary = _run_visibility_astar(env, scene_seed, scenario_mode)
    _print_comparison(dqn_summary, astar_summary)


def _run_dqn_episode(
    env: BattlefieldEnv,
    agent: DoubleDQNAgent,
    checkpoint_path: Path,
    scene_seed: int | None,
    scenario_mode: str,
) -> dict[str, Any]:
    observation = env.reset(scene_seed=scene_seed, scenario_mode=scenario_mode)
    done = False
    total_reward = 0.0
    path = [tuple(env.agent_position.tolist())]
    step_count = 0
    success = False

    print(f"加载 D3QN 模型: {checkpoint_path}")
    while not done:
        action = agent.select_action_masked(observation, env=env)
        result = env.step(action)
        observation = result.observation
        total_reward += result.reward
        step_count += 1
        path.append(tuple(env.agent_position.tolist()))
        print(
            f"[D3QN] step={step_count:03d} pos={tuple(env.agent_position.tolist())} "
            f"visible={result.info['visibility']:.0f} hidden_ratio={result.info['hidden_ratio']:.3f} "
            f"reward={result.reward:.3f}"
        )
        done = result.done
        success = bool(result.info["success"])

    print(
        f"[D3QN] reward={total_reward:.3f} | path_length={env.total_path_length:.3f} | "
        f"hidden_ratio={env.hidden_ratio:.3f} | success={success}"
    )
    return {
        "name": "D3QN",
        "success": success,
        "steps": step_count,
        "reward": total_reward,
        "path_length": env.total_path_length,
        "visible_path_length": env.visible_path_length,
        "hidden_path_length": env.hidden_path_length,
        "hidden_ratio": env.hidden_ratio,
        "path": path,
    }


def _run_visibility_astar(env: BattlefieldEnv, scene_seed: int | None, scenario_mode: str) -> dict[str, Any]:
    env.reset(scene_seed=scene_seed, scenario_mode=scenario_mode)
    planner = VisibilityAwareAStarPlanner(env)
    result = planner.plan()

    print("[Visibility-A*] path:")
    for index, cell in enumerate(result.path):
        print(f"[Visibility-A*] step={index:03d} pos={cell} visible={env.visibility_map[cell]:.0f}")

    print(
        f"[Visibility-A*] total_cost={result.total_cost:.3f} | path_length={result.path_length:.3f} | "
        f"hidden_ratio={result.hidden_ratio:.3f} | success={result.success}"
    )
    return {
        "name": "Visibility-A*",
        "success": result.success,
        "steps": result.steps,
        "reward": None,
        "path_length": result.path_length,
        "visible_path_length": result.visible_path_length,
        "hidden_path_length": result.hidden_path_length,
        "hidden_ratio": result.hidden_ratio,
        "total_cost": result.total_cost,
        "path": result.path,
    }


def _print_comparison(dqn_summary: dict[str, Any] | None, astar_summary: dict[str, Any]) -> None:
    print("\n=== 对比汇总 ===")
    if dqn_summary is not None:
        print(
            f"D3QN         | success={dqn_summary['success']} | steps={dqn_summary['steps']:03d} | "
            f"reward={dqn_summary['reward']:.3f} | path_len={dqn_summary['path_length']:.3f} | "
            f"hidden_ratio={dqn_summary['hidden_ratio']:.3f}"
        )
    print(
        f"Visibility-A* | success={astar_summary['success']} | steps={astar_summary['steps']:03d} | "
        f"path_len={astar_summary['path_length']:.3f} | hidden_ratio={astar_summary['hidden_ratio']:.3f} | "
        f"total_cost={astar_summary['total_cost']:.3f}"
    )


if __name__ == "__main__":
    run_episode(
        checkpoint_name="ddqn_best.pt",
        scene_seed=1031,
        scenario_mode="random",
    )
