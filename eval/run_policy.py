from __future__ import annotations

from pathlib import Path
from typing import Any

from env.battlefield_env import BattlefieldEnv
from planner.risk_astar import RiskAStarPlanner
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

    astar_summary = _run_risk_astar(env, scene_seed, scenario_mode)
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
    total_risk = 0.0
    path = [tuple(env.agent_position.tolist())]
    step_count = 0
    success = False

    print(f"加载 D3QN 模型: {checkpoint_path}")
    while not done:
        action = agent.select_action(observation, epsilon=0.0, env=env)
        result = env.step(action)
        observation = result.observation
        total_reward += result.reward
        total_risk += float(result.info["risk"])
        step_count += 1
        path.append(tuple(env.agent_position.tolist()))
        print(
            f"[D3QN] step={step_count:03d} pos={tuple(env.agent_position.tolist())} "
            f"risk={result.info['risk']:.3f} reward={result.reward:.3f}"
        )
        done = result.done
        success = bool(result.info["success"])

    print(f"[D3QN] reward={total_reward:.3f} | cumulative_risk={total_risk:.3f} | success={success}")
    return {
        "name": "D3QN",
        "success": success,
        "steps": step_count,
        "reward": total_reward,
        "cumulative_risk": total_risk,
        "path": path,
    }


def _run_risk_astar(env: BattlefieldEnv, scene_seed: int | None, scenario_mode: str) -> dict[str, Any]:
    env.reset(scene_seed=scene_seed, scenario_mode=scenario_mode)
    planner = RiskAStarPlanner(env)
    result = planner.plan()

    print("[Risk-A*] path:")
    for index, cell in enumerate(result.path):
        print(f"[Risk-A*] step={index:03d} pos={cell} risk={env.risk_map[cell]:.3f}")

    print(
        f"[Risk-A*] total_cost={result.total_cost:.3f} | "
        f"path_length={result.path_length:.3f} | cumulative_risk={result.cumulative_risk:.3f} | "
        f"success={result.success}"
    )
    return {
        "name": "Risk-A*",
        "success": result.success,
        "steps": result.steps,
        "reward": None,
        "path_length": result.path_length,
        "cumulative_risk": result.cumulative_risk,
        "total_cost": result.total_cost,
        "path": result.path,
    }


def _print_comparison(dqn_summary: dict[str, Any] | None, astar_summary: dict[str, Any]) -> None:
    print("\n=== 对比汇总 ===")
    if dqn_summary is not None:
        print(
            f"D3QN     | success={dqn_summary['success']} | steps={dqn_summary['steps']:03d} | "
            f"reward={dqn_summary['reward']:.3f} | cumulative_risk={dqn_summary['cumulative_risk']:.3f}"
        )
    print(
        f"Risk-A*  | success={astar_summary['success']} | steps={astar_summary['steps']:03d} | "
        f"path_len={astar_summary['path_length']:.3f} | cumulative_risk={astar_summary['cumulative_risk']:.3f} | "
        f"total_cost={astar_summary['total_cost']:.3f}"
    )


if __name__ == "__main__":
    run_episode()
