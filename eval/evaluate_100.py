from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from env.battlefield_env import BattlefieldEnv
from train.dqn_agent import DoubleDQNAgent, TrainingConfig


def evaluate(
    checkpoint_name: str = "ddqn_best.pt",
    num_episodes: int = 1000,
    seed_start: int = 7200,
    output_path: str | None = None,
    scenario_mode: str = "random",
) -> Path:
    root_dir = Path(__file__).resolve().parents[1]
    checkpoint_path = root_dir / "artifacts" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {checkpoint_path}")

    env = BattlefieldEnv()
    agent = DoubleDQNAgent(action_dim=len(BattlefieldEnv.ACTIONS), config=TrainingConfig())
    agent.load(str(checkpoint_path))

    rows: list[dict[str, Any]] = []
    seeds = [seed_start + i for i in range(num_episodes)]

    for idx, scene_seed in enumerate(seeds, start=1):
        observation = env.reset(scene_seed=scene_seed, scenario_mode=scenario_mode)
        done = False
        total_reward = 0.0
        step_count = 0
        success = False
        collisions = 0

        while not done:
            action = agent.select_action_masked(observation, env=env)
            result = env.step(action)
            observation = result.observation
            total_reward += float(result.reward)
            done = result.done
            step_count += 1
            if bool(result.info["collision"]):
                collisions += 1
            success = bool(result.info["success"])

        rows.append(
            {
                "episode": idx,
                "scene_seed": scene_seed,
                "success": success,
                "steps": step_count,
                "total_reward": total_reward,
                "path_length": env.total_path_length,
                "hidden_ratio": env.hidden_ratio,
                "visible_path_length": env.visible_path_length,
                "hidden_path_length": env.hidden_path_length,
                "collisions": collisions,
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "success_rate": float(df["success"].mean()),
        "avg_reward": float(df["total_reward"].mean()),
        "avg_path_length": float(df["path_length"].mean()),
        "avg_hidden_ratio": float(df["hidden_ratio"].mean()),
        "avg_visible_path_length": float(df["visible_path_length"].mean()),
        "avg_hidden_path_length": float(df["hidden_path_length"].mean()),
        "avg_steps": float(df["steps"].mean()),
        "avg_collisions": float(df["collisions"].mean()),
    }

    output_dir = root_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = str(output_dir / f"eval_100_{timestamp}.xlsx")

    # Append a summary row to the episodes sheet for quick lookup.
    summary_row = {
        "episode": "summary",
        "scene_seed": "",
        "success": summary["success_rate"],
        "steps": summary["avg_steps"],
        "total_reward": summary["avg_reward"],
        "path_length": summary["avg_path_length"],
        "hidden_ratio": summary["avg_hidden_ratio"],
        "visible_path_length": summary["avg_visible_path_length"],
        "hidden_path_length": summary["avg_hidden_path_length"],
        "collisions": summary["avg_collisions"],
    }
    df_with_summary = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_with_summary.to_excel(writer, index=False, sheet_name="episodes")
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="summary")

    return Path(output_path)


if __name__ == "__main__":
    output = evaluate()
    print(f"总成功率: {pd.read_excel(output, sheet_name='summary')['success_rate'][0]:.3f}")
    print(f"已保存评估结果: {output}")
