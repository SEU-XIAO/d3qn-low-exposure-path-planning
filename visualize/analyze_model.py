from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from config import EnvConfig
from env.battlefield_env import BattlefieldEnv
from models.actor_critic_cnn import ActorCriticCNN, infer_model_spec


def _pick_scene(data: dict[str, np.ndarray], scene_idx: int | None) -> int:
    if scene_idx is not None:
        return int(scene_idx)

    bfs_lens = data["bfs_lengths"]
    for i in range(min(2000, len(bfs_lens))):
        n_obs = int((data["tags"][i] != 0).sum())
        if 10 <= n_obs <= 200 and 15 <= bfs_lens[i] <= 80:
            return i
    return min(42, len(bfs_lens) - 1)


def _build_env(data: dict[str, np.ndarray], idx: int, model_channels: int) -> BattlefieldEnv:
    grid_size = int(data["heights"].shape[1])
    cfg = EnvConfig(
        grid_size=grid_size,
        local_map_size=grid_size,
        scenario_mode="fixed",
        planner_guide_channel=model_channels > 10,
    )
    env = BattlefieldEnv(cfg)
    env.full_terrain = None
    env.full_visibility_maps = []
    env.enemy_pool = []
    env.current_progress_weight = cfg.progress_weight

    env.height_map = data["heights"][idx].copy()
    env.window_tag_map = data["tags"][idx].copy()
    env.start_position = data["starts"][idx].copy()
    env.goal_position = data["goals"][idx].copy()
    if "visibility" in data:
        env.visibility_map = data["visibility"][idx].copy().astype(np.float32)
        env.cover_map = 1.0 - env.visibility_map
    else:
        env.visibility_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        env.cover_map = np.ones((grid_size, grid_size), dtype=np.float32)
    env.occupancy_map = env.height_map.astype(np.float32) / max(1.0, float(env.height_levels))
    env.window_offset = (0, 0)
    env.enemy_position = np.array([-1, -1, 0], dtype=np.float32)
    env.current_scenario_mode = "full_map"
    env.agent_position = env.start_position.copy()
    env.steps = 0
    env.consecutive_collisions = 0
    env.total_collisions = 0
    return env


def _run_episode(policy: ActorCriticCNN, env: BattlefieldEnv, device: torch.device, max_steps: int) -> dict:
    positions = [tuple(map(int, env.agent_position.tolist()))]
    actions_taken: list[int] = []
    rewards: list[float] = []
    values: list[float] = []
    info: dict = {"result": "unknown", "collisions": 0}

    done = False
    while not done and len(positions) <= max_steps:
        obs_t = torch.from_numpy(env._get_observation()).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(env.get_action_mask()).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, value, _, _ = policy(obs_t, mask_t, deterministic=True)
        _, reward, done, info = env.step(int(action.item()))
        positions.append(tuple(map(int, env.agent_position.tolist())))
        actions_taken.append(int(action.item()))
        rewards.append(float(reward))
        values.append(float(value.item()))

    dists = [
        float(np.linalg.norm(np.array(p, dtype=np.float32) - env.goal_position.astype(np.float32)))
        for p in positions
    ]
    return {
        "positions": positions,
        "actions_taken": actions_taken,
        "rewards": rewards,
        "values": values,
        "distances": dists,
        "result": info.get("result", "unknown"),
        "collisions": int(info.get("collisions", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick checkpoint diagnostic on a local pool scene")
    parser.add_argument("--pool", type=str, default="artifacts/window_pool_10.npz")
    parser.add_argument("--model", type=str, default="checkpoints_local_pool/policy_best.pt")
    parser.add_argument("--scene-idx", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=250)
    args = parser.parse_args()

    data_np = np.load(Path(args.pool))
    data = {k: data_np[k] for k in data_np.files}

    device = torch.device("cpu")
    state_dict = torch.load(Path(args.model), map_location=device, weights_only=True)
    model_spec = infer_model_spec(state_dict)
    policy = ActorCriticCNN.from_state_dict(state_dict).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()

    idx = _pick_scene(data, args.scene_idx)
    env = _build_env(data, idx, model_channels=int(model_spec["in_channels"]))
    bfs_path = env.compute_bfs_path()
    episode = _run_episode(policy, env, device, max_steps=args.max_steps)

    print(f"scene_idx={idx}")
    print(f"grid={env.grid_size} obs_channels={model_spec['in_channels']} backbone={model_spec['backbone_name']}")
    print(f"start={tuple(map(int, env.start_position.tolist()))} goal={tuple(map(int, env.goal_position.tolist()))}")
    print(f"bfs_len={0 if bfs_path is None else len(bfs_path)}")
    print(
        f"result={episode['result']} steps={len(episode['positions']) - 1} "
        f"collisions={episode['collisions']} final_dist={episode['distances'][-1]:.2f}"
    )
    print(f"reward_sum={sum(episode['rewards']):.2f}")
    print(f"distance: {episode['distances'][0]:.2f} -> {episode['distances'][-1]:.2f}")
    print(f"action_hist={dict(Counter(episode['actions_taken']))}")


if __name__ == "__main__":
    main()
