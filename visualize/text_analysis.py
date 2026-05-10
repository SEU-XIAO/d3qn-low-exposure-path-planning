from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from config import EnvConfig
from env.battlefield_env import BattlefieldEnv
from models.actor_critic_cnn import ActorCriticCNN, infer_model_spec


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


def _choose_scene(data: dict[str, np.ndarray]) -> int:
    bfs_lens = data["bfs_lengths"]
    for i in range(min(2000, len(bfs_lens))):
        obs = int((data["tags"][i] != 0).sum())
        if 10 <= obs <= 200 and 15 <= bfs_lens[i] <= 80:
            return i
    return min(42, len(bfs_lens) - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Text-only checkpoint diagnostic")
    parser.add_argument("--pool", type=str, default="artifacts/window_pool_10.npz")
    parser.add_argument("--model", type=str, default="checkpoints_local_pool/policy_best.pt")
    parser.add_argument("--max-steps", type=int, default=250)
    args = parser.parse_args()

    data_np = np.load(Path(args.pool))
    data = {k: data_np[k] for k in data_np.files}
    scene_idx = _choose_scene(data)

    device = torch.device("cpu")
    state_dict = torch.load(Path(args.model), map_location=device, weights_only=True)
    model_spec = infer_model_spec(state_dict)
    policy = ActorCriticCNN.from_state_dict(state_dict).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()

    env = _build_env(data, scene_idx, model_channels=int(model_spec["in_channels"]))
    bfs_path = env.compute_bfs_path()
    action_hist: Counter[int] = Counter()
    rewards: list[float] = []
    info: dict = {"result": "unknown", "collisions": 0}

    print("=== Pool Stats ===")
    bfs_lens = data["bfs_lengths"]
    print(f"scenes={len(bfs_lens)} bfs[min={bfs_lens.min()}, max={bfs_lens.max()}, mean={bfs_lens.mean():.1f}]")
    obs_counts = (data["tags"] != 0).sum(axis=(1, 2))
    print(f"obstacles[min={obs_counts.min()}, max={obs_counts.max()}, mean={obs_counts.mean():.1f}]")
    print()

    print("=== Scene ===")
    print(f"scene_idx={scene_idx} bfs_len={0 if bfs_path is None else len(bfs_path)}")
    print(f"start={tuple(map(int, env.start_position.tolist()))} goal={tuple(map(int, env.goal_position.tolist()))}")
    print(
        f"model_channels={model_spec['in_channels']} "
        f"feature_dim={model_spec['feature_dim']} backbone={model_spec['backbone_name']}"
    )
    print()

    done = False
    steps = 0
    while not done and steps < args.max_steps:
        obs_t = torch.from_numpy(env._get_observation()).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(env.get_action_mask()).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, value, _, _ = policy(obs_t, mask_t, deterministic=True)
        _, reward, done, info = env.step(int(action.item()))
        action_hist[int(action.item())] += 1
        rewards.append(float(reward))
        steps += 1
        dist = float(np.linalg.norm(env.agent_position.astype(np.float32) - env.goal_position.astype(np.float32)))
        print(f"step={steps:03d} action={int(action.item())} value={float(value.item()):7.3f} dist={dist:6.2f}")

    print()
    print("=== Result ===")
    print(f"result={info.get('result', 'unknown')} steps={steps} collisions={info.get('collisions', 0)}")
    print(f"reward_sum={sum(rewards):.2f}")
    print(f"final_pos={tuple(map(int, env.agent_position.tolist()))}")
    print(f"action_hist={dict(action_hist)}")


if __name__ == "__main__":
    main()
