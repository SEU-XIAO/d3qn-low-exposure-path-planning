from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config import EnvConfig
from env.vectorized_env import VectorizedEnv
from experiment_config import add_config_args, parse_args_with_config
from models.actor_critic_cnn import ActorCriticCNN, infer_model_spec


def _split_indices(n_scene: int, val_ratio: float, seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(n_scene, dtype=np.int32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(all_indices)
    n_val = max(1, int(len(perm) * val_ratio))
    val_indices = np.sort(perm[:n_val]).astype(np.int32)
    train_indices = np.sort(perm[n_val:]).astype(np.int32)
    return train_indices, val_indices


def _difficulty_bucket(val_bfs: np.ndarray, bfs_len: int) -> str:
    q1 = float(np.quantile(val_bfs, 0.33))
    q2 = float(np.quantile(val_bfs, 0.66))
    if bfs_len <= q1:
        return "easy"
    if bfs_len <= q2:
        return "mid"
    return "hard"


def _run_episode(policy: ActorCriticCNN, env, device: torch.device) -> dict:
    obs = env._get_observation()
    done = False
    ep_steps = 0
    ep_exposed = 0
    traj: list[list[int]] = [list(map(int, env.agent_position.tolist()))]

    info: dict = {"result": "unknown", "collisions": 0}
    while not done:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(env.get_action_mask()).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _, _ = policy(obs_t, mask_t, deterministic=True)

        obs, _reward, done, info = env.step(int(action.item()))
        ep_steps += 1
        pos = tuple(env.agent_position.tolist())
        traj.append([int(pos[0]), int(pos[1])])
        if env.visibility_map[pos] > 0.5:
            ep_exposed += 1

    rem_dist = float(
        np.linalg.norm(env.agent_position.astype(np.float32) - env.goal_position.astype(np.float32))
    )
    return {
        "result": info.get("result", "unknown"),
        "success": info.get("result") == "success",
        "steps": ep_steps,
        "collisions": int(info.get("collisions", 0)),
        "exposure": ep_exposed / max(1, ep_steps),
        "remaining_dist": rem_dist,
        "trajectory": traj,
    }


def _render_ascii_map(env, trajectory: list[list[int]]) -> str:
    g = int(env.grid_size)
    grid = np.full((g, g), ".", dtype="<U1")
    tags = env.window_tag_map
    if tags is not None:
        grid[tags == 1] = "#"
        grid[tags == 2] = "T"

    for x, y in trajectory:
        if 0 <= x < g and 0 <= y < g and grid[x, y] == ".":
            grid[x, y] = "*"

    sx, sy = map(int, env.start_position.tolist())
    gx, gy = map(int, env.goal_position.tolist())
    ax, ay = map(int, env.agent_position.tolist())
    grid[sx, sy] = "S"
    grid[gx, gy] = "G"
    grid[ax, ay] = "A"

    lines = []
    for x in range(g):
        lines.append("".join(grid[x, y] for y in range(g)))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model and export failure cases on fixed val split", allow_abbrev=False)
    add_config_args(parser, default_section="failure_case_report")
    parser.add_argument("--pool", type=str, default="artifacts/window_pool_10.npz")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="analysis/failure_report")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=70)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means evaluate full val set")
    parser.add_argument("--save-fail-maps", type=int, default=40, help="number of failed cases to render")
    parser.add_argument("--planner-guide-sigma", type=float, default=1.4)
    parser.add_argument("--planner-w-len", type=float, default=1.0)
    parser.add_argument("--planner-w-vis", type=float, default=2.5)
    parser.add_argument("--planner-w-slope", type=float, default=0.8)
    parser.add_argument("--planner-w-turn", type=float, default=0.15)
    args = parse_args_with_config(parser, default_section="failure_case_report")

    data = np.load(Path(args.pool))
    pool = {
        "heights": data["heights"],
        "tags": data["tags"],
        "starts": data["starts"],
        "goals": data["goals"],
        "bfs_lengths": data["bfs_lengths"],
    }
    if "visibility" in data:
        pool["visibility"] = data["visibility"]

    n_scene = len(pool["starts"])
    _train_indices, val_indices = _split_indices(n_scene=n_scene, val_ratio=args.val_ratio, seed=2026)
    val_bfs = pool["bfs_lengths"][val_indices]
    if args.limit > 0:
        val_indices = val_indices[: args.limit]
        val_bfs = val_bfs[: args.limit]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state_dict = torch.load(Path(args.model), map_location=device)
    model_spec = infer_model_spec(state_dict)
    expected_channels = int(model_spec["in_channels"])

    grid_size = int(pool["heights"].shape[1])
    env_config = replace(
        EnvConfig(),
        grid_size=grid_size,
        local_map_size=grid_size,
        max_steps=args.max_steps,
        planner_guide_channel=expected_channels > 10,
        planner_guide_sigma=args.planner_guide_sigma,
        planner_w_len=args.planner_w_len,
        planner_w_vis=args.planner_w_vis,
        planner_w_slope=args.planner_w_slope,
        planner_w_turn=args.planner_w_turn,
    )
    vec_env = VectorizedEnv(env_config, pool, num_envs=1, seed=7, allowed_indices=val_indices)
    env = vec_env.envs[0]

    obs_channels = int(vec_env.get_observations().shape[1])
    if obs_channels != expected_channels:
        raise RuntimeError(f"observation channels mismatch: env={obs_channels}, ckpt={expected_channels}")
    policy = ActorCriticCNN.from_state_dict(state_dict).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()

    out_dir = Path(args.output_dir)
    fail_dir = out_dir / "fail_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    fail_saved = 0

    for rank, scene_idx in enumerate(val_indices.tolist()):
        vec_env.reset_env_to_index(0, int(scene_idx))
        episode = _run_episode(policy, env, device)
        bfs_len = int(pool["bfs_lengths"][scene_idx])
        bucket = _difficulty_bucket(val_bfs, bfs_len)
        row = {
            "rank": rank,
            "scene_idx": int(scene_idx),
            "result": episode["result"],
            "success": int(episode["success"]),
            "steps": int(episode["steps"]),
            "collisions": int(episode["collisions"]),
            "exposure": float(episode["exposure"]),
            "remaining_dist": float(episode["remaining_dist"]),
            "bfs_len": bfs_len,
            "bucket": bucket,
            "start_x": int(env.start_position[0]),
            "start_y": int(env.start_position[1]),
            "goal_x": int(env.goal_position[0]),
            "goal_y": int(env.goal_position[1]),
            "trajectory": episode["trajectory"],
        }
        rows.append(row)

        if (not episode["success"]) and fail_saved < args.save_fail_maps:
            txt = _render_ascii_map(env, episode["trajectory"])
            meta = {
                "scene_idx": int(scene_idx),
                "result": episode["result"],
                "steps": int(episode["steps"]),
                "remaining_dist": float(episode["remaining_dist"]),
                "bfs_len": bfs_len,
                "bucket": bucket,
                "start": [int(env.start_position[0]), int(env.start_position[1])],
                "goal": [int(env.goal_position[0]), int(env.goal_position[1])],
            }
            with open(fail_dir / f"fail_{fail_saved:03d}_scene_{scene_idx}.txt", "w", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False, indent=2))
                f.write("\n\n")
                f.write(txt)
                f.write("\n\ntrajectory:\n")
                f.write(str(episode["trajectory"]))
                f.write("\n")
            fail_saved += 1

    n = len(rows)
    succ = sum(r["success"] for r in rows)
    timeout = sum(1 for r in rows if r["result"] == "timeout")
    stuck = sum(1 for r in rows if r["result"] == "stuck")
    avg_steps = float(np.mean([r["steps"] for r in rows])) if n else 0.0
    avg_exposure = float(np.mean([r["exposure"] for r in rows])) if n else 0.0

    summary = {
        "n_eval": n,
        "success_rate": succ / max(1, n),
        "timeout_rate": timeout / max(1, n),
        "stuck_rate": stuck / max(1, n),
        "avg_steps": avg_steps,
        "avg_exposure": avg_exposure,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "cases.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_dir / "cases.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "scene_idx",
                "result",
                "success",
                "steps",
                "collisions",
                "exposure",
                "remaining_dist",
                "bfs_len",
                "bucket",
                "start_x",
                "start_y",
                "goal_x",
                "goal_y",
            ],
        )
        writer.writeheader()
        for r in rows:
            out_row = {k: r[k] for k in writer.fieldnames}
            writer.writerow(out_row)

    print("Evaluation done")
    print(f"summary: {summary}")
    print(f"output dir: {out_dir}")
    print(f"failed maps saved: {fail_saved}")


if __name__ == "__main__":
    main()
