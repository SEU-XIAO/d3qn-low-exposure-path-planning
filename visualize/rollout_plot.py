from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from config import EnvConfig
from env.battlefield_env import BattlefieldEnv
from env.obs import add_obs_args
from experiment_config import add_config_args, parse_args_with_config
from models.actor_critic_cnn import ActorCriticCNN, infer_model_spec
from planner import StealthCostConfig, WaypointConfig, extract_waypoints, plan_stealth_path


MOVE_TO_ACTION = {
    (-1, 0): 0,
    (1, 0): 1,
    (0, -1): 2,
    (0, 1): 3,
    (-1, -1): 4,
    (-1, 1): 5,
    (1, -1): 6,
    (1, 1): 7,
}


def _to_tuple(pos: np.ndarray) -> tuple[int, int]:
    return int(pos[0]), int(pos[1])


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def _make_waypoints(
    env: BattlefieldEnv,
    start: tuple[int, int],
    final_goal: tuple[int, int],
    global_cfg: StealthCostConfig,
    wp_cfg: WaypointConfig,
    subgoal_max_hop: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], bool]:
    path = plan_stealth_path(env, start, final_goal, global_cfg)
    if not path or len(path) <= 1:
        return [], [], False

    coarse_waypoints = extract_waypoints(path, wp_cfg)
    if not coarse_waypoints:
        coarse_waypoints = [path[-1]]

    index_map: dict[tuple[int, int], int] = {p: i for i, p in enumerate(path)}
    anchor_indices = [index_map[w] for w in coarse_waypoints if w in index_map]
    anchor_indices.append(len(path) - 1)
    anchor_indices = sorted(set(i for i in anchor_indices if i > 0))
    if not anchor_indices:
        anchor_indices = [len(path) - 1]

    max_hop = max(1, int(subgoal_max_hop))
    dense_waypoints: list[tuple[int, int]] = []
    prev_idx = 0
    for end_idx in anchor_indices:
        k = prev_idx + max_hop
        while k < end_idx:
            dense_waypoints.append(path[k])
            k += max_hop
        dense_waypoints.append(path[end_idx])
        prev_idx = end_idx

    cleaned: list[tuple[int, int]] = []
    for w in dense_waypoints:
        if not cleaned or cleaned[-1] != w:
            cleaned.append(w)
    if not cleaned or cleaned[-1] != path[-1]:
        cleaned.append(path[-1])
    return cleaned, path, True


def _run_episode(
    env: BattlefieldEnv,
    policy: ActorCriticCNN,
    device: torch.device,
    global_cfg: StealthCostConfig,
    fallback_cfg: StealthCostConfig,
    wp_cfg: WaypointConfig,
    waypoint_reach_radius: float,
    replan_stagnation: int,
    replan_collisions: int,
    replan_max: int,
    fallback_goal_radius: float,
    fallback_stagnation: int,
    fallback_remaining_steps: int,
    fallback_max_uses: int,
    subgoal_max_hop: int,
) -> dict:
    final_goal = _to_tuple(env.goal_position.copy())
    start = _to_tuple(env.agent_position.copy())
    waypoints, planner_path, ok = _make_waypoints(
        env, start, final_goal, global_cfg, wp_cfg, subgoal_max_hop=subgoal_max_hop
    )
    if not ok:
        return {
            "success": False,
            "result": "plan_fail",
            "steps": 0,
            "exposure": 0.0,
            "trajectory": [list(start)],
            "waypoints": [],
            "planner_path": [],
            "waypoint_hits": [],
            "replan_count": 0,
            "fallback_used_count": 0,
        }

    wp_idx = 0
    env.set_goal(waypoints[wp_idx])
    done = False
    info: dict = {"result": "unknown", "collisions": 0}
    ep_steps = 0
    ep_exposed = 0
    trajectory: list[list[int]] = [list(start)]
    waypoint_hits: list[int] = []
    replan_count = 0
    fallback_used_count = 0

    best_dist = _dist(_to_tuple(env.agent_position), _to_tuple(env.goal_position))
    stagnation = 0

    while not done:
        obs = env._get_observation()
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(env.get_action_mask()).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _, _ = policy(obs_t, mask_t, deterministic=True)
        _, _, done, info = env.step(int(action.item()))
        ep_steps += 1
        pos = _to_tuple(env.agent_position)
        trajectory.append([pos[0], pos[1]])
        if env.visibility_map[pos] > 0.5:
            ep_exposed += 1

        cur_dist = _dist(pos, _to_tuple(env.goal_position))
        if cur_dist + 1e-4 < best_dist:
            best_dist = cur_dist
            stagnation = 0
        else:
            stagnation += 1

        if not done and cur_dist <= waypoint_reach_radius and wp_idx < len(waypoints) - 1:
            waypoint_hits.append(ep_steps)
            wp_idx += 1
            env.set_goal(waypoints[wp_idx])
            best_dist = _dist(_to_tuple(env.agent_position), _to_tuple(env.goal_position))
            stagnation = 0

        near_goal = cur_dist <= fallback_goal_radius
        near_timeout = (env.config.max_steps - env.steps) <= fallback_remaining_steps
        fallback_trigger = (stagnation >= fallback_stagnation) or near_timeout
        if (
            not done
            and near_goal
            and fallback_trigger
            and fallback_used_count < max(1, fallback_max_uses)
        ):
            used, success = _apply_planner_fallback(env, _to_tuple(env.goal_position), fallback_cfg)
            if used:
                fallback_used_count += 1
                ep_steps = int(env.steps)
                pos = _to_tuple(env.agent_position)
                trajectory.append([pos[0], pos[1]])
                if success:
                    if wp_idx < len(waypoints) - 1:
                        waypoint_hits.append(ep_steps)
                        wp_idx += 1
                        env.set_goal(waypoints[wp_idx])
                        best_dist = _dist(_to_tuple(env.agent_position), _to_tuple(env.goal_position))
                        stagnation = 0
                        done = False
                        info = {"result": "waypoint", "collisions": env.total_collisions}
                    else:
                        done = True
                        info = {"result": "success", "collisions": env.total_collisions}
                else:
                    done = True
                    info = {"result": "fallback_fail", "collisions": env.total_collisions}

        if (
            not done
            and replan_count < replan_max
            and (stagnation >= replan_stagnation or env.consecutive_collisions >= replan_collisions)
        ):
            cur = _to_tuple(env.agent_position)
            waypoints_new, planner_path_new, ok_new = _make_waypoints(
                env, cur, final_goal, global_cfg, wp_cfg, subgoal_max_hop=subgoal_max_hop
            )
            if ok_new:
                waypoints = waypoints_new
                planner_path = planner_path_new
                wp_idx = 0
                env.set_goal(waypoints[wp_idx])
                replan_count += 1
                stagnation = 0
                best_dist = _dist(_to_tuple(env.agent_position), _to_tuple(env.goal_position))

        if done and info.get("result") == "success" and wp_idx < len(waypoints) - 1:
            waypoint_hits.append(ep_steps)
            wp_idx += 1
            env.set_goal(waypoints[wp_idx])
            best_dist = _dist(_to_tuple(env.agent_position), _to_tuple(env.goal_position))
            stagnation = 0
            done = False
            info = {"result": "waypoint", "collisions": env.total_collisions}

        if not done and ep_steps >= env.config.max_steps:
            done = True
            info = {"result": "timeout", "collisions": env.total_collisions}

    return {
        "success": info.get("result") == "success",
        "result": info.get("result", "unknown"),
        "steps": ep_steps,
        "exposure": ep_exposed / max(1, ep_steps),
        "trajectory": trajectory,
        "waypoints": [list(w) for w in waypoints],
        "planner_path": [list(p) for p in planner_path],
        "waypoint_hits": waypoint_hits,
        "replan_count": replan_count,
        "fallback_used_count": fallback_used_count,
        "start": [int(start[0]), int(start[1])],
        "final_goal": [int(final_goal[0]), int(final_goal[1])],
    }


def _apply_planner_fallback(env: BattlefieldEnv, goal: tuple[int, int], cfg: StealthCostConfig) -> tuple[bool, bool]:
    start = _to_tuple(env.agent_position)
    path = plan_stealth_path(env, start, goal, cfg)
    if not path or len(path) <= 1:
        return False, False
    for nxt in path[1:]:
        cur = _to_tuple(env.agent_position)
        move = (nxt[0] - cur[0], nxt[1] - cur[1])
        action = MOVE_TO_ACTION.get(move)
        if action is None:
            return True, False
        _, _, done, info = env.step(action)
        if done:
            return True, info.get("result") == "success"
    return True, _to_tuple(env.agent_position) == goal


def _plot_episode(env: BattlefieldEnv, result: dict, out_path: Path, title: str) -> None:
    tags = env.window_tag_map
    vis = env.visibility_map
    g = int(env.grid_size)

    rgba = np.zeros((g, g, 4), dtype=np.float32)
    rgba[:, :] = np.array([0.93, 0.91, 0.84, 1.0], dtype=np.float32)
    rgba[vis > 0.5] = np.array([0.98, 0.80, 0.36, 1.0], dtype=np.float32)
    if tags is not None:
        rgba[tags == 1] = np.array([0.22, 0.22, 0.24, 1.0], dtype=np.float32)
        rgba[tags == 2] = np.array([0.12, 0.35, 0.18, 1.0], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8.8, 8.8))
    ax.imshow(rgba, origin="lower", interpolation="nearest")

    ax.set_xticks(np.arange(-0.5, g + 0.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, g + 0.5, 1), minor=True)
    ax.grid(which="minor", color="#8f8f8f", linewidth=0.35, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    planner_path = np.array(result["planner_path"], dtype=np.float32)
    if len(planner_path) > 1:
        ax.plot(planner_path[:, 1], planner_path[:, 0], color="#4a4a4a", linewidth=2.0, alpha=0.8, label="planner path")

    traj = np.array(result["trajectory"], dtype=np.float32)
    if len(traj) > 1:
        ax.plot(traj[:, 1], traj[:, 0], color="#00bcd4", linewidth=2.6, alpha=0.95, label="rl trajectory")
        ax.scatter(traj[:, 1], traj[:, 0], c=np.linspace(0.0, 1.0, len(traj)), cmap="viridis", s=10, zorder=4)

    waypoints = np.array(result["waypoints"], dtype=np.float32)
    if len(waypoints) > 0:
        ax.scatter(waypoints[:, 1], waypoints[:, 0], c="#ff2d55", s=42, marker="s", label="waypoints", zorder=5)
        for i, (x, y) in enumerate(waypoints.tolist(), start=1):
            ax.text(y + 0.2, x + 0.2, str(i), fontsize=7, color="#111111", zorder=6)

    sx, sy = result["start"]
    gx, gy = result["final_goal"]
    ax.scatter([sy], [sx], c="#00acc1", s=140, marker="o", edgecolors="white", linewidths=1.6, zorder=7, label="start")
    ax.scatter([gy], [gx], c="#f44336", s=180, marker="*", edgecolors="white", linewidths=1.6, zorder=7, label="goal")

    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(-0.5, g - 0.5)
    ax.set_xticks(range(0, g, 5))
    ax.set_yticks(range(0, g, 5))
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.set_xlabel(
        f"result={result['result']}  steps={result['steps']}  "
        f"exposure={result['exposure']:.3f}  waypoints={len(result['waypoints'])}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot successful hierarchical 50x50 rollouts", allow_abbrev=False)
    add_config_args(parser, default_section="hierarchical_waypoint_eval")
    add_obs_args(parser, include_view_size=False)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3, help="number of rollouts to save")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--output-dir", type=str, default="analysis/rollout_plot")
    parser.add_argument("--policy-view-size", type=int, default=10)
    parser.add_argument("--executor", type=str, default="rl", choices=["rl", "planner"])
    parser.add_argument("--subgoal-max-hop", type=int, default=4)
    parser.add_argument("--planner-guide", action="store_true")
    parser.add_argument("--w-len", type=float, default=1.0)
    parser.add_argument("--w-vis", type=float, default=2.5)
    parser.add_argument("--w-slope", type=float, default=0.8)
    parser.add_argument("--w-turn", type=float, default=0.15)
    parser.add_argument("--wp-spacing", type=int, default=7)
    parser.add_argument("--wp-min-segment", type=int, default=3)
    parser.add_argument("--wp-reach-radius", type=float, default=1.5)
    parser.add_argument("--replan-stagnation", type=int, default=10)
    parser.add_argument("--replan-collisions", type=int, default=6)
    parser.add_argument("--replan-max", type=int, default=4)
    parser.add_argument("--fallback-goal-radius", type=float, default=3.0)
    parser.add_argument("--fallback-stagnation", type=int, default=4)
    parser.add_argument("--fallback-remaining-steps", type=int, default=8)
    parser.add_argument("--fallback-max-uses", type=int, default=1)
    parser.add_argument("--fb-w-len", type=float, default=1.0)
    parser.add_argument("--fb-w-vis", type=float, default=2.0)
    parser.add_argument("--fb-w-slope", type=float, default=0.6)
    parser.add_argument("--fb-w-turn", type=float, default=0.1)
    parser.add_argument("--planner-guide-sigma", type=float, default=1.4)
    parser.add_argument("--planner-w-len", type=float, default=1.0)
    parser.add_argument("--planner-w-vis", type=float, default=2.5)
    parser.add_argument("--planner-w-slope", type=float, default=0.8)
    parser.add_argument("--planner-w-turn", type=float, default=0.15)
    args = parse_args_with_config(parser, default_section="hierarchical_waypoint_eval")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state_dict = torch.load(Path(args.model), map_location=device)
    model_spec = infer_model_spec(state_dict)

    env_cfg = replace(
        EnvConfig(),
        grid_size=50,
        local_map_size=50,
        obs_view_size=args.policy_view_size,
        scenario_mode="full_map",
        max_steps=args.max_steps,
        obs_line_guide=args.obs_line_guide,
        obs_line_sigma=args.obs_line_sigma,
        obs_use_visited=args.obs_visited,
        obs_use_remaining=args.obs_remaining,
        obs_use_stagnation=args.obs_stagnation,
        obs_stagnation_cap=args.obs_stagnation_cap,
        obs_use_prev_move=args.obs_prev_move,
        obs_visit_decay=args.obs_visit_decay,
        planner_guide_channel=args.planner_guide,
        planner_guide_sigma=args.planner_guide_sigma,
        planner_w_len=args.planner_w_len,
        planner_w_vis=args.planner_w_vis,
        planner_w_slope=args.planner_w_slope,
        planner_w_turn=args.planner_w_turn,
    )
    env = BattlefieldEnv(env_cfg)
    obs_channels = int(env._get_observation().shape[0])
    if obs_channels != int(model_spec["in_channels"]):
        raise RuntimeError(f"observation channels mismatch: env={obs_channels}, ckpt={model_spec['in_channels']}")

    policy = ActorCriticCNN.from_state_dict(state_dict).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()

    global_cfg = StealthCostConfig(
        w_len=args.w_len, w_vis=args.w_vis, w_slope=args.w_slope, w_turn=args.w_turn
    )
    fallback_cfg = StealthCostConfig(
        w_len=args.fb_w_len, w_vis=args.fb_w_vis, w_slope=args.fb_w_slope, w_turn=args.fb_w_turn
    )
    wp_cfg = WaypointConfig(spacing=args.wp_spacing, keep_turn_points=True, min_segment=args.wp_min_segment)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    metas: list[dict] = []

    for ep in range(args.episodes):
        scene_seed = int(rng.integers(0, 10_000_000))
        env.reset(seed=scene_seed)
        result = _run_episode(
            env,
            policy,
            device,
            global_cfg=global_cfg,
            fallback_cfg=fallback_cfg,
            wp_cfg=wp_cfg,
            waypoint_reach_radius=args.wp_reach_radius,
            replan_stagnation=args.replan_stagnation,
            replan_collisions=args.replan_collisions,
            replan_max=args.replan_max,
            fallback_goal_radius=args.fallback_goal_radius,
            fallback_stagnation=args.fallback_stagnation,
            fallback_remaining_steps=args.fallback_remaining_steps,
            fallback_max_uses=args.fallback_max_uses,
            subgoal_max_hop=args.subgoal_max_hop,
        )
        png_path = out_dir / f"ep_{ep:02d}_seed_{scene_seed}.png"
        _plot_episode(
            env,
            result,
            png_path,
            title=f"ep={ep} seed={scene_seed} success={int(result['success'])}",
        )
        metas.append(
            {
                "episode": ep,
                "scene_seed": scene_seed,
                "image": png_path.name,
                "result": result["result"],
                "success": int(result["success"]),
                "steps": int(result["steps"]),
                "exposure": float(result["exposure"]),
                "n_waypoints": len(result["waypoints"]),
                "replan_count": int(result["replan_count"]),
                "fallback_used_count": int(result["fallback_used_count"]),
            }
        )

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"episodes": metas}, f, ensure_ascii=False, indent=2)

    print(f"saved {len(metas)} rollout plots to: {out_dir}")


if __name__ == "__main__":
    main()
