from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

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


def _render_ascii_map(env: BattlefieldEnv, trajectory: list[list[int]]) -> str:
    g = int(env.grid_size)
    grid = np.full((g, g), ".", dtype="<U1")
    tags = env.window_tag_map
    if tags is not None:
        grid[tags == 1] = "#"
        grid[tags == 2] = "T"
    for x, y in trajectory:
        if 0 <= x < g and 0 <= y < g and grid[x, y] == ".":
            grid[x, y] = "*"
    sx, sy = _to_tuple(env.start_position)
    gx, gy = _to_tuple(env.goal_position)
    ax, ay = _to_tuple(env.agent_position)
    grid[sx, sy] = "S"
    grid[gx, gy] = "G"
    grid[ax, ay] = "A"
    lines = ["".join(grid[x, y] for y in range(g)) for x in range(g)]
    return "\n".join(lines)


def _make_waypoints(
    env: BattlefieldEnv,
    start: tuple[int, int],
    final_goal: tuple[int, int],
    global_cfg: StealthCostConfig,
    wp_cfg: WaypointConfig,
    subgoal_max_hop: int,
) -> tuple[list[tuple[int, int]], bool]:
    path = plan_stealth_path(env, start, final_goal, global_cfg)
    if not path or len(path) <= 1:
        return [], False
    coarse_waypoints = extract_waypoints(path, wp_cfg)
    if not coarse_waypoints:
        coarse_waypoints = [path[-1]]

    index_map: dict[tuple[int, int], int] = {}
    for i, p in enumerate(path):
        index_map[p] = i

    anchor_indices = [index_map[w] for w in coarse_waypoints if w in index_map]
    anchor_indices.append(len(path) - 1)
    anchor_indices = sorted(set(anchor_indices))
    anchor_indices = [i for i in anchor_indices if i > 0]
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
    return cleaned, True


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
    policy_view_size: int,
    executor: str,
    subgoal_max_hop: int,
) -> dict:
    final_goal = _to_tuple(env.goal_position.copy())
    start = _to_tuple(env.agent_position.copy())
    waypoints, ok = _make_waypoints(env, start, final_goal, global_cfg, wp_cfg, subgoal_max_hop=subgoal_max_hop)
    if not ok:
        return {
            "success": False,
            "result": "plan_fail",
            "steps": 0,
            "exposure": 0.0,
            "fallback_used": False,
            "fallback_used_count": 0,
            "near_goal_fallback_count": 0,
            "fallback_success_count": 0,
            "replan_count": 0,
            "trajectory": [list(start)],
            "n_waypoints": 0,
        }

    wp_idx = 0
    env.set_goal(waypoints[wp_idx])
    n_waypoints = len(waypoints)
    done = False
    info: dict = {"result": "unknown", "collisions": 0}
    ep_steps = 0
    ep_exposed = 0
    trajectory: list[list[int]] = [list(start)]

    best_dist = _dist(_to_tuple(env.agent_position), _to_tuple(env.goal_position))
    stagnation = 0
    replan_count = 0
    fallback_used_count = 0
    near_goal_fallback_count = 0
    fallback_success_count = 0

    while not done:
        old_dist = _dist(_to_tuple(env.agent_position), _to_tuple(env.goal_position))
        if executor == "planner":
            p = plan_stealth_path(env, _to_tuple(env.agent_position), _to_tuple(env.goal_position), fallback_cfg)
            if not p or len(p) <= 1:
                done = True
                info = {"result": "plan_fail", "collisions": env.total_collisions}
                _reward = 0.0
            else:
                nxt = p[1]
                cur = _to_tuple(env.agent_position)
                move = (nxt[0] - cur[0], nxt[1] - cur[1])
                action = MOVE_TO_ACTION.get(move)
                if action is None:
                    done = True
                    info = {"result": "plan_fail", "collisions": env.total_collisions}
                    _reward = 0.0
                else:
                    _obs, _reward, done, info = env.step(action)
        else:
            obs = env._get_observation()
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(env.get_action_mask()).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _, _ = policy(obs_t, mask_t, deterministic=True)
            _obs, _reward, done, info = env.step(int(action.item()))
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

        # 瀛愮洰鏍囧崐寰勫埌杈撅細鍒囧埌涓嬩竴涓獁aypoint锛堟渶缁堢洰鏍囬櫎澶栵級
        if not done and cur_dist <= waypoint_reach_radius and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            env.set_goal(waypoints[wp_idx])
            best_dist = _dist(_to_tuple(env.agent_position), _to_tuple(env.goal_position))
            stagnation = 0

        # 杩戠粓鐐圭獎瑙﹀彂鍏滃簳
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
                near_goal_fallback_count += 1
                fallback_success_count += int(success)
                ep_steps = int(env.steps)
                pos = _to_tuple(env.agent_position)
                trajectory.append([pos[0], pos[1]])
                if success:
                    # 到达当前子目标后，继续切换到下一个 waypoint。
                    if wp_idx < len(waypoints) - 1:
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

        # 鍋滄粸鎴栫鎾為噸瑙勫垝锛堥潰鍚戞渶缁堢洰鏍囷級
        if (
            not done
            and replan_count < replan_max
            and (stagnation >= replan_stagnation or env.consecutive_collisions >= replan_collisions)
        ):
            cur = _to_tuple(env.agent_position)
            waypoints_new, ok_new = _make_waypoints(
                env,
                cur,
                final_goal,
                global_cfg,
                wp_cfg,
                subgoal_max_hop=subgoal_max_hop,
            )
            if ok_new:
                waypoints = waypoints_new
                wp_idx = 0
                env.set_goal(waypoints[wp_idx])
                replan_count += 1
                stagnation = 0
                best_dist = _dist(_to_tuple(env.agent_position), _to_tuple(env.goal_position))

        # 闈炴渶缁堝瓙鐩爣琚?env 鍒ゅ畾 success 鏃讹紝鏀逛负缁х画鎵ц
        if done and info.get("result") == "success" and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            env.set_goal(waypoints[wp_idx])
            best_dist = _dist(_to_tuple(env.agent_position), _to_tuple(env.goal_position))
            stagnation = 0
            done = False
            info = {"result": "waypoint", "collisions": env.total_collisions}

        # 閬垮厤娼滃湪姝诲惊鐜繚鎶?        if not done and ep_steps >= env.config.max_steps:
            done = True
            info = {"result": "timeout", "collisions": env.total_collisions}

    return {
        "success": info.get("result") == "success",
        "result": info.get("result", "unknown"),
        "steps": ep_steps,
        "exposure": ep_exposed / max(1, ep_steps),
        "start": [int(start[0]), int(start[1])],
        "final_goal": [int(final_goal[0]), int(final_goal[1])],
        "final_remaining_dist": _dist(_to_tuple(env.agent_position), final_goal),
        "fallback_used": fallback_used_count > 0,
        "fallback_used_count": fallback_used_count,
        "near_goal_fallback_count": near_goal_fallback_count,
        "fallback_success_count": fallback_success_count,
        "replan_count": replan_count,
        "trajectory": trajectory,
        "n_waypoints": n_waypoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchical 50x50 eval: global waypoint planner + local RL", allow_abbrev=False)
    add_config_args(parser, default_section="hierarchical_waypoint_eval")
    add_obs_args(parser, include_view_size=False)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--output-dir", type=str, default="analysis/hierarchical_50_eval")
    parser.add_argument("--save-fail-maps", type=int, default=30)
    parser.add_argument("--policy-view-size", type=int, default=10, help="策略局部观测边长；0表示使用整图")
    parser.add_argument("--executor", type=str, default="rl", choices=["rl", "planner"], help="局部执行器类型")
    parser.add_argument("--subgoal-max-hop", type=int, default=4, help="相邻子目标在全局路径上的最大步长")
    parser.add_argument("--planner-guide", action="store_true", help="在观测中加入 planner 走廊通道")
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
    if args.subgoal_max_hop < 1:
        raise ValueError("--subgoal-max-hop must be >= 1")
    if args.policy_view_size > 0:
        hard_cap = max(1, args.policy_view_size // 2 - 1)
        if args.subgoal_max_hop > hard_cap:
            print(
                f"subgoal_max_hop={args.subgoal_max_hop} 超过局部窗口安全上限 {hard_cap}，"
                f"已自动下调为 {hard_cap}"
            )
            args.subgoal_max_hop = hard_cap

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state_dict = torch.load(Path(args.model), map_location=device)
    model_spec = infer_model_spec(state_dict)
    expected_channels = int(model_spec["in_channels"])
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
    if obs_channels != expected_channels:
        raise RuntimeError(f"observation channels mismatch: env={obs_channels}, ckpt={expected_channels}")
    policy = ActorCriticCNN.from_state_dict(state_dict).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()

    global_cfg = StealthCostConfig(
        w_len=args.w_len,
        w_vis=args.w_vis,
        w_slope=args.w_slope,
        w_turn=args.w_turn,
    )
    fallback_cfg = StealthCostConfig(
        w_len=args.fb_w_len,
        w_vis=args.fb_w_vis,
        w_slope=args.fb_w_slope,
        w_turn=args.fb_w_turn,
    )
    wp_cfg = WaypointConfig(spacing=args.wp_spacing, keep_turn_points=True, min_segment=args.wp_min_segment)

    out_dir = Path(args.output_dir)
    fail_dir = out_dir / "fail_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    fail_saved = 0

    for ep in range(args.episodes):
        scene_seed = int(rng.integers(0, 10_000_000))
        env.reset(seed=scene_seed)
        res = _run_episode(
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
            policy_view_size=args.policy_view_size,
            executor=args.executor,
            subgoal_max_hop=args.subgoal_max_hop,
        )
        row = {
            "episode": ep,
            "scene_seed": scene_seed,
            "success": int(res["success"]),
            "result": res["result"],
            "steps": int(res["steps"]),
            "exposure": float(res["exposure"]),
            "start": res["start"],
            "final_goal": res["final_goal"],
            "final_remaining_dist": float(res["final_remaining_dist"]),
            "fallback_used": int(res["fallback_used"]),
            "fallback_used_count": int(res["fallback_used_count"]),
            "near_goal_fallback_count": int(res["near_goal_fallback_count"]),
            "fallback_success_count": int(res["fallback_success_count"]),
            "replan_count": int(res["replan_count"]),
            "n_waypoints": int(res["n_waypoints"]),
            "trajectory": res["trajectory"],
        }
        rows.append(row)

        if (not res["success"]) and fail_saved < args.save_fail_maps:
            meta = {
                "episode": ep,
                "scene_seed": scene_seed,
                "result": res["result"],
                "steps": int(res["steps"]),
                "replan_count": int(res["replan_count"]),
                "fallback_used_count": int(res["fallback_used_count"]),
                "n_waypoints": int(res["n_waypoints"]),
            }
            text = _render_ascii_map(env, res["trajectory"])
            with open(fail_dir / f"fail_{fail_saved:03d}_ep_{ep}.txt", "w", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False, indent=2))
                f.write("\n\n")
                f.write(text)
                f.write("\n\ntrajectory:\n")
                f.write(str(res["trajectory"]))
                f.write("\n")
            fail_saved += 1

    n = len(rows)
    succ = sum(r["success"] for r in rows)
    fallback_eps = sum(r["fallback_used"] for r in rows)
    fallback_uses = sum(r["fallback_used_count"] for r in rows)
    near_fb_uses = sum(r["near_goal_fallback_count"] for r in rows)
    fallback_succ = sum(r["fallback_success_count"] for r in rows)
    timeout = sum(1 for r in rows if r["result"] == "timeout")
    avg_steps = float(np.mean([r["steps"] for r in rows])) if n else 0.0
    avg_exposure = float(np.mean([r["exposure"] for r in rows])) if n else 0.0
    avg_replan = float(np.mean([r["replan_count"] for r in rows])) if n else 0.0
    avg_wps = float(np.mean([r["n_waypoints"] for r in rows])) if n else 0.0
    raw_success_no_fb = sum(1 for r in rows if r["success"] == 1 and r["fallback_used"] == 0) / max(1, n)

    summary = {
        "n_eval": n,
        "success_rate": succ / max(1, n),
        "timeout_rate": timeout / max(1, n),
        "avg_steps": avg_steps,
        "avg_exposure": avg_exposure,
        "fallback_episode_rate": fallback_eps / max(1, n),
        "fallback_use_per_ep": fallback_uses / max(1, n),
        "near_goal_fallback_use_per_ep": near_fb_uses / max(1, n),
        "fallback_salvage_rate": fallback_succ / max(1, fallback_uses),
        "raw_success_without_fallback": raw_success_no_fb,
        "avg_replan_count": avg_replan,
        "avg_waypoint_count": avg_wps,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Hierarchical eval done")
    print(f"summary: {summary}")
    print(f"output dir: {out_dir}")
    print(f"failed maps saved: {fail_saved}")


if __name__ == "__main__":
    main()
