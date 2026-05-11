from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

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
from planner import StealthCostConfig, WaypointConfig
from visualize.rollout_plot import _run_episode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a hierarchical rollout for blocked cells and corner cuts",
        allow_abbrev=False,
    )
    add_config_args(parser, default_section="hierarchical_waypoint_eval")
    add_obs_args(parser, include_view_size=False)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--scene-seed", type=int, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--policy-view-size", type=int, default=10)
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
    parser.add_argument("--dump-json", type=str, default=None)
    return parser


def _blocked_points(points: list[tuple[int, int]], tags) -> list[dict]:
    bad: list[dict] = []
    for idx, (x, y) in enumerate(points):
        tag = int(tags[x, y])
        if tag != 0:
            bad.append({"idx": idx, "pos": [x, y], "tag": tag})
    return bad


def _illegal_moves(points: list[tuple[int, int]]) -> list[dict]:
    bad: list[dict] = []
    for idx in range(1, len(points)):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        dx, dy = x1 - x0, y1 - y0
        if max(abs(dx), abs(dy)) > 1 or (dx == 0 and dy == 0):
            bad.append(
                {
                    "idx": idx - 1,
                    "from": [x0, y0],
                    "to": [x1, y1],
                    "delta": [dx, dy],
                }
            )
    return bad


def _corner_cuts(points: list[tuple[int, int]], tags) -> list[dict]:
    cuts: list[dict] = []
    for idx in range(1, len(points)):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) == 1 and abs(dy) == 1:
            c1 = (x0 + dx, y0)
            c2 = (x0, y0 + dy)
            tag1 = int(tags[c1])
            tag2 = int(tags[c2])
            if tag1 != 0 or tag2 != 0:
                cuts.append(
                    {
                        "idx": idx - 1,
                        "from": [x0, y0],
                        "to": [x1, y1],
                        "adj1": [c1[0], c1[1]],
                        "adj1_tag": tag1,
                        "adj2": [c2[0], c2[1]],
                        "adj2_tag": tag2,
                        "both_blocked": bool(tag1 != 0 and tag2 != 0),
                    }
                )
    return cuts


def _plot_bg_blocked(points: list[tuple[int, int]], tags) -> list[dict]:
    mismatches: list[dict] = []
    for idx, (row, col) in enumerate(points):
        # rollout_plot.py 当前使用 rgba.transpose(1, 0, 2) 绘底图，
        # 所以显示在 (x=col, y=row) 位置的底色其实来自 tags[col, row]。
        if not (0 <= col < tags.shape[0] and 0 <= row < tags.shape[1]):
            continue
        shown_tag = int(tags[col, row])
        if shown_tag != 0:
            mismatches.append(
                {
                    "idx": idx,
                    "pos": [row, col],
                    "env_tag": int(tags[row, col]),
                    "shown_tag": shown_tag,
                }
            )
    return mismatches


def main() -> None:
    parser = _build_parser()
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
    wp_cfg = WaypointConfig(
        spacing=args.wp_spacing,
        keep_turn_points=True,
        min_segment=args.wp_min_segment,
    )

    env.reset(seed=args.scene_seed)
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

    tags = env.window_tag_map
    traj = [tuple(map(int, p)) for p in result["trajectory"]]
    planner_path = [tuple(map(int, p)) for p in result["planner_path"]]
    waypoints = [tuple(map(int, p)) for p in result["waypoints"]]

    payload = {
        "scene_seed": int(args.scene_seed),
        "result": result["result"],
        "success": bool(result["success"]),
        "steps": int(result["steps"]),
        "start": result["start"],
        "goal": result["final_goal"],
        "trajectory_len": len(traj),
        "planner_len": len(planner_path),
        "waypoint_len": len(waypoints),
        "trajectory_blocked": _blocked_points(traj, tags),
        "planner_blocked": _blocked_points(planner_path, tags),
        "waypoints_blocked": _blocked_points(waypoints, tags),
        "trajectory_illegal_moves": _illegal_moves(traj),
        "planner_illegal_moves": _illegal_moves(planner_path),
        "trajectory_corner_cuts": _corner_cuts(traj, tags),
        "planner_corner_cuts": _corner_cuts(planner_path, tags),
        "trajectory_plot_bg_blocked": _plot_bg_blocked(traj, tags),
        "planner_plot_bg_blocked": _plot_bg_blocked(planner_path, tags),
    }

    print(
        f"scene_seed={payload['scene_seed']} result={payload['result']} "
        f"success={int(payload['success'])} steps={payload['steps']}"
    )
    print(
        f"blocked: traj={len(payload['trajectory_blocked'])} "
        f"planner={len(payload['planner_blocked'])} "
        f"waypoints={len(payload['waypoints_blocked'])}"
    )
    print(
        f"illegal_moves: traj={len(payload['trajectory_illegal_moves'])} "
        f"planner={len(payload['planner_illegal_moves'])}"
    )
    print(
        f"corner_cuts: traj={len(payload['trajectory_corner_cuts'])} "
        f"planner={len(payload['planner_corner_cuts'])}"
    )
    print(
        f"plot_bg_blocked: traj={len(payload['trajectory_plot_bg_blocked'])} "
        f"planner={len(payload['planner_plot_bg_blocked'])}"
    )
    if payload["trajectory_corner_cuts"]:
        print("traj_corner_cuts_first3=", json.dumps(payload["trajectory_corner_cuts"][:3], ensure_ascii=False))
    if payload["planner_corner_cuts"]:
        print("planner_corner_cuts_first3=", json.dumps(payload["planner_corner_cuts"][:3], ensure_ascii=False))
    if payload["trajectory_plot_bg_blocked"]:
        print("traj_plot_bg_blocked_first3=", json.dumps(payload["trajectory_plot_bg_blocked"][:3], ensure_ascii=False))

    if args.dump_json:
        out_path = Path(args.dump_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
