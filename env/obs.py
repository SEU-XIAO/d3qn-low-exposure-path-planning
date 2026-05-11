from __future__ import annotations

import argparse

import numpy as np

from planner import plan_stealth_path


def add_obs_args(parser: argparse.ArgumentParser, include_view_size: bool = True) -> None:
    if include_view_size:
        parser.add_argument("--obs-view-size", type=int, default=0, help="局部观测边长；0表示使用整图")
    parser.add_argument(
        "--obs-line-guide",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否加入 agent->goal 直线引导通道",
    )
    parser.add_argument("--obs-line-sigma", type=float, default=1.4, help="直线引导通道的高斯宽度（格）")
    parser.add_argument("--obs-visited", action="store_true", help="加入访问衰减热图通道")
    parser.add_argument("--obs-remaining", action="store_true", help="加入剩余步数比例通道")
    parser.add_argument("--obs-stagnation", action="store_true", help="加入停滞比例通道")
    parser.add_argument("--obs-stagnation-cap", type=int, default=8, help="停滞比例归一化上限步数")
    parser.add_argument("--obs-prev-move", action="store_true", help="加入上一移动方向通道")
    parser.add_argument("--obs-visit-decay", type=float, default=0.92, help="访问热图每步衰减系数")


def crop_obs(obs: np.ndarray, center: tuple[int, int], view_size: int) -> np.ndarray:
    if view_size <= 0:
        return obs

    c, h, w = obs.shape
    half = view_size // 2
    cx, cy = int(center[0]), int(center[1])
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + view_size
    y1 = y0 + view_size

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(h, x1)
    src_y1 = min(w, y1)

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    out = np.zeros((c, view_size, view_size), dtype=np.float32)
    out[:, dst_x0:dst_x1, dst_y0:dst_y1] = obs[:, src_x0:src_x1, src_y0:src_y1]
    return out


def build_obs(env) -> np.ndarray:
    env._ensure_obs_state()
    h, w = int(env.grid_size), int(env.grid_size)
    ax, ay = int(env.agent_position[0]), int(env.agent_position[1])
    gx, gy = int(env.goal_position[0]), int(env.goal_position[1])

    ch_height = env.height_map.astype(np.float32) / max(1.0, float(env.height_levels))

    if env.window_tag_map is not None:
        ch_ground = (env.window_tag_map == 0).astype(np.float32)
        ch_building = (env.window_tag_map == 1).astype(np.float32)
        ch_tree = (env.window_tag_map == 2).astype(np.float32)
    else:
        ch_ground = np.ones((h, w), dtype=np.float32)
        ch_building = np.zeros((h, w), dtype=np.float32)
        ch_tree = np.zeros((h, w), dtype=np.float32)

    ch_vis = env.visibility_map.astype(np.float32)

    ch_agent = np.zeros((h, w), dtype=np.float32)
    ch_agent[ax, ay] = 1.0

    ch_goal = np.zeros((h, w), dtype=np.float32)
    ch_goal[gx, gy] = 1.0

    rel_scale = _rel_scale(env)
    dx = np.clip((gx - ax) / rel_scale, -1.0, 1.0)
    dy = np.clip((gy - ay) / rel_scale, -1.0, 1.0)
    ch_rel_dx = np.full((h, w), dx, dtype=np.float32)
    ch_rel_dy = np.full((h, w), dy, dtype=np.float32)

    channels = [
        ch_height,
        ch_ground,
        ch_building,
        ch_tree,
        ch_vis,
        ch_agent,
        ch_goal,
        ch_rel_dx,
        ch_rel_dy,
    ]

    if env.config.obs_line_guide:
        channels.append(_line_guide(env, sigma=_line_sigma(env)))

    if env.config.planner_guide_channel:
        channels.append(_planner_corridor(env))

    if env.config.obs_use_visited:
        channels.append(env._obs_visited.copy())

    if env.config.obs_use_remaining:
        remaining = max(0.0, float(env.config.max_steps - env.steps)) / max(1.0, float(env.config.max_steps))
        channels.append(np.full((h, w), remaining, dtype=np.float32))

    if env.config.obs_use_stagnation:
        stagnation = min(1.0, float(env._obs_goal_stagnation) / max(1.0, float(env.config.obs_stagnation_cap)))
        channels.append(np.full((h, w), stagnation, dtype=np.float32))

    if env.config.obs_use_prev_move:
        channels.append(np.full((h, w), float(env._obs_prev_move[0]), dtype=np.float32))
        channels.append(np.full((h, w), float(env._obs_prev_move[1]), dtype=np.float32))

    obs = np.stack(channels, axis=0).astype(np.float32)
    if env.config.obs_view_size > 0:
        obs = crop_obs(obs, center=(ax, ay), view_size=int(env.config.obs_view_size))
    return obs


def _rel_scale(env) -> float:
    if int(env.config.obs_view_size) > 0:
        return max(1.0, float(env.config.obs_view_size - 1))
    return max(1.0, float(env.grid_size - 1))


def _line_sigma(env) -> float:
    if int(env.config.obs_view_size) > 0:
        return max(0.5, float(env.config.obs_line_sigma))
    return max(1.0, float(env.grid_size) * 0.08)


def _line_guide(env, sigma: float) -> np.ndarray:
    h, w = int(env.grid_size), int(env.grid_size)
    ax, ay = int(env.agent_position[0]), int(env.agent_position[1])
    gx, gy = int(env.goal_position[0]), int(env.goal_position[1])

    yy, xx = np.mgrid[0:h, 0:w]
    p0 = np.array([ax, ay], dtype=np.float32)
    p1 = np.array([gx, gy], dtype=np.float32)
    v = p1 - p0
    denom = float(v[0] * v[0] + v[1] * v[1]) + 1e-6
    t = ((xx - p0[0]) * v[0] + (yy - p0[1]) * v[1]) / denom
    t = np.clip(t, 0.0, 1.0)
    proj_x = p0[0] + t * v[0]
    proj_y = p0[1] + t * v[1]
    dist2 = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
    return np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)


def _planner_corridor(env) -> np.ndarray:
    h, w = int(env.grid_size), int(env.grid_size)
    start = tuple(int(v) for v in env.agent_position.tolist())
    goal = tuple(int(v) for v in env.goal_position.tolist())
    if start == goal:
        out = np.zeros((h, w), dtype=np.float32)
        out[start] = 1.0
        return out

    path = plan_stealth_path(env, start, goal, env._planner_cfg())
    if not path:
        return np.zeros((h, w), dtype=np.float32)

    gx, gy = np.indices((h, w), dtype=np.float32)
    pts = np.array(path, dtype=np.float32)
    dx = gx[None, :, :] - pts[:, 0][:, None, None]
    dy = gy[None, :, :] - pts[:, 1][:, None, None]
    dist2 = np.min(dx * dx + dy * dy, axis=0)
    sigma = max(0.5, float(env.config.planner_guide_sigma))
    return np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
