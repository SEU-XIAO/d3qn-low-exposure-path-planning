from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planner import (
    StealthCostConfig,
    WaypointConfig,
    extract_waypoints,
    plan_stealth_path,
)


@dataclass
class WaypointCurriculumConfig:
    enabled: bool = False
    waypoint_spacing: int = 6
    waypoint_reach_radius: float = 1.0
    waypoint_reward: float = 6.0
    replan_stuck_collisions: int = 4
    w_len: float = 1.0
    w_vis: float = 3.0
    w_slope: float = 0.8
    w_turn: float = 0.15


class WaypointManager:
    def __init__(self, cfg: WaypointCurriculumConfig):
        self.cfg = cfg
        self.final_goal: tuple[int, int] | None = None
        self.waypoints: list[tuple[int, int]] = []
        self.index = 0

    def reset_for_episode(self, env) -> None:
        self.final_goal = tuple(env.goal_position.tolist())
        self.waypoints = []
        self.index = 0
        if not self.cfg.enabled:
            return

        start = tuple(env.start_position.tolist())
        stealth_cfg = StealthCostConfig(
            w_len=self.cfg.w_len,
            w_vis=self.cfg.w_vis,
            w_slope=self.cfg.w_slope,
            w_turn=self.cfg.w_turn,
        )
        full_path = plan_stealth_path(env, start, self.final_goal, stealth_cfg)
        if not full_path:
            return
        wp_cfg = WaypointConfig(spacing=self.cfg.waypoint_spacing)
        self.waypoints = extract_waypoints(full_path, wp_cfg)
        self._set_current_goal(env)

    def _set_current_goal(self, env) -> None:
        if not self.cfg.enabled or not self.waypoints or self.index >= len(self.waypoints):
            if self.final_goal is not None:
                env.goal_position = np.array(self.final_goal, dtype=np.int32)
            return
        env.goal_position = np.array(self.waypoints[self.index], dtype=np.int32)

    def maybe_replan(self, env) -> None:
        if not self.cfg.enabled:
            return
        if env.consecutive_collisions < self.cfg.replan_stuck_collisions:
            return
        self.reset_for_episode(env)

    def patch_step_result(self, env, reward: float, done: bool, info: dict) -> tuple[float, bool, dict]:
        """拦截中间航点终止，把其改为非终止并切换下一个航点。"""
        if not self.cfg.enabled or not self.waypoints:
            return reward, done, info
        if not done:
            return reward, done, info
        if info.get("result") != "success":
            return reward, done, info

        pos = tuple(env.agent_position.tolist())
        cur_goal = tuple(env.goal_position.tolist())
        reach = np.linalg.norm(np.array(pos, dtype=np.float32) - np.array(cur_goal, dtype=np.float32))
        if reach > self.cfg.waypoint_reach_radius:
            return reward, done, info

        is_final = self.final_goal is not None and cur_goal == self.final_goal
        if is_final:
            return reward, done, info

        reward = reward - env.config.goal_reward + env.config.waypoint_reached_reward + self.cfg.waypoint_reward
        done = False
        info = dict(info)
        info["result"] = "waypoint"
        self.index += 1
        self._set_current_goal(env)
        return reward, done, info

