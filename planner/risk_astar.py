from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import sqrt

import numpy as np

from env.battlefield_env import BattlefieldEnv


@dataclass(frozen=True)
class PathResult:
    path: list[tuple[int, int]]
    total_cost: float
    path_length: float
    cumulative_risk: float
    steps: int
    success: bool


class RiskAStarPlanner:
    def __init__(self, env: BattlefieldEnv, risk_weight: float = 6.0) -> None:
        self.env = env
        self.risk_weight = risk_weight
        self.moves = BattlefieldEnv.ACTIONS

    def plan(
        self,
        start: tuple[int, int] | None = None,
        goal: tuple[int, int] | None = None,
    ) -> PathResult:
        start = start or tuple(self.env.agent_position.tolist())
        goal = goal or tuple(self.env.goal_position.tolist())

        frontier: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(frontier, (0.0, start))

        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        g_cost: dict[tuple[int, int], float] = {start: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                return self._build_result(came_from, g_cost[goal], current)

            for dx, dy in self.moves:
                neighbor = (current[0] + dx, current[1] + dy)
                if self._is_blocked(neighbor):
                    continue

                move_cost = sqrt(2.0) if abs(dx) + abs(dy) == 2 else 1.0
                risk_cost = self.risk_weight * float(self.env.risk_map[neighbor])
                tentative_cost = g_cost[current] + move_cost + risk_cost

                if neighbor not in g_cost or tentative_cost < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_cost
                    priority = tentative_cost + self._heuristic(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        return PathResult(
            path=[start],
            total_cost=float("inf"),
            path_length=0.0,
            cumulative_risk=0.0,
            steps=0,
            success=False,
        )

    def _build_result(
        self,
        came_from: dict[tuple[int, int], tuple[int, int] | None],
        total_cost: float,
        goal: tuple[int, int],
    ) -> PathResult:
        path: list[tuple[int, int]] = []
        current: tuple[int, int] | None = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()

        path_length = 0.0
        cumulative_risk = 0.0
        for idx, cell in enumerate(path):
            cumulative_risk += float(self.env.risk_map[cell])
            if idx == 0:
                continue
            prev = path[idx - 1]
            dx = abs(cell[0] - prev[0])
            dy = abs(cell[1] - prev[1])
            path_length += sqrt(2.0) if dx + dy == 2 else 1.0

        return PathResult(
            path=path,
            total_cost=total_cost,
            path_length=path_length,
            cumulative_risk=cumulative_risk,
            steps=max(0, len(path) - 1),
            success=True,
        )

    def _heuristic(self, node: tuple[int, int], goal: tuple[int, int]) -> float:
        dx = abs(goal[0] - node[0])
        dy = abs(goal[1] - node[1])
        diagonal = min(dx, dy)
        straight = max(dx, dy) - diagonal
        return diagonal * sqrt(2.0) + straight

    def _is_blocked(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        if x < 0 or y < 0 or x >= self.env.grid_size or y >= self.env.grid_size:
            return True
        return bool(self.env.occupancy_map[cell] > 0)
