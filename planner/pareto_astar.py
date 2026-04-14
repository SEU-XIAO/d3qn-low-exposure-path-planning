from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import sqrt

from env.battlefield_env import BattlefieldEnv
from planner.visibility_astar import PathResult


@dataclass(frozen=True)
class ParetoFrontResult:
    paths: list[PathResult]


@dataclass
class _Label:
    node: tuple[int, int]
    g_len: float
    g_vis: float
    parent: int | None


class ParetoVisibilityAStarPlanner:
    def __init__(self, env: BattlefieldEnv, queue_visible_weight: float = 1.0) -> None:
        self.env = env
        self.moves = BattlefieldEnv.ACTIONS
        self.queue_visible_weight = queue_visible_weight

    def plan(
        self,
        start: tuple[int, int] | None = None,
        goal: tuple[int, int] | None = None,
        max_solutions: int | None = None,
        max_expansions: int | None = None,
        progress_interval: int = 0,
    ) -> ParetoFrontResult:
        start = start or tuple(self.env.agent_position.tolist())
        goal = goal or tuple(self.env.goal_position.tolist())

        labels: list[_Label] = []
        node_labels: dict[tuple[int, int], list[int]] = {}
        goal_labels: list[int] = []
        open_list: list[tuple[float, int, int]] = []
        counter = 0
        expansions = 0

        def add_label(node: tuple[int, int], g_len: float, g_vis: float, parent: int | None) -> int:
            label = _Label(node=node, g_len=g_len, g_vis=g_vis, parent=parent)
            labels.append(label)
            idx = len(labels) - 1
            node_labels.setdefault(node, []).append(idx)
            return idx

        def dominates(a_len: float, a_vis: float, b_len: float, b_vis: float) -> bool:
            return (a_len <= b_len and a_vis <= b_vis) and (a_len < b_len or a_vis < b_vis)

        def is_dominated(g_len: float, g_vis: float, indices: list[int]) -> bool:
            for idx in indices:
                lab = labels[idx]
                if dominates(lab.g_len, lab.g_vis, g_len, g_vis):
                    return True
            return False

        def prune_dominated(g_len: float, g_vis: float, indices: list[int]) -> list[int]:
            kept: list[int] = []
            for idx in indices:
                lab = labels[idx]
                if dominates(g_len, g_vis, lab.g_len, lab.g_vis):
                    continue
                kept.append(idx)
            return kept

        def push_open(label_idx: int) -> None:
            nonlocal counter
            lab = labels[label_idx]
            priority = lab.g_len + self.queue_visible_weight * lab.g_vis + self._heuristic(lab.node, goal)
            heapq.heappush(open_list, (priority, counter, label_idx))
            counter += 1

        start_idx = add_label(start, 0.0, 0.0, None)
        push_open(start_idx)

        while open_list:
            _, _, label_idx = heapq.heappop(open_list)
            lab = labels[label_idx]
            active_indices = node_labels.get(lab.node, [])
            if label_idx not in active_indices:
                continue

            expansions += 1
            if progress_interval > 0 and expansions % progress_interval == 0:
                print(
                    f"[Pareto] expanded={expansions} open={len(open_list)} "
                    f"labels={len(labels)} goals={len(goal_labels)}"
                )
            if max_expansions is not None and expansions >= max_expansions:
                print(
                    f"[Pareto] reached max_expansions={max_expansions}, "
                    f"returning current front (goals={len(goal_labels)})"
                )
                break

            # If goal already dominates this label, no need to expand.
            if goal_labels and is_dominated(lab.g_len, lab.g_vis, goal_labels):
                continue

            if lab.node == goal:
                if not is_dominated(lab.g_len, lab.g_vis, goal_labels):
                    goal_labels = prune_dominated(lab.g_len, lab.g_vis, goal_labels)
                    goal_labels.append(label_idx)
                    if max_solutions is not None and len(goal_labels) >= max_solutions:
                        break
                continue

            for dx, dy in self.moves:
                neighbor = (lab.node[0] + dx, lab.node[1] + dy)
                if self._is_blocked(neighbor):
                    continue

                move_cost = sqrt(2.0) if abs(dx) + abs(dy) == 2 else 1.0
                visibility = float(self.env.visibility_map[neighbor])
                new_len = lab.g_len + move_cost
                new_vis = lab.g_vis + move_cost * visibility

                existing = node_labels.get(neighbor, [])
                if existing and is_dominated(new_len, new_vis, existing):
                    continue

                node_labels[neighbor] = prune_dominated(new_len, new_vis, existing)
                new_idx = add_label(neighbor, new_len, new_vis, label_idx)
                push_open(new_idx)

        return ParetoFrontResult(paths=[self._build_result(labels, idx, goal) for idx in goal_labels])

    def _build_result(self, labels: list[_Label], label_idx: int, goal: tuple[int, int]) -> PathResult:
        path: list[tuple[int, int]] = []
        current = label_idx
        while current is not None:
            path.append(labels[current].node)
            current = labels[current].parent
        path.reverse()

        g_len = labels[label_idx].g_len
        g_vis = labels[label_idx].g_vis
        hidden_len = max(0.0, g_len - g_vis)
        hidden_ratio = hidden_len / max(g_len, 1e-6) if g_len > 0 else 1.0

        return PathResult(
            path=path,
            total_cost=g_len + g_vis,
            path_length=g_len,
            visible_path_length=g_vis,
            hidden_path_length=hidden_len,
            hidden_ratio=hidden_ratio,
            steps=max(0, len(path) - 1),
            success=path[-1] == goal,
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
        if x == int(self.env.enemy_position[0]) and y == int(self.env.enemy_position[1]):
            return True
        return bool(self.env.occupancy_map[cell] > 0)
