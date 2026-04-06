from __future__ import annotations

from planner.visibility_astar import PathResult, VisibilityAwareAStarPlanner


class WeightedVisibilityAStarPlanner(VisibilityAwareAStarPlanner):
    """Weighted sum A* variant: cost = path_length + visible_weight * visible_path_length."""

    def __init__(self, env, visible_weight: float = 6.0) -> None:
        super().__init__(env, visible_weight=visible_weight)


__all__ = ["WeightedVisibilityAStarPlanner", "PathResult"]
