from __future__ import annotations

from planner.visibility_astar import PathResult, VisibilityAwareAStarPlanner


class ScalarizedVisibilityAStarPlanner(VisibilityAwareAStarPlanner):
    """Single-objective A*: J(p)=L(p)+lambda*V(p)."""

    def __init__(self, env, lambda_visibility: float = 6.0) -> None:
        super().__init__(env, visible_weight=lambda_visibility)


class WeightedVisibilityAStarPlanner(ScalarizedVisibilityAStarPlanner):
    """Backward-compatible alias for ScalarizedVisibilityAStarPlanner."""

    def __init__(self, env, visible_weight: float = 6.0) -> None:
        super().__init__(env, lambda_visibility=visible_weight)


__all__ = ["ScalarizedVisibilityAStarPlanner", "WeightedVisibilityAStarPlanner", "PathResult"]
