from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Transition:
    local_map: np.ndarray
    global_features: np.ndarray
    action: int
    reward: float
    next_local_map: np.ndarray
    next_global_features: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        local_map: np.ndarray,
        global_features: np.ndarray,
        action: int,
        reward: float,
        next_local_map: np.ndarray,
        next_global_features: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            Transition(
                local_map=local_map.copy(),
                global_features=global_features.copy(),
                action=action,
                reward=reward,
                next_local_map=next_local_map.copy(),
                next_global_features=next_global_features.copy(),
                done=float(done),
            )
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        return {
            "local_map": np.stack([item.local_map for item in batch], axis=0).astype(np.float32),
            "global_features": np.stack([item.global_features for item in batch], axis=0).astype(np.float32),
            "action": np.array([item.action for item in batch], dtype=np.int64),
            "reward": np.array([item.reward for item in batch], dtype=np.float32),
            "next_local_map": np.stack([item.next_local_map for item in batch], axis=0).astype(np.float32),
            "next_global_features": np.stack([item.next_global_features for item in batch], axis=0).astype(np.float32),
            "done": np.array([item.done for item in batch], dtype=np.float32),
        }
