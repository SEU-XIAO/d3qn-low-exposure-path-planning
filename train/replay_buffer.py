from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """环形缓冲区，支持 n-step TD 采样、episode 边界追踪和优先经验回放 (PER)。

    local_map 用 uint8 存储节省 4× 内存（输入值域 [0,1]）。
    """

    def __init__(self, capacity: int, action_dim: int,
                 local_map_channels: int = 7, global_feature_dim: int = 12) -> None:
        self.capacity = capacity
        self.action_dim = action_dim
        self._local_map_size = (local_map_channels, 50, 50)
        self._global_dim = global_feature_dim

        self.local_maps = np.zeros((capacity, local_map_channels, 50, 50), dtype=np.uint8)
        self.global_features = np.zeros((capacity, self._global_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_local_maps = np.zeros((capacity, local_map_channels, 50, 50), dtype=np.uint8)
        self.next_global_features = np.zeros((capacity, self._global_dim), dtype=np.float32)
        self.next_valid_masks = np.zeros((capacity, action_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.episode_starts = np.zeros(capacity, dtype=np.bool_)
        self.priorities = np.zeros(capacity, dtype=np.float32)

        self._pos = 0
        self._size = 0
        self._pending_episode_start = True
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        local_map: np.ndarray,
        global_features: np.ndarray,
        action: int,
        reward: float,
        next_local_map: np.ndarray,
        next_global_features: np.ndarray,
        done: bool,
        next_valid_actions: list[int] | None = None,
    ) -> None:
        idx = self._pos
        self.local_maps[idx] = (local_map * 255.0).clip(0, 255).astype(np.uint8)
        self.global_features[idx] = global_features.astype(np.float32)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_local_maps[idx] = (next_local_map * 255.0).clip(0, 255).astype(np.uint8)
        self.next_global_features[idx] = next_global_features.astype(np.float32)
        self.dones[idx] = done
        self.episode_starts[idx] = self._pending_episode_start

        mask = np.zeros(self.action_dim, dtype=np.float32)
        if next_valid_actions:
            mask[next_valid_actions] = 1.0
        self.next_valid_masks[idx] = mask

        self.priorities[idx] = self._max_priority

        self._pending_episode_start = done
        self._pos = (idx + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample_per(self, batch_size: int, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
        """优先级采样，返回 (indices, is_weights)。"""
        if self._size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        probs = self.priorities[:self._size] ** alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones(self._size) / self._size
        else:
            probs /= probs_sum

        indices = np.random.choice(self._size, size=min(batch_size, self._size), p=probs, replace=False)

        # IS 权重：w_i = (N * P(i))^(-beta)，归一化除以 max 稳定训练
        weights = (self._size * probs[indices]) ** (-beta)
        weights /= max(weights.max(), 1e-8)
        return indices.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, epsilon: float) -> None:
        """根据 TD 误差更新优先级。"""
        new_priorities = np.abs(td_errors) + epsilon
        for i, idx in enumerate(indices):
            self.priorities[idx] = float(new_priorities[i])
        self._max_priority = max(self._max_priority, float(new_priorities.max()))

    def get_n_step_data(
        self,
        indices: np.ndarray,
        n_step: int,
        gamma: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回 (n_step_return, nth_next_local, nth_next_global, nth_done, nth_valid_mask)。

        nth_valid_mask 是第 n 步状态的合法动作掩码。
        """
        n_returns = np.zeros(len(indices), dtype=np.float32)
        nth_local = np.zeros((len(indices), self._local_map_size[0], 50, 50), dtype=np.uint8)
        nth_global = np.zeros((len(indices), self._global_dim), dtype=np.float32)
        nth_done = np.ones(len(indices), dtype=np.float32)
        nth_mask = np.zeros((len(indices), self.action_dim), dtype=np.float32)

        for b, idx in enumerate(indices):
            ret = 0.0
            final_k = 0
            for k in range(n_step):
                i = (idx + k) % self.capacity
                if i >= self._size:
                    break
                ret += (gamma ** k) * float(self.rewards[i])
                final_k = k
                if self.dones[i]:
                    break

            n_returns[b] = float(ret)

            # 若未在窗口内 done，取第 n-1 步的 next state 用于 bootstrapping（= s_{idx+n_step}）
            n_idx = (idx + n_step - 1) % self.capacity
            last_idx = (idx + final_k) % self.capacity
            if not self.dones[last_idx] and n_idx < self._size and not self._is_cross_episode(idx, n_step):
                nth_local[b] = self.next_local_maps[n_idx]
                nth_global[b] = self.next_global_features[n_idx]
                nth_done[b] = 0.0
                nth_mask[b] = self.next_valid_masks[n_idx]

        return n_returns, nth_local, nth_global, nth_done, nth_mask

    def _is_cross_episode(self, start_idx: int, n_step: int) -> bool:
        """检查从 start_idx 开始的 n_step 窗口是否跨越 episode 边界。"""
        for k in range(1, n_step + 1):
            i = (start_idx + k) % self.capacity
            if i >= self._size:
                return True
            if self.episode_starts[i]:
                return True
        return False
