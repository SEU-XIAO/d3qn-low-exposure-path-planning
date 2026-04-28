from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """环形缓冲区，支持 n-step TD 采样和 episode 边界追踪。

    local_map 用 uint8 存储节省 4× 内存（输入值域 [0,1]）。
    """

    def __init__(self, capacity: int, action_dim: int) -> None:
        self.capacity = capacity
        self.action_dim = action_dim
        self._local_map_size = (5, 50, 50)
        self._global_dim = 8

        self.local_maps = np.zeros((capacity, 5, 50, 50), dtype=np.uint8)
        self.global_features = np.zeros((capacity, self._global_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_local_maps = np.zeros((capacity, 5, 50, 50), dtype=np.uint8)
        self.next_global_features = np.zeros((capacity, self._global_dim), dtype=np.float32)
        self.next_valid_masks = np.zeros((capacity, action_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.episode_starts = np.zeros(capacity, dtype=np.bool_)

        self._pos = 0
        self._size = 0
        self._pending_episode_start = True

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

        self._pending_episode_start = done
        self._pos = (idx + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """均匀采样，返回字典。"""
        indices = np.random.randint(0, self._size, size=batch_size)
        payload: dict[str, np.ndarray] = {
            "local_map": self.local_maps[indices].astype(np.float32) / 255.0,
            "global_features": self.global_features[indices].astype(np.float32),
            "action": self.actions[indices].astype(np.int64),
            "reward": self.rewards[indices].astype(np.float32),
            "next_local_map": self.next_local_maps[indices].astype(np.float32) / 255.0,
            "next_global_features": self.next_global_features[indices].astype(np.float32),
            "next_valid_action_mask": self.next_valid_masks[indices].astype(np.float32),
            "done": self.dones[indices].astype(np.float32),
        }
        return payload

    def sample_n_step_indices(self, batch_size: int) -> np.ndarray:
        """采样不含 episode 边界的起始索引（供 n-step TD 使用）。"""
        if self._size < batch_size * 2:
            return np.random.randint(0, max(1, self._size), size=batch_size)

        valid: list[int] = []
        attempts = 0
        while len(valid) < batch_size and attempts < batch_size * 20:
            attempts += 1
            i = int(np.random.randint(0, self._size))
            ok = True
            for k in range(1, 6):  # check up to 5 steps ahead
                j = (i + k) % self.capacity
                if j >= self._size:
                    break
                if self.episode_starts[j]:
                    ok = False
                    break
            if ok:
                valid.append(i)
        if len(valid) < batch_size:
            valid.extend([int(np.random.randint(0, self._size)) for _ in range(batch_size - len(valid))])
        return np.array(valid[:batch_size], dtype=np.int64)

    def get_n_step_returns(
        self,
        indices: np.ndarray,
        n_step: int,
        gamma: float,
    ) -> np.ndarray:
        """计算 n-step 折现回报 R_i^n + gamma^n * 0（bootstrapping 部分由调用方加）。"""
        n_step_returns = np.zeros(len(indices), dtype=np.float32)
        for b, idx in enumerate(indices):
            ret = 0.0
            actual_n = 0
            for k in range(n_step):
                i = (idx + k) % self.capacity
                if i >= self._size:
                    break
                ret += (gamma ** k) * float(self.rewards[i])
                actual_n = k + 1
                if self.dones[i]:
                    break
            n_step_returns[b] = float(ret)
            n_step_returns[b] = (actual_n, float(ret))  # hack: need to return actual_n too
        return n_step_returns

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
        nth_local = np.zeros((len(indices), 5, 50, 50), dtype=np.uint8)
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

            # 若未在窗口内 done，取第 n 步的 next state 用于 bootstrapping
            n_idx = (idx + n_step) % self.capacity
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
