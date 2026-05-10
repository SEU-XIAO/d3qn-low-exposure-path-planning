"""RolloutBuffer：存储 on-policy 轨迹并计算 GAE。

数据增强只在 rollout 时施加一次，buffer 存储增强后的 (obs, mask, log_prob)，
PPO update 时原样取出，不再重新增强——保证 old_log_prob 与 new_log_prob 可比。
"""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """固定大小的循环 buffer，存 rollout 数据 + GAE 计算 + mini-batch 采样。"""

    def __init__(self, buffer_size: int, obs_shape: tuple[int, ...], device: torch.device):
        self.buffer_size = buffer_size
        self.device = device

        # —— 预分配存储 ——
        self.observations = torch.zeros(buffer_size, *obs_shape, device=device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        self.masks = torch.zeros(buffer_size, 8, dtype=torch.bool, device=device)

        # GAE 结果
        self.advantages = torch.zeros(buffer_size, device=device)
        self.returns = torch.zeros(buffer_size, device=device)

        self.ptr = 0

    # ── 写入 ────────────────────────────────────────────

    def add(
        self,
        obs: np.ndarray | torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
        mask: np.ndarray | torch.Tensor,
    ) -> None:
        """存入一个 transition。obs/mask 可以是 numpy，内部转 tensor 并搬运到 device。"""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        self.observations[self.ptr] = obs.to(self.device)
        self.actions[self.ptr] = action.to(self.device)
        self.log_probs[self.ptr] = log_prob.to(self.device)
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value.to(self.device)
        self.dones[self.ptr] = done
        self.masks[self.ptr] = mask.to(self.device)
        self.ptr += 1

    # ── GAE ────────────────────────────────────────────

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """在收集完 buffer_size 条数据后调用，原地计算 advantages 和 returns。

        Args:
            last_value: V(s_{T+1})，若最后一步 done 则传 0。
            gamma: 折扣因子。
            gae_lambda: GAE λ。

        δ_t = r_t + γ·(1-done_t)·V(s_{t+1}) - V(s_t)
        A_t = δ_t + γλ·(1-done_t)·A_{t+1}
        """
        gae = 0.0
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_value = last_value
            else:
                next_value = float(self.values[t + 1])

            not_done = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def compute_gae_parallel(
        self,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        num_envs: int,
    ) -> None:
        """并行环境 GAE：数据按 env 交织存储，按 stride=num_envs 独立计算。

        buffer 存储顺序: [e0_t0, e1_t0, ..., eN_t0, e0_t1, e1_t1, ..., eN_t1, ...]
        每个 env 的步数 = buffer_size / num_envs。

        Args:
            last_values: (num_envs,) V(s_{T+1})，若某 env 最后一步 done 则传 0。
            gamma: 折扣因子。
            gae_lambda: GAE λ。
            num_envs: 并行环境数。
        """
        steps_per_env = self.buffer_size // num_envs
        for env_idx in range(num_envs):
            gae = 0.0
            for step in reversed(range(steps_per_env)):
                t = env_idx + step * num_envs
                if step == steps_per_env - 1:
                    next_value = float(last_values[env_idx])
                else:
                    next_t = env_idx + (step + 1) * num_envs
                    next_value = float(self.values[next_t])

                not_done = 1.0 - float(self.dones[t])
                delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
                gae = delta + gamma * gae_lambda * not_done * gae
                self.advantages[t] = gae

            for step in range(steps_per_env):
                t = env_idx + step * num_envs
                self.returns[t] = self.advantages[t] + self.values[t]

    def normalize_advantages(self) -> None:
        """对 advantages 做 batch 内标准化（均值 0，标准差 1）。"""
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    # ── 采样 ────────────────────────────────────────────

    def sample(self, batch_size: int):
        """随机打乱索引，按 batch_size 切分，逐个 yield 索引 slice。"""
        indices = torch.randperm(self.buffer_size, device=self.device)
        for start in range(0, self.buffer_size, batch_size):
            yield indices[start : start + batch_size]

    # ── 重置 ────────────────────────────────────────────

    def clear(self) -> None:
        """重置指针（无需清零数据，下次写入会覆盖）。"""
        self.ptr = 0

    def __len__(self) -> int:
        return self.buffer_size
