"""向量化环境包装器：N 个独立环境并行 rollout，批量前向传播加速训练。

每个环境从场景池独立采样，共享同一网络进行批量推理。
"""

from __future__ import annotations

import numpy as np

from config import EnvConfig
from env.battlefield_env import BattlefieldEnv


class VectorizedEnv:
    """N 个并行环境，批量前向传播加速 PPO rollout。

    用法:
        vec_env = VectorizedEnv(env_config, pool, num_envs=8)
        obs_batch = vec_env.get_observations()       # (N, 7, 50, 50)
        masks = vec_env.get_action_masks()            # (N, 8)
        # ... 网络批量推理 ...
        next_obs, rewards, dones, infos = vec_env.step(actions)  # actions: (N,)
    """

    def __init__(
        self,
        env_config: EnvConfig,
        pool: dict,
        num_envs: int = 8,
        seed: int = 42,
    ) -> None:
        self.num_envs = num_envs
        self.pool = pool
        self.n_scenes = len(pool["starts"])
        self.rng = np.random.default_rng(seed)

        self._obs_shape = (7, env_config.grid_size, env_config.grid_size)
        self._observations = np.zeros((num_envs, *self._obs_shape), dtype=np.float32)

        self.envs: list[BattlefieldEnv] = []
        for i in range(num_envs):
            env = BattlefieldEnv(env_config)
            # 清理全图模式残留，使用场景池
            env.full_terrain = None
            env.full_visibility_maps = []
            env.enemy_pool = []
            env.current_progress_weight = env_config.progress_weight
            self.envs.append(env)
            self._observations[i] = self._reset_env(env, i)

    def _reset_env(self, env: BattlefieldEnv, env_idx: int = 0) -> np.ndarray:
        """从场景池采样重置单个环境。"""
        idx = int(self.rng.integers(0, self.n_scenes))
        env.height_map = self.pool["heights"][idx].copy()
        env.window_tag_map = self.pool["tags"][idx].copy()
        env.start_position = self.pool["starts"][idx].copy()
        env.goal_position = self.pool["goals"][idx].copy()

        env.visibility_map = np.zeros(
            (env.grid_size, env.grid_size), dtype=np.float32,
        )
        env.cover_map = np.ones(
            (env.grid_size, env.grid_size), dtype=np.float32,
        )
        env.occupancy_map = (
            env.height_map.astype(np.float32)
            / max(1.0, float(env.height_levels))
        )
        env.window_offset = (0, 0)
        env.enemy_position = np.array([-1, -1, 0], dtype=np.float32)
        env.current_scenario_mode = "full_map"
        env.full_terrain = None

        env.agent_position = env.start_position.copy()
        env.steps = 0
        env.consecutive_collisions = 0
        env.total_collisions = 0
        env.current_progress_weight = env.config.progress_weight
        return env._get_observation()

    def get_observations(self) -> np.ndarray:
        """返回所有环境的观测 (N, 7, H, W)."""
        return self._observations

    def get_action_masks(self) -> np.ndarray:
        """返回所有环境的动作掩码 (N, 8)."""
        masks = np.zeros((self.num_envs, 8), dtype=bool)
        for i, env in enumerate(self.envs):
            masks[i] = env.get_action_mask()
        return masks

    def step(
        self, actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """对所有环境执行一步。

        Args:
            actions: (N,) int, 每个环境的原始空间动作索引

        Returns:
            next_obs: (N, 7, H, W)
            rewards: (N,) float32
            dones: (N,) bool
            infos: list[dict] 长度 N
        """
        next_obs = np.zeros_like(self._observations)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = [{} for _ in range(self.num_envs)]

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, info = env.step(int(action))
            next_obs[i] = obs
            rewards[i] = float(reward)
            dones[i] = done
            infos[i] = info

            if done:
                next_obs[i] = self._reset_env(env, i)

        self._observations = next_obs
        return next_obs, rewards, dones, infos

    def set_progress_weight(self, weight: float) -> None:
        """统一设置所有环境的进度奖励权重。"""
        for env in self.envs:
            env.current_progress_weight = weight
