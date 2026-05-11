from __future__ import annotations

import numpy as np

from config import EnvConfig
from env.battlefield_env import BattlefieldEnv


class VectorizedEnv:
    def __init__(
        self,
        env_config: EnvConfig,
        pool: dict,
        num_envs: int = 8,
        seed: int = 42,
        allowed_indices: np.ndarray | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.pool = pool
        self.n_scenes = len(pool["starts"])
        self.rng = np.random.default_rng(seed)

        if allowed_indices is None:
            self.allowed_indices = np.arange(self.n_scenes, dtype=np.int32)
        else:
            self.allowed_indices = np.array(allowed_indices, dtype=np.int32)
            if self.allowed_indices.size == 0:
                raise ValueError("allowed_indices 不能为空")

        self.envs: list[BattlefieldEnv] = []
        for i in range(num_envs):
            env = BattlefieldEnv(env_config)
            env.full_terrain = None
            env.full_visibility_maps = []
            env.enemy_pool = []
            env.current_progress_weight = env_config.progress_weight
            self.envs.append(env)
        first_obs = self._reset_env(self.envs[0], 0)
        self._obs_shape = first_obs.shape
        self._observations = np.zeros((num_envs, *self._obs_shape), dtype=np.float32)
        self._observations[0] = first_obs
        for i in range(1, num_envs):
            self._observations[i] = self._reset_env(self.envs[i], i)

    def set_allowed_indices(self, indices: np.ndarray) -> None:
        indices = np.array(indices, dtype=np.int32)
        if indices.size == 0:
            raise ValueError("课程切换后 allowed_indices 不能为空")
        self.allowed_indices = indices

    def _reset_env(
        self, env: BattlefieldEnv, env_idx: int = 0, scene_idx: int | None = None
    ) -> np.ndarray:
        if scene_idx is None:
            pick = int(self.rng.integers(0, len(self.allowed_indices)))
            idx = int(self.allowed_indices[pick])
        else:
            idx = int(scene_idx)

        env.height_map = self.pool["heights"][idx].copy()
        env.window_tag_map = self.pool["tags"][idx].copy()
        env.start_position = self.pool["starts"][idx].copy()
        env.set_goal(self.pool["goals"][idx].copy())

        if "visibility" in self.pool:
            env.visibility_map = self.pool["visibility"][idx].copy().astype(np.float32)
            env.cover_map = 1.0 - env.visibility_map
        else:
            env.visibility_map = np.zeros((env.grid_size, env.grid_size), dtype=np.float32)
            env.cover_map = np.ones((env.grid_size, env.grid_size), dtype=np.float32)

        env.occupancy_map = env.height_map.astype(np.float32) / max(1.0, float(env.height_levels))
        env.window_offset = (0, 0)
        env.enemy_position = np.array([-1, -1, 0], dtype=np.float32)
        env.current_scenario_mode = "full_map"
        env.full_terrain = None

        env.agent_position = env.start_position.copy()
        env.steps = 0
        env.consecutive_collisions = 0
        env.total_collisions = 0
        env.current_progress_weight = env.config.progress_weight
        env._reset_obs_state()
        return env._get_observation()

    def reset_env_to_index(self, env_idx: int, scene_idx: int) -> np.ndarray:
        obs = self._reset_env(self.envs[env_idx], env_idx=env_idx, scene_idx=scene_idx)
        self._observations[env_idx] = obs
        return obs

    def get_observations(self) -> np.ndarray:
        return self._observations

    def get_action_masks(self) -> np.ndarray:
        masks = np.zeros((self.num_envs, 8), dtype=bool)
        for i, env in enumerate(self.envs):
            masks[i] = env.get_action_mask()
        return masks

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
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
        for env in self.envs:
            env.current_progress_weight = weight
