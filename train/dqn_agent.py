from __future__ import annotations

from dataclasses import dataclass, field
import random

import numpy as np
import torch
from torch import nn

from config import ExplorationConfig, TrainingDefaults
from env.battlefield_env import BattlefieldEnv
from models.policy_network import HybridPolicyNetwork
from planner.visibility_astar import VisibilityAwareAStarPlanner
from train.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class TrainingConfig:
    device: str = TrainingDefaults().device
    episodes: int = TrainingDefaults().episodes
    batch_size: int = TrainingDefaults().batch_size
    replay_capacity: int = TrainingDefaults().replay_capacity
    gamma: float = TrainingDefaults().gamma
    learning_rate: float = TrainingDefaults().learning_rate
    target_update_interval: int = TrainingDefaults().target_update_interval
    warmup_steps: int = TrainingDefaults().warmup_steps
    train_frequency: int = TrainingDefaults().train_frequency
    epsilon_start: float = TrainingDefaults().epsilon_start
    epsilon_end: float = TrainingDefaults().epsilon_end
    epsilon_decay_steps: int = TrainingDefaults().epsilon_decay_steps
    eval_interval: int = TrainingDefaults().eval_interval
    save_interval: int = TrainingDefaults().save_interval
    max_gradient_norm: float = TrainingDefaults().max_gradient_norm
    seed: int = TrainingDefaults().seed
    exploration: ExplorationConfig = field(default_factory=lambda: TrainingDefaults().exploration)
    early_stop_enabled: bool = TrainingDefaults().early_stop_enabled
    early_stop_eval_episodes: int = TrainingDefaults().early_stop_eval_episodes
    early_stop_success_rate_threshold: float = TrainingDefaults().early_stop_success_rate_threshold
    early_stop_plateau_patience: int = TrainingDefaults().early_stop_plateau_patience
    early_stop_min_delta: float = TrainingDefaults().early_stop_min_delta


class DoubleDQNAgent:
    def __init__(self, action_dim: int, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        requested_device = self.config.device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.action_dim = action_dim
        self.exploration = self.config.exploration

        self.online_net = HybridPolicyNetwork(action_dim=self.action_dim).to(self.device)
        self.target_net = HybridPolicyNetwork(action_dim=self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(self.config.replay_capacity, self.action_dim)
        self.training_steps = 0
        self.last_loss = 0.0
        self.episode_action_stats = {
            "greedy": 0,
            "heuristic": 0,
            "teacher": 0,
            "random": 0,
        }

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def select_action(self, observation: dict[str, np.ndarray], epsilon: float, env: BattlefieldEnv | None = None, global_step: int = 0) -> int:
        if random.random() < epsilon:
            guided_action, source = self._select_guided_exploration_action(env, global_step)
            if guided_action is not None:
                self.episode_action_stats[source] += 1
                return guided_action
            self.episode_action_stats["random"] += 1
            return random.randrange(self.action_dim)

        self.online_net.eval()
        with torch.no_grad():
            local_map = torch.from_numpy(observation["local_map"]).unsqueeze(0).float().to(self.device)
            global_features = torch.from_numpy(observation["global_features"]).unsqueeze(0).float().to(self.device)
            q_values = self.online_net(local_map, global_features)
        if env is not None:
            valid_actions = env.get_valid_actions()
            q_values = self._mask_invalid_actions(q_values, valid_actions)
        self.episode_action_stats["greedy"] += 1
        return int(torch.argmax(q_values, dim=1).item())

    def _select_guided_exploration_action(self, env: BattlefieldEnv | None, global_step: int) -> tuple[int | None, str]:
        if env is None:
            return None, "random"

        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None, "random"

        teacher_prob = self._anneal_probability(
            self.exploration.teacher_action_prob_start,
            self.exploration.teacher_action_prob_end,
            global_step,
        )
        if self.exploration.teacher_enabled and random.random() < teacher_prob:
            teacher_action = self._teacher_action(env)
            if teacher_action is not None:
                return teacher_action, "teacher"

        heuristic_prob = self._anneal_probability(
            self.exploration.heuristic_subset_prob_start,
            self.exploration.heuristic_subset_prob_end,
            global_step,
        )
        if self.exploration.heuristic_subset_enabled and random.random() < heuristic_prob:
            heuristic_actions = self._heuristic_action_subset(env, valid_actions)
            if heuristic_actions:
                return random.choice(heuristic_actions), "heuristic"

        return random.choice(valid_actions), "random"

    def _teacher_action(self, env: BattlefieldEnv) -> int | None:
        start = tuple(env.agent_position.tolist())
        goal = tuple(env.goal_position.tolist())
        result = VisibilityAwareAStarPlanner(env).plan(start=start, goal=goal)
        if not result.success or len(result.path) < 2:
            return None

        next_cell = result.path[1]
        move = (next_cell[0] - start[0], next_cell[1] - start[1])
        for action_idx, action_move in enumerate(BattlefieldEnv.ACTIONS):
            if action_move == move:
                return action_idx
        return None

    def _heuristic_action_subset(self, env: BattlefieldEnv, valid_actions: list[int]) -> list[int]:
        current = env.agent_position
        goal = env.goal_position
        dx = int(goal[0] - current[0])
        dy = int(goal[1] - current[1])

        preferred: set[int] = set()
        for action_idx in valid_actions:
            move_x, move_y = BattlefieldEnv.ACTIONS[action_idx]
            score_x = dx * move_x
            score_y = dy * move_y
            if dx == 0 and move_x == 0:
                score_x = 1
            if dy == 0 and move_y == 0:
                score_y = 1
            if score_x >= 0 and score_y >= 0 and (score_x > 0 or score_y > 0 or (dx == 0 and dy == 0)):
                preferred.add(action_idx)

        if preferred:
            return [action for action in valid_actions if action in preferred]
        return valid_actions

    def _anneal_probability(self, start: float, end: float, global_step: int) -> float:
        if global_step >= self.config.epsilon_decay_steps:
            return end
        ratio = global_step / max(1, self.config.epsilon_decay_steps)
        return start + ratio * (end - start)

    def reset_episode_stats(self) -> None:
        for key in self.episode_action_stats:
            self.episode_action_stats[key] = 0

    def get_episode_stats(self) -> dict[str, int]:
        return dict(self.episode_action_stats)

    def store_transition(
        self,
        observation: dict[str, np.ndarray],
        action: int,
        reward: float,
        next_observation: dict[str, np.ndarray],
        done: bool,
        next_valid_actions: list[int] | None = None,
    ) -> None:
        self.replay_buffer.add(
            local_map=observation["local_map"],
            global_features=observation["global_features"],
            action=action,
            reward=reward,
            next_local_map=next_observation["local_map"],
            next_global_features=next_observation["global_features"],
            done=done,
            next_valid_actions=next_valid_actions,
        )

    def can_train(self, batch_size: int) -> bool:
        return len(self.replay_buffer) >= batch_size

    def train_step(self, batch_size: int) -> float:
        batch = self.replay_buffer.sample(batch_size)
        local_map = torch.from_numpy(batch["local_map"]).float().to(self.device)
        global_features = torch.from_numpy(batch["global_features"]).float().to(self.device)
        actions = torch.from_numpy(batch["action"]).long().to(self.device)
        rewards = torch.from_numpy(batch["reward"]).float().to(self.device)
        next_local_map = torch.from_numpy(batch["next_local_map"]).float().to(self.device)
        next_global_features = torch.from_numpy(batch["next_global_features"]).float().to(self.device)
        dones = torch.from_numpy(batch["done"]).float().to(self.device)

        self.online_net.train()
        current_q = self.online_net(local_map, global_features).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_online_q = self.online_net(next_local_map, next_global_features)
            if "next_valid_action_mask" in batch:
                next_online_q = self._mask_invalid_actions(next_online_q, batch["next_valid_action_mask"])
            next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)
            next_target_q = self.target_net(next_local_map, next_global_features).gather(1, next_actions).squeeze(1)
            td_target = rewards + self.config.gamma * next_target_q * (1.0 - dones)

        loss = self.loss_fn(current_q, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.max_gradient_norm)
        self.optimizer.step()

        self.training_steps += 1
        self.last_loss = float(loss.item())
        if self.training_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return self.last_loss

    def _mask_invalid_actions(
        self,
        q_values: torch.Tensor,
        valid_actions: list[int] | np.ndarray,
    ) -> torch.Tensor:
        if isinstance(valid_actions, np.ndarray) and valid_actions.ndim == 2:
            mask = torch.from_numpy(valid_actions).to(q_values.device)
            return q_values.masked_fill(mask <= 0.0, float("-inf"))

        mask = torch.full((self.action_dim,), float("-inf"), device=q_values.device)
        if valid_actions:
            mask[valid_actions] = 0.0
        return q_values + mask

    def current_epsilon(self, global_step: int) -> float:
        if global_step >= self.config.epsilon_decay_steps:
            return self.config.epsilon_end
        decay_ratio = global_step / max(1, self.config.epsilon_decay_steps)
        return self.config.epsilon_start + decay_ratio * (self.config.epsilon_end - self.config.epsilon_start)

    def save(self, path: str) -> None:
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "last_loss": self.last_loss,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_steps = int(checkpoint.get("training_steps", 0))
        self.last_loss = float(checkpoint.get("last_loss", 0.0))
