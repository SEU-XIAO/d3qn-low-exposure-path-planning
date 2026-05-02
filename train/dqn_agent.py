from __future__ import annotations

from dataclasses import dataclass, field
import random

import numpy as np
import torch
from torch import nn

from config import ExplorationConfig, ModelConfig, TrainingDefaults, WaypointConfig
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
    full_eval_interval: int = TrainingDefaults().full_eval_interval
    save_interval: int = TrainingDefaults().save_interval
    max_gradient_norm: float = TrainingDefaults().max_gradient_norm
    n_step: int = TrainingDefaults().n_step
    bc_reg_weight_start: float = TrainingDefaults().bc_reg_weight_start
    bc_reg_weight_end: float = TrainingDefaults().bc_reg_weight_end
    per_alpha: float = TrainingDefaults().per_alpha
    per_beta_start: float = TrainingDefaults().per_beta_start
    per_beta_end: float = TrainingDefaults().per_beta_end
    per_epsilon: float = TrainingDefaults().per_epsilon
    her_relabel_count: int = TrainingDefaults().her_relabel_count
    waypoint: WaypointConfig = field(default_factory=lambda: TrainingDefaults().waypoint)
    seed: int = TrainingDefaults().seed
    exploration: ExplorationConfig = field(default_factory=lambda: TrainingDefaults().exploration)
    use_lstm: bool = TrainingDefaults().use_lstm
    lstm_sequence_length: int = TrainingDefaults().lstm_sequence_length
    curriculum_enabled: bool = TrainingDefaults().curriculum_enabled
    curriculum_success_threshold: float = TrainingDefaults().curriculum_success_threshold
    curriculum_window: int = TrainingDefaults().curriculum_window
    curriculum_patience: int = TrainingDefaults().curriculum_patience
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

        wp = self.config.waypoint
        lc = 6 if wp.enabled else 5
        gd = 12 if wp.enabled else 8
        model_config = ModelConfig(
            local_channels=lc, global_feature_dim=gd,
            use_lstm=self.config.use_lstm,
        )

        self.use_lstm = self.config.use_lstm
        self.lstm_seq_len = self.config.lstm_sequence_length

        self.online_net = HybridPolicyNetwork(action_dim=self.action_dim, config=model_config).to(self.device)
        self.target_net = HybridPolicyNetwork(action_dim=self.action_dim, config=model_config).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.bc_net = HybridPolicyNetwork(action_dim=self.action_dim, config=model_config).to(self.device)
        self.bc_net.load_state_dict(self.online_net.state_dict())
        for p in self.bc_net.parameters():
            p.requires_grad = False
        self.bc_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(
            self.config.replay_capacity, self.action_dim,
            local_map_channels=lc, global_feature_dim=gd,
        )
        self.training_steps = 0
        self.last_loss = 0.0
        self.episode_action_stats = {
            "greedy": 0,
            "heuristic": 0,
            "teacher": 0,
            "random": 0,
        }
        self._hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def select_action(self, observation: dict[str, np.ndarray], epsilon: float, env: BattlefieldEnv | None = None, global_step: int = 0) -> int:
        self.online_net.eval()
        with torch.no_grad():
            local_map = torch.from_numpy(observation["local_map"]).unsqueeze(0).float().to(self.device)
            global_features = torch.from_numpy(observation["global_features"]).unsqueeze(0).float().to(self.device)
            if self.use_lstm:
                q_values, self._hidden_state = self.online_net(local_map, global_features, self._hidden_state)
            else:
                q_values = self.online_net(local_map, global_features)

        if random.random() < epsilon:
            guided_action, source = self._select_guided_exploration_action(env, global_step)
            if guided_action is not None:
                self.episode_action_stats[source] += 1
                return guided_action
            self.episode_action_stats["random"] += 1
            return random.randrange(self.action_dim)

        if env is not None:
            valid_actions = env.get_valid_actions()
            q_values = self._mask_invalid_actions(q_values, valid_actions)
        self.episode_action_stats["greedy"] += 1
        return int(torch.argmax(q_values, dim=1).item())

    def select_action_masked(self, observation: dict[str, np.ndarray], env: BattlefieldEnv) -> int:
        self.online_net.eval()
        with torch.no_grad():
            local_map = torch.from_numpy(observation["local_map"]).unsqueeze(0).float().to(self.device)
            global_features = torch.from_numpy(observation["global_features"]).unsqueeze(0).float().to(self.device)
            if self.use_lstm:
                q_values, self._hidden_state = self.online_net(local_map, global_features, self._hidden_state)
            else:
                q_values = self.online_net(local_map, global_features)
        valid_actions = env.get_valid_actions()
        q_values = self._mask_invalid_actions(q_values, valid_actions)
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
            teacher_action = self._teacher_action(env, global_step)
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

    def _teacher_action(self, env: BattlefieldEnv, global_step: int) -> int | None:
        start = tuple(env.agent_position.tolist())
        target = tuple(env.current_subgoal.tolist()) if env.current_subgoal is not None else tuple(env.goal_position.tolist())
        teacher_lambda = self._anneal_probability(
            self.exploration.teacher_lambda_start,
            self.exploration.teacher_lambda_end,
            global_step,
        )
        result = VisibilityAwareAStarPlanner(env, visible_weight=teacher_lambda).plan(start=start, goal=target)
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
        target = env.current_subgoal if env.current_subgoal is not None else env.goal_position
        dx = int(target[0] - current[0])
        dy = int(target[1] - current[1])

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

    def _current_bc_reg_weight(self) -> float:
        decay_steps = max(1, self.config.epsilon_decay_steps // self.config.train_frequency)
        if self.training_steps >= decay_steps:
            return self.config.bc_reg_weight_end
        ratio = self.training_steps / decay_steps
        return self.config.bc_reg_weight_start + ratio * (self.config.bc_reg_weight_end - self.config.bc_reg_weight_start)

    def _current_per_beta(self) -> float:
        decay_steps = max(1, self.config.epsilon_decay_steps // self.config.train_frequency)
        if self.training_steps >= decay_steps:
            return self.config.per_beta_end
        ratio = self.training_steps / decay_steps
        return self.config.per_beta_start + ratio * (self.config.per_beta_end - self.config.per_beta_start)

    def reset_episode_stats(self) -> None:
        for key in self.episode_action_stats:
            self.episode_action_stats[key] = 0
        self._hidden_state = self.online_net.init_hidden(1, self.device)

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
        if self.use_lstm:
            return self._train_step_lstm(batch_size)

        n_step = self.config.n_step
        gamma = self.config.gamma

        per_beta = self._current_per_beta()
        indices, is_weights = self.replay_buffer.sample_per(batch_size, self.config.per_alpha, per_beta)
        if len(indices) == 0:
            return 0.0
        n_step_returns, nth_local, nth_global, nth_done, nth_mask = \
            self.replay_buffer.get_n_step_data(indices, n_step, gamma)

        # 加载当前状态和动作
        local_map = torch.from_numpy(
            self.replay_buffer.local_maps[indices].astype(np.float32) / 255.0,
        ).float().to(self.device)
        global_features = torch.from_numpy(
            self.replay_buffer.global_features[indices].astype(np.float32),
        ).float().to(self.device)
        actions = torch.from_numpy(
            self.replay_buffer.actions[indices],
        ).long().to(self.device)
        is_weights_t = torch.from_numpy(is_weights).float().to(self.device)

        self.online_net.train()
        online_q = self.online_net(local_map, global_features)
        current_q = online_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            n_step_ret_t = torch.from_numpy(n_step_returns).float().to(self.device)
            nth_local_t = torch.from_numpy(nth_local.astype(np.float32) / 255.0).float().to(self.device)
            nth_global_t = torch.from_numpy(nth_global.astype(np.float32)).float().to(self.device)
            nth_done_t = torch.from_numpy(nth_done).float().to(self.device)
            nth_mask_t = torch.from_numpy(nth_mask).float().to(self.device)

            td_target = n_step_ret_t.clone()
            bootstrap_mask = nth_done_t < 0.5
            if bootstrap_mask.any():
                nth_local_valid = nth_local_t[bootstrap_mask]
                nth_global_valid = nth_global_t[bootstrap_mask]
                nth_mask_valid = nth_mask_t[bootstrap_mask]

                nth_online_q = self.online_net(nth_local_valid, nth_global_valid)
                nth_online_q = self._mask_invalid_actions(nth_online_q, nth_mask_valid)
                nth_actions = torch.argmax(nth_online_q, dim=1, keepdim=True)

                nth_target_q_full = self.target_net(nth_local_valid, nth_global_valid)
                nth_target_q_full = self._mask_invalid_actions(nth_target_q_full, nth_mask_valid)
                nth_target_q = nth_target_q_full.gather(1, nth_actions).squeeze(1)

                td_target[bootstrap_mask] += (gamma ** n_step) * nth_target_q

            bc_q = self.bc_net(local_map, global_features)
            bc_expert = torch.argmax(bc_q, dim=1)

        # TD 误差（用于 PER 优先级更新）
        td_errors = (current_q - td_target).abs().detach().cpu().numpy()

        # 带 IS 权重的 TD 损失 + 衰减 BC 正则化
        td_loss = self.loss_fn(current_q, td_target)
        weighted_td_loss = (is_weights_t * td_loss).mean()
        bc_reg_weight = self._current_bc_reg_weight()
        ce_loss = nn.functional.cross_entropy(online_q, bc_expert)
        loss = weighted_td_loss + bc_reg_weight * ce_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.max_gradient_norm)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors, self.config.per_epsilon)

        self.training_steps += 1
        self.last_loss = float(td_loss.mean().item())
        if self.training_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return self.last_loss

    def _train_step_lstm(self, batch_size: int) -> float:
        """LSTM 序列训练：采样连续序列，用 forward_sequence 计算 Q 值。"""
        seq_data = self.replay_buffer.sample_sequences(batch_size, self.lstm_seq_len)
        if seq_data is None:
            return 0.0

        gamma = self.config.gamma
        last_indices = seq_data["last_indices"]

        local_seq = torch.from_numpy(seq_data["local_map"]).float().to(self.device)
        global_seq = torch.from_numpy(seq_data["global_features"]).float().to(self.device)
        actions = torch.from_numpy(seq_data["action"]).long().to(self.device)
        rewards = torch.from_numpy(seq_data["reward"]).float().to(self.device)
        dones = torch.from_numpy(seq_data["done"]).float().to(self.device)
        next_local_seq = torch.from_numpy(seq_data["next_local_map"]).float().to(self.device)
        next_global_seq = torch.from_numpy(seq_data["next_global_features"]).float().to(self.device)
        next_masks = torch.from_numpy(seq_data["next_valid_action_mask"]).float().to(self.device)

        B, T = actions.shape

        self.online_net.train()
        online_q = self.online_net.forward_sequence(local_seq, global_seq)  # (B, T, A)
        current_q = online_q.gather(2, actions.unsqueeze(2)).squeeze(2)  # (B, T)

        with torch.no_grad():
            # Double DQN: online 选动作, target 评估
            next_q_online = self.online_net.forward_sequence(next_local_seq, next_global_seq)
            next_q_online_masked = self._mask_invalid_actions(next_q_online, next_masks)
            next_actions = torch.argmax(next_q_online_masked, dim=2, keepdim=True)

            next_q_target = self.target_net.forward_sequence(next_local_seq, next_global_seq)
            next_q_target_masked = self._mask_invalid_actions(next_q_target, next_masks)
            next_q = next_q_target_masked.gather(2, next_actions).squeeze(2)  # (B, T)

            td_target = rewards + gamma * next_q * (1.0 - dones)

            bc_q = self.bc_net.forward_sequence(local_seq, global_seq)
            bc_expert = torch.argmax(bc_q, dim=2)  # (B, T)

        td_errors = (current_q - td_target).abs()
        td_loss = self.loss_fn(current_q, td_target)
        bc_reg_weight = self._current_bc_reg_weight()
        ce_loss = nn.functional.cross_entropy(
            online_q.view(B * T, self.action_dim),
            bc_expert.view(B * T),
        )
        loss = td_loss + bc_reg_weight * ce_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.max_gradient_norm)
        self.optimizer.step()

        # PER 更新：用平均 TD 误差更新序列末尾索引
        mean_td = td_errors.mean(dim=1).detach().cpu().numpy()
        self.replay_buffer.update_priorities(last_indices, mean_td, self.config.per_epsilon)

        self.training_steps += 1
        self.last_loss = float(td_loss.mean().item())
        if self.training_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return self.last_loss

    def _mask_invalid_actions(
        self,
        q_values: torch.Tensor,
        valid_actions: list[int] | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(valid_actions, np.ndarray):
            mask = torch.from_numpy(valid_actions).to(q_values.device)
            return q_values.masked_fill(mask <= 0.0, float("-inf"))
        if isinstance(valid_actions, torch.Tensor):
            return q_values.masked_fill(valid_actions <= 0.0, float("-inf"))

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
        online_missing, online_unexpected = self.online_net.load_state_dict(checkpoint["online_state_dict"], strict=False)
        self.target_net.load_state_dict(checkpoint["target_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass  # optimizer state shape可能不匹配（如LSTM参数新增），跳过
        self.training_steps = int(checkpoint.get("training_steps", 0))
        self.last_loss = float(checkpoint.get("last_loss", 0.0))
        self.bc_net.load_state_dict(checkpoint["online_state_dict"], strict=False)
        if online_missing:
            from warnings import warn
            warn(f"加载 checkpoint 时缺少键（随机初始化）: {online_missing}")
        if online_unexpected:
            from warnings import warn
            warn(f"加载 checkpoint 时多余键（已忽略）: {online_unexpected}")
