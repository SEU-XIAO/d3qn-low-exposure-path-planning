"""PPO 训练器：rollout → GAE → PPO update → eval 的完整训练循环。

数据增强仅在 rollout 时施加一次，buffer 存储增强后的数据。
PPO update 时原样取出，不重新增强——保证 old/new log_prob 可比。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EnvConfig
from env.battlefield_env import BattlefieldEnv
from models.policy_network import ActorCriticCNN, random_augment, deaugment_action
from train.ppo_buffer import RolloutBuffer
from train.ppo_config import PPOConfig


class PPOTrainer:
    def __init__(
        self,
        env_config: EnvConfig | None = None,
        ppo_config: PPOConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.env_config = env_config or EnvConfig()
        self.ppo_config = ppo_config or PPOConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.env = BattlefieldEnv(self.env_config)
        self.policy = ActorCriticCNN().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.ppo_config.learning_rate
        )

        self.buffer = RolloutBuffer(
            self.ppo_config.rollout_steps,
            (7, self.env_config.grid_size, self.env_config.grid_size),
            self.device,
        )

        self.global_step = 0
        self.best_eval_rate = 0.0

    # ── 训练主循环 ──────────────────────────────────────

    def train(self, save_dir: str = "checkpoints") -> None:
        cfg = self.ppo_config
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        obs = self.env.reset()
        episode_reward = 0.0
        episode_count = 0
        next_eval_at = cfg.eval_interval

        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f"设备: {self.device}  参数: {n_params:,}  总步数: {cfg.total_steps:,}")
        t_start = time.time()

        while self.global_step < cfg.total_steps:
            # —— 进度权重衰减（课程学习） ——
            progress = self.global_step / cfg.total_steps
            if progress >= cfg.progress_decay_start:
                frac = min(
                    1.0,
                    (progress - cfg.progress_decay_start)
                    / (cfg.progress_decay_end - cfg.progress_decay_start),
                )
                self.env.current_progress_weight = self.env_config.progress_weight * (
                    1.0 - frac
                )
            else:
                self.env.current_progress_weight = self.env_config.progress_weight

            # ========================================================
            #  1. Rollout — 收集 rollout_steps 条 transition
            # ========================================================
            for _ in range(cfg.rollout_steps):
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                mask_np = self.env.get_action_mask()
                mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

                # 数据增强（仅 rollout 时施加，buffer 存增强后的数据）
                aug_obs, aug_mask, aug_params = random_augment(obs_t, mask_t)

                with torch.no_grad():
                    aug_action, log_prob, value, _entropy, _ = self.policy(aug_obs, aug_mask)

                # 动作逆变换：增强空间 → 原始空间
                orig_action = deaugment_action(aug_action, *aug_params)
                next_obs, reward, done, info = self.env.step(orig_action.item())

                self.buffer.add(
                    aug_obs.squeeze(0),
                    aug_action.squeeze(0),
                    log_prob.squeeze(0),
                    reward,
                    value.squeeze(0),
                    done,
                    aug_mask.squeeze(0),
                )

                episode_reward += reward
                self.global_step += 1

                obs = next_obs
                if done:
                    episode_count += 1
                    obs = self.env.reset()

                if self.global_step >= cfg.total_steps:
                    break

            # ========================================================
            #  2. GAE — 计算 advantages 和 returns
            # ========================================================
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                last_value = self.policy.get_value(obs_t).item()
            self.buffer.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)
            self.buffer.normalize_advantages()

            # ========================================================
            #  3. PPO Update — 多 epoch，mini-batch
            # ========================================================
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            n_updates = 0

            for _epoch in range(cfg.ppo_epochs):
                for indices in self.buffer.sample(cfg.minibatch_size):
                    mb_obs = self.buffer.observations[indices]
                    mb_actions = self.buffer.actions[indices]
                    mb_old_log_probs = self.buffer.log_probs[indices]
                    mb_advantages = self.buffer.advantages[indices]
                    mb_returns = self.buffer.returns[indices]
                    mb_masks = self.buffer.masks[indices]

                    new_log_probs, values, entropy = self.policy.evaluate(
                        mb_obs, mb_actions, mb_masks
                    )

                    # PPO clipped objective
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = (
                        torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon)
                        * mb_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(values, mb_returns)

                    loss = (
                        policy_loss
                        + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy.mean()
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), cfg.max_grad_norm
                    )
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    n_updates += 1

            self.buffer.clear()

            # —— 日志 ——
            elapsed = time.time() - t_start
            avg_pl = total_policy_loss / max(n_updates, 1)
            avg_vl = total_value_loss / max(n_updates, 1)
            avg_ent = total_entropy / max(n_updates, 1)
            avg_rew = episode_reward / max(episode_count, 1)
            print(
                f"Step {self.global_step:>8,} | ep {episode_count:>5} | "
                f"p_loss {avg_pl:>7.4f} | v_loss {avg_vl:>7.4f} | "
                f"ent {avg_ent:.4f} | avg_rew {avg_rew:>7.2f} | "
                f"{elapsed:.0f}s"
            )

            # ========================================================
            #  4. 评估
            # ========================================================
            if self.global_step >= next_eval_at:
                self._evaluate_and_save(save_path)
                next_eval_at += cfg.eval_interval

        # 训练结束保存最终模型
        torch.save(self.policy.state_dict(), save_path / "policy_final.pt")
        print(f"训练完成，模型已保存至 {save_path}")

    # ── 评估 ────────────────────────────────────────────

    def _evaluate_and_save(self, save_path: Path) -> None:
        cfg = self.ppo_config
        results = self._evaluate(cfg.num_eval_episodes)

        print(
            f"  >>> Eval @ {self.global_step:>8,} | "
            f"success {results['success_rate']:.1%} | "
            f"steps {results['avg_steps']:.1f} | "
            f"exposure {results['avg_exposure_ratio']:.3f}"
        )

        if results["success_rate"] >= self.best_eval_rate:
            self.best_eval_rate = results["success_rate"]
            torch.save(self.policy.state_dict(), save_path / "policy_best.pt")
            print(f"  >>> 最佳模型已保存")

        torch.save(
            self.policy.state_dict(), save_path / f"policy_{self.global_step}.pt"
        )

    def _evaluate(self, num_episodes: int) -> dict[str, float]:
        """确定性推理，无数据增强。"""
        successes = 0
        total_steps = 0
        total_exposure_ratio = 0.0

        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            ep_steps = 0
            ep_exposed = 0

            while not done:
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                mask_t = torch.from_numpy(self.env.get_action_mask()).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action, _, _, _, _ = self.policy(obs_t, mask_t, deterministic=True)

                obs, _reward, done, info = self.env.step(action.item())
                ep_steps += 1

                if self.env.visibility_map[tuple(self.env.agent_position)] > 0.5:
                    ep_exposed += 1

            total_steps += ep_steps
            total_exposure_ratio += ep_exposed / max(1, ep_steps)
            if info.get("result") == "success":
                successes += 1

        return {
            "success_rate": successes / num_episodes,
            "avg_steps": total_steps / num_episodes,
            "avg_exposure_ratio": total_exposure_ratio / num_episodes,
        }
