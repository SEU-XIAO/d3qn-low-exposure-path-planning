"""阶段1：障碍地形纯导航 · 并行训练版。

N 个独立环境并行 rollout，批量前向传播，加速约 N 倍。
用法: python -m train.stage1_parallel --steps 500000 --pool artifacts/scene_pool.npz --envs 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EnvConfig
from env.vectorized_env import VectorizedEnv
from models.policy_network import ActorCriticCNN, random_augment, deaugment_action
from train.ppo_buffer import RolloutBuffer
from train.ppo_config import PPOConfig


def _evaluate_vec(
    vec_env: VectorizedEnv,
    policy: ActorCriticCNN,
    device: torch.device,
    num_episodes: int,
) -> dict:
    """确定性评估（无增强），依次评估各环境。"""
    successes = 0
    total_steps = 0
    total_collisions = 0
    total_exposure = 0
    total_path_ratio = 0.0

    # 使用第一个环境做评估（重置它获取新场景）
    env = vec_env.envs[0]

    for _ in range(num_episodes):
        vec_env._reset_env(env)
        done = False
        ep_steps = 0
        ep_exposed = 0

        while not done:
            obs_t = torch.from_numpy(env._get_observation()).unsqueeze(0).to(device)
            mask_t = (
                torch.from_numpy(env.get_action_mask()).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                action, _, _, _, _ = policy(obs_t, mask_t, deterministic=True)

            _, _reward, done, info = env.step(action.item())
            ep_steps += 1

            if env.visibility_map[tuple(env.agent_position)] > 0.5:
                ep_exposed += 1

        ep_collisions = info.get("collisions", 0)
        total_steps += ep_steps
        total_collisions += ep_collisions
        total_exposure += ep_exposed / max(1, ep_steps)

        if info.get("result") == "success":
            successes += 1

    n = num_episodes
    return {
        "success_rate": successes / n,
        "avg_steps": total_steps / n,
        "avg_collisions": total_collisions / n,
        "avg_path_ratio": 0.0,  # 并行版暂不追踪 BFS 路径比
        "avg_exposure_ratio": total_exposure / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段1：障碍地形纯导航（并行训练）")
    parser.add_argument("--steps", type=int, default=500_000, help="总训练步数")
    parser.add_argument("--pool", type=str, default="artifacts/scene_pool.npz",
                        help="场景池 NPZ 文件路径")
    parser.add_argument("--envs", type=int, default=8, help="并行环境数")
    parser.add_argument("--save", type=str, default="checkpoints_stage1",
                        help="模型保存目录")
    parser.add_argument("--device", type=str, default=None, help="设备")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    args = parser.parse_args()

    # —— 加载场景池 ——
    pool_path = Path(args.pool)
    if not pool_path.exists():
        raise FileNotFoundError(
            f"场景池文件不存在: {pool_path}\n"
            f"请先运行: python -m env.scene_pool --num 5000"
        )

    print(f"加载场景池: {pool_path}")
    data = np.load(pool_path)
    pool = {
        "heights": data["heights"],
        "tags": data["tags"],
        "starts": data["starts"],
        "goals": data["goals"],
        "bfs_lengths": data["bfs_lengths"],
    }
    print(f"  {len(pool['starts'])} 个预生成场景")

    # —— 配置 ——
    env_config = EnvConfig()
    num_envs = args.envs
    ppo_cfg = PPOConfig(
        total_steps=args.steps,
        learning_rate=args.lr,
        rollout_steps=2048,  # 必须是 num_envs 的整数倍
        eval_interval=max(10_000, args.steps // 50),
    )
    # 确保 rollout_steps 整除 num_envs
    assert ppo_cfg.rollout_steps % num_envs == 0, \
        f"rollout_steps ({ppo_cfg.rollout_steps}) 必须能整除 num_envs ({num_envs})"

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # —— 创建向量化环境和网络 ——
    vec_env = VectorizedEnv(env_config, pool, num_envs=num_envs, seed=42)
    policy = ActorCriticCNN().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_cfg.learning_rate)

    buffer = RolloutBuffer(
        ppo_cfg.rollout_steps,
        (7, env_config.grid_size, env_config.grid_size),
        device,
    )

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"设备: {device}  参数: {n_params:,}  总步数: {ppo_cfg.total_steps:,}")
    print(f"并行环境: {num_envs}  Rollout/update: {ppo_cfg.rollout_steps}")
    print(f"每轮并行步数: {ppo_cfg.rollout_steps // num_envs}")
    print()

    # —— 训练状态 ——
    global_step = 0
    episode_reward = 0.0
    episode_count = 0
    best_eval_rate = 0.0
    next_eval_at = ppo_cfg.eval_interval
    save_path = Path(args.save)
    save_path.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    obs_batch = vec_env.get_observations()  # (N, 7, 50, 50)

    while global_step < ppo_cfg.total_steps:
        # —— 进度权重衰减 ——
        progress = global_step / ppo_cfg.total_steps
        if progress >= ppo_cfg.progress_decay_start:
            frac = min(
                1.0,
                (progress - ppo_cfg.progress_decay_start)
                / (ppo_cfg.progress_decay_end - ppo_cfg.progress_decay_start),
            )
            vec_env.set_progress_weight(
                env_config.progress_weight * (1.0 - frac)
            )
        else:
            vec_env.set_progress_weight(env_config.progress_weight)

        # ================================================================
        #  1. Rollout — 批量并行收集
        # ================================================================
        parallel_steps = ppo_cfg.rollout_steps // num_envs

        for _ in range(parallel_steps):
            obs_t = torch.from_numpy(obs_batch).to(device)
            mask_np = vec_env.get_action_masks()
            mask_t = torch.from_numpy(mask_np).to(device)

            # 数据增强 + 批量前向传播
            aug_obs, aug_mask, aug_params = random_augment(obs_t, mask_t)

            with torch.no_grad():
                aug_actions, log_probs, values, _, _ = policy(aug_obs, aug_mask)

            # 动作逆变换
            orig_actions = deaugment_action(aug_actions, *aug_params)

            # 并行步进所有环境
            next_obs_batch, rewards, dones, infos = vec_env.step(
                orig_actions.cpu().numpy()
            )

            # 逐环境存入 buffer
            for i in range(num_envs):
                buffer.add(
                    aug_obs[i],
                    aug_actions[i],
                    log_probs[i],
                    float(rewards[i]),
                    values[i],
                    bool(dones[i]),
                    aug_mask[i],
                )
                episode_reward += float(rewards[i])
                global_step += 1

                if dones[i]:
                    episode_count += 1

                if global_step >= ppo_cfg.total_steps:
                    break

            obs_batch = next_obs_batch

            if global_step >= ppo_cfg.total_steps:
                break

        # ================================================================
        #  2. GAE（按 env 独立计算，数据交织存储，stride=num_envs）
        # ================================================================
        obs_last = torch.from_numpy(obs_batch).to(device)
        with torch.no_grad():
            last_values = policy.get_value(obs_last)  # (num_envs,)
        buffer.compute_gae_parallel(
            last_values, ppo_cfg.gamma, ppo_cfg.gae_lambda, num_envs,
        )
        buffer.normalize_advantages()

        # ================================================================
        #  3. PPO Update
        # ================================================================
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _epoch in range(ppo_cfg.ppo_epochs):
            for indices in buffer.sample(ppo_cfg.minibatch_size):
                mb_obs = buffer.observations[indices]
                mb_actions = buffer.actions[indices]
                mb_old_log_probs = buffer.log_probs[indices]
                mb_advantages = buffer.advantages[indices]
                mb_returns = buffer.returns[indices]
                mb_masks = buffer.masks[indices]

                new_log_probs, values, entropy = policy.evaluate(
                    mb_obs, mb_actions, mb_masks,
                )

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - ppo_cfg.clip_epsilon,
                        1.0 + ppo_cfg.clip_epsilon,
                    )
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_returns)
                loss = (
                    policy_loss
                    + ppo_cfg.value_coef * value_loss
                    - ppo_cfg.entropy_coef * entropy.mean()
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    policy.parameters(), ppo_cfg.max_grad_norm,
                )
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        buffer.clear()

        # —— 日志 ——
        elapsed = time.time() - t_start
        avg_pl = total_policy_loss / max(n_updates, 1)
        avg_vl = total_value_loss / max(n_updates, 1)
        avg_ent = total_entropy / max(n_updates, 1)
        avg_rew = episode_reward / max(episode_count, 1)
        print(
            f"Step {global_step:>8,} | ep {episode_count:>5} | "
            f"p_loss {avg_pl:>7.4f} | v_loss {avg_vl:>7.4f} | "
            f"ent {avg_ent:.4f} | avg_rew {avg_rew:>7.2f} | "
            f"{elapsed:.0f}s"
        )

        # ================================================================
        #  4. 评估
        # ================================================================
        if global_step >= next_eval_at:
            results = _evaluate_vec(
                vec_env, policy, device, ppo_cfg.num_eval_episodes,
            )
            print(
                f"  >>> Eval @ {global_step:>8,} | "
                f"success {results['success_rate']:.1%} | "
                f"steps {results['avg_steps']:.1f} | "
                f"collisions {results['avg_collisions']:.2f} | "
                f"exposure {results['avg_exposure_ratio']:.3f}"
            )

            if results["success_rate"] >= best_eval_rate:
                best_eval_rate = results["success_rate"]
                torch.save(policy.state_dict(), save_path / "policy_best.pt")
                print(f"  >>> 最佳模型已保存")

            torch.save(
                policy.state_dict(),
                save_path / f"policy_{global_step}.pt",
            )
            next_eval_at += ppo_cfg.eval_interval
            # 评估修改了 env 0 的状态，刷新观测以避免下一步训练用过期数据
            obs_batch = vec_env.get_observations()

    # —— 最终保存 ——
    torch.save(policy.state_dict(), save_path / "policy_final.pt")
    print(f"\n训练完成，模型已保存至 {save_path}")

    # —— 最终评估 ——
    print()
    print("=" * 60)
    print("最终评估 (100 episodes, 确定性策略)")
    print("=" * 60)
    results = _evaluate_vec(vec_env, policy, device, 100)
    print(f"  到达率:       {results['success_rate']:.1%}")
    print(f"  平均步数:     {results['avg_steps']:.1f}")
    print(f"  平均碰撞:     {results['avg_collisions']:.2f}")
    print(f"  平均暴露率:   {results['avg_exposure_ratio']:.3f}")

    if results["success_rate"] >= 0.6 and results["avg_collisions"] < 2.0:
        print("\n阶段1达标！")
    else:
        print("\n阶段1未达标，建议增加训练步数。")


if __name__ == "__main__":
    main()
