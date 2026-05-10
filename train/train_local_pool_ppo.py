from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EnvConfig
from env.vectorized_env import VectorizedEnv
from models.actor_critic_cnn import ActorCriticCNN, deaugment_action, random_augment
from planner import StealthCostConfig, plan_stealth_path
from train.ppo_buffer import RolloutBuffer
from train.ppo_config import PPOConfig

MOVE_TO_ACTION = {
    (-1, 0): 0,
    (1, 0): 1,
    (0, -1): 2,
    (0, 1): 3,
    (-1, -1): 4,
    (-1, 1): 5,
    (1, -1): 6,
    (1, 1): 7,
}


def _split_curriculum_indices(bfs_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1 = np.quantile(bfs_lengths, 0.33)
    q2 = np.quantile(bfs_lengths, 0.66)
    easy = np.where(bfs_lengths <= q1)[0].astype(np.int32)
    mid = np.where((bfs_lengths > q1) & (bfs_lengths <= q2))[0].astype(np.int32)
    hard = np.where(bfs_lengths > q2)[0].astype(np.int32)
    if len(easy) == 0:
        easy = np.arange(len(bfs_lengths), dtype=np.int32)
    if len(mid) == 0:
        mid = easy
    if len(hard) == 0:
        hard = mid
    return easy, mid, hard


def _sample_with_replacement(rng: np.random.Generator, indices: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0,), dtype=np.int32)
    if len(indices) == 0:
        raise ValueError("cannot sample from empty indices")
    picks = rng.integers(0, len(indices), size=n)
    return indices[picks].astype(np.int32)


def _build_mixed_curriculum_indices(
    rng: np.random.Generator,
    easy_idx: np.ndarray,
    mid_idx: np.ndarray,
    hard_idx: np.ndarray,
    total_size: int,
    progress: float,
) -> np.ndarray:
    # 始终保留 easy 占比，避免阶段切换导致遗忘。
    if progress < 0.4:
        weights = (0.80, 0.20, 0.00)
    elif progress < 0.8:
        weights = (0.65, 0.30, 0.05)
    else:
        weights = (0.50, 0.30, 0.20)

    base_counts = [int(total_size * w) for w in weights]
    base_counts[0] = max(1, base_counts[0])
    remainder = total_size - sum(base_counts)
    if remainder > 0:
        order = np.argsort(np.array(weights))[::-1]
        for i in range(remainder):
            base_counts[int(order[i % len(order)])] += 1

    e_src = easy_idx if len(easy_idx) > 0 else np.concatenate([mid_idx, hard_idx])
    m_src = mid_idx if len(mid_idx) > 0 else e_src
    h_src = hard_idx if len(hard_idx) > 0 else m_src

    mixed = np.concatenate(
        [
            _sample_with_replacement(rng, e_src, base_counts[0]),
            _sample_with_replacement(rng, m_src, base_counts[1]),
            _sample_with_replacement(rng, h_src, base_counts[2]),
        ]
    ).astype(np.int32)
    rng.shuffle(mixed)
    return mixed


def _apply_fallback_if_needed(env, cfg: StealthCostConfig) -> tuple[bool, bool]:
    start = tuple(env.agent_position.tolist())
    goal = tuple(env.goal_position.tolist())
    path = plan_stealth_path(env, start, goal, cfg)
    if not path or len(path) <= 1:
        return False, False

    for nxt in path[1:]:
        cur = tuple(env.agent_position.tolist())
        move = (nxt[0] - cur[0], nxt[1] - cur[1])
        action = MOVE_TO_ACTION.get(move)
        if action is None:
            return True, False
        _, _, done, info = env.step(action)
        if done:
            return True, info.get("result") == "success"
    return True, tuple(env.agent_position.tolist()) == goal


def _run_eval_episode(policy: ActorCriticCNN, env, device: torch.device, use_fallback: bool) -> dict:
    obs = env._get_observation()
    done = False
    ep_steps = 0
    ep_exposed = 0
    fallback_used = False
    best_dist = float(np.linalg.norm(env.agent_position.astype(np.float32) - env.goal_position.astype(np.float32)))
    stagnation = 0

    fallback_cfg = StealthCostConfig(w_len=1.0, w_vis=2.0, w_slope=0.6, w_turn=0.1)

    while not done:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(env.get_action_mask()).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _, _ = policy(obs_t, mask_t, deterministic=True)

        obs, _reward, done, info = env.step(action.item())
        ep_steps += 1
        if env.visibility_map[tuple(env.agent_position)] > 0.5:
            ep_exposed += 1

        dist = float(np.linalg.norm(env.agent_position.astype(np.float32) - env.goal_position.astype(np.float32)))
        if dist + 1e-4 < best_dist:
            best_dist = dist
            stagnation = 0
        else:
            stagnation += 1

        near_timeout = (env.config.max_steps - env.steps) <= 10
        if use_fallback and not done and (stagnation >= 12 or near_timeout):
            used, success = _apply_fallback_if_needed(env, fallback_cfg)
            if used:
                fallback_used = True
                done = True
                info = {"result": "success" if success else "fallback_fail", "collisions": env.total_collisions}

    return {
        "success": info.get("result") == "success",
        "result": info.get("result", "unknown"),
        "steps": ep_steps,
        "collisions": info.get("collisions", 0),
        "exposure": ep_exposed / max(1, ep_steps),
        "fallback_used": fallback_used,
    }


def _evaluate_fixed(
    vec_env: VectorizedEnv,
    policy: ActorCriticCNN,
    device: torch.device,
    eval_indices: np.ndarray,
    num_episodes: int,
    use_fallback: bool,
) -> dict:
    env = vec_env.envs[0]
    n = min(num_episodes, len(eval_indices))
    if n <= 0:
        return {
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_collisions": 0.0,
            "avg_exposure_ratio": 0.0,
            "fallback_rate": 0.0,
        }

    succ = 0
    total_steps = 0
    total_collisions = 0
    total_exposure = 0.0
    fallback_count = 0
    timeout_count = 0
    stuck_count = 0
    fallback_fail_count = 0

    for i in range(n):
        vec_env.reset_env_to_index(0, int(eval_indices[i]))
        out = _run_eval_episode(policy, env, device, use_fallback=use_fallback)
        succ += int(out["success"])
        total_steps += int(out["steps"])
        total_collisions += int(out["collisions"])
        total_exposure += float(out["exposure"])
        fallback_count += int(out["fallback_used"])
        timeout_count += int(out["result"] == "timeout")
        stuck_count += int(out["result"] == "stuck")
        fallback_fail_count += int(out["result"] == "fallback_fail")

    return {
        "success_rate": succ / n,
        "avg_steps": total_steps / n,
        "avg_collisions": total_collisions / n,
        "avg_exposure_ratio": total_exposure / n,
        "fallback_rate": fallback_count / n,
        "timeout_rate": timeout_count / n,
        "stuck_rate": stuck_count / n,
        "fallback_fail_rate": fallback_fail_count / n,
    }


def _evaluate_by_splits(
    vec_env: VectorizedEnv,
    policy: ActorCriticCNN,
    device: torch.device,
    val_indices: np.ndarray,
    val_bfs_lengths: np.ndarray,
    num_episodes: int,
    use_fallback: bool,
) -> dict[str, dict]:
    q1 = np.quantile(val_bfs_lengths, 0.33)
    q2 = np.quantile(val_bfs_lengths, 0.66)

    easy_mask = val_bfs_lengths <= q1
    mid_mask = (val_bfs_lengths > q1) & (val_bfs_lengths <= q2)
    hard_mask = val_bfs_lengths > q2

    split_indices = {
        "easy": val_indices[easy_mask],
        "mid": val_indices[mid_mask],
        "hard": val_indices[hard_mask],
    }
    split_results: dict[str, dict] = {}
    for name, idxs in split_indices.items():
        split_results[name] = _evaluate_fixed(
            vec_env,
            policy,
            device,
            eval_indices=idxs,
            num_episodes=min(num_episodes, len(idxs)),
            use_fallback=use_fallback,
        )
    return split_results


def main() -> None:
    parser = argparse.ArgumentParser(description="局部窗口池 PPO 训练（并行）")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--pool", type=str, default="artifacts/window_pool_15.npz")
    parser.add_argument("--val-pool", type=str, default=None, help="独立验证池；为空则从训练池切分")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="无独立验证池时从训练池切分比例")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--save", type=str, default="checkpoints_local_pool")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--visible-penalty", type=float, default=0.0, help="可见区域惩罚，建议先设0学到达再逐步加大")
    parser.add_argument("--progress-weight", type=float, default=0.25, help="朝目标推进奖励权重")
    parser.add_argument("--progress-decay-start", type=float, default=0.8, help="从训练进度该比例开始衰减 progress reward（默认0.8）")
    parser.add_argument("--progress-decay-end", type=float, default=1.05, help="progress reward 衰减结束的训练进度比例（默认1.05，等效弱衰减）")
    parser.add_argument("--save-every-evals", type=int, default=0)
    parser.add_argument("--keep-last-milestones", type=int, default=3)
    parser.add_argument("--curriculum", action="store_true", help="启用BFS难度课程")
    parser.add_argument("--curriculum-hard-switch", action="store_true", help="课程使用旧版硬切换（默认使用混合采样）")
    parser.add_argument("--timeout-extra-penalty", type=float, default=0.0, help="训练时对 timeout 额外扣分（在环境原始奖励基础上叠加）")
    parser.add_argument("--no-augment", action="store_true", help="关闭训练时随机旋转/翻转增强")
    parser.add_argument("--force-augment", action="store_true", help="即使含方向语义通道也强制开启增强")
    parser.add_argument("--eval-fallback", action="store_true", help="评估时启用规则兜底并统计触发率")
    parser.add_argument("--eval-episodes", type=int, default=50, help="常规评估样本数")
    parser.add_argument("--eval-full-every", type=int, default=10, help="每N次评估做一次全量评估，0关闭")
    args = parser.parse_args()
    if args.progress_decay_end <= args.progress_decay_start:
        raise ValueError("--progress-decay-end 必须大于 --progress-decay-start")

    train_data = np.load(Path(args.pool))
    pool = {
        "heights": train_data["heights"],
        "tags": train_data["tags"],
        "starts": train_data["starts"],
        "goals": train_data["goals"],
        "bfs_lengths": train_data["bfs_lengths"],
    }
    if "visibility" in train_data:
        pool["visibility"] = train_data["visibility"]

    n_scene = len(pool["starts"])
    all_indices = np.arange(n_scene, dtype=np.int32)

    if args.val_pool:
        val_data = np.load(Path(args.val_pool))
        val_pool = {
            "heights": val_data["heights"],
            "tags": val_data["tags"],
            "starts": val_data["starts"],
            "goals": val_data["goals"],
            "bfs_lengths": val_data["bfs_lengths"],
        }
        if "visibility" in val_data:
            val_pool["visibility"] = val_data["visibility"]
        train_indices = all_indices
        val_indices = np.arange(len(val_pool["starts"]), dtype=np.int32)
        val_bfs_lengths = val_pool["bfs_lengths"]
    else:
        rng = np.random.default_rng(2026)
        perm = rng.permutation(all_indices)
        n_val = max(1, int(len(perm) * args.val_ratio))
        val_indices = np.sort(perm[:n_val]).astype(np.int32)
        train_indices = np.sort(perm[n_val:]).astype(np.int32)
        val_pool = pool
        val_bfs_lengths = val_pool["bfs_lengths"][val_indices]

    print(f"训练场景: {len(train_indices)}  验证场景: {len(val_indices)}")

    pool_grid = int(pool["heights"].shape[1])
    adaptive_max_steps = max(60, pool_grid * 6)
    env_config = replace(
        EnvConfig(),
        grid_size=pool_grid,
        local_map_size=pool_grid,
        max_steps=args.max_steps if args.max_steps is not None else adaptive_max_steps,
        visible_penalty=args.visible_penalty,
        progress_weight=args.progress_weight,
    )

    num_envs = args.envs
    ppo_cfg = PPOConfig(
        total_steps=args.steps,
        learning_rate=args.lr,
        entropy_coef=args.entropy,
        ppo_epochs=args.epochs,
        rollout_steps=2048,
        eval_interval=max(10_000, args.steps // 50),
        num_eval_episodes=args.eval_episodes,
        progress_decay_start=args.progress_decay_start,
        progress_decay_end=args.progress_decay_end,
    )
    assert ppo_cfg.rollout_steps % num_envs == 0

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    vec_env = VectorizedEnv(env_config, pool, num_envs=num_envs, seed=42, allowed_indices=train_indices)
    vec_env_val = VectorizedEnv(env_config, val_pool, num_envs=1, seed=7, allowed_indices=val_indices)

    obs_channels = int(vec_env.get_observations().shape[1])
    feature_dim = 256 if env_config.grid_size <= 15 else 512
    policy = ActorCriticCNN(in_channels=obs_channels, feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_cfg.learning_rate)

    if args.no_augment and args.force_augment:
        raise ValueError("--no-augment 与 --force-augment 不能同时使用")
    if args.no_augment:
        use_augment = False
    elif obs_channels > 7 and not args.force_augment:
        use_augment = False
        print("检测到方向语义通道（>7通道），默认关闭增强；如需强制开启请加 --force-augment")
    else:
        use_augment = True

    buffer = RolloutBuffer(
        ppo_cfg.rollout_steps,
        (obs_channels, env_config.grid_size, env_config.grid_size),
        device,
    )

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"设备: {device}  参数: {n_params:,}  总步数: {ppo_cfg.total_steps:,}")

    if args.curriculum:
        easy, mid, hard = _split_curriculum_indices(pool["bfs_lengths"][train_indices])
        # 映射回全局索引
        easy_idx = train_indices[easy]
        mid_idx = train_indices[mid]
        hard_idx = train_indices[hard]
        curriculum_rng = np.random.default_rng(2027)
    else:
        easy_idx = mid_idx = hard_idx = train_indices
        curriculum_rng = np.random.default_rng(2027)

    global_step = 0
    episode_reward = 0.0
    episode_count = 0
    best_eval_rate = 0.0
    next_eval_at = ppo_cfg.eval_interval
    eval_count = 0
    milestone_paths: deque[Path] = deque()
    save_path = Path(args.save)
    save_path.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    obs_batch = vec_env.get_observations()

    while global_step < ppo_cfg.total_steps:
        progress = global_step / ppo_cfg.total_steps
        if args.curriculum:
            if args.curriculum_hard_switch:
                if progress < 0.3:
                    vec_env.set_allowed_indices(easy_idx)
                elif progress < 0.7:
                    vec_env.set_allowed_indices(np.concatenate([easy_idx, mid_idx]))
                else:
                    vec_env.set_allowed_indices(np.concatenate([easy_idx, mid_idx, hard_idx]))
            else:
                mixed_indices = _build_mixed_curriculum_indices(
                    curriculum_rng,
                    easy_idx=easy_idx,
                    mid_idx=mid_idx,
                    hard_idx=hard_idx,
                    total_size=len(train_indices),
                    progress=progress,
                )
                vec_env.set_allowed_indices(mixed_indices)
        else:
            vec_env.set_allowed_indices(train_indices)

        if progress >= ppo_cfg.progress_decay_start:
            frac = min(1.0, (progress - ppo_cfg.progress_decay_start) / (ppo_cfg.progress_decay_end - ppo_cfg.progress_decay_start))
            vec_env.set_progress_weight(env_config.progress_weight * (1.0 - frac))
        else:
            vec_env.set_progress_weight(env_config.progress_weight)

        parallel_steps = ppo_cfg.rollout_steps // num_envs
        for _ in range(parallel_steps):
            obs_t = torch.from_numpy(obs_batch).to(device)
            mask_t = torch.from_numpy(vec_env.get_action_masks()).to(device)
            if use_augment:
                aug_obs, aug_mask, aug_params = random_augment(obs_t, mask_t)
                with torch.no_grad():
                    aug_actions, log_probs, values, _, _ = policy(aug_obs, aug_mask)
                orig_actions = deaugment_action(aug_actions, *aug_params)
            else:
                aug_obs = obs_t
                aug_mask = mask_t
                with torch.no_grad():
                    aug_actions, log_probs, values, _, _ = policy(aug_obs, aug_mask)
                orig_actions = aug_actions

            next_obs_batch, rewards, dones, infos = vec_env.step(orig_actions.cpu().numpy())
            if args.timeout_extra_penalty > 0.0:
                for i in range(num_envs):
                    if bool(dones[i]) and infos[i].get("result") == "timeout":
                        rewards[i] -= float(args.timeout_extra_penalty)

            for i in range(num_envs):
                buffer.add(aug_obs[i], aug_actions[i], log_probs[i], float(rewards[i]), values[i], bool(dones[i]), aug_mask[i])
                episode_reward += float(rewards[i])
                global_step += 1
                if dones[i]:
                    episode_count += 1
                if global_step >= ppo_cfg.total_steps:
                    break

            obs_batch = next_obs_batch
            if global_step >= ppo_cfg.total_steps:
                break

        obs_last = torch.from_numpy(obs_batch).to(device)
        with torch.no_grad():
            last_values = policy.get_value(obs_last)
        buffer.compute_gae_parallel(last_values, ppo_cfg.gamma, ppo_cfg.gae_lambda, num_envs)
        buffer.normalize_advantages()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(ppo_cfg.ppo_epochs):
            for indices in buffer.sample(ppo_cfg.minibatch_size):
                mb_obs = buffer.observations[indices]
                mb_actions = buffer.actions[indices]
                mb_old_log_probs = buffer.log_probs[indices]
                mb_advantages = buffer.advantages[indices]
                mb_returns = buffer.returns[indices]
                mb_masks = buffer.masks[indices]

                new_log_probs, values, entropy = policy.evaluate(mb_obs, mb_actions, mb_masks)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - ppo_cfg.clip_epsilon, 1.0 + ppo_cfg.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_returns)
                loss = policy_loss + ppo_cfg.value_coef * value_loss - ppo_cfg.entropy_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        buffer.clear()

        elapsed = time.time() - t_start
        print(
            f"Step {global_step:>8,} | ep {episode_count:>5} | "
            f"p_loss {total_policy_loss / max(1, n_updates):>7.4f} | "
            f"v_loss {total_value_loss / max(1, n_updates):>7.4f} | "
            f"ent {total_entropy / max(1, n_updates):.4f} | "
            f"avg_rew {episode_reward / max(1, episode_count):>7.2f} | {elapsed:.0f}s"
        )

        if global_step >= next_eval_at:
            eval_count += 1
            results = _evaluate_fixed(
                vec_env_val,
                policy,
                device,
                eval_indices=val_indices,
                num_episodes=min(ppo_cfg.num_eval_episodes, len(val_indices)),
                use_fallback=args.eval_fallback,
            )
            print(
                f"  >>> Eval @ {global_step:>8,} | success {results['success_rate']:.1%} | "
                f"steps {results['avg_steps']:.1f} | collisions {results['avg_collisions']:.2f} | "
                f"exposure {results['avg_exposure_ratio']:.3f} | fallback {results['fallback_rate']:.1%} | "
                f"timeout {results['timeout_rate']:.1%} | stuck {results['stuck_rate']:.1%} | "
                f"fb_fail {results['fallback_fail_rate']:.1%}"
            )
            split_results = _evaluate_by_splits(
                vec_env_val,
                policy,
                device,
                val_indices=val_indices,
                val_bfs_lengths=val_bfs_lengths,
                num_episodes=min(ppo_cfg.num_eval_episodes, len(val_indices)),
                use_fallback=args.eval_fallback,
            )
            print(
                "      split | "
                f"easy {split_results['easy']['success_rate']:.1%} | "
                f"mid {split_results['mid']['success_rate']:.1%} | "
                f"hard {split_results['hard']['success_rate']:.1%}"
            )

            if args.eval_full_every > 0 and (eval_count % args.eval_full_every == 0):
                full_results = _evaluate_fixed(
                    vec_env_val,
                    policy,
                    device,
                    eval_indices=val_indices,
                    num_episodes=len(val_indices),
                    use_fallback=args.eval_fallback,
                )
                print(
                    "      full  | "
                    f"success {full_results['success_rate']:.1%} | "
                    f"steps {full_results['avg_steps']:.1f} | "
                    f"exposure {full_results['avg_exposure_ratio']:.3f} | "
                    f"timeout {full_results['timeout_rate']:.1%} | "
                    f"stuck {full_results['stuck_rate']:.1%} | "
                    f"fb_fail {full_results['fallback_fail_rate']:.1%}"
                )

            if results["success_rate"] >= best_eval_rate:
                best_eval_rate = results["success_rate"]
                torch.save(policy.state_dict(), save_path / "policy_best.pt")

            if args.save_every_evals > 0 and (eval_count % args.save_every_evals == 0):
                milestone = save_path / f"policy_step_{global_step}.pt"
                torch.save(policy.state_dict(), milestone)
                milestone_paths.append(milestone)
                while len(milestone_paths) > args.keep_last_milestones:
                    old = milestone_paths.popleft()
                    if old.exists():
                        old.unlink()

            next_eval_at += ppo_cfg.eval_interval
            obs_batch = vec_env.get_observations()

    torch.save(policy.state_dict(), save_path / "policy_final.pt")
    print(f"\n训练完成，模型已保存至 {save_path}")


if __name__ == "__main__":
    main()
