from __future__ import annotations

from datetime import datetime
from pathlib import Path
import random
import statistics
from typing import Callable

import numpy as np

from env.battlefield_env import BattlefieldEnv
from planner.visibility_astar import VisibilityAwareAStarPlanner
from train.dqn_agent import DoubleDQNAgent, TrainingConfig

# 路径采样：多 λ 候选值，评估时尝试多个可见性权重
LAMBDA_CANDIDATES = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0]


def train(config: TrainingConfig | None = None, bc_pretrain_path: str = "artifacts/ddqn_bc.pt", multi_lambda: bool = False) -> None:
    config = config or TrainingConfig()
    env = BattlefieldEnv()
    agent = DoubleDQNAgent(action_dim=len(BattlefieldEnv.ACTIONS), config=config)
    rng = random.Random(config.seed)

    bc_path = Path(bc_pretrain_path)
    if bc_path.exists():
        agent.load(str(bc_path))
        print(f"已加载 BC 预训练权重: {bc_path}")

    output_dir = Path(__file__).resolve().parents[1] / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_fp = log_path.open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message)
        log_fp.write(message + "\n")
        log_fp.flush()

    checkpoint_path = output_dir / "ddqn_latest.pt"
    best_path = output_dir / "ddqn_best.pt"
    interrupt_path = output_dir / "ddqn_interrupt.pt"

    log(f"训练设备: {agent.device}")
    log(f"训练日志: {log_path}")
    use_waypoints = config.waypoint.enabled
    if use_waypoints:
        log(f"航点模式: interval={config.waypoint.interval} max_segment_multiplier={config.waypoint.max_segment_multiplier}")

    global_step = 0
    best_eval_reward = float("-inf")
    best_eval_success_rate = float("-inf")
    plateau_count = 0
    last_completed_episode = 0
    recent_rewards: list[float] = []

    try:
        for episode in range(1, config.episodes + 1):
            train_scene_seed = rng.choice(env.config.train_scene_seeds)
            observation = env.reset(scene_seed=train_scene_seed, scenario_mode=env.config.scenario_mode)
            agent.reset_episode_stats()
            done = False
            episode_reward = 0.0
            episode_loss: list[float] = []
            success = False

            if use_waypoints:
                # 航点模式：先生成 A* 路径并采样航点
                waypoints = _generate_waypoints(env, config)
                segment_reward, segment_loss, done, success, global_step = _run_waypoint_episode(
                    env, agent, waypoints, config, global_step, log,
                )
                episode_reward += segment_reward
                episode_loss.extend(segment_loss)
            else:
                # 标准模式：单段 episode
                episode_positions: list[tuple[int, int]] = [tuple(env.agent_position.tolist())]
                episode_transitions: list[tuple] = []

                while not done:
                    epsilon = agent.current_epsilon(global_step)
                    action = agent.select_action(observation, epsilon, env=env, global_step=global_step)
                    result = env.step(action)
                    next_valid_actions = env.get_valid_actions()
                    agent.store_transition(
                        observation,
                        action,
                        result.reward,
                        result.observation,
                        result.done,
                        next_valid_actions=next_valid_actions,
                    )

                    episode_positions.append(tuple(env.agent_position.tolist()))
                    episode_transitions.append((
                        observation, action, result.reward,
                        result.observation, result.done, next_valid_actions,
                    ))

                    observation = result.observation
                    episode_reward += result.reward
                    done = result.done
                    success = bool(result.info["success"])
                    global_step += 1

                    if global_step >= config.warmup_steps and global_step % config.train_frequency == 0 and agent.can_train(config.batch_size):
                        loss = agent.train_step(config.batch_size)
                        if loss > 0:
                            episode_loss.append(loss)

                # HER: 失败 episode 用已访问位置重新标记奖励（航点模式关闭）
                if not success:
                    _relabel_her(agent, env, episode_positions, episode_transitions,
                                 k=config.her_relabel_count, log_fn=log)

            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 20:
                recent_rewards.pop(0)

            mean_reward = statistics.mean(recent_rewards)
            mean_loss = statistics.mean(episode_loss) if episode_loss else 0.0
            action_stats = agent.get_episode_stats()
            log(
                f"Episode {episode:04d} | reward={episode_reward:7.3f} | mean20={mean_reward:7.3f} | "
                f"loss={mean_loss:6.4f} | epsilon={agent.current_epsilon(global_step):5.3f} | "
                f"path_len={env.total_path_length:6.3f} | hidden_ratio={env.hidden_ratio:5.3f} | "
                f"scene_seed={train_scene_seed} | success={success} | "
                f"greedy={action_stats['greedy']:03d} heuristic={action_stats['heuristic']:03d} "
                f"teacher={action_stats['teacher']:03d} random={action_stats['random']:03d}"
            )
            last_completed_episode = episode

            if episode % config.save_interval == 0:
                agent.save(str(checkpoint_path))

            if episode % config.eval_interval == 0:
                eval_summary = evaluate_policy(
                    agent,
                    env,
                    scene_seeds=env.config.val_scene_seeds[: config.early_stop_eval_episodes],
                    scenario_mode=env.config.scenario_mode,
                    log_fn=log,
                    use_waypoints=use_waypoints,
                    multi_lambda=multi_lambda,
                )

                selection_summary: dict[str, float] | None = None
                selection_scope = "none"

                if config.full_eval_interval > 0 and episode % config.full_eval_interval == 0:
                    full_eval_summary = evaluate_policy(
                        agent,
                        env,
                        scene_seeds=env.config.val_scene_seeds,
                        scenario_mode=env.config.scenario_mode,
                        log_fn=log,
                        use_waypoints=use_waypoints,
                        multi_lambda=multi_lambda,
                    )
                    log(
                        f"[Eval-Full] scenes={len(env.config.val_scene_seeds)} | "
                        f"avg_reward={full_eval_summary['avg_reward']:7.3f} | "
                        f"success_rate={full_eval_summary['success_rate']:.2f} | "
                        f"avg_hidden_ratio={full_eval_summary['avg_hidden_ratio']:.3f} | "
                        f"avg_path_len={full_eval_summary['avg_path_length']:.3f}"
                    )
                    selection_summary = full_eval_summary
                    selection_scope = f"full-{len(env.config.val_scene_seeds)}"

                if selection_summary is None:
                    continue

                if _is_better_eval(selection_summary, best_eval_success_rate, best_eval_reward, config.early_stop_min_delta):
                    best_eval_reward = selection_summary["avg_reward"]
                    best_eval_success_rate = selection_summary["success_rate"]
                    plateau_count = 0
                    agent.save(str(best_path))
                    log(
                        f"[Best] success_rate={best_eval_success_rate:.2f} | avg_reward={best_eval_reward:7.3f} | "
                        f"avg_hidden_ratio={selection_summary['avg_hidden_ratio']:.3f} | "
                        f"scope={selection_scope} | saved={best_path}"
                    )
                elif (
                    config.early_stop_enabled
                    and selection_summary["success_rate"] >= config.early_stop_success_rate_threshold
                    and selection_summary["avg_reward"] <= best_eval_reward + config.early_stop_min_delta
                ):
                    plateau_count += 1
                    log(
                        f"[EarlyStop Check] plateau={plateau_count}/{config.early_stop_plateau_patience} | "
                        f"best_avg_reward={best_eval_reward:7.3f}"
                    )
                    if plateau_count >= config.early_stop_plateau_patience:
                        log(f"[EarlyStop] 在 episode {episode} 提前停止训练")
                        break
                else:
                    plateau_count = 0

        agent.save(str(checkpoint_path))
        log(f"训练结束，最新模型已保存到: {checkpoint_path}")
        log(f"最佳模型已保存到: {best_path}")
    except KeyboardInterrupt:
        agent.save(str(interrupt_path))
        log(f"\n训练被中断，已保存中断检查点: {interrupt_path}")
        log(f"最后完成的 episode: {last_completed_episode}")
    finally:
        log_fp.close()


def _generate_waypoints(env: BattlefieldEnv, config: TrainingConfig) -> list[tuple[int, int]]:
    """用 A* 生成完整路径并等间隔采样航点。"""
    return _generate_waypoints_with_lambda(env, config.waypoint.interval, config.waypoint.waypoint_visible_weight)


def _generate_waypoints_with_lambda(env: BattlefieldEnv, interval: int, lam: float) -> list[tuple[int, int]]:
    """用指定 λ 的 Visibility-A* 生成航点。"""
    start = tuple(env.agent_position.tolist())
    goal = tuple(env.goal_position.tolist())
    try:
        result = VisibilityAwareAStarPlanner(env, visible_weight=lam).plan(start=start, goal=goal)
        if result.success and len(result.path) >= 2:
            return _sample_waypoints(result.path, interval)
    except Exception:
        pass
    return [goal]


def _sample_waypoints(path: list[tuple[int, int]], interval: int) -> list[tuple[int, int]]:
    """沿 A* 路径每隔 interval 步采样一个航点，最后一个一定是终点。"""
    waypoints: list[tuple[int, int]] = []
    for i in range(interval, len(path), interval):
        waypoints.append(path[i])
    if not waypoints or waypoints[-1] != path[-1]:
        waypoints.append(path[-1])
    return waypoints


def _run_waypoint_episode(
    env: BattlefieldEnv,
    agent: DoubleDQNAgent,
    waypoints: list[tuple[int, int]],
    config: TrainingConfig,
    global_step: int,
    log_fn: Callable[[str], None],
) -> tuple[float, list[float], bool, bool, int]:
    """运行一个航点式 episode：逐段导航到每个航点。

    Returns:
        (total_reward, loss_list, done, success, final_global_step)
    """
    wp = config.waypoint
    max_segment_steps = int(wp.interval * wp.max_segment_multiplier)
    total_reward = 0.0
    losses: list[float] = []
    success = False
    gs = global_step
    total_waypoints = len(waypoints)

    for wp_idx, waypoint in enumerate(waypoints):
        is_final = (wp_idx == total_waypoints - 1)
        env.set_subgoal(waypoint, is_final=is_final)
        observation = env.get_observation()

        segment_steps = 0
        segment_done = False

        while not segment_done and segment_steps < max_segment_steps:
            epsilon = agent.current_epsilon(gs)
            action = agent.select_action(observation, epsilon, env=env, global_step=gs)
            result = env.step(action)
            next_valid_actions = env.get_valid_actions()
            agent.store_transition(
                observation,
                action,
                result.reward,
                result.observation,
                result.done,
                next_valid_actions=next_valid_actions,
            )

            observation = result.observation
            total_reward += result.reward
            segment_steps += 1
            gs += 1
            segment_done = result.done

            if result.info["success"]:
                success = True
            if result.info["waypoint_reached"]:
                break  # 到达航点，进入下一段

            if gs >= config.warmup_steps and gs % config.train_frequency == 0 and agent.can_train(config.batch_size):
                losses.append(agent.train_step(config.batch_size))

        if segment_done:
            return total_reward, losses, True, success, gs

        if segment_steps >= max_segment_steps:
            total_reward -= wp.segment_timeout_penalty
            log_fn(f"  [WP] wp={wp_idx + 1}/{total_waypoints} timeout after {segment_steps} steps")
            return total_reward, losses, True, False, gs

        log_fn(f"  [WP] wp={wp_idx + 1}/{total_waypoints} reached in {segment_steps} steps")

    return total_reward, losses, True, success, gs


def _relabel_her(
    agent: DoubleDQNAgent,
    env: BattlefieldEnv,
    positions: list[tuple[int, int]],
    transitions: list[tuple],
    k: int,
    log_fn: Callable[[str], None] | None = None,
) -> None:
    """HER 'future': 失败 episode 中，对每步采样未来位置，若动作靠近未来位置则加进度奖励。"""
    T = len(transitions)
    if T < 3 or k <= 0:
        return

    progress_reward = env.config.goal_reward * 0.05
    relabeled = 0

    for t in range(T - 1):
        obs, action, reward, next_obs, done, valid_actions = transitions[t]
        current_pos = np.array(positions[t], dtype=np.float32)
        next_pos = np.array(positions[t + 1], dtype=np.float32)

        future_candidates = positions[t + 2:]
        if not future_candidates:
            continue
        n_sample = min(k, len(future_candidates))
        sampled = random.sample(future_candidates, n_sample)

        made_progress = False
        for future_pos in sampled:
            fp = np.array(future_pos, dtype=np.float32)
            if np.linalg.norm(next_pos - fp) < np.linalg.norm(current_pos - fp):
                made_progress = True
                break

        if made_progress:
            new_reward = reward + progress_reward
            agent.store_transition(obs, action, new_reward, next_obs, False, valid_actions)
            relabeled += 1

    if relabeled > 0 and log_fn is not None:
        log_fn(f"[HER-future] relabeled {relabeled}/{T} transitions")


def evaluate_policy(
    agent: DoubleDQNAgent,
    env: BattlefieldEnv,
    scene_seeds: tuple[int, ...] | list[int] | None = None,
    scenario_mode: str = "fixed",
    log_fn: Callable[[str], None] | None = None,
    use_waypoints: bool = False,
    multi_lambda: bool = False,
) -> dict[str, float]:
    rewards: list[float] = []
    hidden_ratios: list[float] = []
    path_lengths: list[float] = []
    successes = 0
    seeds = tuple(scene_seeds) if scene_seeds is not None else tuple([None] * 3)
    lambdas = LAMBDA_CANDIDATES if multi_lambda else [agent.config.waypoint.waypoint_visible_weight]
    interval = agent.config.waypoint.interval

    for scene_seed in seeds:
        best_reward = float("-inf")
        best_hidden = 0.0
        best_path_len = 0.0
        scene_success = False

        agent.reset_episode_stats()
        if use_waypoints:
            for lam in lambdas:
                env.reset(scene_seed=scene_seed, scenario_mode=scenario_mode)
                agent.reset_episode_stats()
                waypoints = _generate_waypoints_with_lambda(env, interval, lam)
                total_reward, _, success = _eval_waypoint_episode(env, agent, waypoints)
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_hidden = env.hidden_ratio
                    best_path_len = env.total_path_length
                if success:
                    scene_success = True
                    break  # 找到一条能走通的路径，不再尝试更多 λ
        else:
            env.reset(scene_seed=scene_seed, scenario_mode=scenario_mode)
            observation = env.get_observation()
            done = False
            total_reward = 0.0
            success = False
            while not done:
                action = agent.select_action_masked(observation, env=env)
                result = env.step(action)
                observation = result.observation
                total_reward += result.reward
                done = result.done
                if result.done and bool(result.info["success"]):
                    success = True
            best_reward = total_reward
            best_hidden = env.hidden_ratio
            best_path_len = env.total_path_length
            scene_success = success

        if scene_success:
            successes += 1
        rewards.append(best_reward)
        hidden_ratios.append(best_hidden)
        path_lengths.append(best_path_len)

    avg_reward = statistics.mean(rewards) if rewards else 0.0
    avg_hidden_ratio = statistics.mean(hidden_ratios) if hidden_ratios else 0.0
    avg_path_length = statistics.mean(path_lengths) if path_lengths else 0.0
    success_rate = successes / max(1, len(seeds))
    eval_tag = "[Eval]" if len(seeds) <= config_default_eval_count() else "[Eval-Full]"
    ml_tag = " [multi-λ]" if multi_lambda else ""
    eval_message = (
        f"{eval_tag}{ml_tag} avg_reward={avg_reward:7.3f} | success_rate={success_rate:.2f} | "
        f"avg_hidden_ratio={avg_hidden_ratio:.3f} | avg_path_len={avg_path_length:.3f}"
    )
    if log_fn is None:
        print(eval_message)
    else:
        log_fn(eval_message)
    return {
        "avg_reward": avg_reward,
        "success_rate": success_rate,
        "avg_hidden_ratio": avg_hidden_ratio,
        "avg_path_length": avg_path_length,
    }


def _eval_waypoint_episode(
    env: BattlefieldEnv,
    agent: DoubleDQNAgent,
    waypoints: list[tuple[int, int]],
) -> tuple[float, bool, bool]:
    """评估用：运行一个航点式 episode（关闭探索）。"""
    wp = agent.config.waypoint
    max_segment_steps = int(wp.interval * wp.max_segment_multiplier)
    total_reward = 0.0
    success = False
    total_waypoints = len(waypoints)

    for wp_idx, waypoint in enumerate(waypoints):
        is_final = (wp_idx == total_waypoints - 1)
        env.set_subgoal(waypoint, is_final=is_final)
        observation = env.get_observation()

        segment_steps = 0
        while segment_steps < max_segment_steps:
            action = agent.select_action_masked(observation, env=env)
            result = env.step(action)
            observation = result.observation
            total_reward += result.reward
            segment_steps += 1

            if result.info["success"]:
                success = True
                return total_reward, True, True
            if result.info["waypoint_reached"]:
                break
            if result.done:
                return total_reward, True, success

        if segment_steps >= max_segment_steps:
            return total_reward, True, False

    return total_reward, True, success


def config_default_eval_count() -> int:
    return TrainingConfig().early_stop_eval_episodes


def _is_better_eval(
    eval_summary: dict[str, float],
    best_eval_success_rate: float,
    best_eval_reward: float,
    min_delta: float,
) -> bool:
    if eval_summary["success_rate"] > best_eval_success_rate:
        return True
    if eval_summary["success_rate"] == best_eval_success_rate and eval_summary["avg_reward"] > best_eval_reward + min_delta:
        return True
    return False


def _eval_only(config: TrainingConfig, bc_path: str, multi_lambda: bool = False) -> None:
    """仅评估模式：加载已有模型，在验证集上评估并退出。"""
    env = BattlefieldEnv()
    agent = DoubleDQNAgent(action_dim=len(BattlefieldEnv.ACTIONS), config=config)
    bc = Path(bc_path)
    if bc.exists():
        agent.load(str(bc))
        print(f"已加载模型: {bc}")
    else:
        print(f"模型文件不存在: {bc}")
        return

    print(f"评估配置: multi_lambda={multi_lambda}, waypoints={config.waypoint.enabled}")
    print(f"λ 候选: {LAMBDA_CANDIDATES if multi_lambda else [config.waypoint.waypoint_visible_weight]}")

    summary = evaluate_policy(
        agent, env,
        scene_seeds=env.config.val_scene_seeds,
        scenario_mode=env.config.scenario_mode,
        use_waypoints=config.waypoint.enabled,
        multi_lambda=multi_lambda,
    )
    print(
        f"[Eval-Only] scenes={len(env.config.val_scene_seeds)} | "
        f"success_rate={summary['success_rate']:.2f} | "
        f"avg_reward={summary['avg_reward']:7.3f} | "
        f"avg_hidden_ratio={summary['avg_hidden_ratio']:.3f} | "
        f"avg_path_len={summary['avg_path_length']:.3f}"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train D3QN agent")
    parser.add_argument("--use-waypoints", action="store_true", help="Enable waypoint-based hierarchical RL")
    parser.add_argument("--bc-path", default="artifacts/ddqn_bc.pt", help="BC pretrain checkpoint path")
    parser.add_argument("--episodes", type=int, default=None, help="Override training episodes")
    parser.add_argument("--multi-lambda", action="store_true", help="Evaluate with multiple λ values for waypoint generation")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on existing model, skip training")
    args = parser.parse_args()

    cfg = TrainingConfig()
    if args.use_waypoints:
        from config import WaypointConfig
        cfg = TrainingConfig(
            waypoint=WaypointConfig(enabled=True),
        )
    if args.episodes is not None:
        cfg = TrainingConfig(
            episodes=args.episodes,
            waypoint=cfg.waypoint,
        )

    if args.eval_only:
        _eval_only(cfg, args.bc_path, multi_lambda=args.multi_lambda)
    else:
        train(config=cfg, bc_pretrain_path=args.bc_path, multi_lambda=args.multi_lambda)
