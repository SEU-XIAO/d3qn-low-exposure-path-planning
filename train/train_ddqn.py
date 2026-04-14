from __future__ import annotations

from datetime import datetime
from pathlib import Path
import random
import statistics
from typing import Callable

from env.battlefield_env import BattlefieldEnv
from train.dqn_agent import DoubleDQNAgent, TrainingConfig


def train(config: TrainingConfig | None = None) -> None:
    config = config or TrainingConfig()
    env = BattlefieldEnv()
    agent = DoubleDQNAgent(action_dim=len(BattlefieldEnv.ACTIONS), config=config)
    rng = random.Random(config.seed)

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

    global_step = 0
    best_eval_reward = float("-inf")
    best_eval_success_rate = float("-inf")
    plateau_count = 0
    last_completed_episode = 0
    recent_rewards: list[float] = []

    try:
        for episode in range(1, config.episodes + 1):
            train_scene_seed = rng.choice(env.config.train_scene_seeds)
            observation = env.reset(scene_seed=train_scene_seed, scenario_mode="random")
            agent.reset_episode_stats()
            done = False
            episode_reward = 0.0
            episode_loss: list[float] = []
            success = False

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

                observation = result.observation
                episode_reward += result.reward
                done = result.done
                success = bool(result.info["success"])
                global_step += 1

                if global_step >= config.warmup_steps and global_step % config.train_frequency == 0 and agent.can_train(config.batch_size):
                    episode_loss.append(agent.train_step(config.batch_size))

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
                    scenario_mode="random",
                    log_fn=log,
                )

                if config.full_eval_interval > 0 and episode % config.full_eval_interval == 0:
                    full_eval_summary = evaluate_policy(
                        agent,
                        env,
                        scene_seeds=env.config.val_scene_seeds,
                        scenario_mode="random",
                        log_fn=log,
                    )
                    log(
                        f"[Eval-Full] scenes={len(env.config.val_scene_seeds)} | "
                        f"avg_reward={full_eval_summary['avg_reward']:7.3f} | "
                        f"success_rate={full_eval_summary['success_rate']:.2f} | "
                        f"avg_hidden_ratio={full_eval_summary['avg_hidden_ratio']:.3f} | "
                        f"avg_path_len={full_eval_summary['avg_path_length']:.3f}"
                    )

                if _is_better_eval(eval_summary, best_eval_success_rate, best_eval_reward, config.early_stop_min_delta):
                    best_eval_reward = eval_summary["avg_reward"]
                    best_eval_success_rate = eval_summary["success_rate"]
                    plateau_count = 0
                    agent.save(str(best_path))
                    log(
                        f"[Best] success_rate={best_eval_success_rate:.2f} | avg_reward={best_eval_reward:7.3f} | "
                        f"avg_hidden_ratio={eval_summary['avg_hidden_ratio']:.3f} | saved={best_path}"
                    )
                elif (
                    config.early_stop_enabled
                    and eval_summary["success_rate"] >= config.early_stop_success_rate_threshold
                    and eval_summary["avg_reward"] <= best_eval_reward + config.early_stop_min_delta
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


def evaluate_policy(
    agent: DoubleDQNAgent,
    env: BattlefieldEnv,
    scene_seeds: tuple[int, ...] | list[int] | None = None,
    scenario_mode: str = "fixed",
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, float]:
    rewards: list[float] = []
    hidden_ratios: list[float] = []
    path_lengths: list[float] = []
    successes = 0
    seeds = tuple(scene_seeds) if scene_seeds is not None else tuple([None] * 3)

    for scene_seed in seeds:
        observation = env.reset(scene_seed=scene_seed, scenario_mode=scenario_mode)
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.select_action_masked(observation, env=env)
            result = env.step(action)
            observation = result.observation
            episode_reward += result.reward
            done = result.done
            if result.done and bool(result.info["success"]):
                successes += 1

        rewards.append(episode_reward)
        hidden_ratios.append(env.hidden_ratio)
        path_lengths.append(env.total_path_length)

    avg_reward = statistics.mean(rewards) if rewards else 0.0
    avg_hidden_ratio = statistics.mean(hidden_ratios) if hidden_ratios else 0.0
    avg_path_length = statistics.mean(path_lengths) if path_lengths else 0.0
    success_rate = successes / max(1, len(seeds))
    eval_message = (
        f"[Eval] avg_reward={avg_reward:7.3f} | success_rate={success_rate:.2f} | "
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


if __name__ == "__main__":
    train()
