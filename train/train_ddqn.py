from __future__ import annotations

from pathlib import Path
import statistics

from env.battlefield_env import BattlefieldEnv
from train.dqn_agent import DoubleDQNAgent, TrainingConfig


def train(config: TrainingConfig | None = None) -> None:
    config = config or TrainingConfig()
    env = BattlefieldEnv()
    agent = DoubleDQNAgent(action_dim=len(BattlefieldEnv.ACTIONS), config=config)
    print(f"训练设备: {agent.device}")

    output_dir = Path(__file__).resolve().parents[1] / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "ddqn_latest.pt"
    best_path = output_dir / "ddqn_best.pt"

    global_step = 0
    best_reward = float("-inf")
    recent_rewards: list[float] = []

    for episode in range(1, config.episodes + 1):
        observation = env.reset()
        agent.reset_episode_stats()
        done = False
        episode_reward = 0.0
        episode_loss: list[float] = []
        success = False

        while not done:
            epsilon = agent.current_epsilon(global_step)
            action = agent.select_action(observation, epsilon, env=env, global_step=global_step)
            result = env.step(action)
            agent.store_transition(observation, action, result.reward, result.observation, result.done)

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
        print(
            f"Episode {episode:04d} | reward={episode_reward:7.3f} | "
            f"mean20={mean_reward:7.3f} | loss={mean_loss:6.4f} | "
            f"epsilon={agent.current_epsilon(global_step):5.3f} | success={success} | "
            f"greedy={action_stats['greedy']:03d} heuristic={action_stats['heuristic']:03d} "
            f"teacher={action_stats['teacher']:03d} random={action_stats['random']:03d}"
        )

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(str(best_path))

        if episode % config.save_interval == 0:
            agent.save(str(checkpoint_path))

        if episode % config.eval_interval == 0:
            evaluate_policy(agent, env)

    agent.save(str(checkpoint_path))
    print(f"训练结束，最新模型已保存到: {checkpoint_path}")
    print(f"最佳模型已保存到: {best_path}")


def evaluate_policy(agent: DoubleDQNAgent, env: BattlefieldEnv, episodes: int = 3) -> None:
    rewards: list[float] = []
    successes = 0

    for _ in range(episodes):
        observation = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.select_action(observation, epsilon=0.0, env=env)
            result = env.step(action)
            observation = result.observation
            episode_reward += result.reward
            done = result.done
            if result.done and bool(result.info["success"]):
                successes += 1

        rewards.append(episode_reward)

    avg_reward = statistics.mean(rewards) if rewards else 0.0
    print(f"[Eval] avg_reward={avg_reward:7.3f} | success_rate={successes / max(1, episodes):.2f}")


if __name__ == "__main__":
    train()
