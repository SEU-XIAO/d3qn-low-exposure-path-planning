"""PPO 训练入口脚本。"""

from __future__ import annotations

import argparse

from config import EnvConfig
from train.ppo_config import PPOConfig
from train.ppo_trainer import PPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO 战场寻路训练")
    parser.add_argument("--steps", type=int, default=None, help="总训练步数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--rollout", type=int, default=None, help="每次 rollout 步数")
    parser.add_argument("--epochs", type=int, default=None, help="PPO update epoch 数")
    parser.add_argument("--batch", type=int, default=None, help="Minibatch size")
    parser.add_argument("--save", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    args = parser.parse_args()

    # 构建配置（命令行参数覆盖默认值）
    env_config = EnvConfig()
    ppo_kwargs = {}
    if args.steps is not None:
        ppo_kwargs["total_steps"] = args.steps
    if args.lr is not None:
        ppo_kwargs["learning_rate"] = args.lr
    if args.rollout is not None:
        ppo_kwargs["rollout_steps"] = args.rollout
    if args.epochs is not None:
        ppo_kwargs["ppo_epochs"] = args.epochs
    if args.batch is not None:
        ppo_kwargs["minibatch_size"] = args.batch
    ppo_config = PPOConfig(**ppo_kwargs)

    trainer = PPOTrainer(env_config, ppo_config, device=args.device)
    trainer.train(save_dir=args.save)


if __name__ == "__main__":
    main()
