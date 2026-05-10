"""PPO 训练超参数配置（独立于 EnvConfig）。"""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    # —— 折扣与优势估计 ——
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # —— PPO 裁剪 ——
    clip_epsilon: float = 0.2

    # —— 优化器 ——
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5

    # —— 损失系数 ——
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # —— 训练规模 ——
    rollout_steps: int = 2048
    minibatch_size: int = 64
    ppo_epochs: int = 10
    total_steps: int = 2_000_000

    # —— 泛化 ——
    eval_interval: int = 10_000
    num_eval_episodes: int = 50

    # —— 进度奖励衰减 ——
    progress_decay_start: float = 0.5   # 训练进度 50% 时开始衰减
    progress_decay_end: float = 0.9     # 训练进度 90% 时完全衰减为 0
