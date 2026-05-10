"""冒烟测试：平坦开阔地形上验证 PPO 算法可行性。

场景：50×50 全平地，无建筑/树木/敌人，起点 (0,0) → 终点 (49,49)。
验证目标：几千步内 policy_loss、value_loss 明显下降，至少偶尔能到达终点。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from config import EnvConfig
from train.ppo_config import PPOConfig
from train.ppo_trainer import PPOTrainer


def _setup_flat_scene(env) -> None:
    """将 env 覆盖为全平坦开阔场景（无敌人、无建筑、无遮挡）。"""
    env.height_map = np.zeros((50, 50), dtype=np.int32)
    env.window_tag_map = None  # _cell_passable → 全部可通行
    env.visibility_map = np.zeros((50, 50), dtype=np.float32)  # 无可见性惩罚
    env.cover_map = np.ones((50, 50), dtype=np.float32)
    env.occupancy_map = np.zeros((50, 50), dtype=np.float32)
    env.height_levels = 1
    env.start_position = np.array([0, 0], dtype=np.int32)
    env.goal_position = np.array([49, 49], dtype=np.int32)
    env.enemy_position = np.array([0, 0, 0], dtype=np.float32)
    env.window_offset = (0, 0)
    env.current_scenario_mode = "smoke_test"
    env.current_progress_weight = 0.1  # 稍低，鼓励探索短路径


def _make_simple_reset(env):
    """返回一个不重新生成场景的 reset 函数，只重置 agent 位置和计数器。"""
    original_reset = env.reset

    def simple_reset(seed: int | None = None) -> np.ndarray:
        env.agent_position = env.start_position.copy()
        env.steps = 0
        env.consecutive_collisions = 0
        return env._get_observation()

    return simple_reset


def main() -> None:
    # —— 配置：小规模快速验证 ——
    env_config = EnvConfig()

    ppo_config = PPOConfig(
        rollout_steps=512,           # 小 rollout，更新更频繁
        minibatch_size=32,
        ppo_epochs=5,
        total_steps=10_000,          # 总共 1 万步（≈20 次 PPO update）
        eval_interval=2_048,         # 每 4 次 update 评估一次
        num_eval_episodes=20,
        learning_rate=1e-3,          # 稍高学习率，加速收敛
        entropy_coef=0.02,           # 稍高熵，鼓励探索
        progress_decay_start=2.0,    # 不衰减
        progress_decay_end=3.0,
    )

    trainer = PPOTrainer(env_config, ppo_config, device="cpu")
    env = trainer.env

    # —— 覆盖为平坦场景 ——
    _setup_flat_scene(env)
    env.reset = _make_simple_reset(env)

    # 验证场景设置
    obs = env.reset()
    mask = env.get_action_mask()
    valid_count = int(mask.sum())
    assert obs.shape == (7, 50, 50), f"obs shape 异常: {obs.shape}"
    assert valid_count > 0, f"无有效动作！"
    print(f"场景就绪: 起点={tuple(env.start_position)}, 终点={tuple(env.goal_position)}")
    print(f"起点有效动作数: {valid_count}/8 (预期 3: 下/右/右下)")
    print(f"obs range: [{obs.min():.2f}, {obs.max():.2f}]")
    print()

    # —— 训练 ——
    print("=" * 60)
    print("开始冒烟训练 (平坦场景, 10k 步, ~20 次 PPO update)")
    print("=" * 60)
    trainer.train(save_dir="checkpoints_smoke")

    # —— 最终评估 ——
    print()
    print("=" * 60)
    print("最终评估 (100 episodes, 确定性策略)")
    print("=" * 60)
    results = trainer._evaluate(100)
    print(f"  到达率:      {results['success_rate']:.1%}")
    print(f"  平均步数:    {results['avg_steps']:.1f}")
    print(f"  平均暴露率:  {results['avg_exposure_ratio']:.3f}")

    if results["success_rate"] > 0.0:
        print()
        print("冒烟测试通过: 算法能学会到达终点！")
    else:
        print()
        print("冒烟测试未通过: 100 次评估均未到达终点，需排查。")
        sys.exit(1)


if __name__ == "__main__":
    main()
