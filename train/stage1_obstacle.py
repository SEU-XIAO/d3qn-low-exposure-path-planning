"""阶段1：障碍地形 · 纯导航训练（无敌人）。

验证动作掩码在建筑/树木/陡坡约束下有效，智能体学会绕行，不会撞墙卡死。

用法: python -m train.stage1_obstacle --steps 500000 --pool artifacts/scene_pool.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from config import EnvConfig
from env.battlefield_env import BattlefieldEnv
from models.policy_network import ActorCriticCNN
from train.ppo_config import PPOConfig
from train.ppo_trainer import PPOTrainer


def _make_pool_reset(env: BattlefieldEnv, pool: dict, rng: np.random.Generator):
    """返回从场景池采样的 reset 函数。"""
    n_scenes = len(pool["starts"])
    original_reset = env.reset

    def pool_reset(seed: int | None = None) -> np.ndarray:
        idx = int(rng.integers(0, n_scenes))
        env.height_map = pool["heights"][idx].copy()
        env.window_tag_map = pool["tags"][idx].copy()
        env.start_position = pool["starts"][idx].copy()
        env.goal_position = pool["goals"][idx].copy()
        env._bfs_optimal_length = int(pool["bfs_lengths"][idx])

        env.visibility_map = np.zeros((env.grid_size, env.grid_size), dtype=np.float32)
        env.cover_map = np.ones((env.grid_size, env.grid_size), dtype=np.float32)
        env.occupancy_map = (
            env.height_map.astype(np.float32) / max(1.0, float(env.height_levels))
        )
        env.window_offset = (0, 0)
        env.enemy_position = np.array([-1, -1, 0], dtype=np.float32)
        env.current_scenario_mode = "full_map"  # 跳过 _is_blocked 中的敌人位置检查

        env.agent_position = env.start_position.copy()
        env.steps = 0
        env.consecutive_collisions = 0
        env.total_collisions = 0
        return env._get_observation()

    return pool_reset


def _make_pool_evaluate(env: BattlefieldEnv, policy: ActorCriticCNN, device: torch.device):
    """返回增强版评估函数，额外统计碰撞次数和路径比。"""

    def pool_evaluate(num_episodes: int) -> dict:
        successes = 0
        total_steps = 0
        total_collisions = 0
        total_exposure = 0
        total_path_ratio = 0.0  # actual_steps / bfs_optimal

        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            ep_steps = 0
            ep_exposed = 0
            ep_collisions = 0
            bfs_opt = getattr(env, "_bfs_optimal_length", None)

            while not done:
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                mask_t = (
                    torch.from_numpy(env.get_action_mask()).unsqueeze(0).to(device)
                )

                with torch.no_grad():
                    action, _, _, _, _ = policy(obs_t, mask_t, deterministic=True)

                obs, _reward, done, info = env.step(action.item())
                ep_steps += 1

                if env.visibility_map[tuple(env.agent_position)] > 0.5:
                    ep_exposed += 1

            ep_collisions = info.get("collisions", 0)
            total_steps += ep_steps
            total_collisions += ep_collisions
            total_exposure += ep_exposed / max(1, ep_steps)

            if info.get("result") == "success":
                successes += 1
                if bfs_opt and bfs_opt > 0:
                    total_path_ratio += ep_steps / bfs_opt
            elif bfs_opt and bfs_opt > 0:
                # 失败的 episode 也计入路径比（用实际步数上限）
                total_path_ratio += ep_steps / bfs_opt

        n = num_episodes
        return {
            "success_rate": successes / n,
            "avg_steps": total_steps / n,
            "avg_collisions": total_collisions / n,
            "avg_path_ratio": total_path_ratio / n,
            "avg_exposure_ratio": total_exposure / n,
        }

    return pool_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段1：障碍地形纯导航训练")
    parser.add_argument("--steps", type=int, default=500_000, help="总训练步数")
    parser.add_argument("--pool", type=str, default="artifacts/scene_pool.npz",
                        help="场景池 NPZ 文件路径")
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
        "heights": data["heights"],        # (N, 50, 50) int32
        "tags": data["tags"],               # (N, 50, 50) int32
        "starts": data["starts"],           # (N, 2) int32
        "goals": data["goals"],             # (N, 2) int32
        "bfs_lengths": data["bfs_lengths"], # (N,) int32
    }
    n_scenes = len(pool["starts"])
    print(f"  {n_scenes} 个预生成场景")
    print(f"  BFS 长度范围: [{pool['bfs_lengths'].min()}, {pool['bfs_lengths'].max()}]")

    # —— 配置 ——
    env_config = EnvConfig()
    ppo_config = PPOConfig(
        total_steps=args.steps,
        learning_rate=args.lr,
        eval_interval=max(10_000, args.steps // 50),  # 约 50 次评估
    )

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # —— 创建 trainer ——
    trainer = PPOTrainer(env_config, ppo_config, device=device_str)
    env = trainer.env

    # —— 覆盖 env 为场景池模式 ——
    rng = np.random.default_rng(42)
    env.reset = _make_pool_reset(env, pool, rng)
    env._bfs_optimal_length = 0  # 占位，每次 reset 更新

    # 关闭可见性惩罚（无敌人）
    env.current_progress_weight = env_config.progress_weight

    # 先 reset 一次验证
    obs = env.reset()
    mask = env.get_action_mask()
    valid_count = int(mask.sum())
    assert obs.shape == (7, 50, 50), f"obs shape 异常: {obs.shape}"
    assert valid_count > 0, "无有效动作！"
    print(f"场景就绪: 起点={tuple(env.start_position)}, "
          f"终点={tuple(env.goal_position)}")
    print(f"起点有效动作数: {valid_count}/8")
    print(f"BFS 最优路径: {env._bfs_optimal_length} 步")
    print(f"建筑格数: {int((env.window_tag_map == 1).sum())}, "
          f"树木格数: {int((env.window_tag_map == 2).sum())}")
    print()

    # —— 覆盖评估函数 ——
    trainer._evaluate = _make_pool_evaluate(env, trainer.policy, trainer.device)
    # 也覆盖 _evaluate_and_save 以打印新指标
    original_eval_and_save = trainer._evaluate_and_save

    def enhanced_eval_and_save(save_path):
        cfg = trainer.ppo_config
        results = trainer._evaluate(cfg.num_eval_episodes)
        print(
            f"  >>> Eval @ {trainer.global_step:>8,} | "
            f"success {results['success_rate']:.1%} | "
            f"steps {results['avg_steps']:.1f} | "
            f"collisions {results['avg_collisions']:.2f} | "
            f"path_ratio {results['avg_path_ratio']:.2f} | "
            f"exposure {results['avg_exposure_ratio']:.3f}"
        )
        if results["success_rate"] >= trainer.best_eval_rate:
            trainer.best_eval_rate = results["success_rate"]
            torch.save(trainer.policy.state_dict(), save_path / "policy_best.pt")
            print(f"  >>> 最佳模型已保存")
        torch.save(
            trainer.policy.state_dict(),
            save_path / f"policy_{trainer.global_step}.pt",
        )
        return results

    trainer._evaluate_and_save = enhanced_eval_and_save

    # —— 训练 ——
    print("=" * 60)
    print(f"阶段1：障碍地形纯导航训练")
    print(f"  总步数: {args.steps:,}  场景池: {n_scenes} 场景")
    print(f"  爬坡约束: tan ≤ {env_config.max_climb_tan}")
    print(f"  无敌人，可见性惩罚 = 0")
    print("=" * 60)
    print()

    trainer.train(save_dir=args.save)

    # —— 最终评估 ——
    print()
    print("=" * 60)
    print("最终评估 (100 episodes, 确定性策略)")
    print("=" * 60)
    results = trainer._evaluate(100)
    print(f"  到达率:       {results['success_rate']:.1%}")
    print(f"  平均步数:     {results['avg_steps']:.1f}")
    print(f"  平均碰撞:     {results['avg_collisions']:.2f}")
    print(f"  路径比:       {results['avg_path_ratio']:.2f} (实际/BFS最优)")
    print(f"  平均暴露率:   {results['avg_exposure_ratio']:.3f}")

    # 达标检查
    if results["success_rate"] >= 0.6 and results["avg_collisions"] < 2.0:
        print()
        print("阶段1达标: 成功率≥60% 且 平均碰撞<2！可以进入阶段2。")
    elif results["success_rate"] >= 0.6:
        print()
        print("成功率已达标，但碰撞数偏高(≥2)，mask 可能有问题，需排查。")
    else:
        print()
        print("阶段1未达标: 成功率不足60%。建议：")
        print("  1. 增加训练步数到 100 万")
        print("  2. 用 A* 生成专家轨迹做 BC 预训练")


if __name__ == "__main__":
    main()
