"""Behavioral Cloning 预训练：用 Visibility-A* 生成专家路径，监督学习初始化网络。

训练目标：MSE 拟合 Monte Carlo 回报（校准 Q 值量级）+ 小权重 CE 保动作正确。

用法：
    python -m train.bc_pretrain [--episodes 2000] [--epochs 50] [--lr 1e-3]

输出 artifacts/ddqn_bc.pt 可直接由 DoubleDQNAgent.load() 加载。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EnvConfig, TrainingDefaults
from env.battlefield_env import BattlefieldEnv
from models.policy_network import HybridPolicyNetwork
from planner.visibility_astar import VisibilityAwareAStarPlanner

GAMMA = TrainingDefaults().gamma  # 0.99，与 DQN 一致
CE_WEIGHT = 1.0  # CE 损失权重，与 MSE 平衡（MSE 校准量级，CE 保序）


def generate_expert_data(episodes: int = 2000) -> list[dict[str, np.ndarray]]:
    """生成专家轨迹数据集，每步附带 Monte Carlo 回报。"""
    config = EnvConfig()
    env = BattlefieldEnv(config)
    dataset: list[dict[str, np.ndarray]] = []
    path_count = 0

    sp = config.step_penalty
    vp = config.visible_penalty
    pw = config.progress_weight
    gr = config.goal_reward

    print(f"生成专家数据 (目标 {episodes} 条路径)...")
    for seed in range(1000, 1000 + episodes * 3):
        if path_count >= episodes:
            break
        try:
            obs = env.reset(scene_seed=seed, scenario_mode="full_map")
        except RuntimeError:
            continue

        planner = VisibilityAwareAStarPlanner(env, visible_weight=3.0)
        result = planner.plan()
        if not result.success or len(result.path) < 2:
            continue

        # 第一遍：收集路径上的每步信息
        step_records: list[dict] = []
        goal_pos = np.array(result.path[-1], dtype=np.int32)
        for i in range(len(result.path) - 1):
            current = result.path[i]
            nxt = result.path[i + 1]
            move = (nxt[0] - current[0], nxt[1] - current[1])
            try:
                action = env.ACTIONS.index(move)
            except ValueError:
                break

            env.agent_position = np.array(current, dtype=np.int32)
            obs = env.get_observation()
            move_cost = 1.414 if move[0] != 0 and move[1] != 0 else 1.0
            vis = float(env.visibility_map[tuple(nxt)])
            prev_dist = float(np.linalg.norm(np.array(current, dtype=np.float32) - goal_pos.astype(np.float32)))
            cur_dist = float(np.linalg.norm(np.array(nxt, dtype=np.float32) - goal_pos.astype(np.float32)))
            step_records.append({
                "local_map": obs["local_map"].copy(),
                "global_features": obs["global_features"].copy(),
                "action": action,
                "move_cost": move_cost,
                "visibility": vis,
                "prev_dist": prev_dist,
                "cur_dist": cur_dist,
            })

        # 第二遍：反向计算 MC 回报
        mc_return = 0.0  # 终局后无未来收益
        for idx, rec in enumerate(reversed(step_records)):
            step_r = -(sp + vp * rec["visibility"]) * rec["move_cost"] \
                     + (rec["prev_dist"] - rec["cur_dist"]) * pw
            if idx == 0:  # 最后一步到达目标
                step_r += gr
            mc_return = step_r + GAMMA * mc_return
            rec["mc_return"] = float(mc_return)

        for rec in step_records:
            dataset.append({
                "local_map": rec["local_map"],
                "global_features": rec["global_features"],
                "action": rec["action"],
                "mc_return": rec["mc_return"],
            })

        path_count += 1

        if len(dataset) % 5000 == 0 or (len(dataset) > 0 and len(dataset) <= 100):
            print(f"  已收集 {len(dataset)} 个状态-动作对 ({path_count} / {episodes} 条路径)")

    print(f"专家数据生成完成: {len(dataset)} 个样本")
    return dataset


def train_bc(
    dataset: list[dict[str, np.ndarray]],
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 512,
    device: str = "cuda",
) -> HybridPolicyNetwork:
    """BC 训练：MSE 回归 MC 回报 + 小权重 CE 保序。"""
    net = HybridPolicyNetwork(action_dim=8)
    if device.startswith("cuda") and torch.cuda.is_available():
        net = net.to(device)
    else:
        device = "cpu"
        net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_mse = nn.MSELoss()
    loss_ce = nn.CrossEntropyLoss()

    n = len(dataset)
    indices = np.arange(n)

    print(f"\nBC 训练: {epochs} epochs, {n} 样本, lr={lr}, batch={batch_size}")
    print(f"  目标: MSE(Q[expert] → MC_return) + {CE_WEIGHT} * CE")
    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_mse = 0.0
        total_ce = 0.0
        correct = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            local_batch = np.stack([dataset[i]["local_map"] for i in batch_idx])
            global_batch = np.stack([dataset[i]["global_features"] for i in batch_idx])
            action_batch = np.array([dataset[i]["action"] for i in batch_idx], dtype=np.int64)
            mc_return_batch = np.array([dataset[i]["mc_return"] for i in batch_idx], dtype=np.float32)

            local_t = torch.from_numpy(local_batch).float().to(device)
            global_t = torch.from_numpy(global_batch).float().to(device)
            action_t = torch.from_numpy(action_batch).long().to(device)
            mc_t = torch.from_numpy(mc_return_batch).float().to(device)

            net.train()
            q_values = net(local_t, global_t)  # [B, 8]

            q_expert = q_values[range(len(action_t)), action_t]
            mse = loss_mse(q_expert, mc_t)
            ce = loss_ce(q_values, action_t)
            loss = mse + CE_WEIGHT * ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_batch = len(batch_idx)
            total_mse += mse.item() * n_batch
            total_ce += ce.item() * n_batch
            correct += int((torch.argmax(q_values, dim=1) == action_t).sum().item())

        acc = correct / n
        avg_mse = total_mse / n
        avg_ce = total_ce / n
        print(f"  Epoch {epoch + 1:3d}/{epochs} | mse={avg_mse:.4f} | ce={avg_ce:.4f} | acc={acc:.4f}")

    return net


def save_bc_checkpoint(net: HybridPolicyNetwork, path: str) -> None:
    torch.save(
        {
            "online_state_dict": net.state_dict(),
            "target_state_dict": net.state_dict(),
            "training_steps": 0,
            "last_loss": 0.0,
        },
        path,
    )
    print(f"\nBC 权重已保存到 {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BC 预训练")
    parser.add_argument("--episodes", type=int, default=2000, help="专家路径条数")
    parser.add_argument("--epochs", type=int, default=50, help="BC 训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="BC 学习率")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="artifacts/ddqn_bc.pt")
    args = parser.parse_args()

    dataset = generate_expert_data(args.episodes)

    net = train_bc(dataset, epochs=args.epochs, lr=args.lr,
                   batch_size=args.batch_size, device=args.device)
    save_bc_checkpoint(net, args.output)


if __name__ == "__main__":
    main()
