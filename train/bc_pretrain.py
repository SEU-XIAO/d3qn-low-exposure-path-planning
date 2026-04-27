"""Behavioral Cloning 预训练：用 Visibility-A* 生成专家路径，监督学习初始化网络。

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
from config import EnvConfig
from env.battlefield_env import BattlefieldEnv
from models.policy_network import HybridPolicyNetwork
from planner.visibility_astar import VisibilityAwareAStarPlanner


def generate_expert_data(episodes: int = 2000) -> list[dict[str, np.ndarray]]:
    """生成专家轨迹数据集。"""
    config = EnvConfig()
    env = BattlefieldEnv(config)
    dataset: list[dict[str, np.ndarray]] = []
    path_count = 0

    print(f"生成专家数据 (目标 {episodes} 条路径)...")
    for seed in range(1000, 1000 + episodes * 3):
        if path_count >= episodes:
            break
        try:
            obs = env.reset(scene_seed=seed, scenario_mode="full_map")
        except RuntimeError:
            continue

        planner = VisibilityAwareAStarPlanner(env, visible_weight=8.0)
        result = planner.plan()
        if not result.success or len(result.path) < 2:
            continue

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
            dataset.append({
                "local_map": obs["local_map"].copy(),
                "global_features": obs["global_features"].copy(),
                "action": action,
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
    """BC 训练：交叉熵损失拟合专家动作分布。"""
    net = HybridPolicyNetwork(action_dim=8)
    if device.startswith("cuda") and torch.cuda.is_available():
        net = net.to(device)
    else:
        device = "cpu"
        net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = len(dataset)
    indices = np.arange(n)

    print(f"\nBC 训练: {epochs} epochs, {n} 样本, lr={lr}, batch={batch_size}")
    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        correct = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            local_batch = np.stack([dataset[i]["local_map"] for i in batch_idx])
            global_batch = np.stack([dataset[i]["global_features"] for i in batch_idx])
            action_batch = np.array([dataset[i]["action"] for i in batch_idx], dtype=np.int64)

            local_t = torch.from_numpy(local_batch).float().to(device)
            global_t = torch.from_numpy(global_batch).float().to(device)
            action_t = torch.from_numpy(action_batch).long().to(device)

            net.train()
            q_values = net(local_t, global_t)
            loss = loss_fn(q_values, action_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)
            correct += int((torch.argmax(q_values, dim=1) == action_t).sum().item())

        acc = correct / n
        avg_loss = total_loss / n
        print(f"  Epoch {epoch + 1:3d}/{epochs} | loss={avg_loss:.4f} | acc={acc:.4f}")

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
    parser.add_argument("--data-cache", type=str, default="artifacts/expert_data.pt",
                        help="缓存专家数据的路径（若存在则跳过生成）")
    args = parser.parse_args()

    cache_path = Path(args.data_cache)
    if cache_path.exists():
        print(f"加载缓存的专家数据: {cache_path}")
        raw = torch.load(cache_path, map_location="cpu", weights_only=False)
        dataset = [{"local_map": lm, "global_features": gf, "action": int(a)}
                   for lm, gf, a in zip(raw["local_maps"], raw["global_features"], raw["actions"])]
    else:
        dataset = generate_expert_data(args.episodes)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "local_maps": np.stack([d["local_map"] for d in dataset]),
                "global_features": np.stack([d["global_features"] for d in dataset]),
                "actions": np.array([d["action"] for d in dataset], dtype=np.int64),
            },
            cache_path,
        )
        print(f"专家数据缓存至: {cache_path}")

    net = train_bc(dataset, epochs=args.epochs, lr=args.lr,
                   batch_size=args.batch_size, device=args.device)
    save_bc_checkpoint(net, args.output)


if __name__ == "__main__":
    main()
