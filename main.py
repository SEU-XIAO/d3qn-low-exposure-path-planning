from env.battlefield_env import BattlefieldEnv
from models.policy_network import HybridPolicyNetwork


def main() -> None:
    env = BattlefieldEnv()
    obs = env.reset()
    model = HybridPolicyNetwork()
    q_values = model.forward_numpy(obs)

    print("环境已初始化")
    print(f"局部地图张量形状: {obs['local_map'].shape}")
    print(f"全局特征维度: {obs['global_features'].shape}")
    print(f"动作价值输出形状: {q_values.shape}")


if __name__ == "__main__":
    main()
