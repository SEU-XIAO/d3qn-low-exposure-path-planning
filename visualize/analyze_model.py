"""鍒嗘瀽鍦烘櫙姹犲拰妯″瀷琛屼负鐨勫彲瑙嗗寲鑴氭湰銆?""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from pathlib import Path
from collections import Counter

from config import EnvConfig
from env.battlefield_env import BattlefieldEnv
from models.actor_critic_cnn import ActorCriticCNN
from visualize.visualizer import C_GROUND, C_BUILDING, C_TREE, C_START, C_GOAL, _draw_cell_grid


def main():
    # 鍔犺浇鍦烘櫙姹?    data = np.load('artifacts/scene_pool.npz')
    bfs_lens = data['bfs_lengths']

    # 鎵句竴涓腑绛夐毦搴﹀満鏅?    candidates = []
    for i in range(min(2000, len(bfs_lens))):
        n_obs = (data['tags'][i] != 0).sum()
        if 10 <= n_obs <= 200 and 40 <= bfs_lens[i] <= 80:
            candidates.append(i)
        if len(candidates) >= 10:
            break
    print(f'Found {len(candidates)} moderate scenes')
    idx = candidates[3] if len(candidates) > 3 else 42
    print(f'Scene {idx}: bfs_len={bfs_lens[idx]}, obstacles={(data["tags"][idx]!=0).sum()}')
    print(f'start={data["starts"][idx]}, goal={data["goals"][idx]}')

    # 鍦烘櫙缁熻
    total_obs = sum((data['tags'][i] != 0).sum() for i in range(len(bfs_lens)))
    print(f'鍦烘櫙姹? {len(bfs_lens)} scenes, avg obstacles: {total_obs/len(bfs_lens):.1f}')
    print(f'BFS length range: {bfs_lens.min()} - {bfs_lens.max()}, avg: {bfs_lens.mean():.1f}')

    # 鍒濆鍖栫幆澧?    cfg = EnvConfig()
    env = BattlefieldEnv(cfg)
    env.full_terrain = None
    env.full_visibility_maps = []
    env.enemy_pool = []
    env.current_progress_weight = cfg.progress_weight

    env.height_map = data['heights'][idx].copy()
    env.window_tag_map = data['tags'][idx].copy()
    env.start_position = data['starts'][idx].copy()
    env.goal_position = data['goals'][idx].copy()
    env.visibility_map = np.zeros((50, 50), dtype=np.float32)
    env.cover_map = np.ones((50, 50), dtype=np.float32)
    env.occupancy_map = env.height_map.astype(np.float32) / max(1.0, float(env.height_levels))
    env.window_offset = (0, 0)
    env.enemy_position = np.array([-1, -1, 0], dtype=np.float32)
    env.current_scenario_mode = 'full_map'
    env.full_terrain = None
    env.agent_position = env.start_position.copy()
    env.steps = 0
    env.consecutive_collisions = 0
    env.total_collisions = 0

    bfs_path = env.compute_bfs_path()
    print(f'BFS path: {len(bfs_path) if bfs_path else "unreachable"}')

    # 鍔犺浇妯″瀷
    device = torch.device('cpu')
    policy = ActorCriticCNN().to(device)
    ckpt = torch.load('checkpoints_local_pool/policy_best.pt', map_location=device, weights_only=True)
    policy.load_state_dict(ckpt)
    policy.eval()

    # 璺戣建杩?    positions = [tuple(env.agent_position.copy())]
    actions_taken = []
    rewards = []
    values = []
    done = False
    while not done and len(positions) <= 250:
        obs_t = torch.from_numpy(env._get_observation()).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(env.get_action_mask()).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, value, _, logits = policy(obs_t, mask_t, deterministic=True)
        _, reward, done, info = env.step(action.item())
        positions.append(tuple(env.agent_position.copy()))
        actions_taken.append(action.item())
        rewards.append(reward)
        values.append(value.item())

    result = info.get('result', 'timeout')
    print(f'Trajectory: {len(positions)-1} steps, sum reward: {sum(rewards):.1f}')
    print(f'Result: {result}, collisions: {info.get("collisions", "?")}')
    print(f'Reached goal: {np.array_equal(env.agent_position, env.goal_position)}')
    print(f'Final pos: {env.agent_position}, goal: {env.goal_position}')
    print(f'Action dist: {Counter(actions_taken)}')
    print(f'Value range: {min(values):.3f} - {max(values):.3f}')

    # 璁＄畻姣忎釜浣嶇疆鍒扮粓鐐圭殑璺濈
    dists = [np.linalg.norm(np.array(p, dtype=np.float32) - env.goal_position.astype(np.float32))
             for p in positions]
    print(f'Distance to goal: start={dists[0]:.1f} -> end={dists[-1]:.1f}')

    # ===== 鍙鍖?=====
    fig, axes = plt.subplots(1, 3, figsize=(27, 8.5))

    ACTIONS_NAMES = ['涓?, '涓?, '宸?, '鍙?, '宸︿笂', '鍙充笂', '宸︿笅', '鍙充笅']

    for ax_i, (title, path_data) in enumerate([
        ("BFS 鏈€鐭矾寰?, bfs_path),
        ("鏅鸿兘浣撹建杩?(best model)", positions),
    ]):
        ax = axes[ax_i]
        tag_map = env.window_tag_map
        H, W = tag_map.shape
        rgba = np.zeros((H, W, 4), dtype=np.float32)
        rgba[tag_map == 0] = C_GROUND
        rgba[tag_map == 1] = C_BUILDING
        rgba[tag_map == 2] = C_TREE

        ax.imshow(rgba.transpose(1, 0, 2), origin="lower", interpolation="nearest")
        _draw_cell_grid(ax, env.grid_size)

        sx, sy = int(env.start_position[0]), int(env.start_position[1])
        gx, gy = int(env.goal_position[0]), int(env.goal_position[1])
        ax.scatter(sy, sx, c=C_START, s=200, marker="o", edgecolors="white", linewidths=2.0, zorder=20)
        ax.scatter(gy, gx, c=C_GOAL, s=260, marker="*", edgecolors="white", linewidths=2.0, zorder=20)

        if path_data:
            px = [p[0] for p in path_data]
            py = [p[1] for p in path_data]
            n = len(path_data)
            ax.plot(py, px, color="white", linewidth=4.5, zorder=11, solid_capstyle="round")
            for i in range(n - 1):
                t = i / max(n - 2, 1)
                r, g, b = 0.9 * t, 0.75 * (1 - t) + 0.1 * t, 0.85 * (1 - t) + 0.55 * t
                ax.plot([py[i], py[i + 1]], [px[i], px[i + 1]], color=(r, g, b), linewidth=2.5, zorder=12)
            ax.scatter(py, px, c="white", s=6, zorder=13)

        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        ax.set_xticks(range(0, env.grid_size, 5))
        ax.set_yticks(range(0, env.grid_size, 5))
        ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=13, fontweight="bold")

        legend_elements = [
            mpatches.Patch(facecolor=C_GROUND, label="鍦伴潰"),
            mpatches.Patch(facecolor=C_BUILDING, label="寤虹瓚"),
            mpatches.Patch(facecolor=C_TREE, label="鏍戞湪"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_START, markersize=10, label="璧风偣"),
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=C_GOAL, markersize=12, label="缁堢偣"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7.5, framealpha=0.92)

    # 绗笁涓瓙鍥撅細濂栧姳鍜屼环鍊?    ax3 = axes[2]
    steps_range = range(len(rewards))
    ax3.plot(steps_range, np.cumsum(rewards), 'b-', linewidth=1.5, label='绱Н濂栧姳')
    ax3.plot(steps_range, values, 'r--', linewidth=1, alpha=0.7, label='V(s)')
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('姝ユ暟', fontsize=10)
    ax3.set_ylabel('鍊?, fontsize=10)
    ax3.set_title('绱Н濂栧姳涓庝环鍊间及璁?, fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path("artifacts/plots/model_analysis.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f'\n鍥剧墖宸蹭繚瀛? {save_path}')


if __name__ == "__main__":
    main()


