"""绾枃鏈垎鏋愶細鍦烘櫙缁撴瀯 + 妯″瀷琛屼负璇婃柇銆?""
import numpy as np
import torch
from collections import Counter
import sys
sys.path.insert(0, '.')

from config import EnvConfig
from env.battlefield_env import BattlefieldEnv
from models.actor_critic_cnn import ActorCriticCNN

# 1. 鍦烘櫙姹犵粺璁?data = np.load('artifacts/scene_pool.npz')
bfs_lens = data['bfs_lengths']
print(f'=== 鍦烘櫙姹犵粺璁?===')
print(f'鎬绘暟: {len(bfs_lens)}')
print(f'BFS: min={bfs_lens.min()}, max={bfs_lens.max()}, mean={bfs_lens.mean():.1f}')
n_obs = (data['tags'] != 0).sum(axis=(1, 2))
print(f'闅滅鐗╂暟: min={n_obs.min()}, max={n_obs.max()}, mean={n_obs.mean():.0f}')
print()

# 2. 閫変竴涓満鏅?idx = None
for i in range(min(2000, len(bfs_lens))):
    obs = (data['tags'][i] != 0).sum()
    if 10 <= obs <= 200 and 40 <= bfs_lens[i] <= 80:
        idx = i
        break
if idx is None:
    idx = 42
print(f'=== 鍦烘櫙 {idx} ===')
print(f'BFS length: {bfs_lens[idx]}')
print(f'闅滅鐗? ground={(data["tags"][idx]==0).sum()} build={(data["tags"][idx]==1).sum()} tree={(data["tags"][idx]==2).sum()}')
print(f'start={data["starts"][idx]}, goal={data["goals"][idx]}')
print(f'楂樺害鑼冨洿: {data["heights"][idx].min()} - {data["heights"][idx].max()}')
print()

# 3. 鍒濆鍖栫幆澧?cfg = EnvConfig()
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
print(f'BFS 璺緞闀垮害: {len(bfs_path) if bfs_path else "涓嶅彲杈?}')

# 鏆撮湶鐜囷紙闃舵1鏃犳晫浜烘墍浠ュ叏涓?锛屼絾鐪嬬湅mask锛?action_mask = env.get_action_mask()
print(f'Action mask @ start: {action_mask} (鏈夋晥: {action_mask.sum()}/8)')

# 4. 鍔犺浇妯″瀷
print('\n=== 鍔犺浇 best model ===')
device = torch.device('cpu')
policy = ActorCriticCNN().to(device)
ckpt = torch.load('checkpoints_local_pool/policy_best.pt', map_location=device, weights_only=True)
policy.load_state_dict(ckpt)
policy.eval()
print('妯″瀷鍔犺浇瀹屾垚')

# 5. 璺戣建杩?print('\n=== 妯″瀷杞ㄨ抗 ===')
positions = [tuple(env.agent_position.copy())]
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
print(f'姝ユ暟: {len(positions)-1}')
print(f'缁撴灉: {result}')
print(f'纰版挒: {info.get("collisions", "?")}')
print(f'鍒拌揪缁堢偣: {np.array_equal(env.agent_position, env.goal_position)}')
print(f'绱濂栧姳: {sum(rewards):.1f}')
print(f'鏈€缁堜綅缃? {env.agent_position}, 缁堢偣: {env.goal_position}')

ACT_NAMES = ['涓?,'涓?,'宸?,'鍙?,'宸︿笂','鍙充笂','宸︿笅','鍙充笅']
counter = Counter(actions_taken)
print(f'鍔ㄤ綔鍒嗗竷:')
for a in range(8):
    print(f'  {ACT_NAMES[a]}: {counter.get(a,0)}')

dists = [np.linalg.norm(np.array(p, dtype=np.float32) - env.goal_position.astype(np.float32))
         for p in positions]
print(f'璺濈粓鐐? {dists[0]:.1f} -> {dists[-1]:.1f} (鍙樺寲: {dists[-1]-dists[0]:.1f})')
print(f'浠峰€艰寖鍥? {min(values):.3f} - {max(values):.3f}')

# 6. 鎵撳嵃鍦烘櫙 ASCII锛堝皬鍥撅級
print('\n=== 鍦烘櫙 ASCII (寤虹瓚=B, 鏍戞湪=T, 鍦伴潰=., S=璧风偣, G=缁堢偣) ===')
tag_map = env.window_tag_map
sx, sy = int(env.start_position[0]), int(env.start_position[1])
gx, gy = int(env.goal_position[0]), int(env.goal_position[1])

# 閲囨牱鎵撳嵃 (姣?鏍煎悎骞?
for x in range(0, 50, 2):
    line = ''
    for y in range(0, 50, 2):
        cell_tags = tag_map[x:min(x+2,50), y:min(y+2,50)]
        if x <= sx < x+2 and y <= sy < y+2:
            ch = 'S'
        elif x <= gx < x+2 and y <= gy < y+2:
            ch = 'G'
        elif (cell_tags == 1).sum() > (cell_tags == 2).sum() and (cell_tags == 1).sum() > 0:
            ch = 'B'
        elif (cell_tags == 2).sum() > 0:
            ch = 'T'
        else:
            ch = '.'
        line += ch
    print(f'{x:2d} {line}')

# 鏍囪杞ㄨ抗
print('\n=== 杞ㄨ抗璺緞 (鍓?5姝? ===')
for i, (x, y) in enumerate(positions[:15]):
    tag = tag_map[x, y]
    print(f'  绗瑊i}姝? ({x},{y}) tag={tag} 璺濈洰鏍?{dists[i]:.1f}')
if len(positions) > 15:
    print(f'  ... 鍏眥len(positions)-1}姝?)

# 7. 鐪媜bs涓璦gent/goal閫氶亾
obs = env._get_observation()
print(f'\n=== 瑙傛祴閫氶亾 ===')
print(f'obs shape: {obs.shape}')
for c in range(7):
    print(f'  閫氶亾{c}: min={obs[c].min():.3f} max={obs[c].max():.3f} sum={obs[c].sum():.3f} nonzero={(obs[c]!=0).sum()}')

# agent/goal one-hot浣嶇疆
agent_idx = np.unravel_index(obs[5].argmax(), obs[5].shape)
goal_idx = np.unravel_index(obs[6].argmax(), obs[6].shape)
print(f'瑙傛祴涓璦gent浣嶇疆: {agent_idx}, 瀹為檯浣嶇疆: {env.start_position}')
print(f'瑙傛祴涓璯oal浣嶇疆:  {goal_idx}, 瀹為檯浣嶇疆: {env.goal_position}')
print(f'agent閫氶亾鍊? {obs[5, agent_idx[0], agent_idx[1]]:.3f}')
print(f'goal閫氶亾鍊?  {obs[6, goal_idx[0], goal_idx[1]]:.3f}')


