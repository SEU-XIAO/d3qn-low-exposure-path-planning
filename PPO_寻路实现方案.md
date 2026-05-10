# PPO 瀵昏矾瀹炵幇鏂规锛堜慨璁㈢増锛?
## 鐩爣

鐢?PPO 绠楁硶璁粌涓€涓湪 50脳50 鎴樺満涓粠璧风偣璧板埌缁堢偣鐨勬櫤鑳戒綋锛屾渶灏忓寲琚晫浜虹湅鍒扮殑鏆撮湶鏃堕棿銆?
---

## 涓€銆佹€讳綋鎶€鏈矾绾?
**绠楁硶**: PPO (Proximal Policy Optimization) + GAE (Generalized Advantage Estimation)
**妗嗘灦**: PyTorch锛屼粠闆跺疄鐜帮紙涓嶄緷璧?stable-baselines3锛夛紝渚夸簬鑷畾涔?action masking 鍜?observation 鏋勫缓
**鐜**: `BattlefieldEnv`锛岃ˉ鍏?RL 鎺ュ彛锛坄reset`/`step`/`get_obs`锛?**娉涘寲**: 姣?episode 鍦ㄥ叏鍥句笂闅忔満婊戝姩绐楀彛锛岃缁冩椂鍔犳暟鎹寮猴紙鏃嬭浆/缈昏浆锛?
---

## 浜屻€佺缁忕綉缁滆璁?
### 2.1 杈撳叆锛氬閫氶亾鍏ㄥ浘瑙傛祴 (7 脳 50 脳 50)

鍏ㄥ浘 CNN 鏂规锛?0脳50 涓嶇畻澶э紝鏅鸿兘浣撻渶瑕佺湅鍒板畬鏁寸殑鍙鎬у簳鍥惧拰鐩爣浣嶇疆鎵嶈兘瑙勫垝闅愯斀璺緞銆?
| 閫氶亾 | 鍐呭 | 鑼冨洿 |
|------|------|------|
| 0 | 褰掍竴鍖栭珮搴﹀浘 `height / max(1, height_levels)` | [0, 1] |
| 1 | 鍦伴潰鎺╃爜 `tag == 0` | {0, 1} |
| 2 | 寤虹瓚鎺╃爜 `tag == 1` | {0, 1} |
| 3 | 鏍戞湪鎺╃爜 `tag == 2` | {0, 1} |
| 4 | 鍙鎬у簳鍥?`visibility_map` | {0, 1} |
| 5 | 鏅鸿兘浣撲綅缃?鈥?楂樻柉鏂戯紝蟽=2锛屼腑蹇冨湪 agent 鍧愭爣 | [0, 1] |
| 6 | 缁堢偣浣嶇疆 鈥?楂樻柉鏂戯紝蟽=2锛屼腑蹇冨湪 goal 鍧愭爣 | [0, 1] |

**鍏抽敭鏀硅繘**锛氶€氶亾 5/6 涓嶄娇鐢?one-hot 鑰岀敤楂樻柉鏂戯紙蟽=2锛夛紝澶╃劧鎼哄甫鐩稿璺濈淇℃伅锛岄伩鍏嶇綉缁滆蹇嗙粷瀵瑰潗鏍囷紝鍒╀簬璺ㄧ獥鍙ｆ硾鍖栥€?
鏁屾柟浣嶇疆闅愬惈鍦ㄩ€氶亾 4 鍙鎬у簳鍥句腑锛屾棤闇€鍗曠嫭閫氶亾銆?
### 2.2 鏁版嵁澧炲己锛堣缁冩椂锛?
姣忎釜 batch 鐨勮娴嬮殢鏈烘柦鍔狅細
- 90掳/180掳/270掳 鏃嬭浆锛堢瓑姒傜巼 4 閫?1锛?- 姘村钩缈昏浆锛?0% 姒傜巼锛?- 鍨傜洿缈昏浆锛?0% 姒傜巼锛?
**鍔ㄤ綔鏄犲皠**锛氬寮哄悗鐨勮娴嬪搴旂殑鏈夋晥鍔ㄤ綔涔熷仛鐩稿簲鍙樻崲锛堝鏃嬭浆 90掳 鍚庯紝"涓?鍙樻垚"鍙?锛夈€傝繖鍑犱箮涓嶅鍔犺绠楁垚鏈紝浣嗗己鍒剁綉缁滃涔犵浉瀵规柟鍚戝叧绯汇€?
### 2.3 缃戠粶缁撴瀯锛欳NN Backbone + Actor/Critic 鍙屽ご

```
Input: (7, 50, 50)
  鈫?Conv2d(7鈫?2, k=5, s=2, p=2) + ReLU     鈫?(32, 25, 25)
Conv2d(32鈫?4, k=3, s=2, p=1) + ReLU    鈫?(64, 13, 13)
Conv2d(64鈫?4, k=3, s=1, p=1) + ReLU    鈫?(64, 13, 13)
Conv2d(64鈫?28, k=3, s=1, p=1) + ReLU   鈫?(128, 13, 13)
AdaptiveAvgPool2d((8, 8))               鈫?(128, 8, 8)
Flatten                                 鈫?8192
Linear(8192鈫?12) + ReLU                 鈫?512
  鈹溾攢 Actor:  Linear(512鈫?)              鈫?8 涓姩浣?logits
  鈹斺攢 Critic: Linear(512鈫?)              鈫?鐘舵€佷环鍊?V(s)
```

**鍏抽敭鏀硅繘**锛氬彧鍋氫袱灞?stride-2 涓嬮噰鏍凤紙50鈫?5鈫?3锛夛紝淇濈暀 13脳13 鐗瑰緛鍥撅紝鍐?AdaptiveAvgPool 鍒?8脳8銆傛瘮涔嬪墠鐨勪笁灞備笅閲囨牱锛?脳7锛変繚鐣欐洿澶氱簿缁嗙┖闂翠俊鎭紝閫傚悎璐存帺浣撶粫琛岀瓑鎿嶄綔銆?
### 2.4 杈撳嚭锛? 涓鏁ｅ姩浣?
```python
ACTIONS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),       # 涓婁笅宸﹀彸
    (-1, -1), (-1, 1), (1, -1), (1, 1),      # 瀵硅绾?)
```

### 2.5 Action Masking

鍦ㄥ墠鍚戜紶鎾椂璁＄畻 invalid action mask锛屽皢涓嶅彲鎵ц鍔ㄤ綔鐨?logit 璁句负 `-inf`锛宻oftmax 鍚庢鐜囦负 0銆?
涓嶅彲鎵ц鍒ゅ畾锛堝鐢?`battlefield_env._is_blocked`锛夛細
- 鍑虹晫
- 鐩爣鏍?tag 鈮?0锛堝缓绛?鏍戞湪锛?- 鐖潯 tan > 0.3

**閲囨牱鍜岃缁冩椂鍧囦娇鐢?mask**锛屼繚璇佷粠涓嶉€夋棤鏁堝姩浣滐紝鏃犳晥鍔ㄤ綔涓嶅弬涓庢搴︺€?
---

## 涓夈€佸鍔卞嚱鏁拌璁?
鍏ㄩ儴鍙傛暟浠?`EnvConfig` 涓鍙栵紙宸叉湁瀹氫箟锛屼笉鍔?`EnvConfig`锛夛細

| 濂栧姳椤?| 鍊?| 璇存槑 |
|--------|-----|------|
| 姝ユ暟鎯╃綒 | `-step_penalty` = -0.05 | 姣忔鍥哄畾鎴愭湰锛岄紦鍔辫蛋鎹峰緞 |
| 闈犺繎濂栧姳 | `+progress_weight * 螖dist` = +0.1 脳 螖dist | 绋犲瘑寮曞锛堟潈閲嶈緝鍘?cfg 鍑忓崐锛夛紝鍚庢湡鍙“鍑?|
| 鏆撮湶鎯╃綒 | `-visible_penalty * I(visible)` = -0.4 | 澶勪簬鍙鏍煎瓙鐨勯澶栨儵缃?|
| 纰版挒鎯╃綒 | `-collision_penalty` = -1.0 | 灏濊瘯鏃犳晥鍔ㄤ綔锛坢ask 涓嬫瀬灏戝彂鐢燂級 |
| 鍒拌揪濂栧姳 | `+goal_reward` = +100 | 鍒拌揪缁堢偣 |
| 瓒呮椂鎯╃綒 | `-5.0` | 瓒呰繃 200 姝ヤ粛鏈埌杈撅紙杞绘儵缃氾紝閬垮厤浠峰€肩綉缁滈渿鑽★級 |

**鍏抽敭鏀硅繘**锛?- `progress_weight` 浠?0.2 闄嶄负 0.1锛屽噺灏戝蹇呰缁曡鐨勬儵缃氾紱璁粌鍚庢湡鍙繘涓€姝ヨ“鍑忚嚦 0
- 瓒呮椂鎯╃綒浠?-50 闄嶄负 -5锛屽け璐ヤ富瑕侀€氳繃绱Н姝ユ暟鎯╃綒浣撶幇锛岄伩鍏嶅崟娆″ぇ璐熷€奸渿鑽′环鍊肩綉缁?
### Episode 缁堟鏉′欢

1. 鍒拌揪缁堢偣 鈫?鎴愬姛
2. `steps >= max_steps` (200) 鈫?瓒呮椂
3. `consecutive_collisions >= max_consecutive_collisions` (15) 鈫?鍗℃锛坢ask 涓嬫瀬灏戝彂鐢燂級

---

## 鍥涖€丳PO 璁粌缁嗚妭

### 4.1 瓒呭弬鏁?
| 鍙傛暟 | 鍊?|
|------|-----|
| 纬 (鎶樻墸鍥犲瓙) | 0.99 |
| 位 (GAE) | 0.95 |
| 蔚 (PPO clip) | 0.2 |
| 瀛︿範鐜?| 3e-4 |
| 鐔电郴鏁?| 0.01 |
| Value loss 绯绘暟 | 0.5 |
| 鏈€澶ф搴﹁寖鏁?| 0.5 |
| Rollout 姝ユ暟/update | 2048 |
| Minibatch size | 64 |
| Epochs/update | 10 |
| 鎬昏缁冩鏁?| 2,000,000 |

### 4.2 璁粌娴佺▼

```
for total_steps:
    # 1. Rollout锛堟敹闆嗙粡楠岋級
    for step in range(2048):
        obs = env.get_obs()          # (7, 50, 50)
        action_mask = env.get_action_mask()
        obs_aug, mask_aug = random_augment(obs, action_mask)
        action, log_prob, value = policy(obs_aug, mask_aug)
        next_obs, reward, done, _ = env.step(action)
        buffer.add(obs, action, log_prob, reward, value, done, mask)
        if done: env.reset()         # 鏂扮獥鍙ｃ€佹柊璧风偣缁堢偣

    # 2. GAE 璁＄畻 advantages 鍜?returns
    advantages, returns = compute_gae(buffer, 纬=0.99, 位=0.95)

    # 3. PPO update锛堝 epoch锛?    for epoch in range(10):
        for minibatch in buffer:
            mb_obs, mb_mask = random_augment(minibatch)
            new_log_prob, new_value, entropy = policy(mb_obs, mb_mask)
            ratio = exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = clip(ratio, 1-蔚, 1+蔚) * advantage
            policy_loss = -min(surr1, surr2).mean()
            value_loss = 0.5 * (new_value - returns)^2
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            loss.backward()
        clip_grad_norm_(0.5)
        optimizer.step()
```

### 4.3 娉涘寲鏈哄埗

- **姣?episode 闅忔満鏂扮獥鍙?*锛歚env.reset()` 璋冪敤 `generate_scene()` 鍦ㄥ叏鍥撅紙501脳499锛変笂闅忔満閲囨牱 50脳50 绐楀彛锛岃捣鐐?缁堢偣鍦ㄧ獥鍙ｅ瑙掑尯鍩熼殢鏈洪€夊彇
- **璇勪及鐢ㄥ浐瀹氱獥鍙ｉ泦**锛氶閫?50 涓缁冧腑鏈嚭鐜扮殑绐楀彛鍧愭爣 + 绉嶅瓙锛屾瘡 10,000 姝ヨ瘎浼颁竴娆?- **璇勪及鎸囨爣**锛氬埌杈剧巼銆佸钩鍧囪矾寰勯暱搴︺€佸钩鍧囨毚闇茬巼

---

## 浜斻€佹枃浠剁粨鏋?
### 鏂板鏂囦欢

```
models/
  __init__.py           鈥?瀵煎嚭 ActorCriticCNN
  actor_critic_cnn.py     鈥?CNN backbone + Actor/Critic 澶?+ action masking

train/
  __init__.py
  ppo_config.py         鈥?PPOConfig dataclass锛堢嫭绔嬩簬 EnvConfig锛?  ppo_buffer.py         鈥?RolloutBuffer锛堝瓨鍌?trajectories + GAE 璁＄畻锛?  ppo_trainer.py        鈥?PPO 璁粌寰幆
  train_ppo.py          鈥?鍏ュ彛鑴氭湰
```

### 淇敼鏂囦欢

```
env/battlefield_env.py  鈥?琛ュ厖 RL 鎺ュ彛锛坮eset/step/get_obs/get_action_mask锛?requirements.txt        鈥?娣诲姞 torch
```

### 涓嶆敼鐨勬枃浠?
```
config.py               鈥?EnvConfig 涓嶅姩
env/occlusion.py        鈥?涓嶅姩
env/enemy_search.py     鈥?涓嶅姩
env/terrain_loader.py   鈥?涓嶅姩
env/__init__.py         鈥?涓嶅姩
visualize/              鈥?涓嶅姩
```

---

## 鍏€乣BattlefieldEnv` 闇€瑕佽ˉ鍏呯殑 RL 鏂规硶

```python
def reset(self, seed=None) -> np.ndarray:
    self.generate_scene(scene_seed=seed)
    self.agent_position = self.start_position.copy()
    self.steps = 0
    self.consecutive_collisions = 0
    return self._get_observation()

def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
    old_pos = self.agent_position.copy()
    move = np.array(self.ACTIONS[action], dtype=np.int32)
    candidate = old_pos + move
    if self._is_blocked(candidate):
        self.consecutive_collisions += 1
        reward = -self.config.collision_penalty
    else:
        self.agent_position = candidate.copy()
        self.consecutive_collisions = 0
        reward = -self.config.step_penalty
        old_dist = np.linalg.norm(old_pos.astype(np.float32) - self.goal_position.astype(np.float32))
        new_dist = np.linalg.norm(self.agent_position.astype(np.float32) - self.goal_position.astype(np.float32))
        reward += self.config.progress_weight * (old_dist - new_dist)
        if self.visibility_map[tuple(self.agent_position)] > 0.5:
            reward -= self.config.visible_penalty

    self.steps += 1
    done = False
    if np.array_equal(self.agent_position, self.goal_position):
        reward += self.config.goal_reward
        done = True
    elif self.steps >= self.config.max_steps:
        reward -= 5.0  # 杞昏秴鏃舵儵缃?        done = True
    elif self.consecutive_collisions >= self.config.max_consecutive_collisions:
        done = True

    return self._get_observation(), reward, done, {}

def _get_observation(self) -> np.ndarray:
    """鏋勫缓 7脳50脳50 澶氶€氶亾瑙傛祴锛岃 2.1 鑺?""

def get_action_mask(self) -> np.ndarray:
    """杩斿洖闀垮害涓?8 鐨?bool 鏁扮粍锛孴rue=鍙墽琛?""
```

---

## 涓冦€侀獙璇佹柟妗?
1. 瀵煎叆妫€鏌? `python -c "from models import ActorCriticCNN; from train.ppo_config import PPOConfig; from train import RolloutBuffer, PPOTrainer"`
2. 缃戠粶鍓嶅悜浼犳挱: 鏋勯€犻殢鏈?(7,50,50) 杈撳叆锛岄獙璇佽緭鍑?(8,) logits + (1,) value
3. Action masking: 鏋勯€犲叏闃诲鍦烘櫙锛岄獙璇?softmax 姒傜巼鍙湪鏈夋晥鍔ㄤ綔涓婇潪闆?4. 鐜 step 闂幆: `obs = env.reset(); obs2, r, d, _ = env.step(0); assert obs.shape == (7,50,50)`
5. 鏁版嵁澧炲己: 楠岃瘉鏃嬭浆/缈昏浆鍚庡姩浣滄槧灏勬纭?6. 璁粌鍚姩: `python -m train.train_fullmap_ppo --steps 10000` 灏忚妯¤繍琛岋紝纭 loss 涓嬮檷銆佹棤 NaN
7. BFS 瀵规瘮: 璁粌鍚庢彁鍙栫瓥鐣ヨ蛋 100 涓満鏅紝鍜?BFS 鏈€鐭矾寰勫姣旀毚闇茬巼


