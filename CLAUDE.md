# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 闂鍩?
50脳50 鏍呮牸鎴樺満锛? 鏂瑰悜绉诲姩銆傚瓨鍦ㄥ湴褰㈤珮搴︼紙褰卞搷鍙€氳鎬э級鍜屼簩鍊煎彲瑙佹€э紙鏁屼汉鐬湜鐐规湁 FOV锛夈€傜敤 PPO 璁粌鏅鸿兘浣撲粠璧风偣璧板埌缁堢偣锛屾渶灏忓寲琚晫浜虹湅鍒扮殑鏆撮湶鏃堕棿銆?
## 椤圭洰缁撴瀯

```
config.py               鈥?EnvConfig 鐜鍙傛暟锛坒rozen dataclass锛屼竴瀛椾笉鏀癸級
MyPath_Data417.txt      鈥?501脳499 鍏ㄥ湴褰㈡暟鎹紝姣忔牸 (height,tag)锛宼ag: 0=鍦伴潰 1=寤虹瓚 2=鏍戞湪

env/
  terrain_loader.py     鈥?鍦板舰鏂囦欢瑙ｆ瀽
  occlusion.py          鈥?3D 鍏夌嚎杩借釜閬尅鍒ゅ畾锛堢嫭绔嬫ā鍧楋紝鏃犵姸鎬侊級
  enemy_search.py       鈥?鐗瑰緛浠ｇ悊璇勫垎 + 绌洪棿鎶戝埗 鈫?8 涓灜鏈涚偣 + 鍏ㄥ浘鍙鎬у簳鍥?  battlefield_env.py    鈥?鍦烘櫙鐢熸垚 + 閫氳妫€鏌?+ RL 鎺ュ彛锛坮eset/step/get_obs/get_action_mask锛?  scene_pool.py         鈥?棰勮绠楅殢鏈洪殰纰嶅満鏅睜锛堥樁娈?鐢紝鏃犳晫浜猴級
  vectorized_env.py     鈥?N 涓苟琛岀幆澧冨寘瑁呭櫒锛屾壒閲忓墠鍚戜紶鎾姞閫?
models/
  actor_critic_cnn.py     鈥?CNN backbone + Actor/Critic 鍙屽ご + Action Masking + 鏁版嵁澧炲己 + 鍔ㄤ綔閫嗗彉鎹?
train/
  ppo_config.py         鈥?PPOConfig dataclass锛堢嫭绔嬩簬 EnvConfig锛?  ppo_buffer.py         鈥?RolloutBuffer锛堝瓨鍌?transitions + GAE 璁＄畻锛?  ppo_trainer.py        鈥?PPO 璁粌寰幆锛坮ollout 鈫?GAE 鈫?PPO update 鈫?eval锛?  train_ppo.py          鈥?鍏ㄥ姛鑳借缁冨叆鍙ｏ紙鍏ㄥ浘妯″紡 + 鏁屼汉锛?  smoke_test.py         鈥?骞冲潶鍦板舰鍐掔儫娴嬭瘯锛堟棤鏁屼汉鏃犲缓绛戯紝楠岃瘉绠楁硶鍙鎬э級
  train_local_pool_ppo.py    鈥?闃舵1锛氶殰纰嶅湴褰㈢函瀵艰埅璁粌锛堟棤鏁屼汉锛?  train_local_pool_ppo.py    鈥?闃舵1骞惰鐗堬紙澶氱幆澧冩壒閲忔帹鐞嗭級

artifacts/
  enemy_pool.json       鈥?棰勮绠楃殑 8 涓晫浜虹灜鏈涚偣
  visibility_maps.npz   鈥?棰勮绠楃殑 8 寮犲叏鍥惧彲瑙佹€у簳鍥?  scene_pool.npz        鈥?棰勮绠楃殑 N 涓殢鏈洪殰纰嶅満鏅紙闃舵1鐢級

visualize/
  visualizer.py         鈥?璁粌缁撴灉鍙鍖?  find_dense_scenes.py  鈥?鏌ユ壘瀵嗛泦闅滅鍦烘櫙
```

## 鍚勬ā鍧楀姛鑳?
### `env/battlefield_env.py` 鈥?鍦烘櫙鐢熸垚涓庨€氳妫€鏌?
**`BattlefieldEnv`** 绫伙紝鏋勯€犳椂鑷姩璋?`generate_scene()`銆?
RL 鎺ュ彛锛?- `reset(seed)` 鈫?obs (7,50,50) float32
- `step(action: int)` 鈫?(obs, reward, done, info)
- `get_action_mask()` 鈫?(8,) bool锛孴rue=鍙墽琛?- `_get_observation()` 鈫?(7,50,50) 7閫氶亾锛歨eight + ground/building/tree + visibility + agent/goal one-hot 缂栫爜
- `compute_bfs_path()` 鈫?list[tuple] 鎴?None

鍦烘櫙妯″紡锛歚"full_map"`锛堝叏鍥炬粦鍔ㄧ獥鍙ｏ級銆乣"random"`锛堢▼搴忓寲鍦板舰锛夈€乣"fixed"`锛堝浐瀹氶殰纰嶇墿锛夈€?
閫氳妫€鏌ワ細`_is_blocked` 缁煎悎鍒ゆ柇杈圭晫/鏍囩/鐖潯(tan鈮?.3)/鏁屼汉浣嶇疆銆?
### `env/occlusion.py` 鈥?閬尅鍒ゅ畾锛堢嫭绔嬫ā鍧楋紝鏃犵姸鎬侊級

- `is_occluded(start, end, height_map, config)` 鈫?bool: 3D 鍏夌嚎杩借釜銆?*start/end 蹇呴』涓?height_map 鍚屽潗鏍囩郴**
- `compute_cell_visibility(observer, cell, height_map, config)` 鈫?float
- `compute_visibility_map(observer, terrain, config)` 鈫?(vis_map, visible_count)

### `env/enemy_search.py` 鈥?鏁屼汉鐬湜鐐规悳绱?
- `compute_feature_scores(terrain)` 鈫?np.ndarray: 鐗瑰緛浠ｇ悊璇勫垎锛堥珮搴︽帓鍚?.3 + 寮€闃斿害0.3 + 鏀厤鍔?.4锛?- `spatial_suppression(scores, k, radius)` 鈫?list[tuple]: NMS 绌洪棿鎶戝埗

### `models/actor_critic_cnn.py` 鈥?CNN 绛栫暐-浠峰€肩綉缁?
**`ActorCriticCNN`**锛?- CNN backbone: Conv(7鈫?2,k5,s2) 鈫?Conv(32鈫?4,k3,s2) 鈫?Conv(64鈫?4,k3,s1) 鈫?Conv(64鈫?28,k3,s1) 鈫?AdaptiveAvgPool(8,8) 鈫?FC(8192鈫?12)
- Actor: Linear(512鈫?), Critic: Linear(512鈫?)
- `forward(obs, action_mask, deterministic)` 鈫?(action, log_prob, value, entropy, logits)
- `evaluate(obs, action, action_mask)` 鈫?(log_probs, values, entropy) 鈥?PPO update 鐢?- `get_value(obs)` 鈫?value 鈥?GAE bootstrap 鐢?- Action masking: 鏃犳晥鍔ㄤ綔 logit = -1e9锛堥潪 -inf锛岄伩鍏?softmax NaN锛?
**鏁版嵁澧炲己** (`random_augment`)锛?- 闅忔満鏃嬭浆锛?/90/180/270锛? 姘村钩/鍨傜洿缈昏浆锛堝悇50%锛?- 杩斿洖 `(aug_obs, aug_mask, (k, flip_h, flip_v))`
- **鍏抽敭**锛氬寮哄湪 rollout 鏃舵柦鍔犱竴娆★紝buffer 瀛樺偍澧炲己鍚庢暟鎹€侾PO update 鏃跺師鏍峰彇鍑猴紝淇濊瘉 old/new log_prob 鍙瘮銆?
**鍔ㄤ綔閫嗗彉鎹?* (`deaugment_action(aug_action, k, flip_h, flip_v)`)锛?- **蹇呴』璋冪敤锛?* 澧炲己绌洪棿鐨勫姩浣滅储寮曞繀椤诲厛閫嗗彉鎹㈠洖鍘熷绌洪棿锛屽啀浜ょ粰 `env.step()`銆?- 閫嗗簭锛氶€?flip_v 鈫?閫?flip_h 鈫?閫嗘棆杞?k 娆?forward 鏄犲皠)

### `train/ppo_buffer.py` 鈥?Rollout Buffer

棰勫垎閰嶆墍鏈?tensor锛宍add()` 閫愭潯瀛樺偍 transition銆?
GAE 璁＄畻鏈変袱绉嶆ā寮忥細
- `compute_gae(last_value, gamma, gae_lambda)`: 鍗曠幆澧冧覆琛岀増锛屽€掑簭閬嶅巻涓€鏉¤繛缁建杩?- `compute_gae_parallel(last_values, gamma, gae_lambda, num_envs)`: **骞惰鐗?*锛屾暟鎹寜 `[e0_t0, e1_t0, ..., eN_t0, e0_t1, ...]` 浜ょ粐瀛樺偍锛屾寜 `stride=num_envs` 鐙珛璁＄畻鍚勭幆澧冪殑 GAE

`normalize_advantages()` 鍋?z-score 鏍囧噯鍖栵紝`sample()` 杩斿洖闅忔満 mini-batch 绱㈠紩銆?
### `train/ppo_trainer.py` 鈥?PPO 璁粌鍣?
**`PPOTrainer`**锛氬畬鏁磋缁冨惊鐜€?- 杩涘害濂栧姳琛板噺锛氳缁冭繘搴?50%-90% 鏈熼棿 `progress_weight` 绾挎€ц“鍑忓埌 0
- 璇勪及锛氱‘瀹氭€ф帹鐞嗭紙鏃犲寮恒€乤rgmax锛夛紝姣?`eval_interval` 姝ヤ竴娆?
### `env/vectorized_env.py` 鈥?骞惰鐜

**`VectorizedEnv`**锛歂 涓嫭绔?`BattlefieldEnv` 瀹炰緥锛屾瘡涓粠鍦烘櫙姹犵嫭绔嬮噰鏍枫€?- `get_observations()` 鈫?(N,7,50,50)
- `step(actions)` 鈫?瀵规墍鏈夌幆澧冨悇鎵ц涓€姝ワ紝鑷姩 reset 宸插畬鎴愮殑
- `_reset_env()` 灏?`scenario_mode` 璁句负 `"full_map"`锛岃繖鏄湁鎰忎负涔嬧€斺€旇烦杩?`_is_blocked` 涓殑鏁屼汉浣嶇疆妫€鏌ワ紙闃舵1鏃犳晫浜猴級
- 閰嶅悎鎵归噺 CNN 鍓嶅悜浼犳挱锛屽姞閫熺害 N 鍊?- **娉ㄦ剰**锛氳瘎浼板嚱鏁颁細淇敼 env 0 鐨勫唴閮ㄧ姸鎬侊紙`_reset_env` 鎹㈠満鏅級锛岃瘎浼板悗蹇呴』鍒锋柊 `obs_batch`锛屽惁鍒欎笅涓€姝ヨ缁冧細鐢ㄥ埌杩囨湡瑙傛祴

## 璁粌闃舵浣撶郴

| 闃舵 | 鑴氭湰 | 鍦烘櫙 | 鏁屼汉 | 鐩殑 |
|------|------|------|------|------|
| 鍐掔儫 | `train/smoke_test.py` | 骞冲潶鍦板舰(0,0)鈫?49,49) | 鏃?| 楠岃瘉绠楁硶鍙鎬?|
| 闃舵1 | `train/train_local_pool_ppo.py` | 鍦烘櫙姹狅紙寤虹瓚+鏍戞湪+楂樺害锛?| 鏃?| 楠岃瘉 action masking + 缁曡 |
| 闃舵2 | `train/train_ppo.py` | 鍏ㄥ浘婊戝姩绐楀彛 | 鏈?| 瀹屾暣闅愯斀瀵昏矾 |

## 甯哥敤鍛戒护

```bash
# 鐢熸垚鍦烘櫙姹狅紙闃舵1鐢紝涓€娆℃€э級
python -m env.scene_pool --num 5000

# 闃舵1璁粌锛堝苟琛岀増锛屾帹鑽愶級
python -m train.train_local_pool_ppo --steps 500000 --envs 8 --pool artifacts/scene_pool.npz

# 闃舵1璁粌锛堜覆琛岀増锛岃皟璇曠敤锛?python -m train.train_local_pool_ppo --steps 500000 --pool artifacts/scene_pool.npz

# 鍏ㄥ姛鑳借缁冿紙鍏ㄥ浘妯″紡 + 鏁屼汉锛?python -m train.train_fullmap_ppo --steps 2000000 --save checkpoints

# 鍐掔儫娴嬭瘯锛堝钩鍧﹀湴褰㈠揩閫熼獙璇侊級
python -m train.smoke_test

# 閲嶆柊鐢熸垚鏁屼汉姹?+ 鍙鎬у簳鍥?python -m env.enemy_search

# 蹇€熷鍏ラ獙璇?python -c "from models import ActorCriticCNN, random_augment, deaugment_action; from train import PPOConfig, RolloutBuffer, PPOTrainer; print('OK')"
```

## 鍧愭爣绯荤粺绾﹀畾

- 鍏ㄥ眬鍧愭爣: 鍦ㄥ叏鍦板舰 `height_map` (501脳499) 涓婄殑鍧愭爣
- 绐楀彛鍧愭爣: 鍦?50脳50 婊戝姩绐楀彛鍐呯殑鍧愭爣
- `occlusion.py` 涓墍鏈夊嚱鏁颁娇鐢ㄥ悓涓€鍧愭爣绯伙紙start/end 涓?height_map 瀵瑰簲锛?- `BattlefieldEnv._is_occluded_global` 璐熻矗绐楀彛鈫掑叏灞€鍧愭爣杞崲

## 鍏抽敭瀹炵幇缁嗚妭

- **Action Masking**: 鐢?`-1e9` 鑰岄潪 `-inf`锛岄伩鍏嶅叏 mask 鏃?softmax NaN
- **鏁版嵁澧炲己鏃舵満**: 浠呭湪 rollout 鏃舵柦鍔犱竴娆★紝buffer 瀛樺寮哄悗鏁版嵁锛孭PO update 涓嶉噸鏂板寮?- **鍔ㄤ綔閫嗗彉鎹?*: `random_augment` 鍙樻崲浜?obs 鍜?mask锛孋NN 杈撳嚭澧炲己绌洪棿鍔ㄤ綔锛屽繀椤?`deaugment_action()` 杩樺師鍚庡啀 `env.step()`
- **GAE bootstrap**: 鐢?rollout 鏈€鍚庝竴姝ョ殑 obs 璁＄畻 `last_value`锛宒one=True 鏃?`not_done` 鍥犲瓙鑷姩褰掗浂
- **杩涘害濂栧姳琛板噺**: `progress_weight` 鍦ㄨ缁冭繘搴?50%-90% 鏈熼棿绾挎€ц“鍑忓埌 0锛堣绋嬪涔狅級
- **瓒呮椂鎯╃綒纭紪鐮?*: `step()` 涓秴鏃舵儵缃氬啓姝?`-5.0`锛岃€岄潪浣跨敤 `config.timeout_penalty`锛?0.0锛夈€傝交瓒呮椂鎯╃綒閬垮厤浠峰€肩綉缁滈渿鑽★紝澶辫触涓昏閫氳繃绱Н姝ユ暟鎯╃綒浣撶幇
- **EnvConfig 涓嶅彲淇敼**: 鎵€鏈夎缁冭秴鍙傚湪 `PPOConfig` 涓厤缃?- **骞惰 GAE 浜ょ粐瀛樺偍**: buffer 鎸?`[e0_t0, e1_t0, ..., eN_t0, e0_t1, ...]` 椤哄簭瀛樺偍锛宍compute_gae_parallel` 鎸?`t = env_idx + step * num_envs` 璺ㄦ闀胯闂紝纭繚鍚?env 鐨?GAE 鐙珛璁＄畻鑰屼笉涓叉壈
- **璇勪及鍚庡埛鏂拌娴?*: `_evaluate_vec` 浼?reset env 0 鎹㈠満鏅紝璇勪及鍚庡繀椤?`vec_env.get_observations()` 鍒锋柊 `obs_batch`锛屽惁鍒欎笅涓€姝ヨ缁冪敤杩囨湡鏁版嵁



