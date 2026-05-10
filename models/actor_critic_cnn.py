"""CNN 策略-价值网络：Actor-Critic 双头 + Action Masking + 数据增强。

输入 (7, 50, 50) 多通道观测 → CNN backbone → 512 维特征 → Actor(8) / Critic(1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 8 方向动作定义 ──────────────────────────────────────
ACTIONS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),          # 上下左右
    (-1, -1), (-1, 1), (1, -1), (1, 1),         # 对角线
)

_NUM_ACTIONS = len(ACTIONS)


# ── CNN Backbone ────────────────────────────────────────

class _CNNBackbone(nn.Module):
    """特征提取器：7×50×50 → 8192 → 512."""

    def __init__(self, in_channels: int = 7, feature_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),   # (32, 25, 25)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (64, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (128, 13, 13)
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))                   # (128, 8, 8)
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# ── Actor-Critic ────────────────────────────────────────

class ActorCriticCNN(nn.Module):
    """CNN 策略-价值网络，内置 action masking。

    forward() 返回：action, log_prob, value, entropy, action_logits
    evaluate() 返回：new_log_probs, values, entropy（用于 PPO update）
    """

    def __init__(self, in_channels: int = 7, feature_dim: int = 512):
        super().__init__()
        self.backbone = _CNNBackbone(in_channels=in_channels, feature_dim=feature_dim)
        self.actor = nn.Linear(feature_dim, _NUM_ACTIONS)
        self.critic = nn.Linear(feature_dim, 1)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def _logits_and_value(self, features: torch.Tensor):
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作。

        Args:
            obs: (B, 7, H, W) 观测
            action_mask: (B, 8) bool，True=有效动作
            deterministic: True 时取 argmax 而非采样

        Returns:
            action: (B,) int
            log_prob: (B,)
            value: (B,)
            entropy: (B,) —— 策略熵
            logits: (B, 8) —— masked logits（无效动作为 -inf）
        """
        features = self._features(obs)
        logits, value = self._logits_and_value(features)

        # Action masking（用 -1e9 替代 -inf，避免全 mask 时 softmax NaN）
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy()

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value, entropy, logits

    def evaluate(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """给定观测和动作，计算 log_prob、value、entropy。

        Args:
            obs: (B, 7, H, W)
            action: (B,) int
            action_mask: (B, 8) bool

        Returns:
            log_probs: (B,)
            values: (B,)
            entropy: (B,)
        """
        features = self._features(obs)
        logits, value = self._logits_and_value(features)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return log_probs, value, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """仅计算 V(s)，用于 GAE 最后一步 bootstrap."""
        features = self._features(obs)
        _, value = self._logits_and_value(features)
        return value


# ── 数据增强 ────────────────────────────────────────────

def random_augment(
    obs: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[int, bool, bool]]:
    """对一批观测随机施加旋转/翻转，返回增强后的 (obs, mask, aug_params)。

    增强方式（等概率，独立施加）：
    - rot90: 0 / 90 / 180 / 270 度
    - flip_h: 50% 概率
    - flip_v: 50% 概率

    aug_params = (k, flip_h, flip_v) 用于后续 deaugment_action() 逆变换。

    约定：动作索引 0=上,1=下,2=左,3=右,4=左上,5=右上,6=左下,7=右下
    """
    B = obs.size(0)
    device = obs.device

    # —— 随机旋转 k ∈ {0, 1, 2, 3} ——
    k = torch.randint(0, 4, (1,), device=device).item()

    # —— 随机翻转 ——
    flip_h = torch.rand(1, device=device).item() < 0.5
    flip_v = torch.rand(1, device=device).item() < 0.5

    # 对观测应用变换
    aug_obs = obs
    if k != 0:
        aug_obs = torch.rot90(aug_obs, k, dims=(2, 3))
    if flip_h:
        aug_obs = torch.flip(aug_obs, dims=(3,))
    if flip_v:
        aug_obs = torch.flip(aug_obs, dims=(2,))

    # 对 action_mask 应用相同动作映射
    if action_mask is not None:
        aug_mask = action_mask.clone()
        for _ in range(k):
            aug_mask = _rotate_action_mask(aug_mask)
        if flip_h:
            aug_mask = _flip_h_action_mask(aug_mask)
        if flip_v:
            aug_mask = _flip_v_action_mask(aug_mask)
    else:
        aug_mask = None

    return aug_obs, aug_mask, (k, flip_h, flip_v)


def deaugment_action(
    aug_action: torch.Tensor,
    k: int,
    flip_h: bool,
    flip_v: bool,
) -> torch.Tensor:
    """逆变换动作索引：增强空间 → 原始空间。

    按增强的逆序施加：先逆 flip_v → 逆 flip_h → 逆旋转。
    flip_h/flip_v 自逆，旋转通过再转 (4-k) 次顺时针还原。
    """
    action = aug_action
    # 逆 flip_v（自逆，同 forward 映射）
    if flip_v:
        idx_map = torch.tensor(
            [1, 0, 2, 3, 6, 7, 4, 5], dtype=torch.long, device=action.device,
        )
        action = idx_map[action]
    # 逆 flip_h（自逆）
    if flip_h:
        idx_map = torch.tensor(
            [0, 1, 3, 2, 5, 4, 7, 6], dtype=torch.long, device=action.device,
        )
        action = idx_map[action]
    # 逆旋转：mask 增强时应用了 k 次 forward 映射，动作逆变换时同样应用 k 次
    # forward_map[i] = 原始动作在第 i 位的增强动作
    for _ in range(k):
        idx_map = torch.tensor(
            [3, 2, 0, 1, 5, 7, 4, 6], dtype=torch.long, device=action.device,
        )
        action = idx_map[action]
    return action


def _rotate_action_mask(mask: torch.Tensor) -> torch.Tensor:
    """旋转 90 度顺时针后的动作索引映射。

    原 → 旋转90°：
    0(上)→3(右), 1(下)→2(左), 2(左)→0(上), 3(右)→1(下)
    4(左上)→5(右上), 5(右上)→7(右下), 6(左下)→4(左上), 7(右下)→6(左下)
    """
    idx_map = torch.tensor([3, 2, 0, 1, 5, 7, 4, 6], dtype=torch.long, device=mask.device)
    return mask[:, idx_map]


def _flip_h_action_mask(mask: torch.Tensor) -> torch.Tensor:
    """水平翻转 (左右镜像): 左↔右."""
    idx_map = torch.tensor([0, 1, 3, 2, 5, 4, 7, 6], dtype=torch.long, device=mask.device)
    return mask[:, idx_map]


def _flip_v_action_mask(mask: torch.Tensor) -> torch.Tensor:
    """垂直翻转 (上下镜像): 上↔下."""
    idx_map = torch.tensor([1, 0, 2, 3, 6, 7, 4, 5], dtype=torch.long, device=mask.device)
    return mask[:, idx_map]
