"""Actor-critic CNN with action masking and optional higher-resolution backbones."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


ACTIONS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)

_NUM_ACTIONS = len(ACTIONS)
BACKBONE_LEGACY = "legacy"
BACKBONE_RES_SMALL = "res_small"


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = x + identity
        return self.act(x)


class _LegacyCNNBackbone(nn.Module):
    def __init__(self, in_channels: int = 10, feature_dim: int = 512) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class _ResSmallBackbone(nn.Module):
    """A small residual CNN that keeps more local detail for 10x10 tasks."""

    def __init__(self, in_channels: int = 10, feature_dim: int = 512) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block1 = _ResidualBlock(32)
        self.down = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block2 = _ResidualBlock(64)
        self.proj = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block3 = _ResidualBlock(96)
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc = nn.Sequential(
            nn.Linear(96 * 5 * 5, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.down(x)
        x = self.block2(x)
        x = self.proj(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


_BACKBONE_FACTORY = {
    BACKBONE_LEGACY: _LegacyCNNBackbone,
    BACKBONE_RES_SMALL: _ResSmallBackbone,
}


def infer_model_spec(state_dict: Mapping[str, torch.Tensor]) -> dict[str, int | str]:
    actor_weight = state_dict.get("actor.weight")
    if actor_weight is None:
        raise RuntimeError("checkpoint missing actor.weight")

    if "backbone.conv.0.weight" in state_dict:
        in_channels = int(state_dict["backbone.conv.0.weight"].shape[1])
        backbone_name = BACKBONE_LEGACY
    elif "backbone.stem.0.weight" in state_dict:
        in_channels = int(state_dict["backbone.stem.0.weight"].shape[1])
        backbone_name = BACKBONE_RES_SMALL
    else:
        raise RuntimeError("cannot infer backbone from checkpoint keys")

    return {
        "in_channels": in_channels,
        "feature_dim": int(actor_weight.shape[1]),
        "backbone_name": backbone_name,
    }


class ActorCriticCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        feature_dim: int = 512,
        backbone_name: str = BACKBONE_LEGACY,
    ) -> None:
        super().__init__()
        if backbone_name not in _BACKBONE_FACTORY:
            raise ValueError(f"unsupported backbone: {backbone_name}")
        self.in_channels = int(in_channels)
        self.feature_dim = int(feature_dim)
        self.backbone_name = backbone_name

        backbone_cls = _BACKBONE_FACTORY[backbone_name]
        self.backbone = backbone_cls(in_channels=self.in_channels, feature_dim=self.feature_dim)
        self.actor = nn.Linear(self.feature_dim, _NUM_ACTIONS)
        self.critic = nn.Linear(self.feature_dim, 1)
        self._init_weights()

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, torch.Tensor]) -> "ActorCriticCNN":
        spec = infer_model_spec(state_dict)
        return cls(
            in_channels=int(spec["in_channels"]),
            feature_dim=int(spec["feature_dim"]),
            backbone_name=str(spec["backbone_name"]),
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def _logits_and_value(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._features(obs)
        logits, value = self._logits_and_value(features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy()
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, entropy, logits

    def evaluate(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._features(obs)
        logits, value = self._logits_and_value(features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return log_probs, value, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self._features(obs)
        _, value = self._logits_and_value(features)
        return value


def random_augment(
    obs: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[int, bool, bool]]:
    device = obs.device
    k = torch.randint(0, 4, (1,), device=device).item()
    flip_h = torch.rand(1, device=device).item() < 0.5
    flip_v = torch.rand(1, device=device).item() < 0.5

    aug_obs = obs
    if k != 0:
        aug_obs = torch.rot90(aug_obs, k, dims=(2, 3))
    if flip_h:
        aug_obs = torch.flip(aug_obs, dims=(3,))
    if flip_v:
        aug_obs = torch.flip(aug_obs, dims=(2,))

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
    action = aug_action
    if flip_v:
        idx_map = torch.tensor(
            [1, 0, 2, 3, 6, 7, 4, 5], dtype=torch.long, device=action.device,
        )
        action = idx_map[action]
    if flip_h:
        idx_map = torch.tensor(
            [0, 1, 3, 2, 5, 4, 7, 6], dtype=torch.long, device=action.device,
        )
        action = idx_map[action]
    for _ in range(k):
        idx_map = torch.tensor(
            [3, 2, 0, 1, 5, 7, 4, 6], dtype=torch.long, device=action.device,
        )
        action = idx_map[action]
    return action


def _rotate_action_mask(mask: torch.Tensor) -> torch.Tensor:
    idx_map = torch.tensor([3, 2, 0, 1, 5, 7, 4, 6], dtype=torch.long, device=mask.device)
    return mask[:, idx_map]


def _flip_h_action_mask(mask: torch.Tensor) -> torch.Tensor:
    idx_map = torch.tensor([0, 1, 3, 2, 5, 4, 7, 6], dtype=torch.long, device=mask.device)
    return mask[:, idx_map]


def _flip_v_action_mask(mask: torch.Tensor) -> torch.Tensor:
    idx_map = torch.tensor([1, 0, 2, 3, 6, 7, 4, 5], dtype=torch.long, device=mask.device)
    return mask[:, idx_map]
