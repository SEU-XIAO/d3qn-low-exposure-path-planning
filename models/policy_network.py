from __future__ import annotations

import numpy as np
import torch
from torch import nn

from config import ModelConfig


class HybridPolicyNetwork(nn.Module):
    def __init__(self, action_dim: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.action_dim = action_dim

        self.local_encoder = nn.Sequential(
            nn.Conv2d(self.config.local_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(self.config.global_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 * 16 + 64, 128),
            nn.ReLU(),
        )

        head_input = 128

        self.value_head = nn.Sequential(
            nn.Linear(head_input, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(head_input, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
        )

    def _encode(self, local_map: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        local_feature = self.local_encoder(local_map)
        global_feature = self.global_encoder(global_features)
        fused = torch.cat((local_feature, global_feature), dim=1)
        return self.fusion(fused)

    def _dueling(self, hidden: torch.Tensor) -> torch.Tensor:
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def forward(self, local_map: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """单步前向。返回 Q 值。"""
        fused = self._encode(local_map, global_features)
        return self._dueling(fused)

    def forward_numpy(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            local_map = torch.from_numpy(observation["local_map"]).unsqueeze(0)
            global_features = torch.from_numpy(observation["global_features"]).unsqueeze(0)
            output = self.forward(local_map.float(), global_features.float())
        return output.squeeze(0).cpu().numpy()
