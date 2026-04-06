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
            nn.Conv2d(self.config.local_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(self.config.global_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(64 * 4 + 64, 128),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
        )

    def forward(self, local_map: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        local_feature = self.local_encoder(local_map)
        global_feature = self.global_encoder(global_features)
        fused = torch.cat((local_feature, global_feature), dim=1)
        hidden = self.fusion(fused)
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def forward_numpy(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            local_map = torch.from_numpy(observation["local_map"]).unsqueeze(0)
            global_features = torch.from_numpy(observation["global_features"]).unsqueeze(0)
            output = self.forward(local_map.float(), global_features.float())
        return output.squeeze(0).cpu().numpy()
