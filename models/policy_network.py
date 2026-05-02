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
        self.use_lstm = self.config.use_lstm

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

        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=128,
                hidden_size=self.config.lstm_hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.lstm_scale = nn.Parameter(torch.zeros(1))
            head_input = self.config.lstm_hidden_size
        else:
            self.lstm = None
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

    def forward(self, local_map: torch.Tensor, global_features: torch.Tensor,
                hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """单步前向。返回 (Q, (h, c)) 若有 LSTM，否则返回 Q。"""
        fused = self._encode(local_map, global_features)  # [B, 128] 或 [B, T, 128]

        if self.use_lstm:
            lstm_out, new_hidden = self.lstm(fused.unsqueeze(1), hidden_state)
            hidden = fused + self.lstm_scale * lstm_out.squeeze(1)  # 残差：初始 scale=0，保留 BC 行为
            return self._dueling(hidden), new_hidden

        return self._dueling(fused)

    def forward_sequence(self, local_map: torch.Tensor, global_features: torch.Tensor
                         ) -> torch.Tensor:
        """序列前向（训练用）。输入 [B, T, ...]，输出 [B, T, A]。

        若未启用 LSTM，逐时间步独立计算。
        """
        B, T = local_map.shape[:2]
        if self.use_lstm:
            local_flat = local_map.view(B * T, *local_map.shape[2:])
            global_flat = global_features.view(B * T, *global_features.shape[2:])
            fused = self._encode(local_flat, global_flat)  # [B*T, 128]
            fused_seq = fused.view(B, T, -1)  # [B, T, 128]
            lstm_out, _ = self.lstm(fused_seq)  # [B, T, H]
            return self._dueling(fused_seq + self.lstm_scale * lstm_out)  # 残差连接
        else:
            local_flat = local_map.view(B * T, *local_map.shape[2:])
            global_flat = global_features.view(B * T, *global_features.shape[2:])
            fused = self._encode(local_flat, global_flat)  # [B*T, 128]
            q = self._dueling(fused)  # [B*T, A]
            return q.view(B, T, -1)

    def init_hidden(self, batch_size: int = 1, device: torch.device | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """返回零初始化的 LSTM 隐状态。无 LSTM 时返回 None。"""
        if not self.use_lstm:
            return None
        h = torch.zeros(1, batch_size, self.config.lstm_hidden_size)
        c = torch.zeros(1, batch_size, self.config.lstm_hidden_size)
        if device is not None:
            h, c = h.to(device), c.to(device)
        return (h, c)

    def forward_numpy(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            local_map = torch.from_numpy(observation["local_map"]).unsqueeze(0)
            global_features = torch.from_numpy(observation["global_features"]).unsqueeze(0)
            if self.use_lstm:
                output, _ = self.forward(local_map.float(), global_features.float())
            else:
                output = self.forward(local_map.float(), global_features.float())
        return output.squeeze(0).cpu().numpy()
