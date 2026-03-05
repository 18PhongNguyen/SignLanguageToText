"""Bi-LSTM + CTC model cho nhận diện ngôn ngữ ký hiệu."""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMCTC(nn.Module):
    """Bidirectional LSTM với CTC output layer.

    Input:  (batch, time_steps, feature_dim)
    Output: (batch, time_steps, num_classes + 1)   — log-softmax
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 100,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Linear projection: feature_dim → hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Bi-LSTM stack
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # FC output: hidden_dim*2 (bidirectional) → num_classes + 1 (blank)
        self.fc = nn.Linear(hidden_dim * 2, num_classes + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, feature_dim)
        Returns:
            log_probs: (batch, T, num_classes + 1)
        """
        x = self.projection(x)       # (B, T, hidden)
        x, _ = self.lstm(x)          # (B, T, hidden*2)
        x = self.dropout(x)
        x = self.fc(x)               # (B, T, C)
        return x.log_softmax(dim=-1)
