"""CRNN (Conv1D + Bi-LSTM) + CTC model cho nhận diện ngôn ngữ ký hiệu."""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMCTC(nn.Module):
    """Conv1D + Bidirectional LSTM với CTC output (word-level translation).

    Input:  (batch, time_steps, feature_dim)
    Output: (batch, time_steps, num_classes)  — logits per frame
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 100,
        num_layers: int = 2,
        dropout: float = 0.3,
        input_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Input dropout: ngẫu nhiên zero toàn bộ feature dim → model robust hơn
        self.input_dropout = nn.Dropout(input_dropout)

        # Linear projection: feature_dim → hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Conv1D block: kernel=3, pad=1, stride=1 → T unchanged (CTC-safe)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
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

        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        # FC: hidden_dim*2 (bidirectional) → num_classes (per frame)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, feature_dim)
        Returns:
            logits: (batch, T, num_classes)
        """
        x = self.input_dropout(x)    # Input regularization
        x = self.projection(x)       # (B, T, hidden)
        x = x.transpose(1, 2)        # (B, hidden, T)
        x = self.conv(x)             # (B, hidden, T) — T unchanged
        x = x.transpose(1, 2)        # (B, T, hidden)
        x, _ = self.lstm(x)          # (B, T, hidden*2)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)               # (B, T, num_classes)
        return x
