"""SplitConformerCTC — kiến trúc Conformer tách luồng Pose/Hand.

Thay thế BiLSTMCTC trong thử nghiệm. KHÔNG thay đổi model.py.

Đặc điểm:
- Input 602-dim (toàn bộ feature vector từ extractor, kể cả eyebrow)
- Tách cứng: Pose [0:132]+velocity[301:433] → 264 dim → proj 64
            Hand [132:268]+velocity[433:569] → 272 dim → proj 80
            Eyebrow [268:301]+velocity[569:602] → BỎ QUA hoàn toàn
- d_model = 64 + 80 = 144
- Conformer stack: FF(½) → MHSA → DepthwiseConv → FF(½)

Feature layout từ extractor.py (USE_VELOCITY=True, 602-dim):
  [0:132]   = pose raw
  [132:268] = hand raw  (lh_coords[63] + lh_angles[5] + rh_coords[63] + rh_angles[5] = 136 — thực ra 132:268=136 dim)
  [268:301] = eyebrow raw
  [301:433] = pose velocity
  [433:569] = hand velocity
  [569:602] = eyebrow velocity
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# HELPER: Modality Dropout
# =====================================================================

def modality_dropout(tensor: torch.Tensor, p: float = 0.15) -> torch.Tensor:
    """Dropout toàn bộ một modality với xác suất p (dùng khi training).

    Thay vì dropout từng neuron, zero toàn bộ stream → buộc model
    học cách inference ngay cả khi thiếu một modality.
    """
    if random.random() < p:
        return torch.zeros_like(tensor)
    return tensor


# =====================================================================
# FeedForward Sub-Block
# =====================================================================

class FeedForward(nn.Module):
    """FFN với SiLU activation, dùng trong Conformer block."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================================================================
# ConvModule Sub-Block
# =====================================================================

class ConvModule(nn.Module):
    """Depthwise Separable Conv Module trong Conformer.

    Cấu trúc: Pointwise expand → GLU → Depthwise → BN → SiLU → Pointwise
    """

    def __init__(self, d_model: int, kernel_size: int = 9, dropout: float = 0.1) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size phải lẻ để giữ nguyên T"
        self.pw_conv1 = nn.Conv1d(d_model, d_model * 2, 1)
        self.dw_conv  = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.bn       = nn.BatchNorm1d(d_model)
        self.pw_conv2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x.transpose(1, 2)    # (B, d_model, T)
        x = self.pw_conv1(x)     # (B, d_model*2, T)
        x = F.glu(x, dim=1)      # (B, d_model, T)
        x = self.dw_conv(x)      # (B, d_model, T)
        x = self.bn(x)
        x = F.silu(x)
        x = self.pw_conv2(x)     # (B, d_model, T)
        x = self.dropout(x)
        return x.transpose(1, 2) # (B, T, d_model)


# =====================================================================
# ConformerBlock
# =====================================================================

class ConformerBlock(nn.Module):
    """Conformer block: FF(½) → MHSA → DepthwiseConv → FF(½).

    Pre-norm + residual ở mỗi sub-block (theo paper gốc).
    """

    def __init__(
        self,
        d_model: int = 144,
        num_heads: int = 4,
        conv_kernel: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff1_norm = nn.LayerNorm(d_model)
        self.ff1      = FeedForward(d_model, expansion=4, dropout=dropout)

        self.attn_norm    = nn.LayerNorm(d_model)
        self.attn         = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.conv_norm = nn.LayerNorm(d_model)
        self.conv      = ConvModule(d_model, conv_kernel, dropout=dropout)

        self.ff2_norm = nn.LayerNorm(d_model)
        self.ff2      = FeedForward(d_model, expansion=4, dropout=dropout)

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FF sub-block (scaled ½)
        x = x + 0.5 * self.ff1(self.ff1_norm(x))

        # MHSA sub-block
        n = self.attn_norm(x)
        attn_out, _ = self.attn(n, n, n)
        x = x + self.attn_dropout(attn_out)

        # DepthwiseConv sub-block
        x = x + self.conv(self.conv_norm(x))

        # FF sub-block (scaled ½)
        x = x + 0.5 * self.ff2(self.ff2_norm(x))

        return self.final_norm(x)


# =====================================================================
# SplitConformerCTC — Model chính
# =====================================================================

class SplitConformerCTC(nn.Module):
    """Conformer CTC với 2 luồng tách biệt (Pose + Hand).

    Nhận input 602-dim từ extractor, tự slice bên trong.
    Eyebrow dims (268:301 và 569:602) bị bỏ qua hoàn toàn.

    Args:
        feature_dim: Tổng feature dim từ extractor (phải là 602 khi USE_VELOCITY=True).
        num_conformer_layers: Số Conformer block xếp chồng.
        num_heads: Số attention heads (d_model=144 phải chia hết cho num_heads).
        conv_kernel: Kernel size cho DepthwiseConv (phải lẻ).
        num_classes: Số lớp CTC output (gồm blank).
        dropout: Dropout chung.
        use_aux_loss: Bật auxiliary loss head (mean-pooled → cross-entropy).
    """

    # Index cứng theo feature layout của extractor.py
    _POSE_RAW   = (0, 132)
    _HAND_RAW   = (132, 268)
    _POSE_VEL   = (301, 433)
    _HAND_VEL   = (433, 569)
    # Eyebrow: [268:301] và [569:602] — không dùng

    def __init__(
        self,
        feature_dim: int = 602,
        num_conformer_layers: int = 2,
        num_heads: int = 4,
        conv_kernel: int = 9,
        num_classes: int = 35,
        dropout: float = 0.3,
        use_aux_loss: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.use_aux_loss = use_aux_loss

        # --- Stream projections ---
        # Pose: 132 (raw) + 132 (vel) = 264 dim → 64
        self.pose_proj = nn.Sequential(
            nn.Linear(264, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )
        # Hand: 136 (raw) + 136 (vel) = 272 dim → 80
        self.hand_proj = nn.Sequential(
            nn.Linear(272, 80),
            nn.ReLU(),
            nn.LayerNorm(80),
        )

        # d_model = 64 + 80 = 144
        d_model = 144
        self.input_norm = nn.LayerNorm(d_model)

        # --- Conformer stack ---
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                conv_kernel=conv_kernel,
                dropout=dropout,
            )
            for _ in range(num_conformer_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(d_model, num_classes)

        # --- Auxiliary head (optional) ---
        if use_aux_loss:
            self.aux_fc = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init cho Linear layers, uniform cho Conv."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d,)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, T, 602) — feature tensor đầy đủ từ extractor
            training: True khi đang train → bật modality dropout

        Returns:
            logits: (B, T, num_classes)  nếu use_aux_loss=False
            (logits, aux_logits): tuple nếu use_aux_loss=True
                aux_logits: (B, num_classes) từ mean-pooled representation
        """
        pr, pr_e = self._POSE_RAW
        hr, hr_e = self._HAND_RAW
        pv, pv_e = self._POSE_VEL
        hv, hv_e = self._HAND_VEL

        # ── Tách stream ───────────────────────────────────────
        pose = torch.cat([x[..., pr:pr_e], x[..., pv:pv_e]], dim=-1)  # (B,T,264)
        hand = torch.cat([x[..., hr:hr_e], x[..., hv:hv_e]], dim=-1)  # (B,T,272)
        # Eyebrow [268:301], [569:602] — bỏ qua

        # ── Modality dropout (chỉ training) ───────────────────
        if training and self.training:
            pose = modality_dropout(pose, p=0.15)
            # Hand KHÔNG dropout — quan trọng nhất

        # ── Project từng stream ───────────────────────────────
        p = self.pose_proj(pose)   # (B,T,64)
        h = self.hand_proj(hand)   # (B,T,80)

        # ── Fuse + Input Norm ─────────────────────────────────
        out = self.input_norm(torch.cat([p, h], dim=-1))  # (B,T,144)

        # ── Conformer stack ───────────────────────────────────
        for layer in self.conformer_layers:
            out = layer(out)

        out = self.dropout(self.output_norm(out))  # (B,T,144)
        logits = self.fc(out)                       # (B,T,num_classes)

        if self.use_aux_loss and training:
            # Aux head chỉ dùng khi training — không cần lúc inference
            pooled     = out.mean(dim=1)              # (B,144)
            aux_logits = self.aux_fc(pooled)          # (B,num_classes)
            return logits, aux_logits

        return logits
