"""Pipeline chính: landmarks → text + audio.

Class ``Pipeline`` đóng vai trò trung tâm — buffer frames, chạy
Bi-LSTM inference khi đủ window, decode CTC, gọi TTS.
"""
from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from typing import Any

import numpy as np
import torch
from loguru import logger

# ── re-export các thành phần con ───────────────────────────────
from .model import BiLSTMCTC
from .decoder import greedy_decode, normalize_vietnamese
from .extractor import landmarks_to_features, landmarks_json_to_array
from .tts import synthesize


class Pipeline:
    """Orchestrator cho toàn bộ inference pipeline."""

    def __init__(
        self,
        model_path: str,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        window_size: int = 30,
        window_stride: int = 5,
        tts_voice: str = "vi-VN-HoaiMyNeural",
        use_face: bool = True,
        blank_idx: int = 0,
        idx_to_char: dict[int, str] | None = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = window_size
        self.window_stride = window_stride
        self.tts_voice = tts_voice
        self.use_face = use_face
        self.blank_idx = blank_idx
        self.idx_to_char = idx_to_char or {}

        self.frame_buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._frames_since_infer: int = 0

        # ── Load model ────────────────────────────────────────
        self.model: BiLSTMCTC | None = None
        if os.path.exists(model_path):
            self.model = BiLSTMCTC(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=0.0,  # không dropout khi inference
            ).to(self.device)
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            logger.info(f"Model loaded từ {model_path} trên {self.device}")
        else:
            logger.warning(
                f"Chưa có model tại {model_path}. "
                "Pipeline chạy ở chế độ demo (trả text rỗng)."
            )

    # ──────────────────────────────────────────────────────────
    def add_frame(self, landmarks_json: dict[str, Any]) -> None:
        """Thêm một frame landmarks vào buffer."""
        features = landmarks_json_to_array(landmarks_json, use_face=self.use_face)
        self.frame_buffer.append(features)
        self._frames_since_infer += 1

    def should_infer(self) -> bool:
        """Kiểm tra đã đủ window và stride chưa."""
        return (
            self.model is not None
            and len(self.frame_buffer) >= self.window_size
            and self._frames_since_infer >= self.window_stride
        )

    @torch.no_grad()
    def infer(self) -> str | None:
        """Chạy inference trên window hiện tại."""
        if not self.should_infer():
            return None

        self._frames_since_infer = 0

        window = np.array(list(self.frame_buffer))  # (T, feature_dim)
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)

        log_probs = self.model(x)  # (1, T, C)
        decoded = greedy_decode(log_probs, blank_idx=self.blank_idx)

        if not decoded[0]:
            return None

        text = "".join(self.idx_to_char.get(i, "") for i in decoded[0] if i != self.blank_idx)
        return normalize_vietnamese(text) if text.strip() else None

    async def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Xử lý hoàn chỉnh một frame: buffer → infer → TTS.

        Returns:
            dict với keys: text, audio, confidence, latency_ms
        """
        t0 = time.perf_counter()

        self.add_frame(data)
        text = self.infer()

        result: dict[str, Any] = {
            "text": text or "",
            "audio": "",
            "confidence": 0.0,
            "latency_ms": 0,
        }

        if text:
            try:
                audio_b64 = await synthesize(text, self.tts_voice)
                result["audio"] = audio_b64
                result["confidence"] = 0.9  # placeholder — cải thiện khi có beam search
            except Exception as e:
                logger.error(f"TTS error: {e}")

        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result
