"""Pipeline chính: landmarks → text + audio.

Class ``Pipeline`` đóng vai trò trung tâm — buffer frames, chạy
Bi-LSTM inference khi đủ window, decode CTC word-level, gọi TTS.
"""
from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

# ── re-export các thành phần con ───────────────────────────────
from .model import BiLSTMCTC
from .decoder import decode_to_text, normalize_vietnamese
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
        use_eyebrow: bool = False,
        blank_idx: int = 0,
        idx_to_char: dict[int, str] | None = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = window_size
        self.window_stride = window_stride
        self.tts_voice = tts_voice
        self.use_face = use_face
        self.use_eyebrow = use_eyebrow
        self.blank_idx = blank_idx
        self.idx_to_char = idx_to_char or {}

        self.frame_buffer: deque[np.ndarray] = deque(maxlen=150)
        self._frames_since_infer: int = 0
        self._confidence_threshold: float = 0.40  # Chỉ emit khi model đủ tự tin

        # Silence detection: đếm frame liên tiếp không có tay
        self._silence_frames: int = 0
        self._silence_trigger: int = 15   # frames im lặng → trigger infer
        self._min_infer_frames: int = 10  # buffer tối thiểu để infer

        # Voting nhẹ: chỉ emit khi predict 1 lần (không cần majority vì đã infer toàn gesture)
        self._last_emitted: str = ""

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
    def _hands_visible(self, landmarks_json: dict[str, Any]) -> bool:
        """Trả True nếu có ít nhất một tay được phát hiện trong frame.

        Bỏ filter y < 0.85 vì gây mất frame khi camera góc thấp
        hoặc người dùng làm cử chỉ ở vùng thấp của frame.
        """
        lm = landmarks_json.get("landmarks", {})
        left_hand = lm.get("left_hand")
        right_hand = lm.get("right_hand")

        if left_hand and len(left_hand) > 0:
            return True
        if right_hand and len(right_hand) > 0:
            return True
        return False

    def add_frame(self, landmarks_json: dict[str, Any]) -> None:
        """Thêm một frame landmarks vào buffer."""
        features = landmarks_json_to_array(
            landmarks_json, use_face=self.use_face, use_eyebrow=self.use_eyebrow
        )
        self.frame_buffer.append(features)
        self._frames_since_infer += 1

    def add_features(self, features: np.ndarray) -> None:
        """Thêm feature vector đã tính sẵn vào buffer (dùng khi backend MediaPipe)."""
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
    def _run_infer_on_buffer(self) -> tuple[str | None, float]:
        """Chạy inference trên toàn bộ buffer hiện tại, trả về (text, confidence).

        CTC word-level: model output (1, T, vocab_size) → greedy decode → câu.
        """
        if self.model is None or len(self.frame_buffer) < self._min_infer_frames:
            self.frame_buffer.clear()
            return None, 0.0

        window = np.array(list(self.frame_buffer))
        self.frame_buffer.clear()
        self._frames_since_infer = 0
        self._silence_frames = 0

        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(x)                          # (1, T, num_classes)

        # CTC greedy decode → text
        results = decode_to_text(logits, self.idx_to_char, blank_idx=self.blank_idx)
        text, confidence = results[0]

        logger.info(f"[infer] decoded={text!r} conf={confidence:.2f} frames={len(window)}")

        if not text or text.strip().lower() == "blank":
            return None, confidence

        if confidence < self._confidence_threshold:
            logger.info(f"[infer] Bỏ qua — confidence {confidence:.2f} < {self._confidence_threshold}")
            return None, confidence

        return text, confidence

    async def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Xử lý hoàn chỉnh một frame: buffer → infer → TTS.

        Trigger inference khi:
        - Tay biến mất khỏi frame đủ lâu (silence detection), HOẶC
        - Buffer tích lũy đủ window_size frames.

        Returns:
            dict với keys: text, audio, confidence, latency_ms
        """
        t0 = time.perf_counter()

        hands_active = self._hands_visible(data)

        result: dict[str, Any] = {
            "text": "",
            "audio": "",
            "confidence": 0.0,
            "latency_ms": 0,
        }

        raw_text: str | None = None
        confidence: float = 0.0

        if hands_active:
            # Tay đang hiện → thêm frame vào buffer, reset silence counter
            if self._silence_frames > 0:
                # Vừa bắt đầu gesture mới → cho phép emit lại cùng nhãn
                self._last_emitted = ""
            self.add_frame(data)
            self._silence_frames = 0

            # Safety valve: buffer đầy kịch kim → ngắt bắt buộc
            if len(self.frame_buffer) == 150:
                raw_text, confidence = self._run_infer_on_buffer()
                logger.warning(f"[infer:timeout] Câu quá dài, ngắt bắt buộc. raw={raw_text!r}")

            # Log tiến trình buffer mỗi 30 frame
            buf_len = len(self.frame_buffer)
            if buf_len % 30 == 0 and buf_len > 0:
                logger.debug(f"[buffer] {buf_len} frames tích lũy")

        else:
            # Tay không hiện → đếm silence
            self._silence_frames += 1
            logger.debug(
                f"[silence] frame={self._silence_frames}/{self._silence_trigger} | buffer={len(self.frame_buffer)}"
            )

            # Người dùng VỪA DỪNG TAY → dịch toàn bộ câu
            if (
                self._silence_frames >= self._silence_trigger      # >= thay vì == để tránh miss 1 frame
                and self._silence_frames < self._silence_trigger + 5  # chỉ trigger 1 lần
                and len(self.frame_buffer) >= self._min_infer_frames
            ):
                raw_text, confidence = self._run_infer_on_buffer()
                logger.info(f"[infer:silence_boundary] Dịch câu hoàn chỉnh. raw={raw_text!r}")

            # Đã im lặng quá lâu → reset rác nếu có
            elif self._silence_frames > self._silence_trigger + 5:
                if self.frame_buffer:
                    logger.debug("[reset] Buffer rác xóa sau im lặng dài")
                self.frame_buffer.clear()
                self._last_emitted = ""

        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return await self._emit(result, raw_text, confidence)

    async def process_features(
        self, features: np.ndarray, has_hands: bool
    ) -> dict[str, Any]:
        """Xử lý một frame khi backend đã chạy MediaPipe (feature vector có sẵn).

        Dùng bởi /ws/video endpoint thay cho process().
        Logic silence detection và emit giống hệt process().
        """
        t0 = time.perf_counter()

        result: dict[str, Any] = {
            "text": "",
            "audio": "",
            "confidence": 0.0,
            "latency_ms": 0,
        }

        raw_text: str | None = None
        confidence: float = 0.0

        if has_hands:
            if self._silence_frames > 0:
                self._last_emitted = ""
            self.add_features(features)
            self._silence_frames = 0

            if len(self.frame_buffer) == 150:
                raw_text, confidence = self._run_infer_on_buffer()
                logger.warning(f"[infer:timeout] Câu quá dài, ngắt bắt buộc. raw={raw_text!r}")

            buf_len = len(self.frame_buffer)
            if buf_len % 30 == 0 and buf_len > 0:
                logger.debug(f"[buffer] {buf_len} frames tích lũy")

        else:
            self._silence_frames += 1
            logger.debug(
                f"[silence] frame={self._silence_frames}/{self._silence_trigger} | buffer={len(self.frame_buffer)}"
            )

            if (
                self._silence_frames >= self._silence_trigger
                and self._silence_frames < self._silence_trigger + 5
                and len(self.frame_buffer) >= self._min_infer_frames
            ):
                raw_text, confidence = self._run_infer_on_buffer()
                logger.info(f"[infer:silence_boundary] Dịch câu hoàn chỉnh. raw={raw_text!r}")

            elif self._silence_frames > self._silence_trigger + 5:
                if self.frame_buffer:
                    logger.debug("[reset] Buffer rác xóa sau im lặng dài")
                self.frame_buffer.clear()
                self._last_emitted = ""

        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return await self._emit(result, raw_text, confidence)

    async def _emit(
        self,
        result: dict[str, Any],
        raw_text: str | None,
        confidence: float,
    ) -> dict[str, Any]:
        """Hoàn thiện result dict: lọc rác, gọi TTS nếu cần."""
        if raw_text:
            clean = raw_text.strip().lower()
            if "blank" in clean or clean == "x":
                raw_text = None

        if raw_text and raw_text != self._last_emitted:
            self._last_emitted = raw_text
            result["text"] = raw_text
            result["confidence"] = round(confidence, 3)
            try:
                audio_b64 = await synthesize(raw_text, self.tts_voice)
                result["audio"] = audio_b64
            except Exception as e:
                logger.error(f"TTS error: {e}")

        return result
