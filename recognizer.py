"""Nhận diện ngôn ngữ ký hiệu tiếng Việt — Desktop App.

Chạy hoàn toàn offline trên desktop (giống data_collector.py),
sử dụng Python MediaPipe (nhanh hơn nhiều so với JS trên browser).

Pipeline: Webcam → MediaPipe → BiLSTM CTC → Text + TTS

Chạy:
    python recognizer.py
    python recognizer.py --no-tts        # Tắt giọng nói
    python recognizer.py --no-eyebrow    # Bỏ eyebrow features
"""
from __future__ import annotations

import argparse
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, Future

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config as cfg
from pipeline.model import BiLSTMCTC
from pipeline.decoder import decode_to_text, normalize_vietnamese
from pipeline.extractor import landmarks_to_features, augment_sequence_with_velocity
from pipeline.tts import TTSPlayer

try:
    from pipeline.model_conformer import SplitConformerCTC
    _CONFORMER_AVAILABLE = True
except ImportError:
    _CONFORMER_AVAILABLE = False

try:
    from PIL import Image as _PilImage, ImageDraw as _PilDraw, ImageFont as _PilFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ==========================================
# TẢI MODEL NẾU CHƯA CÓ
# ==========================================
_MODEL_DOWNLOADS: list[tuple[str, str]] = [
    (cfg.POSE_MODEL_URL, cfg.POSE_MODEL_PATH),
    (cfg.HAND_MODEL_URL, cfg.HAND_MODEL_PATH),
    (cfg.FACE_MODEL_URL, cfg.FACE_MODEL_PATH),
]


def download_if_missing(url: str, path: str) -> None:
    if not os.path.exists(path):
        print(f"Đang tải model: {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[OK] Đã tải: {os.path.basename(path)}")


def ensure_models(use_face: bool = True) -> None:
    os.makedirs(cfg.MEDIAPIPE_DIR, exist_ok=True)
    for url, path in _MODEL_DOWNLOADS:
        if "face_landmarker" in path and not use_face:
            continue
        download_if_missing(url, path)


# ==========================================
# CONNECTIONS ĐỂ VẼ
# ==========================================
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]


# ==========================================
# HÀM VẼ
# ==========================================
def draw_landmarks(
    image: np.ndarray,
    pose_result: mp_vision.PoseLandmarkerResult,
    hand_result: mp_vision.HandLandmarkerResult,
    face_result: mp_vision.FaceLandmarkerResult | None = None,
) -> None:
    h, w = image.shape[:2]

    def _draw(
        landmarks_list: list,
        dot_color: tuple[int, int, int],
        line_color: tuple[int, int, int],
        connections: list[tuple[int, int]],
        dot_radius: int = 3,
    ) -> None:
        for landmarks in landmarks_list:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for s, e in connections:
                if s < len(pts) and e < len(pts):
                    cv2.line(image, pts[s], pts[e], line_color, 1, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(image, pt, dot_radius, dot_color, -1)

    _draw(pose_result.pose_landmarks, (0, 255, 0), (0, 180, 0), POSE_CONNECTIONS)
    _draw(hand_result.hand_landmarks, (255, 255, 255), (180, 180, 180), HAND_CONNECTIONS, 4)

    if face_result and face_result.face_landmarks:
        for face_lm in face_result.face_landmarks:
            for lm in face_lm:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 1, (0, 200, 255), -1)


def _put_vn_text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_size: int = 20,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    if _PIL_AVAILABLE:
        pil = _PilImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = _PilDraw.Draw(pil)
        font = None
        for name in ("arial.ttf", "DejaVuSans.ttf", "NotoSans-Regular.ttf"):
            try:
                font = _PilFont.truetype(name, font_size)
                break
            except OSError:
                pass
        if font is None:
            font = _PilFont.load_default()
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        safe = text.encode("ascii", errors="replace").decode()
        cv2.putText(img, safe, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 32.0, color, 1, cv2.LINE_AA)


# ==========================================
# TRÍCH XUẤT ĐẶC TRƯNG
# ==========================================
def extract_keypoints(
    pose_result: mp_vision.PoseLandmarkerResult,
    hand_result: mp_vision.HandLandmarkerResult,
    face_result: mp_vision.FaceLandmarkerResult | None = None,
    use_face: bool = True,
    use_eyebrow: bool = False,
) -> np.ndarray:
    pose_raw = None
    if pose_result.pose_landmarks:
        pose_raw = [
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in pose_result.pose_landmarks[0]
        ]

    lh_raw, rh_raw = None, None
    for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
        label = hand_result.handedness[i][0].category_name
        coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
        if label == "Left":
            lh_raw = coords
        else:
            rh_raw = coords

    face_raw = None
    if (use_face or use_eyebrow) and face_result and face_result.face_landmarks:
        face_raw = [[lm.x, lm.y, lm.z] for lm in face_result.face_landmarks[0]]

    return landmarks_to_features(
        pose=pose_raw,
        left_hand=lh_raw,
        right_hand=rh_raw,
        face=face_raw,
        use_face=use_face,
        use_eyebrow=use_eyebrow,
    )


def hands_visible(hand_result: mp_vision.HandLandmarkerResult) -> bool:
    return len(hand_result.hand_landmarks) > 0


def extract_hand_activity(
    hand_result: mp_vision.HandLandmarkerResult,
) -> float:
    """Lấy tọa độ Y tuyệt đối thấp nhất (gần đỉnh đầu nhất) của cổ tay.

    MediaPipe hand landmark 0 = wrist.  Y trong [0,1], 0=trên, 1=dưới.
    Nếu nhiều tay: lấy min Y (tay cao nhất — đang active nhất).
    Nếu không có tay: trả về 1.0 (= "dưới đáy frame").
    """
    if not hand_result.hand_landmarks:
        return 1.0
    wrist_ys: list[float] = []
    for hand_lm in hand_result.hand_landmarks:
        wrist_ys.append(hand_lm[0].y)  # landmark 0 = wrist
    return min(wrist_ys)


# ==========================================
# INFERENCE ENGINE — HAD State Machine + Rejection Stack
# ==========================================

# Offset tay trong feature vector 301-dim (USE_FACE=False, USE_EYEBROW=True):
#   pose: [0:132], lh_coords: [132:195], lh_angles: [195:200],
#   rh_coords: [200:263], rh_angles: [263:268], eyebrow: [268:301]
_HAND_SLICE = slice(132, 268)  # 136 dims: cả 2 tay (coords + angles)


class InferenceEngine:
    """HAD (Hand Activity Detection) State Machine cho real-time sign recognition.

    Thay thế Sliding Window cố định bằng Dynamic Window:
    - IDLE: chờ tay bắt đầu di chuyển (energy > threshold)
    - SIGNING: buffer frames liên tục cho đến khi phát hiện ranh giới ký hiệu
    - Ranh giới (End-of-Sign):
        A. Tay biến mất khỏi màn hình (SILENCE_TRIGGER frames)
        B. Tay đứng yên (energy < threshold liên tục PAUSE_FRAMES frames)
        C. Fail-safe: buffer >= MAX_BUFFER_FRAMES
    - Flush: chạy model 1 LẦN trên toàn bộ buffer → decode → emit → reset

    Lưu ý: Y_DROP (tay hạ xuống) đã bị loại bỏ — gây infer sớm khi người dùng
    còn đang ký hiệu ở vị trí thấp.
    """

    def __init__(
        self,
        use_face: bool,
        use_eyebrow: bool,
        model_override: str = "auto",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_face = use_face
        self.use_eyebrow = use_eyebrow

        # ── HAD State Machine ─────────────────────────────────────
        self._state: str = "IDLE"  # "IDLE" hoặc "SIGNING"
        self.frame_buffer: list[np.ndarray] = []
        self._prev_features: np.ndarray | None = None  # frame trước, tính energy
        self._current_energy: float = 0.0               # energy frame hiện tại (debug/UI)
        self._low_energy_count: int = 0                  # đếm frame liên tiếp energy thấp
        self._no_hand_count: int = 0                     # đếm frame liên tiếp không thấy tay

        # ── Emit cooldown ─────────────────────────────────────────
        self._last_emitted: str = ""
        self._last_emit_time: float = 0.0

        # Phrase list từ dataset để snap kết quả decode
        self.phrase_list: list[str] = self._load_phrase_list()

        # ── Load model ────────────────────────────────────────────
        self.model: BiLSTMCTC | None = None
        self.idx2word: dict[int, str] = cfg.IDX_TO_CHAR
        self._model_type: str = "bilstm"   # dùng khi gọi forward

        _bilstm_path    = cfg.TRAINED_MODEL_PATH       # models/sltt/bilstm/bilstm_ctc.pt
        _conformer_path = cfg.CONFORMER_MODEL_PATH     # models/sltt/conformer/conformer_ctc.pt

        # Chọn đúng checkpoint path TRƯỚC khi torch.load
        #   auto      → conformer nếu có, else bilstm
        #   conformer → conformer_ctc.pt (fallback bilstm nếu chưa train)
        #   bilstm    → bilstm_ctc.pt
        if model_override == "conformer":
            _ckpt_path = _conformer_path
        elif model_override == "bilstm":
            _ckpt_path = _bilstm_path
        else:  # "auto"
            _ckpt_path = _conformer_path if os.path.exists(_conformer_path) else _bilstm_path

        if os.path.exists(_ckpt_path):
            self._load_model(_ckpt_path, model_override)
        elif model_override == "conformer" and os.path.exists(_bilstm_path):
            print("[MODEL] ⚠ Chưa có conformer_ctc.pt — fallback về bilstm_ctc.pt")
            self._load_model(_bilstm_path, "bilstm")
        elif os.path.exists(_bilstm_path):
            self._load_model(_bilstm_path, "bilstm")
        else:
            print("[MODEL] Chưa có model nào. Hãy train trước: python train.py")

    def _load_model(self, path: str, model_override: str = "auto") -> None:
        """Load checkpoint (dict mới hoặc state_dict thuần) và khởi tạo model phù hợp."""
        raw = torch.load(path, map_location=self.device, weights_only=False)

        # Phân biệt checkpoint dict mới và state_dict thuần (backward compat)
        if isinstance(raw, dict) and "model_state" in raw:
            state        = raw["model_state"]
            ckpt_type    = raw.get("model_type", "bilstm")
            num_classes  = raw.get("num_classes", state.get("fc.weight", torch.zeros(35, 1)).shape[0])
            # Đọc use_aux_loss từ metadata — phải khởi tạo model khớp với checkpoint
            ckpt_aux_loss = raw.get("use_aux_loss", False)
        else:
            # Legacy: state_dict thuần từ BiLSTMCTC
            state         = raw
            ckpt_type     = "bilstm"
            num_classes   = state["fc.weight"].shape[0]
            ckpt_aux_loss = False

        # Override nếu user chỉ định --model tường minh
        model_type = ckpt_type if model_override == "auto" else model_override

        if model_type == "conformer":
            if not _CONFORMER_AVAILABLE:
                print("[MODEL] ⚠ pipeline/model_conformer.py không tìm thấy! Fallback về BiLSTMCTC.")
                model_type = "bilstm"
            else:
                self.model = SplitConformerCTC(
                    feature_dim=cfg.FEATURE_DIM,
                    num_classes=num_classes,
                    use_aux_loss=ckpt_aux_loss,  # khớp với checkpoint (có thể có aux_fc)
                ).to(self.device)

        if model_type != "conformer":
            self.model = BiLSTMCTC(
                feature_dim=cfg.FEATURE_DIM,
                hidden_dim=cfg.HIDDEN_DIM,
                num_classes=num_classes,
                num_layers=cfg.NUM_LSTM_LAYERS,
                dropout=0.0,
            ).to(self.device)

        self.model.load_state_dict(state)
        self.model.eval()
        self._model_type = model_type

        from vocab import load_vocab, vocab_to_dicts
        _vocab = load_vocab()
        _, self.idx2word = vocab_to_dicts(_vocab[:num_classes])

        print(
            f"[MODEL] Loaded từ {path} | type={model_type} | "
            f"device={self.device} | num_classes={num_classes} "
            f"(vocab hiện tại: {cfg.NUM_CLASSES})"
        )
        if num_classes != cfg.NUM_CLASSES:
            print(
                f"[MODEL] ⚠ Checkpoint ({num_classes} classes) khác vocab hiện tại "
                f"({cfg.NUM_CLASSES} classes). Cần train lại để nhận diện đầy đủ từ vựng mới."
            )

    @staticmethod
    def _load_phrase_list() -> list[str]:
        """Đọc danh sách câu duy nhất từ labels.csv để dùng cho phrase snapping."""
        import unicodedata
        import pandas as pd
        label_path = cfg.LABEL_FILE
        try:
            df = pd.read_csv(label_path)
            phrases = sorted({
                unicodedata.normalize("NFC", str(t).strip().lower())
                for t in df["text"]
                if str(t).strip()
            })
            print(f"[PHRASE] Loaded {len(phrases)} phrases: {phrases}")
            return phrases
        except Exception as e:
            print(f"[PHRASE] Không thể load phrases: {e}")
            return []

    # ──────────────────────────────────────────────────────────
    # KINEMATIC ENERGY — tính từ feature vector raw (301-dim)
    # ──────────────────────────────────────────────────────────

    def _compute_energy(self, features: np.ndarray) -> float:
        """Động năng tay = sum(Δhand²) giữa frame hiện tại và frame trước.

        Dùng 136 dims hand data (coords + angles, cả 2 tay) trong feature vector.
        Tọa độ là relative-to-nose nhưng velocity vẫn phản ánh chuyển động tay.
        """
        if self._prev_features is None:
            return 0.0
        delta = features[_HAND_SLICE] - self._prev_features[_HAND_SLICE]
        return float(np.sum(delta ** 2))

    # ──────────────────────────────────────────────────────────
    # REJECTION STACK — chạy TRƯỚC beam search decode (nhanh, O(T))
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _blank_ratio(logits: torch.Tensor, blank_idx: int = 0) -> float:
        """Tỷ lệ frame predict blank. Noise/idle thường >95%."""
        preds = logits.argmax(dim=-1)  # (T,)
        return float((preds == blank_idx).sum().item()) / max(preds.size(0), 1)

    @staticmethod
    def _mean_entropy(logits: torch.Tensor) -> float:
        """Shannon entropy chuẩn hóa trung bình tại non-blank frames."""
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        non_blank = preds != 0
        if not non_blank.any():
            return 1.0
        p = probs[non_blank]
        log_p = torch.log(p + 1e-10)
        entropy = -(p * log_p).sum(dim=-1)
        max_entropy = math.log(logits.size(-1))
        normalized = entropy / max_entropy
        return float(normalized.mean().item())

    @staticmethod
    def _non_blank_count(logits: torch.Tensor, blank_idx: int = 0) -> int:
        """Số frame predict non-blank."""
        preds = logits.argmax(dim=-1)
        return int((preds != blank_idx).sum().item())

    def _rejection_gate(self, logits: torch.Tensor) -> tuple[bool, str]:
        """Chạy rejection stack 3 lớp trước decode."""
        br = self._blank_ratio(logits, cfg.BLANK_IDX)
        if br > cfg.BLANK_RATIO_THRESHOLD:
            return False, f"blank_ratio={br:.2f}>{cfg.BLANK_RATIO_THRESHOLD}"
        nb = self._non_blank_count(logits, cfg.BLANK_IDX)
        if nb < cfg.MIN_NON_BLANK_FRAMES:
            return False, f"non_blank={nb}<{cfg.MIN_NON_BLANK_FRAMES}"
        ent = self._mean_entropy(logits)
        if ent > cfg.ENTROPY_THRESHOLD:
            return False, f"entropy={ent:.2f}>{cfg.ENTROPY_THRESHOLD}"
        return True, "ok"

    # ──────────────────────────────────────────────────────────
    # INFERENCE — chạy trên toàn bộ buffer
    # ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_infer_window(self, frames: np.ndarray) -> tuple[str | None, float]:
        """Infer trên một chuỗi frames. KHÔNG xóa buffer.

        Args:
            frames: (T, raw_feature_dim) numpy array

        Returns:
            (text, confidence) hoặc (None, 0.0)
        """
        if self.model is None or len(frames) < cfg.MIN_SEQUENCE_LENGTH:
            return None, 0.0

        window = frames.copy()
        if cfg.USE_VELOCITY:
            window = augment_sequence_with_velocity(window)

        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        # Conformer nhận training=False để tắt modality dropout
        if self._model_type == "conformer":
            out = self.model(x, training=False)
            logits = out[0] if isinstance(out, tuple) else out
        else:
            logits = self.model(x)  # (1, T, C)
        logits_single = logits[0]  # (T, C)

        # ── Debug info ──────────────────────────────────────
        _br = self._blank_ratio(logits_single, cfg.BLANK_IDX)
        _nb = self._non_blank_count(logits_single, cfg.BLANK_IDX)
        _ent = self._mean_entropy(logits_single)
        _feat_mean = float(np.abs(frames).mean())
        print(
            f"[INFER] T={len(frames)} feat_mean={_feat_mean:.4f} | "
            f"br={_br:.2f} nb={_nb} ent={_ent:.2f}"
        )

        # ── Rejection gate ──────────────────────────────────
        passed, reason = self._rejection_gate(logits_single)
        if not passed:
            print(f"[REJECT] {reason}")
            return None, 0.0

        # ── Decode (beam search + phrase snap) ──────────────
        results = decode_to_text(
            logits,
            self.idx2word,
            blank_idx=cfg.BLANK_IDX,
            confidence_threshold=cfg.CONFIDENCE_THRESHOLD,
            phrase_list=self.phrase_list,
            min_frames=cfg.MIN_DECODE_FRAMES,
        )
        text, confidence = results[0]

        # ── Debug decode output ─────────────────────────────
        _probs_np = torch.softmax(logits_single, dim=-1).cpu().numpy()
        _preds = _probs_np.argmax(axis=-1)
        _tokens = [self.idx2word.get(int(p), f"?{p}") for p in _preds]
        _summary, _prev = [], None
        for t in _tokens:
            if t != _prev:
                _summary.append(t)
                _prev = t
        print(
            f"[CTC raw] {' → '.join(_summary)} | "
            f"decoded={text!r} conf={confidence:.2f}"
        )

        if not text or text.strip().lower() == "blank":
            return None, confidence

        return text, confidence

    # ──────────────────────────────────────────────────────────
    # FLUSH + RESET
    # ──────────────────────────────────────────────────────────

    def _flush_and_infer(self) -> tuple[str | None, float]:
        """Flush toàn bộ buffer vào model, trả về kết quả."""
        if len(self.frame_buffer) < cfg.MIN_SIGN_FRAMES:
            print(f"[SKIP] Buffer quá ngắn: {len(self.frame_buffer)} < {cfg.MIN_SIGN_FRAMES}")
            return None, 0.0
        frames = np.array(self.frame_buffer)
        text, conf = self._run_infer_window(frames)
        if text:
            return self._try_emit(text, conf)
        return None, 0.0

    def _reset_to_idle(self) -> None:
        """Reset state machine về IDLE, clear buffer."""
        self._state = "IDLE"
        self.frame_buffer.clear()
        self._low_energy_count = 0
        self._no_hand_count = 0
        # Giữ _prev_features cho transition tiếp theo

    # ──────────────────────────────────────────────────────────
    # EMIT — cooldown + duplicate check
    # ──────────────────────────────────────────────────────────

    def _try_emit(self, text: str, confidence: float) -> tuple[str | None, float]:
        """Kiểm tra cooldown + duplicate trước khi emit."""
        now = time.time()
        if text == self._last_emitted and (now - self._last_emit_time) < cfg.EMIT_COOLDOWN:
            return None, 0.0
        clean = text.strip().lower()
        if "blank" in clean or clean == "x" or clean == "neutral":
            return None, 0.0
        self._last_emitted = text
        self._last_emit_time = now
        return text, confidence

    # ──────────────────────────────────────────────────────────
    # FEED FRAME — entry point chính, gọi mỗi frame từ main loop
    # ──────────────────────────────────────────────────────────

    def feed_frame(
        self,
        features: np.ndarray,
        has_hands: bool,
        wrist_y: float,
    ) -> tuple[str | None, float]:
        """Thêm 1 frame vào HAD state machine, trả về kết quả nếu phát hiện ranh giới.

        Args:
            features: 301-dim raw feature vector (chưa velocity)
            has_hands: MediaPipe có detect tay không
            wrist_y: tọa độ Y tuyệt đối cổ tay (0=trên, 1=dưới), 1.0 nếu không có tay

        Returns: (text, confidence) hoặc (None, 0.0)
        """
        # ── Tính động năng tay ────────────────────────────────
        energy = self._compute_energy(features)
        self._current_energy = energy
        self._prev_features = features.copy()

        # ══════════════════════════════════════════════════════
        # STATE: IDLE — chờ tay bắt đầu signing
        # ══════════════════════════════════════════════════════
        if self._state == "IDLE":
            # Chỉ cần có tay + cử động → bắt đầu ghi
            # KHÔNG kiểm tra wrist_y ở đây — Y_DROP chỉ dùng để KẾT THÚC
            if has_hands and energy > cfg.ENERGY_THRESHOLD:
                self._state = "SIGNING"
                self.frame_buffer.append(features)
                self._low_energy_count = 0
                self._no_hand_count = 0
                print(
                    f"[STATE] IDLE → SIGNING "
                    f"(energy={energy:.4f}, wrist_y={wrist_y:.2f})"
                )
            return None, 0.0

        # ══════════════════════════════════════════════════════
        # STATE: SIGNING — đang buffer frames
        # ══════════════════════════════════════════════════════
        self.frame_buffer.append(features)
        end_reason: str | None = None

        # Condition A: Tay biến mất
        if not has_hands:
            self._no_hand_count += 1
            if self._no_hand_count >= cfg.SILENCE_TRIGGER:
                end_reason = "silence"
        else:
            self._no_hand_count = 0

        # Condition B: Pause — tay đứng yên (energy thấp liên tục)
        if end_reason is None:
            if energy < cfg.ENERGY_THRESHOLD:
                self._low_energy_count += 1
                if self._low_energy_count >= cfg.PAUSE_FRAMES:
                    end_reason = "pause"
            else:
                self._low_energy_count = 0

        # Fail-safe: buffer quá dài
        if end_reason is None and len(self.frame_buffer) >= cfg.MAX_BUFFER_FRAMES:
            end_reason = "max_buffer"

        # ── Flush nếu phát hiện ranh giới ─────────────────────
        if end_reason:
            buf_len = len(self.frame_buffer)
            print(
                f"[STATE] SIGNING → IDLE "
                f"(reason={end_reason}, buffer={buf_len}f)"
            )
            result = self._flush_and_infer()
            self._reset_to_idle()
            return result

        return None, 0.0


# ==========================================
# MAIN LOOP
# ==========================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Nhận diện ngôn ngữ ký hiệu tiếng Việt — Desktop")
    parser.add_argument("--no-face", action="store_true", help="Không dùng face landmarks")
    parser.add_argument("--no-eyebrow", action="store_true", help="Không dùng eyebrow features")
    parser.add_argument("--no-tts", action="store_true", help="Tắt giọng nói TTS")
    parser.add_argument(
        "--model", type=str, default="auto", choices=["auto", "bilstm", "conformer"],
        help="Override model type (mặc định 'auto' = đọc từ checkpoint)"
    )
    args = parser.parse_args()

    # Mặc định theo config (khớp với lúc train model)
    use_face: bool = cfg.USE_FACE if not args.no_face else False
    use_eyebrow: bool = cfg.USE_EYEBROW if not args.no_eyebrow else False

    feature_dim = cfg.compute_feature_dim(use_face, use_eyebrow)
    print(f"[CONFIG] USE_FACE={use_face} | USE_EYEBROW={use_eyebrow} | feature_dim={feature_dim}")

    _need_face = use_face or use_eyebrow
    ensure_models(_need_face)

    # ── Inference engine ───────────────────────────────────────
    engine = InferenceEngine(use_face, use_eyebrow, model_override=args.model)
    tts = TTSPlayer(model_path=cfg.TTS_MODEL_PATH, enabled=not args.no_tts)

    # ── Camera ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể mở Webcam.")
        return

    # ── MediaPipe Landmarkers ──────────────────────────────────
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=cfg.POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.25,
    )
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=cfg.HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.25,
    )
    face_options = None
    if _need_face:
        face_options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=cfg.FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.35,
            min_face_presence_confidence=0.35,
            min_tracking_confidence=0.25,
            output_face_blendshapes=False,
        )

    pose_lm = mp_vision.PoseLandmarker.create_from_options(pose_options)
    hand_lm = mp_vision.HandLandmarker.create_from_options(hand_options)
    face_lm = (
        mp_vision.FaceLandmarker.create_from_options(face_options)
        if face_options else None
    )

    # ── Performance optimizations (giống data_collector) ──────
    _start_time = time.time()
    _last_ts_ms = 0
    _executor = ThreadPoolExecutor(max_workers=2)
    _cached_pose_result = None
    _pose_frame_counter = 0
    _POSE_SKIP = 2

    # ── State hiển thị ─────────────────────────────────────────
    recognized_history: list[str] = []  # Lịch sử các câu nhận diện
    MAX_HISTORY = 5
    current_text = ""
    current_conf = 0.0
    text_display_time = 0.0  # Thời gian hiển thị text mới nhất

    cv2.namedWindow("SL2Text — Recognizer", cv2.WINDOW_NORMAL)

    print("\n--- NHẬN DIỆN NGÔN NGỮ KÝ HIỆU ---")
    print("Thực hiện cử chỉ trước camera — kết quả hiển thị TRONG LÚC ra ký hiệu.")
    print("Nhấn [C] để xóa lịch sử | [Q] để thoát.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Giảm resolution 2× cho MediaPipe (giống data_collector)
            _H, _W = image_rgb.shape[:2]
            mp_small = cv2.resize(image_rgb, (_W // 2, _H // 2), interpolation=cv2.INTER_AREA)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_small)

            # Timestamp
            _now_ms = int((time.time() - _start_time) * 1000)
            if _now_ms <= _last_ts_ms:
                _now_ms = _last_ts_ms + 1
            _last_ts_ms = _now_ms

            # Pose (skip mỗi 2 frame) + Hand song song
            _pose_frame_counter += 1
            if _pose_frame_counter % _POSE_SKIP == 0 or _cached_pose_result is None:
                pose_future: Future = _executor.submit(
                    pose_lm.detect_for_video, mp_image, _now_ms
                )
            else:
                pose_future = None

            hand_result = hand_lm.detect_for_video(mp_image, _now_ms)

            if pose_future is not None:
                _cached_pose_result = pose_future.result()
            pose_result = _cached_pose_result

            face_result = (
                face_lm.detect_for_video(mp_image, _now_ms) if face_lm else None
            )

            # Vẽ skeleton
            draw_landmarks(image, pose_result, hand_result, face_result)

            # ── Inference ──────────────────────────────────────
            features = extract_keypoints(
                pose_result, hand_result, face_result, use_face, use_eyebrow
            )
            has_hands = hands_visible(hand_result)
            wrist_y = extract_hand_activity(hand_result)
            text, confidence = engine.feed_frame(features, has_hands, wrist_y)

            if text:
                current_text = text
                current_conf = confidence
                text_display_time = time.time()
                recognized_history.append(text)
                if len(recognized_history) > MAX_HISTORY:
                    recognized_history.pop(0)
                print(f"[RESULT] {text} (conf={confidence:.2f})")
                tts.speak(text)

            # ── Hiển thị UI ────────────────────────────────────
            buf_len = len(engine.frame_buffer)

            # Status bar trên cùng — HAD state machine
            if engine._state == "SIGNING":
                status = (
                    f"SIGNING | Buffer: {buf_len} | "
                    f"Energy: {engine._current_energy:.4f}"
                )
                status_color = (0, 255, 255)
            else:
                status = "IDLE — Hay lam cu chi"
                status_color = (0, 255, 0)

            cv2.putText(image, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # ── Panel kết quả ở dưới ────────────────────────────
            _PANEL_H = 120
            panel = np.full((_PANEL_H, image.shape[1], 3), (30, 30, 30), dtype=np.uint8)

            # Kết quả hiện tại (nổi bật trong 3 giây)
            if current_text and (time.time() - text_display_time) < 3.0:
                cv2.rectangle(panel, (0, 0), (panel.shape[1], 40), (0, 120, 0), -1)
                _put_vn_text(panel, f"  {current_text}  ({current_conf:.0%})",
                             (4, 6), font_size=22, color=(255, 255, 255))
            else:
                _put_vn_text(panel, "  Đang chờ cử chỉ...",
                             (4, 6), font_size=18, color=(150, 150, 150))

            # Lịch sử
            _put_vn_text(panel, "  Lịch sử:", (4, 45), font_size=14, color=(100, 100, 100))
            history_text = " | ".join(recognized_history[-4:]) if recognized_history else "(trống)"
            _put_vn_text(panel, f"  {history_text}", (4, 68), font_size=14, color=(200, 200, 200))

            # Phím tắt
            _put_vn_text(panel, "  [C] Xóa lịch sử  [Q] Thoát",
                         (4, 95), font_size=12, color=(100, 100, 100))

            image = np.vstack([image, panel])
            cv2.imshow("SL2Text — Recognizer", image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("c"), ord("C")):
                recognized_history.clear()
                current_text = ""
                engine._last_emitted = ""
                engine._reset_to_idle()
                print("[INFO] Đã xóa lịch sử.")

    finally:
        tts.shutdown()
        _executor.shutdown(wait=False)
        pose_lm.close()
        hand_lm.close()
        if face_lm:
            face_lm.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
