"""Cấu hình trung tâm cho toàn bộ pipeline SL2Text.

Mọi hyperparameter, đường dẫn, vocabulary đều tập trung tại đây.
Các module khác import config rồi truyền giá trị vào hàm/class — không
import config trực tiếp trong các module thư viện (pipeline/*).
"""
from __future__ import annotations

import os
import unicodedata
from pathlib import Path

# ==========================================
# ĐƯỜNG DẪN (PATHS)
# ==========================================
PROJECT_ROOT: str = str(Path(__file__).resolve().parent)
DATA_DIR: str = os.path.join(PROJECT_ROOT, "Dataset")
FEATURES_DIR: str = os.path.join(DATA_DIR, "features")
LABEL_FILE: str = os.path.join(DATA_DIR, "labels.csv")
MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
TRAINED_MODEL_PATH: str = os.path.join(MODELS_DIR, "bilstm_ctc.pt")

# MediaPipe task model files
POSE_MODEL_PATH: str = os.path.join(MODELS_DIR, "pose_landmarker_full.task")
HAND_MODEL_PATH: str = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_MODEL_PATH: str = os.path.join(MODELS_DIR, "face_landmarker.task")

POSE_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
HAND_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)
FACE_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)

# ==========================================
# TRÍCH XUẤT ĐẶC TRƯNG (FEATURE EXTRACTION)
# ==========================================
USE_FACE: bool = False  # Tắt face — 1434-dim face là nhiễu cho nhận diện cử chỉ tay
USE_EYEBROW: bool = True  # Bật lông mày — quan trọng cho biểu cảm và ngữ pháp VSL
USE_VELOCITY: bool = True  # Nối velocity (delta frame) → feature_dim × 2

NUM_POSE_LANDMARKS: int = 33
NUM_HAND_LANDMARKS: int = 21
NUM_FACE_LANDMARKS: int = 478

POSE_DIMS: int = 4   # x, y, z, visibility
HAND_DIMS: int = 3   # x, y, z
FACE_DIMS: int = 3   # x, y, z

POSE_FEATURE_DIM: int = NUM_POSE_LANDMARKS * POSE_DIMS   # 132
HAND_FEATURE_DIM: int = NUM_HAND_LANDMARKS * HAND_DIMS   # 63
HAND_ANGLE_DIM: int = 5                                   # 5 góc PIP mỗi bàn tay
FACE_FEATURE_DIM: int = NUM_FACE_LANDMARKS * FACE_DIMS   # 1434
EYEBROW_FEATURE_DIM: int = 33  # 5×3 + 5×3 + 3 (coords lông mày + raise + furrow)


def compute_feature_dim(
    use_face: bool = USE_FACE,
    use_eyebrow: bool = USE_EYEBROW,
    use_velocity: bool = USE_VELOCITY,
) -> int:
    """Tính tổng số chiều đặc trưng mỗi frame."""
    # pose + (tọa độ + góc ngón tay) × 2 tay
    dim = POSE_FEATURE_DIM + 2 * (HAND_FEATURE_DIM + HAND_ANGLE_DIM)
    if use_face:
        dim += FACE_FEATURE_DIM
    if use_eyebrow:
        dim += EYEBROW_FEATURE_DIM
    if use_velocity:
        dim *= 2  # velocity doubles feature dim
    return dim


FEATURE_DIM: int = compute_feature_dim()
# USE_FACE=False, USE_EYEBROW=True  → 132 + 2×(63+5) + 33 = 301
# USE_FACE=False, USE_EYEBROW=False → 132 + 2×(63+5)      = 268
# USE_FACE=True,  USE_EYEBROW=True  → 132 + 2×(63+5) + 1434 + 33 = 1735

# ==========================================
# MÔ HÌNH BI-LSTM
# ==========================================
HIDDEN_DIM: int = 128
NUM_LSTM_LAYERS: int = 2
DROPOUT: float = 0.4

# ==========================================
# HUẤN LUYỆN (TRAINING)
# ==========================================
LEARNING_RATE: float = 3e-4
BATCH_SIZE: int = 16
NUM_EPOCHS: int = 400
# SCHEDULER_PATIENCE / FACTOR không còn dùng (đã chuyển sang OneCycleLR)
# SCHEDULER_PATIENCE: int = 5
# SCHEDULER_FACTOR: float = 0.5
MIN_SEQUENCE_LENGTH: int = 10  # Số frame tối thiểu
WEIGHT_DECAY: float = 1e-3     # L2 regularization (AdamW)
INPUT_DROPOUT: float = 0.1     # Dropout trên input features trước projection

# ==========================================
# SLIDING WINDOW (INFERENCE THỜI GIAN THỰC)
# ==========================================
WINDOW_SIZE: int = 60   # frames — đủ bao phủ cử chỉ dài nhất (~99 max, mean ~53)
WINDOW_STRIDE: int = 10  # frames

# Số lần inference liên tiếp phải cho cùng kết quả trước khi emit
# (tránh kết quả lộn xộn khi model đang ở giữa cử chỉ)
STABLE_THRESHOLD: int = 3

# ==========================================
# VOCABULARY — Word-level CTC
# Từ điển tự động xây dựng từ labels.csv → vocab.txt
# Mỗi từ đơn là một token, index 0 = <blank> (CTC blank).
#
# Khi thu thêm dữ liệu: chạy build_vocab() hoặc
# data_collector.py tự gọi khi lưu nhãn mới.
# ==========================================
from vocab import load_vocab, vocab_to_dicts, BLANK_TOKEN  # noqa: E402

VOCAB: list[str] = load_vocab()

BLANK_IDX: int = 0
NUM_CLASSES: int = len(VOCAB)   # blank + N words

CHAR_TO_IDX: dict[str, int]
IDX_TO_CHAR: dict[int, str]
CHAR_TO_IDX, IDX_TO_CHAR = vocab_to_dicts(VOCAB)


def reload_vocab() -> None:
    """Tải lại vocab sau khi build_vocab() cập nhật vocab.txt."""
    global VOCAB, NUM_CLASSES, CHAR_TO_IDX, IDX_TO_CHAR
    VOCAB = load_vocab()
    NUM_CLASSES = len(VOCAB)
    CHAR_TO_IDX, IDX_TO_CHAR = vocab_to_dicts(VOCAB)


# ==========================================
# TTS (Text-to-Speech)
# ==========================================
TTS_VOICE: str = "vi-VN-HoaiMyNeural"

# ==========================================
# SERVER
# ==========================================
WS_HOST: str = "0.0.0.0"
WS_PORT: int = 8000
