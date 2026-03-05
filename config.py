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
USE_FACE: bool = True  # Bật/tắt face landmarks

NUM_POSE_LANDMARKS: int = 33
NUM_HAND_LANDMARKS: int = 21
NUM_FACE_LANDMARKS: int = 478

POSE_DIMS: int = 4   # x, y, z, visibility
HAND_DIMS: int = 3   # x, y, z
FACE_DIMS: int = 3   # x, y, z

POSE_FEATURE_DIM: int = NUM_POSE_LANDMARKS * POSE_DIMS   # 132
HAND_FEATURE_DIM: int = NUM_HAND_LANDMARKS * HAND_DIMS   # 63
FACE_FEATURE_DIM: int = NUM_FACE_LANDMARKS * FACE_DIMS   # 1434


def compute_feature_dim(use_face: bool = USE_FACE) -> int:
    """Tính tổng số chiều đặc trưng mỗi frame."""
    dim = POSE_FEATURE_DIM + 2 * HAND_FEATURE_DIM  # pose + 2 tay
    if use_face:
        dim += FACE_FEATURE_DIM
    return dim


FEATURE_DIM: int = compute_feature_dim()
# USE_FACE=True  → 132 + 126 + 1434 = 1692
# USE_FACE=False → 132 + 126       = 258

# ==========================================
# MÔ HÌNH BI-LSTM
# ==========================================
HIDDEN_DIM: int = 256
NUM_LSTM_LAYERS: int = 2
DROPOUT: float = 0.3

# ==========================================
# HUẤN LUYỆN (TRAINING)
# ==========================================
LEARNING_RATE: float = 1e-4
BATCH_SIZE: int = 16
NUM_EPOCHS: int = 100
SCHEDULER_PATIENCE: int = 5
SCHEDULER_FACTOR: float = 0.5
MIN_SEQUENCE_LENGTH: int = 10  # Số frame tối thiểu

# ==========================================
# SLIDING WINDOW (INFERENCE THỜI GIAN THỰC)
# ==========================================
WINDOW_SIZE: int = 30   # frames
WINDOW_STRIDE: int = 5  # frames

# ==========================================
# VOCABULARY — Tiếng Việt
# Blank token ở index 0 theo chuẩn CTC.
# ==========================================
_VIETNAMESE_CHARS: str = (
    " "
    "aàáảãạăằắẳẵặâầấẩẫậ"
    "bcd"
    "đ"
    "eèéẻẽẹêềếểễệ"
    "fgh"
    "iìíỉĩị"
    "jklmn"
    "oòóỏõọôồốổỗộơờớởỡợ"
    "pqrst"
    "uùúủũụưừứửữự"
    "vwx"
    "yỳýỷỹỵ"
    "z"
    "0123456789"
    ".,!?-"
)

BLANK_TOKEN: str = "<blank>"
VOCAB: list[str] = [BLANK_TOKEN] + list(_VIETNAMESE_CHARS)
CHAR_TO_IDX: dict[str, int] = {ch: i for i, ch in enumerate(VOCAB)}
IDX_TO_CHAR: dict[int, str] = {i: ch for i, ch in enumerate(VOCAB)}
NUM_CLASSES: int = len(VOCAB) - 1   # không tính blank
BLANK_IDX: int = 0


def text_to_indices(text: str) -> list[int]:
    """Chuyển chuỗi tiếng Việt → danh sách index (bỏ ký tự ngoài vocab)."""
    text = unicodedata.normalize("NFC", text.lower())
    return [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]


def indices_to_text(indices: list[int]) -> str:
    """Chuyển danh sách index → chuỗi (bỏ blank)."""
    return "".join(IDX_TO_CHAR.get(i, "") for i in indices if i != BLANK_IDX)


# ==========================================
# TTS (Text-to-Speech)
# ==========================================
TTS_VOICE: str = "vi-VN-HoaiMyNeural"

# ==========================================
# SERVER
# ==========================================
WS_HOST: str = "0.0.0.0"
WS_PORT: int = 8000
