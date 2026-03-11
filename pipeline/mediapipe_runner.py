"""MediaPipe backend runner — chạy Pose + Hand landmarker trên server.

Nhận raw JPEG/PNG bytes từ WebSocket, trả về feature vector numpy.
- Dùng mediapipe.tasks (Tasks API) nhất quán với data_collector.py
- Flip ảnh ngang (cv2.flip) giống data_collector để tọa độ khớp training
"""
from __future__ import annotations

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from loguru import logger

import config as cfg
from .extractor import landmarks_to_features


# ── Download model nếu chưa có ────────────────────────────────────────────────
def _ensure_model(url: str, path: str) -> None:
    if not os.path.exists(path):
        logger.info(f"Tải model MediaPipe: {os.path.basename(path)} ...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
        logger.info(f"Đã tải: {os.path.basename(path)}")


class MediaPipeRunner:
    """Singleton-friendly wrapper quanh MediaPipe Tasks Landmarkers.

    Tất cả landmarker chạy ở RunningMode.IMAGE (stateless mỗi frame).
    Dùng IMAGE thay vì VIDEO vì WebSocket frame có thể bị trễ không đều.
    """

    def __init__(self, use_face: bool = False, use_eyebrow: bool = False) -> None:
        self.use_face = use_face
        self.use_eyebrow = use_eyebrow
        self._need_face = use_face or use_eyebrow
        self._frame_counter: int = 0  # dùng làm timestamp giả tăng dần

        # Tải models
        _ensure_model(cfg.POSE_MODEL_URL, cfg.POSE_MODEL_PATH)
        _ensure_model(cfg.HAND_MODEL_URL, cfg.HAND_MODEL_PATH)
        if self._need_face:
            _ensure_model(cfg.FACE_MODEL_URL, cfg.FACE_MODEL_PATH)

        # ── Pose ──────────────────────────────────────────────────────────────
        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=cfg.POSE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.5,
            num_poses=1,
        )
        self._pose = mp_vision.PoseLandmarker.create_from_options(pose_opts)

        # ── Hands ─────────────────────────────────────────────────────────────
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=cfg.HAND_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.5,
        )
        self._hands = mp_vision.HandLandmarker.create_from_options(hand_opts)

        # ── Face (dùng cho full face hoặc eyebrow) ──────────────────────────
        self._face = None
        if self._need_face:
            face_opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=cfg.FACE_MODEL_PATH),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.55,
                min_face_presence_confidence=0.55,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
            )
            self._face = mp_vision.FaceLandmarker.create_from_options(face_opts)

        logger.info(f"MediaPipeRunner ready | use_face={use_face}")

    def process_jpeg(self, jpeg_bytes: bytes) -> tuple[np.ndarray, bool]:
        """Nhận JPEG bytes → feature vector + flag có tay không.

        Ảnh được flip ngang (giống data_collector) trước khi xử lý
        để tọa độ nhất quán với training data.

        Returns:
            (features: np.ndarray shape=(feature_dim,), has_hands: bool)
        """
        # Decode JPEG → BGR → flip → RGB
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("Không decode được JPEG frame")
            return np.zeros(cfg.FEATURE_DIM, dtype=np.float32), False

        # Flip ngang — khớp với data_collector.py
        bgr = cv2.flip(bgr, 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Chạy landmarkers (IMAGE mode — stateless)
        pose_result = self._pose.detect(mp_image)
        hand_result = self._hands.detect(mp_image)
        face_result = self._face.detect(mp_image) if self._face else None

        # ── Parse pose ────────────────────────────────────────────────────────
        pose_raw = None
        if pose_result.pose_landmarks:
            pose_raw = [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in pose_result.pose_landmarks[0]
            ]

        # ── Parse hands ───────────────────────────────────────────────────────
        lh_raw, rh_raw = None, None
        for i, hand_lms in enumerate(hand_result.hand_landmarks):
            label = hand_result.handedness[i][0].category_name
            coords = [[lm.x, lm.y, lm.z] for lm in hand_lms]
            if label == "Left":
                lh_raw = coords
            else:
                rh_raw = coords

        has_hands = (lh_raw is not None) or (rh_raw is not None)

        # ── Parse face (cần cho cả use_face và use_eyebrow) ─────────────────
        face_raw = None
        if self._need_face and face_result and face_result.face_landmarks:
            face_raw = [[lm.x, lm.y, lm.z] for lm in face_result.face_landmarks[0]]

        features = landmarks_to_features(
            pose=pose_raw,
            left_hand=lh_raw,
            right_hand=rh_raw,
            face=face_raw,
            use_face=self.use_face,
            use_eyebrow=self.use_eyebrow,
        )
        return features, has_hands

    def close(self) -> None:
        self._pose.close()
        self._hands.close()
        if self._face:
            self._face.close()
        logger.info("MediaPipeRunner closed")
