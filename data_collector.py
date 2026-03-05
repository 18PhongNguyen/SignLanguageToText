"""Thu thập dữ liệu ngôn ngữ ký hiệu từ webcam.

Sử dụng MediaPipe Tasks API (Pose + Hand + Face Landmarker)
để trích xuất landmarks, sau đó lưu thành file .npy + labels.csv.

Feature vector mỗi frame (khi USE_FACE=True):
    pose(33×4) + lh(21×3) + rh(21×3) + face(478×3) = 1692

Chạy:
    python data_collector.py
    python data_collector.py --no-face   # Bỏ face → 258 features
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config as cfg
from pipeline.extractor import landmarks_to_features

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
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
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
# HÀM VẼ LANDMARK
# ==========================================
def draw_landmarks(
    image: np.ndarray,
    pose_result: mp_vision.PoseLandmarkerResult,
    hand_result: mp_vision.HandLandmarkerResult,
    face_result: mp_vision.FaceLandmarkerResult | None = None,
) -> None:
    """Vẽ khung xương lên frame để quan sát."""
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

    # Face — chỉ vẽ điểm (quá nhiều connection)
    if face_result and face_result.face_landmarks:
        for face_lm in face_result.face_landmarks:
            for lm in face_lm:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 1, (0, 200, 255), -1)


# ==========================================
# TRÍCH XUẤT ĐẶC TRƯNG — bridge MediaPipe result → pipeline.extractor
# ==========================================
def extract_keypoints(
    pose_result: mp_vision.PoseLandmarkerResult,
    hand_result: mp_vision.HandLandmarkerResult,
    face_result: mp_vision.FaceLandmarkerResult | None = None,
    use_face: bool = True,
) -> np.ndarray:
    """Chuyển kết quả MediaPipe Tasks → feature vector qua pipeline.extractor."""

    # Pose → list of [x, y, z, visibility]
    pose_raw = None
    if pose_result.pose_landmarks:
        pose_raw = [
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in pose_result.pose_landmarks[0]
        ]

    # Hands → phân loại Left/Right
    lh_raw, rh_raw = None, None
    for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
        label = hand_result.handedness[i][0].category_name
        coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
        if label == "Left":
            lh_raw = coords
        else:
            rh_raw = coords

    # Face
    face_raw = None
    if use_face and face_result and face_result.face_landmarks:
        face_raw = [[lm.x, lm.y, lm.z] for lm in face_result.face_landmarks[0]]

    return landmarks_to_features(
        pose=pose_raw,
        left_hand=lh_raw,
        right_hand=rh_raw,
        face=face_raw,
        use_face=use_face,
    )


# ==========================================
# MAIN LOOP
# ==========================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Thu thập dữ liệu ngôn ngữ ký hiệu")
    parser.add_argument(
        "--no-face", action="store_true",
        help="Không trích xuất face landmarks (giảm feature_dim về 258)",
    )
    args = parser.parse_args()
    use_face: bool = not args.no_face

    feature_dim = cfg.compute_feature_dim(use_face)
    print(f"[CONFIG] USE_FACE={use_face} | feature_dim={feature_dim}")

    # Tải model MediaPipe nếu chưa có
    ensure_models(use_face)

    # Khởi tạo thư mục + CSV
    os.makedirs(cfg.FEATURES_DIR, exist_ok=True)
    if not os.path.exists(cfg.LABEL_FILE):
        pd.DataFrame(columns=["filename", "text"]).to_csv(cfg.LABEL_FILE, index=False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể mở Webcam.")
        return

    recording = False
    frames_buffer: list[np.ndarray] = []

    existing_files = [f for f in os.listdir(cfg.FEATURES_DIR) if f.endswith(".npy")]
    sequence_count = len(existing_files) + 1

    print("\n--- HƯỚNG DẪN SỬ DỤNG ---")
    print("1. Nhấn [SPACE] để BẮT ĐẦU ghi hình động tác.")
    print("2. Thực hiện câu ngôn ngữ ký hiệu.")
    print("3. Nhấn [SPACE] lần nữa để KẾT THÚC.")
    print("4. Chuyển sang cửa sổ Terminal/Console để gõ nhãn văn bản.")
    print("5. Nhấn [Q] để thoát chương trình.\n")

    # ── Tạo Landmarker options ────────────────────────────────
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=cfg.POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=cfg.HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    face_options = None
    if use_face:
        face_options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=cfg.FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
        )

    # ── Khởi tạo Landmarkers ─────────────────────────────────
    pose_lm = mp_vision.PoseLandmarker.create_from_options(pose_options)
    hand_lm = mp_vision.HandLandmarker.create_from_options(hand_options)
    face_lm = (
        mp_vision.FaceLandmarker.create_from_options(face_options)
        if face_options
        else None
    )

    # Đảm bảo timestamp luôn tăng đều (dùng counter thay vì time.time)
    frame_ts_ms = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            frame_ts_ms += 33  # ~30 FPS

            pose_result = pose_lm.detect_for_video(mp_image, frame_ts_ms)
            hand_result = hand_lm.detect_for_video(mp_image, frame_ts_ms)
            face_result = (
                face_lm.detect_for_video(mp_image, frame_ts_ms)
                if face_lm
                else None
            )

            draw_landmarks(image, pose_result, hand_result, face_result)

            # ── Logic thu thập ────────────────────────────────
            if recording:
                cv2.putText(
                    image, "REC ● PRESS SPACE TO STOP", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
                keypoints = extract_keypoints(pose_result, hand_result, face_result, use_face)
                frames_buffer.append(keypoints)
                cv2.putText(
                    image, f"Frames: {len(frames_buffer)} | dim: {feature_dim}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                )
            else:
                cv2.putText(
                    image, "IDLE - PRESS SPACE TO START", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                cv2.putText(
                    image, f"Saved: {sequence_count - 1} | dim: {feature_dim}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                )

            cv2.imshow("SL2Text — Data Collector", image)
            key = cv2.waitKey(10) & 0xFF

            if key == 32:  # Space
                if not recording:
                    recording = True
                    frames_buffer = []
                    print("-> Đang ghi hình...")
                else:
                    recording = False
                    if len(frames_buffer) < cfg.MIN_SEQUENCE_LENGTH:
                        print(
                            f"-> Lỗi: Video quá ngắn "
                            f"(<{cfg.MIN_SEQUENCE_LENGTH} frames). Đã hủy."
                        )
                        continue

                    filename = f"seq_{sequence_count:04d}.npy"
                    print(f"\n-> Đã dừng ghi. Thu được {len(frames_buffer)} frames.")

                    label_text = input(
                        f"Nhập câu văn cho [{filename}] (hoặc gõ 'huy' để bỏ qua): "
                    )

                    if label_text.strip().lower() != "huy":
                        save_path = os.path.join(cfg.FEATURES_DIR, filename)
                        np.save(save_path, np.array(frames_buffer, dtype=np.float32))

                        with open(cfg.LABEL_FILE, "a", encoding="utf-8") as f:
                            f.write(f"{filename},{label_text}\n")

                        print(f"[OK] Đã lưu: {filename} -> '{label_text}'\n")
                        sequence_count += 1
                    else:
                        print("[HUỶ] Đã bỏ qua sequence này.\n")

            elif key == ord("q"):
                break
    finally:
        pose_lm.close()
        hand_lm.close()
        if face_lm:
            face_lm.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()