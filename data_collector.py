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
from concurrent.futures import ThreadPoolExecutor, Future

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config as cfg
from pipeline.extractor import landmarks_to_features
from vocab import build_vocab

try:
    from PIL import Image as _PilImage, ImageDraw as _PilDraw, ImageFont as _PilFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

import tkinter as tk
from tkinter import simpledialog

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
# TIỆN ÍCH VẼ TEXT UNICODE + NHẬP NHÃN
# ==========================================
def _put_vn_text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_size: int = 20,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Vẽ text Unicode (tiếng Việt) lên ảnh BGR, dùng PIL nếu có."""
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


def _ask_label(current: str = "") -> str:
    """Mở hộp thoại Tkinter để nhập nhãn với hỗ trợ IME tiếng Việt đầy đủ."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    result = simpledialog.askstring(
        "Nhãn cử chỉ",
        "Nhập câu tiếng Việt cho cử chỉ này\n(bấm Cancel để giữ nguyên nhãn cũ):",
        initialvalue=current,
        parent=root,
    )
    root.destroy()
    return result.strip() if result is not None else current


# ==========================================
# TRÍCH XUẤT ĐẶC TRƯNG — bridge MediaPipe result → pipeline.extractor
# ==========================================
def extract_keypoints(
    pose_result: mp_vision.PoseLandmarkerResult,
    hand_result: mp_vision.HandLandmarkerResult,
    face_result: mp_vision.FaceLandmarkerResult | None = None,
    use_face: bool = True,
    use_eyebrow: bool = False,
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

    # Face (dùng cho full face hoặc chỉ eyebrow)
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


# ==========================================
# MAIN LOOP
# ==========================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Thu thập dữ liệu ngôn ngữ ký hiệu")
    parser.add_argument(
        "--no-face", action="store_true",
        help="Không trích xuất face landmarks đầy đủ (giảm feature_dim)",
    )
    parser.add_argument(
        "--no-eyebrow", action="store_true",
        help="Không trích xuất đặc trưng lông mày (bỏ 33 features)",
    )
    args = parser.parse_args()
    use_face: bool = not args.no_face
    use_eyebrow: bool = not args.no_eyebrow

    feature_dim = cfg.compute_feature_dim(use_face, use_eyebrow)
    print(f"[CONFIG] USE_FACE={use_face} | USE_EYEBROW={use_eyebrow} | feature_dim={feature_dim}")

    # Cần FaceLandmarker nếu dùng face đầy đủ hoặc lông mày
    _need_face = use_face or use_eyebrow

    # Tải model MediaPipe nếu chưa có
    ensure_models(_need_face)

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

    # Lấy số thứ tự seq cuối cùng từ labels.csv thay vì đếm file .npy
    # → Đúng ngay cả khi người dùng xóa một số seq bị lỗi
    sequence_count = 1
    if os.path.exists(cfg.LABEL_FILE):
        df_existing = pd.read_csv(cfg.LABEL_FILE)
        if not df_existing.empty:
            import re as _re
            nums = df_existing["filename"].str.extract(r"seq_(\d+)\.npy").dropna()[0].astype(int)
            if not nums.empty:
                sequence_count = int(nums.max()) + 1

    print("\n--- HƯỚNG DẪN SỬ DỤNG ---")
    print("1. Nhấn [E] trên cửa sổ video để nhập/sửa nhãn (hỗ trợ tiếng Việt đầy đủ).")
    print("2. Nhấn [SPACE] để BẮT ĐẦU ghi hình — nhãn hiện tại sẽ được dùng.")
    print("3. Thực hiện câu ngôn ngữ ký hiệu.")
    print("4. Nhấn [SPACE] lần nữa để DỪNG — tự động lưu với nhãn hiện tại.")
    print("5. Đổi nhãn bất cứ lúc nào bằng [E] để ghi cử chỉ tiếp theo.")
    print("6. Nhấn [Q] để thoát chương trình.\n")

    # ── Tạo Landmarker options ────────────────────────────────
    # num_cpu_threads=2 cho mỗi landmarker → tận dụng đa nhân tốt hơn
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=cfg.POSE_MODEL_PATH,
        ),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.25,
    )
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=cfg.HAND_MODEL_PATH,
        ),
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

    # ── Khởi tạo Landmarkers ─────────────────────────────────
    pose_lm = mp_vision.PoseLandmarker.create_from_options(pose_options)
    hand_lm = mp_vision.HandLandmarker.create_from_options(hand_options)
    face_lm = (
        mp_vision.FaceLandmarker.create_from_options(face_options)
        if face_options
        else None
    )

    # Dùng timestamp thực tế để MediaPipe tracking khớp với tốc độ frame thật
    _start_time = time.time()
    _last_ts_ms = 0

    # Parallel inference: pose + hand chạy đồng thời trên 2 thread
    # Pose thay đổi chậm → chỉ update mỗi 2 frame để giải phóng CPU cho hand
    _executor = ThreadPoolExecutor(max_workers=2)
    _cached_pose_result = None
    _pose_frame_counter = 0
    _POSE_SKIP = 2  # Cập nhật pose mỗi N frame

    # ── Cửa sổ có thể resize tự do (kéo góc hoặc full screen) ──
    cv2.namedWindow("SL2Text — Data Collector", cv2.WINDOW_NORMAL)

    # ── Nhãn hiện tại — dùng chung cho mọi lần ghi ────────────
    current_label: str = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Giảm resolution trước khi đưa vào MediaPipe:
            # 4× ít pixel hơn → inference nhanh hơn ~2–3×
            # Tọa độ landmark vẫn chuẩn vì MediaPipe normalize về [0,1]
            _H, _W = image_rgb.shape[:2]
            mp_small = cv2.resize(image_rgb, (_W // 2, _H // 2), interpolation=cv2.INTER_AREA)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_small)

            # Timestamp thực tế (ms), đảm bảo luôn tăng ít nhất 1ms
            _now_ms = int((time.time() - _start_time) * 1000)
            if _now_ms <= _last_ts_ms:
                _now_ms = _last_ts_ms + 1
            _last_ts_ms = _now_ms

            # Chạy pose và hand song song trên 2 thread riêng biệt
            # (mỗi landmarker instance độc lập → thread-safe)
            _pose_frame_counter += 1
            if _pose_frame_counter % _POSE_SKIP == 0 or _cached_pose_result is None:
                pose_future: Future = _executor.submit(
                    pose_lm.detect_for_video, mp_image, _now_ms
                )
            else:
                pose_future = None  # Dùng kết quả pose từ frame trước

            hand_result = hand_lm.detect_for_video(mp_image, _now_ms)

            # Đợi pose nếu có submit
            if pose_future is not None:
                _cached_pose_result = pose_future.result()
            pose_result = _cached_pose_result

            face_result = (
                face_lm.detect_for_video(mp_image, _now_ms)
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
                keypoints = extract_keypoints(
                    pose_result, hand_result, face_result, use_face, use_eyebrow
                )
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

            # ── Textbox nhãn ở dưới video ──────────────────────
            _BAR_H = 65
            bar = np.full((_BAR_H, image.shape[1], 3), (25, 25, 25), dtype=np.uint8)
            cv2.rectangle(bar, (2, 2), (bar.shape[1] - 3, _BAR_H - 3), (0, 200, 255), 2)
            label_display = (
                current_label if current_label else "<chưa có nhãn — nhấn E để nhập>"
            )
            _put_vn_text(bar, f"  Nhãn: {label_display}", (4, 6), font_size=18)
            _put_vn_text(
                bar,
                "  [E] Sửa nhãn  [SPACE] Bắt đầu/Dừng ghi  [Q] Thoát",
                (4, 37), font_size=13, color=(150, 150, 150),
            )
            image = np.vstack([image, bar])

            cv2.imshow("SL2Text — Data Collector", image)
            key = cv2.waitKey(10) & 0xFF

            # [E] → mở hộp thoại nhập nhãn (hỗ trợ IME tiếng Việt đầy đủ)
            if key in (ord("e"), ord("E")):
                current_label = _ask_label(current_label)
                print(f"[LABEL] Nhãn hiện tại: '{current_label}'")

            elif key == 32:  # Space — bắt đầu / dừng ghi
                if not recording:
                    if not current_label.strip():
                        print("-> Cảnh báo: Nhãn đang trống! Nhấn [E] để nhập nhãn trước.")
                    else:
                        recording = True
                        frames_buffer = []
                        print(f"-> Đang ghi hình cho nhãn: '{current_label}' ...")
                else:
                    recording = False
                    if len(frames_buffer) < cfg.MIN_SEQUENCE_LENGTH:
                        print(
                            f"-> Lỗi: Video quá ngắn "
                            f"(<{cfg.MIN_SEQUENCE_LENGTH} frames). Đã hủy."
                        )
                        continue

                    filename = f"seq_{sequence_count:04d}.npy"
                    save_path = os.path.join(cfg.FEATURES_DIR, filename)
                    np.save(save_path, np.array(frames_buffer, dtype=np.float32))
                    with open(cfg.LABEL_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{filename},{current_label}\n")

                    # Tự động cập nhật vocab.txt khi có nhãn mới
                    build_vocab()

                    print(
                        f"[OK] Tự động lưu: {filename} → '{current_label}' "
                        f"({len(frames_buffer)} frames)"
                    )
                    sequence_count += 1

            elif key == ord("q"):
                break
    finally:
        _executor.shutdown(wait=False)
        pose_lm.close()
        hand_lm.close()
        if face_lm:
            face_lm.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()