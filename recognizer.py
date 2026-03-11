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
import asyncio
import os
import threading
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config as cfg
from pipeline.model import BiLSTMCTC
from pipeline.decoder import decode_to_text, normalize_vietnamese
from pipeline.extractor import landmarks_to_features, augment_sequence_with_velocity

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


# ==========================================
# TTS BACKGROUND PLAYER
# ==========================================
class TTSPlayer:
    """Phát giọng nói TTS trong background thread (non-blocking)."""

    def __init__(self, voice: str = cfg.TTS_VOICE, enabled: bool = True) -> None:
        self.voice = voice
        self.enabled = enabled
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        if enabled:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def speak(self, text: str) -> None:
        if not self.enabled or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._speak_async(text), self._loop)

    async def _speak_async(self, text: str) -> None:
        try:
            import edge_tts
            import tempfile
            communicate = edge_tts.Communicate(text, self.voice)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            await communicate.save(tmp_path)
            # Phát bằng system command (cross-platform)
            if os.name == "nt":
                # Windows: dùng PowerShell Media.SoundPlayer hoặc mpv/ffplay
                os.system(f'start /min "" "cmd /c (for /F %I in () do @echo off) & powershell -c "(New-Object Media.SoundPlayer \'{tmp_path}\').PlaySync()" & del \"{tmp_path}\""')
                # Fallback đơn giản hơn:
                os.startfile(tmp_path)
            else:
                os.system(f"mpv --no-video --really-quiet '{tmp_path}' 2>/dev/null && rm -f '{tmp_path}' &")
        except Exception as e:
            print(f"[TTS] Lỗi: {e}")


# ==========================================
# INFERENCE ENGINE
# ==========================================
class InferenceEngine:
    """Quản lý buffer + model inference cho real-time recognition."""

    def __init__(
        self,
        use_face: bool,
        use_eyebrow: bool,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_face = use_face
        self.use_eyebrow = use_eyebrow

        # Buffer
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=150)
        self._silence_frames: int = 0
        self._silence_trigger: int = 15
        self._min_infer_frames: int = 10
        self._confidence_threshold: float = 0.30
        self._last_emitted: str = ""

        # Phrase list từ dataset để snap kết quả decode
        self.phrase_list: list[str] = self._load_phrase_list()

        # Load model
        self.model: BiLSTMCTC | None = None
        self.idx2word: dict[int, str] = cfg.IDX_TO_CHAR  # fallback; overridden khi load checkpoint
        if os.path.exists(cfg.TRAINED_MODEL_PATH):
            # Đọc state_dict trước để lấy num_classes thực tế từ checkpoint
            # (tránh lỗi size mismatch khi vocab thay đổi sau khi train xong)
            state = torch.load(cfg.TRAINED_MODEL_PATH, map_location=self.device, weights_only=True)
            ckpt_num_classes: int = state["fc.weight"].shape[0]

            self.model = BiLSTMCTC(
                feature_dim=cfg.FEATURE_DIM,
                hidden_dim=cfg.HIDDEN_DIM,
                num_classes=ckpt_num_classes,
                num_layers=cfg.NUM_LSTM_LAYERS,
                dropout=0.0,
            ).to(self.device)
            self.model.load_state_dict(state)
            self.model.eval()

            # Build idx2word khớp đúng với số class trong checkpoint
            from vocab import load_vocab, vocab_to_dicts
            _vocab = load_vocab()
            _, self.idx2word = vocab_to_dicts(_vocab[:ckpt_num_classes])

            print(
                f"[MODEL] Loaded từ {cfg.TRAINED_MODEL_PATH} | "
                f"device={self.device} | num_classes={ckpt_num_classes} "
                f"(vocab hiện tại: {cfg.NUM_CLASSES})"
            )
            if ckpt_num_classes != cfg.NUM_CLASSES:
                print(
                    f"[MODEL] ⚠ Checkpoint ({ckpt_num_classes} classes) khác vocab hiện tại "
                    f"({cfg.NUM_CLASSES} classes). Cần train lại để nhận diện đầy đủ từ vựng mới."
                )
        else:
            print(f"[MODEL] Chưa có model tại {cfg.TRAINED_MODEL_PATH}. Hãy train trước!")

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

    @torch.no_grad()
    def _run_infer(self) -> tuple[str | None, float]:
        if self.model is None or len(self.frame_buffer) < self._min_infer_frames:
            self.frame_buffer.clear()
            return None, 0.0

        window = np.array(list(self.frame_buffer))
        self.frame_buffer.clear()
        self._silence_frames = 0

        if cfg.USE_VELOCITY:
            window = augment_sequence_with_velocity(window)

        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(x)

        results = decode_to_text(logits, self.idx2word, blank_idx=cfg.BLANK_IDX, phrase_list=self.phrase_list)
        text, confidence = results[0]

        # Debug: in ra chuỗi token CTC để chẩn đoán xem "khỏe" có bị bỏ qua không
        _probs_np = torch.softmax(logits[0], dim=-1).cpu().numpy()  # (T, C)
        _preds = _probs_np.argmax(axis=-1)  # (T,)
        _tokens = [self.idx2word.get(int(p), f"?{p}") for p in _preds]
        _summary, _prev = [], None
        for t in _tokens:
            if t != _prev:
                _summary.append(t)
                _prev = t
        print(f"[CTC raw] {' → '.join(_summary)} | decoded={text!r} conf={confidence:.2f}")

        if not text or text.strip().lower() == "blank":
            return None, confidence

        if confidence < self._confidence_threshold:
            return None, confidence

        return text, confidence

    def feed_frame(self, features: np.ndarray, has_hands: bool) -> tuple[str | None, float]:
        """Thêm 1 frame và trả về kết quả nếu có.

        Returns: (text, confidence) hoặc (None, 0.0)
        """
        raw_text: str | None = None
        confidence: float = 0.0

        if has_hands:
            if self._silence_frames > 0:
                self._last_emitted = ""
            self.frame_buffer.append(features)
            self._silence_frames = 0

            # Buffer overflow → infer bắt buộc
            if len(self.frame_buffer) == 150:
                raw_text, confidence = self._run_infer()

        else:
            self._silence_frames += 1

            # Dừng tay → dịch
            if (
                self._silence_frames >= self._silence_trigger
                and self._silence_frames < self._silence_trigger + 5
                and len(self.frame_buffer) >= self._min_infer_frames
            ):
                raw_text, confidence = self._run_infer()

            # Im lặng quá lâu → clear
            elif self._silence_frames > self._silence_trigger + 5:
                self.frame_buffer.clear()
                self._last_emitted = ""

        # Filter
        if raw_text:
            clean = raw_text.strip().lower()
            if "blank" in clean or clean == "x" or clean == "neutral":
                raw_text = None

        if raw_text and raw_text != self._last_emitted:
            self._last_emitted = raw_text
            return raw_text, confidence

        return None, 0.0


# ==========================================
# MAIN LOOP
# ==========================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Nhận diện ngôn ngữ ký hiệu tiếng Việt — Desktop")
    parser.add_argument("--no-face", action="store_true", help="Không dùng face landmarks")
    parser.add_argument("--no-eyebrow", action="store_true", help="Không dùng eyebrow features")
    parser.add_argument("--no-tts", action="store_true", help="Tắt giọng nói TTS")
    args = parser.parse_args()

    # Mặc định theo config (khớp với lúc train model)
    use_face: bool = cfg.USE_FACE if not args.no_face else False
    use_eyebrow: bool = cfg.USE_EYEBROW if not args.no_eyebrow else False

    feature_dim = cfg.compute_feature_dim(use_face, use_eyebrow)
    print(f"[CONFIG] USE_FACE={use_face} | USE_EYEBROW={use_eyebrow} | feature_dim={feature_dim}")

    _need_face = use_face or use_eyebrow
    ensure_models(_need_face)

    # ── Inference engine ───────────────────────────────────────
    engine = InferenceEngine(use_face, use_eyebrow)
    tts = TTSPlayer(enabled=not args.no_tts)

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
    print("Thực hiện cử chỉ trước camera, dừng tay để nhận diện.")
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
            text, confidence = engine.feed_frame(features, has_hands)

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
            silence = engine._silence_frames

            # Status bar trên cùng
            if has_hands:
                status = f"Dang ghi... | Buffer: {buf_len} frames"
                status_color = (0, 255, 255)
            elif buf_len > 0:
                status = f"Dang cho... | Buffer: {buf_len} | Silence: {silence}"
                status_color = (0, 200, 200)
            else:
                status = "San sang — Hay lam cu chi"
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
                print("[INFO] Đã xóa lịch sử.")

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
