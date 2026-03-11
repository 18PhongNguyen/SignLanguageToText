"""FastAPI server với WebSocket endpoint cho inference thời gian thực.

Chạy:
    uvicorn server:app --host 0.0.0.0 --port 8000
    # hoặc
    python server.py

Có 2 endpoint:
    /ws/video      — nhận raw JPEG bytes (backend MediaPipe) [DÙNG CHÍNH]
    /ws/recognize  — nhận landmarks JSON (frontend MediaPipe) [legacy]
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import time
import uvicorn

# Fix MIME types trên Windows (registry không có .mjs/.wasm)
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from loguru import logger

import config as cfg
from pipeline import Pipeline
from pipeline.mediapipe_runner import MediaPipeRunner

# =====================================================================
# APP
# =====================================================================
app = FastAPI(title="SL2Text — Vietnamese Sign Language Recognition")

# Lưu stats để debug
_debug_stats: dict = {
    "frames_received": 0,
    "frames_with_hands": 0,
    "frames_no_hands": 0,
    "inferences": 0,
    "last_texts": [],
    "last_frame_time": 0.0,
}


@app.on_event("startup")
async def startup() -> None:
    """Khởi tạo pipeline + MediaPipe runner khi server start."""
    app.state.pipeline = Pipeline(
        model_path=cfg.TRAINED_MODEL_PATH,
        feature_dim=cfg.FEATURE_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_classes=cfg.NUM_CLASSES,
        num_layers=cfg.NUM_LSTM_LAYERS,
        window_size=cfg.WINDOW_SIZE,
        window_stride=cfg.WINDOW_STRIDE,
        tts_voice=cfg.TTS_VOICE,
        use_face=cfg.USE_FACE,
        use_eyebrow=cfg.USE_EYEBROW,
        blank_idx=cfg.BLANK_IDX,
        idx_to_char=cfg.IDX_TO_CHAR,
    )
    # MediaPipe runner (backend) — xử lý raw video frames
    app.state.mp_runner = MediaPipeRunner(use_face=cfg.USE_FACE, use_eyebrow=cfg.USE_EYEBROW)
    logger.info(
        f"Pipeline ready | feature_dim={cfg.FEATURE_DIM} | "
        f"window={cfg.WINDOW_SIZE} | stride={cfg.WINDOW_STRIDE}"
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    if hasattr(app.state, "mp_runner"):
        app.state.mp_runner.close()


# =====================================================================
# WEBSOCKET ENDPOINT
# =====================================================================
@app.websocket("/ws/recognize")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Nhận landmarks JSON mỗi frame, trả về text + audio.

    Client gửi::

        {
            "frame_id": 1042,
            "timestamp": 1234567890.123,
            "landmarks": {
                "pose":       [[x,y,z,v], ...],
                "left_hand":  [[x,y,z], ...] | null,
                "right_hand": [[x,y,z], ...] | null,
                "face":       [[x,y,z], ...] | null
            }
        }

    Server trả về::

        {
            "text": "Xin chào",
            "confidence": 0.92,
            "audio": "<base64>",
            "latency_ms": 45
        }
    """
    await websocket.accept()
    pipeline: Pipeline = app.state.pipeline
    logger.info("WebSocket connected")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            # Debug stats
            _debug_stats["frames_received"] += 1
            _debug_stats["last_frame_time"] = time.time()
            lm = data.get("landmarks", {})
            has_hand = bool(lm.get("left_hand") or lm.get("right_hand"))
            if has_hand:
                _debug_stats["frames_with_hands"] += 1
            else:
                _debug_stats["frames_no_hands"] += 1

            if _debug_stats["frames_received"] % 30 == 1:
                logger.info(
                    f"[WS] frames={_debug_stats['frames_received']} "
                    f"with_hands={_debug_stats['frames_with_hands']} "
                    f"no_hands={_debug_stats['frames_no_hands']} "
                    f"has_face={bool(lm.get('face'))}"
                )

            result = await pipeline.process(data)

            if result.get("text"):
                _debug_stats["inferences"] += 1
                _debug_stats["last_texts"].append(result["text"])
                if len(_debug_stats["last_texts"]) > 5:
                    _debug_stats["last_texts"].pop(0)
                logger.info(f"[WS] RESULT: {result['text']!r}")

            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


# =====================================================================
# WEBSOCKET ENDPOINT — Backend MediaPipe (nhận raw JPEG)
# =====================================================================
@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket) -> None:
    """Nhận raw JPEG bytes mỗi frame, chạy MediaPipe trên server, trả text+audio.

    Client gửi: binary message chứa JPEG bytes của mỗi frame webcam.

    Server trả về JSON::

        {
            "text": "Xin chào",
            "confidence": 0.92,
            "audio": "<base64>",
            "latency_ms": 45,
            "has_hands": true
        }
    """
    await websocket.accept()
    pipeline: Pipeline = app.state.pipeline
    mp_runner: MediaPipeRunner = app.state.mp_runner
    loop = asyncio.get_event_loop()
    logger.info("[video] WebSocket connected")

    try:
        while True:
            jpeg_bytes = await websocket.receive_bytes()
            t0 = time.perf_counter()

            # Chạy MediaPipe ở thread pool (CPU-bound, không block event loop)
            features, has_hands = await loop.run_in_executor(
                None, mp_runner.process_jpeg, jpeg_bytes
            )

            # Đưa feature vào pipeline giống flow cũ
            data = {
                "_features": features,      # bypass landmarks_json_to_array
                "_has_hands": has_hands,
            }
            result = await pipeline.process_features(features, has_hands)

            result["has_hands"] = has_hands
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            if result.get("text"):
                logger.info(f"[video] RESULT: {result['text']!r} | {result['latency_ms']}ms")

            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("[video] WebSocket disconnected")
    except Exception as e:
        logger.error(f"[video] WebSocket error: {e}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


# =====================================================================
# HEALTH CHECK
# =====================================================================
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug")
async def debug_stats() -> dict:
    """Trả về thống kê để debug."""
    pipeline = app.state.pipeline if hasattr(app.state, "pipeline") else None
    return {
        **_debug_stats,
        "buffer_size": len(pipeline.frame_buffer) if pipeline else -1,
        "silence_frames": pipeline._silence_frames if pipeline else -1,
        "last_emitted": pipeline._last_emitted if pipeline else "",
    }


# =====================================================================
# STATIC FILES — SPA frontend
# (mount SAU các route khác để không che WebSocket / API)
# =====================================================================

# Serve @mediapipe/tasks-vision từ node_modules local (tránh CDN chậm)
app.mount(
    "/mediapipe",
    StaticFiles(directory="frontend/node_modules/@mediapipe/tasks-vision"),
    name="mediapipe",
)

# Serve model files (.task) cho frontend JS
app.mount(
    "/models",
    StaticFiles(directory="models"),
    name="models_static",
)

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=cfg.WS_HOST,
        port=cfg.WS_PORT,
        reload=True,
    )
