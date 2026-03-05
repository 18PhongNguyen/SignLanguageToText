"""FastAPI server với WebSocket endpoint cho inference thời gian thực.

Chạy:
    uvicorn server:app --host 0.0.0.0 --port 8000
    # hoặc
    python server.py
"""
from __future__ import annotations

import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from loguru import logger

import config as cfg
from pipeline import Pipeline

# =====================================================================
# APP
# =====================================================================
app = FastAPI(title="SL2Text — Vietnamese Sign Language Recognition")


@app.on_event("startup")
async def startup() -> None:
    """Khởi tạo pipeline khi server start."""
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
        blank_idx=cfg.BLANK_IDX,
        idx_to_char=cfg.IDX_TO_CHAR,
    )
    logger.info(
        f"Pipeline ready | feature_dim={cfg.FEATURE_DIM} | "
        f"window={cfg.WINDOW_SIZE} | stride={cfg.WINDOW_STRIDE}"
    )


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

            result = await pipeline.process(data)
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
# HEALTH CHECK
# =====================================================================
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# =====================================================================
# STATIC FILES — SPA frontend
# (mount SAU các route khác để không che WebSocket / API)
# =====================================================================
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
