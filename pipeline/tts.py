"""Text-to-Speech wrapper sử dụng edge-tts (Microsoft Edge TTS API).

Hỗ trợ tiếng Việt với giọng HoaiMy / NamMinh.
"""
from __future__ import annotations

import base64
import io

import edge_tts


async def synthesize(
    text: str,
    voice: str = "vi-VN-HoaiMyNeural",
) -> str:
    """Chuyển văn bản tiếng Việt → audio base64 (MP3).

    Args:
        text:  Văn bản cần đọc.
        voice: Tên giọng edge-tts (mặc định: vi-VN-HoaiMyNeural).

    Returns:
        Chuỗi base64 của file MP3.
    """
    communicate = edge_tts.Communicate(text, voice)
    buffer = io.BytesIO()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])

    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
