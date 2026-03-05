"""CTC Decoder + chuẩn hóa văn bản tiếng Việt."""
from __future__ import annotations

import unicodedata

import torch


def greedy_decode(
    log_probs: torch.Tensor,
    blank_idx: int = 0,
) -> list[list[int]]:
    """Greedy CTC decoding — collapse repeated tokens và loại blank.

    Args:
        log_probs: (batch, T, C) hoặc (T, C)
        blank_idx: index của blank token

    Returns:
        Danh sách các chuỗi index đã decode (một phần tử mỗi batch item).
    """
    if log_probs.dim() == 2:
        log_probs = log_probs.unsqueeze(0)

    predictions = log_probs.argmax(dim=-1)  # (batch, T)

    results: list[list[int]] = []
    for pred in predictions:
        decoded: list[int] = []
        prev = blank_idx
        for idx in pred.tolist():
            if idx != blank_idx and idx != prev:
                decoded.append(idx)
            prev = idx
        results.append(decoded)

    return results


def normalize_vietnamese(text: str) -> str:
    """Chuẩn hóa Unicode NFC + viết hoa đầu câu + strip."""
    text = unicodedata.normalize("NFC", text.strip())
    if text:
        text = text[0].upper() + text[1:]
    return text
