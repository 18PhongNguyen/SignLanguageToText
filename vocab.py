"""Quản lý từ điển word-level cho CTC decoding.

Từ điển tự động xây dựng từ labels.csv bằng cách tách nhãn thành từ đơn.
Ví dụ: "bạn khỏe không" → ["bạn", "khỏe", "không"]

File vocab.txt được lưu cùng thư mục Dataset/, mỗi dòng một từ.
Dòng đầu luôn là <blank> (CTC blank token, index 0).

Usage:
    from vocab import build_vocab, load_vocab
    build_vocab()           # Quét labels.csv → ghi vocab.txt
    vocab = load_vocab()    # Đọc vocab.txt → dict
"""
from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import pandas as pd
from loguru import logger

# Đường dẫn mặc định (không import config để tránh circular import)
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "Dataset")
_DEFAULT_LABEL_FILE = os.path.join(_DATA_DIR, "labels.csv")
_DEFAULT_VOCAB_PATH = os.path.join(_DATA_DIR, "vocab.txt")

BLANK_TOKEN: str = "<blank>"


def build_vocab(label_file: str = _DEFAULT_LABEL_FILE, out_path: str = _DEFAULT_VOCAB_PATH) -> list[str]:
    """Quét labels.csv, tách từ, ghi vocab.txt.

    Returns:
        Danh sách token (index 0 = <blank>).
    """
    if not os.path.exists(label_file):
        logger.warning(f"Chưa có {label_file}, tạo vocab rỗng.")
        tokens = [BLANK_TOKEN]
        _write(tokens, out_path)
        return tokens

    df = pd.read_csv(label_file)
    word_set: set[str] = set()

    for text in df["text"]:
        text = unicodedata.normalize("NFC", str(text).strip().lower())
        for word in text.split():
            if word and word != "blank":
                word_set.add(word)

    # Sắp xếp ổn định để index không đổi giữa các lần build
    sorted_words = sorted(word_set)
    tokens = [BLANK_TOKEN] + sorted_words

    _write(tokens, out_path)
    logger.info(f"Vocab built: {len(tokens)} tokens (1 blank + {len(sorted_words)} words) → {out_path}")
    return tokens


def load_vocab(path: str = _DEFAULT_VOCAB_PATH) -> list[str]:
    """Đọc vocab.txt → danh sách token.

    Nếu file chưa tồn tại, tự build trước.
    """
    if not os.path.exists(path):
        logger.info(f"vocab.txt chưa tồn tại, tự build từ labels.csv...")
        return build_vocab(out_path=path)

    with open(path, encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]

    if not tokens or tokens[0] != BLANK_TOKEN:
        logger.warning("vocab.txt thiếu <blank> ở đầu, tự build lại...")
        return build_vocab(out_path=path)

    logger.info(f"Loaded vocab: {len(tokens)} tokens từ {path}")
    return tokens


def vocab_to_dicts(tokens: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Chuyển danh sách token → (word2idx, idx2word)."""
    word2idx = {w: i for i, w in enumerate(tokens)}
    idx2word = {i: w for i, w in enumerate(tokens)}
    return word2idx, idx2word


def text_to_word_indices(text: str, word2idx: dict[str, int]) -> list[int]:
    """Chuyển nhãn câu → chuỗi word indices (bỏ qua từ không có trong vocab).

    Ví dụ: "bạn khỏe không" → [2, 5, 4]  (tùy thứ tự trong vocab)
    """
    text = unicodedata.normalize("NFC", text.strip().lower())
    indices: list[int] = []
    for word in text.split():
        idx = word2idx.get(word)
        if idx is not None:
            indices.append(idx)
    return indices


def indices_to_text(indices: list[int], idx2word: dict[int, str]) -> str:
    """Chuyển chuỗi word indices → câu tiếng Việt.

    Bỏ blank token, nối từ bằng dấu cách.
    """
    words = [idx2word[i] for i in indices if i != 0 and i in idx2word]
    return " ".join(words)


def _write(tokens: list[str], path: str) -> None:
    """Ghi danh sách token ra file, mỗi dòng một token."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for token in tokens:
            f.write(token + "\n")
