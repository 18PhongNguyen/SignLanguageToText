"""CTC Decoder + chuẩn hóa văn bản tiếng Việt.

Beam Search CTC decoding ở cấp độ từ (word-level):
- Model output: (B, T, vocab_size) — log-probs per frame
- Decode: CTC beam search → chuỗi word indices tốt nhất
- Convert: word indices → câu tiếng Việt
"""
from __future__ import annotations

import unicodedata
from collections import defaultdict
from typing import Callable

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
        Danh sách các chuỗi word index đã decode (một list mỗi batch item).
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


def _beam_search_single(
    log_probs_tc: torch.Tensor,
    blank_idx: int = 0,
    beam_width: int = 10,
) -> list[int]:
    """CTC Beam Search decoding cho một sequence (T, C).

    Thuật toán: prefix-beam-search chính xác từ bài báo CTC gốc.
    Mỗi beam lưu prefix đã decode (đã collapse blank) cùng hai xác suất:
      - p_blank:     path kết thúc bằng blank
      - p_non_blank: path kết thúc bằng non-blank

    Args:
        log_probs_tc: (T, C) log-probabilities
        blank_idx:    index của blank token
        beam_width:   số beam giữ lại mỗi bước

    Returns:
        Decoded word index list tốt nhất.
    """
    T, C = log_probs_tc.shape
    probs = log_probs_tc.exp().cpu().float()  # (T, C)

    # beams: prefix_tuple → [p_blank, p_non_blank]
    def new_beam_dict() -> defaultdict:
        return defaultdict(lambda: [0.0, 0.0])

    # ── Khởi tạo từ frame 0 ──────────────────────────────────
    beams = new_beam_dict()
    beams[()][0] = float(probs[0, blank_idx])          # prefix rỗng + blank
    for c in range(C):
        if c == blank_idx:
            continue
        p = float(probs[0, c])
        if p > 1e-10:
            beams[(c,)][1] = p

    def _prune(d: defaultdict, k: int) -> defaultdict:
        scored = {pfx: v[0] + v[1] for pfx, v in d.items()}
        top = sorted(scored, key=scored.__getitem__, reverse=True)[:k]
        out = new_beam_dict()
        for pfx in top:
            out[pfx] = d[pfx]
        return out

    beams = _prune(beams, beam_width)

    # ── Mở rộng qua từng frame ────────────────────────────────
    for t in range(1, T):
        new_beams = new_beam_dict()
        p_blank_t = float(probs[t, blank_idx])

        for prefix, (p_b, p_nb) in list(beams.items()):
            p_total = p_b + p_nb
            if p_total < 1e-15:
                continue

            # Mở rộng với blank → giữ nguyên prefix
            new_beams[prefix][0] += p_total * p_blank_t

            # Mở rộng với non-blank c
            for c in range(C):
                if c == blank_idx:
                    continue
                p_c = float(probs[t, c])
                if p_c < 1e-10:
                    continue

                if prefix and prefix[-1] == c:
                    # Cùng ký tự cuối:
                    #   pnb → giữ prefix (AA collapse → A)
                    #   pb  → mở rộng prefix thêm c (_A → một A mới)
                    new_beams[prefix][1] += p_nb * p_c
                    new_beams[prefix + (c,)][1] += p_b * p_c
                else:
                    # Ký tự khác → mở rộng prefix
                    new_beams[prefix + (c,)][1] += p_total * p_c

        beams = _prune(new_beams, beam_width)
        if not beams:
            return []

    # Chọn beam tốt nhất
    best = max(beams, key=lambda pfx: beams[pfx][0] + beams[pfx][1])
    return list(best)


def decode_to_text(
    log_probs: torch.Tensor,
    idx2word: dict[int, str],
    blank_idx: int = 0,
    beam_width: int = 10,
    min_frames: int = 25,
    confidence_threshold: float = 0.5,
) -> list[tuple[str, float]]:
    """Beam Search decode + chuyển thành câu tiếng Việt.

    Args:
        log_probs:            (B, T, C) hoặc (T, C)
        idx2word:             map từ index → từ
        blank_idx:            index blank token
        beam_width:           số beam (10 mặc định)
        min_frames:           số frame tối thiểu để inference hợp lệ;
                              bỏ qua nếu T < min_frames (default 25)
        confidence_threshold: ngưỡng tự tin tối thiểu để chấp nhận kết quả;
                              bỏ qua nếu confidence < threshold (default 0.5)

    Returns:
        List[(text, confidence)] cho mỗi item trong batch.
        Trả về ("", 0.0) nếu không đủ frame hoặc dưới ngưỡng tự tin.
    """
    if log_probs.dim() == 2:
        log_probs = log_probs.unsqueeze(0)

    probs = torch.softmax(log_probs, dim=-1)

    results: list[tuple[str, float]] = []
    for i in range(log_probs.size(0)):
        T = log_probs.size(1)

        # ── Chặn: không đủ frame liên tiếp ─────────────────────
        if min_frames > 0 and T < min_frames:
            results.append(("", 0.0))
            continue

        # ── Beam Search decode ──────────────────────────────────
        indices = _beam_search_single(log_probs[i], blank_idx, beam_width)

        if not indices:
            results.append(("", 0.0))
            continue

        # ── Confidence: mean prob tại các frame phát ra non-blank ────────
        # (tránh inflate do blank frames có prob ~0.95)
        frame_preds = log_probs[i].argmax(dim=-1)       # (T,)
        non_blank_mask = frame_preds != blank_idx
        if non_blank_mask.any():
            confidence = float(
                probs[i][non_blank_mask].max(dim=-1).values.mean().item()
            )
        else:
            confidence = 0.0

        # ── Chặn: dưới ngưỡng tự tin ───────────────────────────
        if confidence_threshold > 0 and confidence < confidence_threshold:
            results.append(("", confidence))
            continue

        words = [idx2word[idx] for idx in indices if idx in idx2word]
        text = " ".join(words)
        text = normalize_vietnamese(text)
        results.append((text, confidence))

    return results


def normalize_vietnamese(text: str) -> str:
    """Chuẩn hóa Unicode NFC + viết hoa đầu câu + strip."""
    text = unicodedata.normalize("NFC", text.strip())
    if text:
        text = text[0].upper() + text[1:]
    return text
