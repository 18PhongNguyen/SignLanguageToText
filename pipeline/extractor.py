"""Trích xuất và biến đổi đặc trưng từ landmarks.

Module này KHÔNG import config — nhận tham số qua hàm.
Dùng chung cho cả data_collector (offline) và inference (realtime).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def landmarks_to_features(
    pose: Sequence[Sequence[float]] | None,
    left_hand: Sequence[Sequence[float]] | None,
    right_hand: Sequence[Sequence[float]] | None,
    face: Sequence[Sequence[float]] | None = None,
    *,
    use_face: bool = True,
    num_pose: int = 33,
    num_hand: int = 21,
    num_face: int = 478,
    pose_dims: int = 4,
    hand_dims: int = 3,
    face_dims: int = 3,
) -> np.ndarray:
    """Chuyển đổi raw landmark coords → numpy feature vector.

    Gốc tọa độ = điểm mũi (pose landmark 0).  Tất cả tọa độ được
    tịnh tiến theo gốc này để bất biến vị trí.

    Returns:
        1-D float32 array với kích thước = feature_dim.
    """
    # ── Gốc tọa độ (mũi) ──────────────────────────────────────
    origin = np.zeros(3, dtype=np.float64)
    if pose is not None and len(pose) >= 1:
        origin = np.array(pose[0][:3], dtype=np.float64)

    # ── Pose (33 × 4) ─────────────────────────────────────────
    if pose is not None and len(pose) == num_pose:
        pose_arr = np.array(
            [
                [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]]
                + ([p[3]] if len(p) > 3 else [1.0])
                for p in pose
            ],
            dtype=np.float32,
        ).flatten()
    else:
        pose_arr = np.zeros(num_pose * pose_dims, dtype=np.float32)

    # ── Tay trái (21 × 3) ─────────────────────────────────────
    if left_hand is not None and len(left_hand) == num_hand:
        lh_arr = np.array(
            [[p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]] for p in left_hand],
            dtype=np.float32,
        ).flatten()
    else:
        lh_arr = np.zeros(num_hand * hand_dims, dtype=np.float32)

    # ── Tay phải (21 × 3) ─────────────────────────────────────
    if right_hand is not None and len(right_hand) == num_hand:
        rh_arr = np.array(
            [[p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]] for p in right_hand],
            dtype=np.float32,
        ).flatten()
    else:
        rh_arr = np.zeros(num_hand * hand_dims, dtype=np.float32)

    parts = [pose_arr, lh_arr, rh_arr]

    # ── Khuôn mặt (478 × 3) — tuỳ chọn ───────────────────────
    if use_face:
        if face is not None and len(face) == num_face:
            face_arr = np.array(
                [[p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]] for p in face],
                dtype=np.float32,
            ).flatten()
        else:
            face_arr = np.zeros(num_face * face_dims, dtype=np.float32)
        parts.append(face_arr)

    return np.concatenate(parts)


def landmarks_json_to_array(
    data: dict[str, Any],
    *,
    use_face: bool = True,
) -> np.ndarray:
    """Shortcut: nhận dict JSON từ frontend WebSocket → feature array.

    Expected JSON::

        {
            "landmarks": {
                "pose":       [[x,y,z,v], ...] | null,
                "left_hand":  [[x,y,z], ...]   | null,
                "right_hand": [[x,y,z], ...]   | null,
                "face":       [[x,y,z], ...]   | null
            }
        }
    """
    lm = data.get("landmarks", {})
    return landmarks_to_features(
        pose=lm.get("pose"),
        left_hand=lm.get("left_hand"),
        right_hand=lm.get("right_hand"),
        face=lm.get("face"),
        use_face=use_face,
    )
