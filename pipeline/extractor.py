"""Trích xuất và biến đổi đặc trưng từ landmarks.

Module này KHÔNG import config — nhận tham số qua hàm.
Dùng chung cho cả data_collector (offline) và inference (realtime).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# ── Chỉ số lông mày — MediaPipe FaceMesh 478 điểm ─────────
# Góc ngoài (cạnh thái dương) → góc trong (cạnh sống mũi)
_EYEBROW_LEFT_IDX:  list[int] = [46, 53, 52, 65, 55]    # lông mày trái ảnh
_EYEBROW_RIGHT_IDX: list[int] = [276, 283, 282, 295, 285] # lông mày phải ảnh
_EYEBROW_L_INNER: int = 55    # đầu trong lông mày trái (sát sống mũi)
_EYEBROW_R_INNER: int = 285   # đầu trong lông mày phải
_EYE_LEFT_UPPER:  int = 159   # điểm mí mắt trên trái
_EYE_RIGHT_UPPER: int = 386   # điểm mí mắt trên phải
_FACE_NOSE_TIP:   int = 4     # đầu mũi — gốc tọa độ nội tại mặt
_EYEBROW_FEATURE_DIM: int = (
    len(_EYEBROW_LEFT_IDX) * 3 + len(_EYEBROW_RIGHT_IDX) * 3 + 3
)  # 5×3 + 5×3 + 3 = 33

# ── Chỉ số khớp để tính góc PIP của từng ngón tay ──────────
# (điểm_gốc, điểm_giữa, điểm_đầu) — góc tính tại điểm giữa
_FINGER_PIP_JOINTS: list[tuple[int, int, int]] = [
    (2,  3,  4),   # Ngón cái   — CMC→MCP→IP
    (5,  6,  7),   # Ngón trỏ   — MCP→PIP→DIP
    (9,  10, 11),  # Ngón giữa  — MCP→PIP→DIP
    (13, 14, 15),  # Ngón nhẫn  — MCP→PIP→DIP
    (17, 18, 19),  # Ngón út    — MCP→PIP→DIP
]


def _angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Góc (radian) tại điểm b, tạo bởi hai vector ba và bc.

    π (180°) = ngón thẳng duỗi, ~0–π/3 = ngón cong gập.
    """
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return float(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)))


def _finger_curl_angles(hand_xyz: np.ndarray) -> np.ndarray:
    """5 góc khớp PIP (radian) — mỗi ngón một giá trị.

    Args:
        hand_xyz: shape (21, 3) — tọa độ 21 landmark của một bàn tay.

    Returns:
        float32 array shape (5,), giá trị trong [0, π].
        Giá trị nhỏ ≈ ngón cong (nắm tay), giá trị lớn ≈ ngón duỗi (mở tay).
    """
    return np.array(
        [_angle_at_joint(hand_xyz[a], hand_xyz[b], hand_xyz[c])
         for a, b, c in _FINGER_PIP_JOINTS],
        dtype=np.float32,
    )


def eyebrow_features(face_xyz: np.ndarray) -> np.ndarray:
    """Trích xuất 33 đặc trưng lông mày từ 478 face landmarks.

    Tọa độ được biểu diễn tương đối so với đầu mũi (face landmark 4)
    để bất biến với vị trí đầu trong frame — chỉ phản ánh biểu cảm thuần túy.

    Args:
        face_xyz: shape (478, 3) — tọa độ normalize [0,1] từ FaceLandmarker.

    Returns float32 (33,):
        [0:15]  5 điểm lông mày trái × 3 coords (relative to nose tip)
        [15:30] 5 điểm lông mày phải × 3 coords
        [30]    left_raise   — lông mày trái cao hơn mí mắt bao nhiêu (+ = nhướng)
        [31]    right_raise  — lông mày phải
        [32]    inner_gap    — khoảng cách 2 đầu trong (nhỏ = nhíu mày)
    """
    anchor = face_xyz[_FACE_NOSE_TIP]  # gốc tọa độ nội tại mặt

    lb = face_xyz[_EYEBROW_LEFT_IDX]  - anchor  # (5, 3) tọa độ lông mày trái
    rb = face_xyz[_EYEBROW_RIGHT_IDX] - anchor  # (5, 3) tọa độ lông mày phải

    # Raise metric: trong image coords y tăng xuống dưới,
    # lông mày ở trên mắt nên eyebrow_y < eye_y.
    # raise = (eye_y - anchor_y) - mean(eyebrow_y - anchor_y)
    #       → lớn dương = lông mày nhướng cao; tiệm cận 0 = lông mày hạ thấp.
    eye_l_rel = face_xyz[_EYE_LEFT_UPPER,  1] - anchor[1]
    eye_r_rel = face_xyz[_EYE_RIGHT_UPPER, 1] - anchor[1]
    left_raise  = float(eye_l_rel - float(np.mean(lb[:, 1])))
    right_raise = float(eye_r_rel - float(np.mean(rb[:, 1])))

    # Furrow metric: khoảng cách 2D giữa 2 đầu trong lông mày
    inner_l = face_xyz[_EYEBROW_L_INNER, :2] - anchor[:2]
    inner_r = face_xyz[_EYEBROW_R_INNER, :2] - anchor[:2]
    inner_gap = float(np.linalg.norm(inner_l - inner_r))

    return np.concatenate([
        lb.flatten().astype(np.float32),
        rb.flatten().astype(np.float32),
        np.array([left_raise, right_raise, inner_gap], dtype=np.float32),
    ])


def augment_sequence_with_velocity(sequence: np.ndarray) -> np.ndarray:
    """Nối vector vận tốc liên frame vào mỗi frame để mã hóa quỹ đạo chuyển động.

    velocity[t] = sequence[t] - sequence[t-1],  với velocity[0] = zeros.

    Tác dụng:
        - Model nhìn thấy cả VỊ TRÍ và HƯỚNG/TỐC ĐỘ CHUYỂN ĐỘNG tại mỗi timestep
        - Phân biệt được các ký hiệu có pose đầu/cuối giống nhau nhưng quỹ đạo khác
        - Đặc biệt quan trọng khi một số ký hiệu chỉ khác nhau ở chiều chuyển động

    Usage (trong train.py sau khi load .npy):
        from pipeline.extractor import augment_sequence_with_velocity
        seq = np.load("seq_0001.npy")          # (T, D)
        seq_v = augment_sequence_with_velocity(seq)  # (T, 2*D)

    Usage (realtime inference — rolling buffer):
        prev_frame = np.zeros(D)
        for frame in stream:
            delta = frame - prev_frame
            augmented = np.concatenate([frame, delta])
            prev_frame = frame

    Args:
        sequence: shape (T, D)

    Returns:
        shape (T, 2*D) — mỗi frame gồm [vị_trí | vận_tốc]
    """
    velocity = np.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    return np.concatenate([sequence, velocity], axis=1)


def landmarks_to_features(
    pose: Sequence[Sequence[float]] | None,
    left_hand: Sequence[Sequence[float]] | None,
    right_hand: Sequence[Sequence[float]] | None,
    face: Sequence[Sequence[float]] | None = None,
    *,
    use_face: bool = True,
    use_eyebrow: bool = False,
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

    use_face:    True  → nối toàn bộ 478×3 face landmarks
    use_eyebrow: True  → nối 33 đặc trưng lông mày (raise/furrow/coords)
                         Yêu cầu face != None nhưng KHÔNG cần use_face=True.
                         Tọa độ lông mày được chuẩn hóa theo đầu mũi mặt (landmark 4).

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

    # ── Tay trái (21 × 3 tọa độ + 5 góc ngón tay) ────────────
    _null_angles = np.zeros(len(_FINGER_PIP_JOINTS), dtype=np.float32)
    if left_hand is not None and len(left_hand) == num_hand:
        lh_xyz = np.array(left_hand, dtype=np.float32)
        lh_arr = (lh_xyz - origin).flatten()
        lh_angles = _finger_curl_angles(lh_xyz)
    else:
        lh_arr = np.zeros(num_hand * hand_dims, dtype=np.float32)
        lh_angles = _null_angles.copy()

    # ── Tay phải (21 × 3 tọa độ + 5 góc ngón tay) ────────────
    if right_hand is not None and len(right_hand) == num_hand:
        rh_xyz = np.array(right_hand, dtype=np.float32)
        rh_arr = (rh_xyz - origin).flatten()
        rh_angles = _finger_curl_angles(rh_xyz)
    else:
        rh_arr = np.zeros(num_hand * hand_dims, dtype=np.float32)
        rh_angles = _null_angles.copy()

    parts = [pose_arr, lh_arr, lh_angles, rh_arr, rh_angles]

    # ── Khuôn mặt đầy đủ (478 × 3) — tuỳ chọn ───────────────
    if use_face:
        if face is not None and len(face) == num_face:
            face_arr = np.array(
                [[p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]] for p in face],
                dtype=np.float32,
            ).flatten()
        else:
            face_arr = np.zeros(num_face * face_dims, dtype=np.float32)
        parts.append(face_arr)

    # ── Đặc trưng lông mày (33) — độc lập với use_face ────────
    if use_eyebrow:
        if face is not None and len(face) == num_face:
            eb = eyebrow_features(np.array(face, dtype=np.float32))
        else:
            eb = np.zeros(_EYEBROW_FEATURE_DIM, dtype=np.float32)
        parts.append(eb)

    return np.concatenate(parts)


def landmarks_json_to_array(
    data: dict[str, Any],
    *,
    use_face: bool = True,
    use_eyebrow: bool = False,
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
        use_eyebrow=use_eyebrow,
        right_hand=lm.get("right_hand"),
        face=lm.get("face"),
        use_face=use_face,
    )
