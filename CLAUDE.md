# CLAUDE.md — Ngữ Cảnh Dự Án: Nhận Diện Ngôn Ngữ Ký Hiệu Tiếng Việt

## Tổng Quan Dự Án

Hệ thống nhận diện ngôn ngữ ký hiệu tiếng Việt theo thời gian thực, chuyển đổi cử chỉ tay và biểu cảm khuôn mặt từ luồng webcam thành văn bản và giọng nói tiếng Việt. Pipeline xử lý hoàn toàn bất đồng bộ qua WebSocket.

---

## Kiến Trúc Tổng Thể

```
Webcam (Frontend)
    │
    ▼ (Video frames)
MediaPipe (Feature Extraction)
    │
    ▼ (Landmark coordinates)
Bi-LSTM Network (Sequence Modeling)
    │
    ▼ (Frame-level predictions)
CTC Decoder + Language Model (Alignment & Normalization)
    │
    ▼ (Normalized Vietnamese text)
TTS Engine (Speech Synthesis)
    │
    ▼ (Audio stream)
Frontend (Text + Audio output — Real-time)
```

---

## Chi Tiết Từng Module

### 1. Trích Xuất Đặc Trưng — MediaPipe

- **Thư viện:** `mediapipe` (Python)
- **Mục tiêu:** Trích xuất tọa độ không gian 3D từ mỗi frame video
- **Landmarks cần lấy:**
  - `FaceMesh` — 468 điểm khuôn mặt (biểu cảm, miệng, mắt, lông mày)
  - `Hands` — 21 điểm mỗi bàn tay × 2 tay = 42 điểm
  - `Pose` (tùy chọn) — 33 điểm thân trên để bổ trợ ngữ cảnh cử chỉ
- **Output mỗi frame:** Vector tọa độ `(x, y, z)` được flatten và chuẩn hóa
- **Lưu ý kỹ thuật:**
  - Chuẩn hóa tọa độ về `[0, 1]` theo kích thước frame
  - Xử lý trường hợp thiếu tay (padding bằng 0)
  - Tần suất xử lý: target 25–30 FPS

---

### 2. Mô Hình Hóa Chuỗi — Bi-LSTM

- **Framework:** PyTorch (ưu tiên) hoặc TensorFlow/Keras
- **Kiến trúc mạng:**

```python
Input:  (batch, time_steps, feature_dim)
  │
  ├─ Linear Projection Layer (feature_dim → hidden_dim)
  │
  ├─ Bi-LSTM Layer 1  (hidden=256, bidirectional=True)
  ├─ Dropout (0.3)
  │
  ├─ Bi-LSTM Layer 2  (hidden=256, bidirectional=True)
  ├─ Dropout (0.3)
  │
  └─ Linear FC Layer  (hidden_dim*2 → num_classes + 1)  # +1 cho blank token CTC

Output: (batch, time_steps, num_classes + 1)
```

- **Tham số quan trọng:**
  - `feature_dim`: tổng số tọa độ sau khi flatten (ví dụ: 1662 nếu dùng full MediaPipe)
  - `num_classes`: số ký tự/từ tiếng Việt trong vocabulary + 1 blank
  - Optimizer: Adam, lr=1e-4 với scheduler ReduceLROnPlateau
  - Batch size: 16–32
- **Lý do dùng Bi-LSTM:** Ngôn ngữ ký hiệu có tính ngữ cảnh hai chiều — chuyển động trước và sau đều ảnh hưởng đến nghĩa của ký hiệu hiện tại.

---

### 3. Căn Chỉnh Tự Động — CTC Loss

- **Mục đích:** Giải quyết bài toán input sequence (frames) dài hơn output sequence (text) mà không cần phân đoạn thủ công
- **Triển khai:**

```python
import torch.nn as nn

ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

# Trong training loop:
log_probs = model(x)           # (T, N, C) — time-first format
loss = ctc_loss(
    log_probs,
    targets,                    # chuỗi nhãn đã encode
    input_lengths,              # độ dài thực của mỗi input trong batch
    target_lengths              # độ dài thực của mỗi target
)
```

- **Blank token:** Index 0 được dùng làm blank (separator)
- **Constraint:** `T >= L` (số frames phải ≥ độ dài text) — cần đảm bảo khi cắt window
- **Decoding lúc inference:** Greedy decoding hoặc Beam Search

---

### 4. Giải Mã và Chuẩn Hóa — Language Model

- **Phương pháp:** CTC Beam Search kết hợp n-gram Language Model
- **Thư viện gợi ý:** `pyctcdecode` + KenLM hoặc `transformers` (PhoBERT re-scoring)
- **Quy trình:**

```
CTC raw output
    │
    ▼ Beam Search (top-k beams)
Candidate sequences
    │
    ▼ LM rescoring (n-gram hoặc PhoBERT)
Best sequence
    │
    ▼ Vietnamese text normalization
      - Chuẩn hóa dấu thanh tổ hợp → dấu thanh dựng sẵn (NFC)
      - Xử lý viết hoa câu đầu
      - Loại bỏ ký tự nhiễu
Final Vietnamese text
```

- **Vocabulary:** Xây dựng từ corpus tiếng Việt + tập ký hiệu ngôn ngữ ký hiệu
- **Lưu ý tiếng Việt:** Phải xử lý đúng Unicode tổ hợp dấu tiếng Việt (NFD vs NFC)

---

### 5. Tổng Hợp Giọng Nói — TTS

- **Input:** Văn bản tiếng Việt đã chuẩn hóa
- **Mô hình TTS gợi ý:**
  `VietTTS` (open-source, tiếng Việt native)
- **Streaming audio:** Trả về audio dạng chunk để giảm latency
- **Format output:** PCM 16-bit / WAV / MP3 tùy frontend

---

### 6. Nền Tảng Triển Khai

#### Backend — FastAPI

```
backend/
├── main.py              # FastAPI app, WebSocket endpoint
├── pipeline/
│   ├── extractor.py     # MediaPipe feature extraction
│   ├── model.py         # Bi-LSTM model definition & inference
│   ├── decoder.py       # CTC decoder + LM integration
│   └── tts.py           # TTS wrapper
├── models/
│   ├── bilstm.pt        # trained model weights
│   └── lm.binary        # KenLM language model
├── config.py            # hyperparams, paths, constants
└── requirements.txt
```

**WebSocket endpoint:**

```python
@app.websocket("/ws/recognize")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async for data in websocket.iter_bytes():
        # data = raw landmark coordinates (JSON hoặc binary)
        result = await pipeline.process(data)
        await websocket.send_json({
            "text": result.text,
            "audio": result.audio_b64,  # base64 encoded audio chunk
            "confidence": result.score
        })
```

#### Frontend

```
frontend/
├── index.html
├── app.js               # Webcam capture, MediaPipe JS, WebSocket client
└── style.css
```

**Frontend flow:**

```javascript
// 1. Khởi tạo MediaPipe trên browser (hoặc gửi raw frame lên server)
// 2. Gửi landmarks qua WebSocket mỗi frame
// 3. Nhận { text, audio } từ server
// 4. Hiển thị text realtime + phát audio
```

**Lựa chọn xử lý MediaPipe:**
- **Option A (recommended):** MediaPipe chạy trên **browser** (JS) → chỉ gửi coordinates qua WS → giảm tải bandwidth và latency
- **Option B:** Gửi raw video frames lên server, MediaPipe chạy trên **Python** → đơn giản hơn nhưng tốn bandwidth hơn

---

## Stack Công Nghệ

| Layer | Technology |
|---|---|
| Feature Extraction | MediaPipe (Python hoặc JS) |
| Deep Learning | PyTorch |
| Sequence Model | Bi-LSTM + CTC Loss |
| Language Model | KenLM / PhoBERT |
| TTS | VietTTS / edge-tts |
| Backend | FastAPI + uvicorn |
| Real-time Transport | WebSocket |
| Frontend | Vanilla JS / React |
| Deployment | Docker + nginx (production) |

---

## Cấu Trúc Dữ Liệu

### Input Frame (gửi qua WebSocket)

```json
{
  "frame_id": 1042,
  "timestamp": 1234567890.123,
  "landmarks": {
    "face": [[x, y, z], ...],     // 468 điểm
    "left_hand": [[x, y, z], ...], // 21 điểm (null nếu không thấy)
    "right_hand": [[x, y, z], ...] // 21 điểm (null nếu không thấy)
  }
}
```

### Output từ Server

```json
{
  "text": "Xin chào",
  "confidence": 0.92,
  "audio": "<base64_encoded_wav>",
  "latency_ms": 45
}
```

---

## Quy Ước Code

- **Ngôn ngữ:** Python 3.10+, type hints bắt buộc
- **Style:** PEP8, dùng `black` formatter
- **Async:** Ưu tiên `async/await` cho mọi I/O trong FastAPI
- **Logging:** Dùng `loguru` thay `print`
- **Config:** Tất cả hyperparameter và path để trong `config.py` hoặc `.env`
- **Error handling:** Mọi WebSocket handler phải có `try/except` và graceful disconnect
- **Model inference:** Wrap trong `torch.no_grad()` và batch khi có thể

---

## Các Vấn Đề Kỹ Thuật Cần Lưu Ý

1. **Latency:** Toàn bộ pipeline target < 200ms end-to-end
2. **Missing landmarks:** Luôn xử lý trường hợp một hoặc hai tay không được phát hiện (padding zeros)
3. **Sliding window:** Dùng sliding window (ví dụ: 30 frames, stride 5) để inference liên tục
4. **Unicode tiếng Việt:** Chuẩn hóa NFC trước mọi xử lý text
5. **CTC constraint:** Đảm bảo số frames đầu vào luôn ≥ độ dài sequence đầu ra
6. **Thread safety:** Model inference nên chạy trong thread pool (`run_in_executor`) để không block event loop
7. **Memory:** Giải phóng GPU memory sau mỗi inference nếu batch nhỏ

---

## Trạng Thái Dự Án & Ưu Tiên

Khi Claude hỗ trợ dự án này, hãy ưu tiên theo thứ tự:
1. **Correctness** — Pipeline phải chạy đúng end-to-end trước
2. **Latency** — Tối ưu để đạt real-time (< 200ms)
3. **Accuracy** — Cải thiện model và language model
4. **UX** — Giao diện người dùng và trải nghiệm

---

*File này là context cố định cho mọi session làm việc với dự án. Cập nhật khi có thay đổi kiến trúc.*