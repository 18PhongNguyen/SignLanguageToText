/**
 * SL2Text Frontend — Frontend MediaPipe mode
 *
 * Dùng @mediapipe/tasks-vision JS để extract landmarks ngay trên browser,
 * gửi JSON qua WebSocket /ws/recognize.
 *
 * === NHẤT QUÁN VỚI data_collector.py ===
 * 1. Frame được flip ngang TRƯỚC khi đưa vào MediaPipe
 *    → giống `image = cv2.flip(frame, 1)` trong Python
 * 2. Handedness "Left"/"Right" từ image-perspective (sau flip)
 *    → khớp với cách data_collector.py phân loại tay
 * 3. Pose: [x, y, z, visibility] × 33
 * 4. Hand: [x, y, z] × 21
 * 5. Face: [x, y, z] × 478 (server dùng để tính eyebrow features)
 */

// ── DOM ──────────────────────────────────────────────────────────────────────
const video       = document.getElementById("webcam");
const canvas      = document.getElementById("overlay");
const ctx         = canvas.getContext("2d");
const btnStart    = document.getElementById("btn-start");
const btnStop     = document.getElementById("btn-stop");
const statusBadge = document.getElementById("status-badge");
const textOutput  = document.getElementById("text-output");
const audioPlayer = document.getElementById("audio-player");
const statFps     = document.getElementById("stat-fps");
const statLatency = document.getElementById("stat-latency");
const statFrames  = document.getElementById("stat-frames");

// ── Config ────────────────────────────────────────────────────────────────────
const TARGET_FPS     = 15;
const FRAME_INTERVAL = Math.round(1000 / TARGET_FPS);
const MIN_POSE_CONF  = 0.35;
const MIN_HAND_CONF  = 0.35;
const MIN_FACE_CONF  = 0.35;
const MIN_TRACKING   = 0.30;

// Serve từ local (tránh CDN chậm)
const WASM_BASE  = "/mediapipe/wasm";
const MODEL_BASE = "/models";

// ── State ─────────────────────────────────────────────────────────────────────
let ws          = null;
let running     = false;
let frameCount  = 0;
let fpsCount    = 0;
let lastFpsTime = performance.now();
let frameTimer  = null;
let mpReady     = false;

// MediaPipe objects — được gán sau dynamic import
let PoseLandmarker, HandLandmarker, FaceLandmarker, DrawingUtils;
let poseLandmarker = null;
let handLandmarker = null;
let faceLandmarker = null;

// Canvas ẩn để detect (flip ngang + giảm resolution 2× trước khi detect)
// Giống data_collector.py: resize về W//2 × H//2 → 4× ít pixel → nhanh hơn ~2-3×
const detectCanvas = document.createElement("canvas");
const detectCtx    = detectCanvas.getContext("2d", { willReadFrequently: true });

// Pose thay đổi chậm → cache kết quả, chỉ chạy mỗi 2 frame
let _cachedPoseResult = null;
let _poseFrameCounter = 0;
const POSE_SKIP = 2;

// ── Khởi tạo MediaPipe Tasks Vision ──────────────────────────────────────────
async function initMediaPipe() {
    statusBadge.textContent = "Đang tải MediaPipe...";
    statusBadge.className   = "badge loading";

    let vision;
    try {
        const mod = await import("/mediapipe/vision_bundle.mjs");
        PoseLandmarker = mod.PoseLandmarker;
        HandLandmarker = mod.HandLandmarker;
        FaceLandmarker = mod.FaceLandmarker;
        DrawingUtils   = mod.DrawingUtils;
        const { FilesetResolver } = mod;
        vision = await FilesetResolver.forVisionTasks(WASM_BASE);
    } catch (err) {
        console.error("[MP] Lỗi tải thư viện:", err);
        statusBadge.textContent = "Lỗi tải MediaPipe";
        statusBadge.className   = "badge error";
        btnStart.disabled = false;
        throw err;
    }

    try {
        [poseLandmarker, handLandmarker, faceLandmarker] = await Promise.all([
            PoseLandmarker.createFromOptions(vision, {
                baseOptions: { modelAssetPath: `${MODEL_BASE}/pose_landmarker_full.task`, delegate: "CPU" },
                runningMode: "VIDEO",
                numPoses: 1,
                minPoseDetectionConfidence: MIN_POSE_CONF,
                minPosePresenceConfidence:  MIN_POSE_CONF,
                minTrackingConfidence:      MIN_TRACKING,
            }),
            HandLandmarker.createFromOptions(vision, {
                baseOptions: { modelAssetPath: `${MODEL_BASE}/hand_landmarker.task`, delegate: "CPU" },
                runningMode: "VIDEO",
                numHands: 2,
                minHandDetectionConfidence: MIN_HAND_CONF,
                minHandPresenceConfidence:  MIN_HAND_CONF,
                minTrackingConfidence:      MIN_TRACKING,
            }),
            FaceLandmarker.createFromOptions(vision, {
                baseOptions: { modelAssetPath: `${MODEL_BASE}/face_landmarker.task`, delegate: "CPU" },
                runningMode: "VIDEO",
                numFaces: 1,
                minFaceDetectionConfidence: MIN_FACE_CONF,
                minFacePresenceConfidence:  MIN_FACE_CONF,
                minTrackingConfidence:      MIN_TRACKING,
            }),
        ]);
    } catch (err) {
        console.error("[MP] Lỗi tải model:", err);
        statusBadge.textContent = "Lỗi tải model";
        statusBadge.className   = "badge error";
        btnStart.disabled = false;
        throw err;
    }

    mpReady = true;
    btnStart.disabled = false;
    statusBadge.textContent = "Sẵn sàng";
    statusBadge.className   = "badge ready";
    console.log("[MP] MediaPipe ready — local models");
}

// ── Webcam ────────────────────────────────────────────────────────────────────
async function startWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
    });
    video.srcObject = stream;
    await video.play();
    const w = video.videoWidth  || 640;
    const h = video.videoHeight || 480;
    canvas.width  = w;
    canvas.height = h;
    // detectCanvas chạy ở half resolution — 4× ít pixel, landmark vẫn chuẩn vì normalize [0,1]
    detectCanvas.width  = Math.round(w / 2);
    detectCanvas.height = Math.round(h / 2);
}

function stopWebcam() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWS() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${location.host}/ws/recognize`);

    ws.onopen = () => {
        statusBadge.textContent = "Đang nhận diện";
        statusBadge.className   = "badge connected";
        console.log("[WS] Connected → /ws/recognize");
    };

    ws.onmessage = ({ data }) => {
        try {
            const msg = JSON.parse(data);
            if (msg.text)  appendText(msg.text);
            if (msg.audio) playAudio(msg.audio);
            if (msg.latency_ms != null)
                statLatency.textContent = "Latency: " + msg.latency_ms + " ms";
        } catch (e) {
            console.error("[WS] parse error:", e);
        }
    };

    ws.onclose = (ev) => {
        statusBadge.textContent = "Mất kết nối";
        statusBadge.className   = "badge disconnected";
        console.warn("[WS] Closed", ev.code, ev.reason);
    };

    ws.onerror = (err) => console.error("[WS] Error:", err);
}

function disconnectWS() {
    if (ws) { ws.close(); ws = null; }
}

// ── Detect + Send frame ───────────────────────────────────────────────────────
function sendFrame() {
    if (!running || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (video.readyState < 2) return;

    const now = performance.now();

    if (!mpReady) {
        // MediaPipe chưa sẵn sàng — chỉ hiển thị video mirror
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
        ctx.restore();
        frameCount++;
        statFrames.textContent = "Frames: " + frameCount;
        return;
    }

    // QUAN TRỌNG: flip ngang + downscale 2× trước khi detect
    // Khớp với data_collector.py: resize về W//2 × H//2 rồi flip
    detectCtx.save();
    detectCtx.scale(-1, 1);
    detectCtx.drawImage(video, -detectCanvas.width, 0, detectCanvas.width, detectCanvas.height);
    detectCtx.restore();

    // Pose chỉ cập nhật mỗi POSE_SKIP frame (thay đổi chậm)
    _poseFrameCounter++;
    if (_poseFrameCounter % POSE_SKIP === 0 || _cachedPoseResult === null) {
        _cachedPoseResult = poseLandmarker.detectForVideo(detectCanvas, now);
    }
    const poseResult = _cachedPoseResult;
    const handResult = handLandmarker.detectForVideo(detectCanvas, now);
    const faceResult = faceLandmarker.detectForVideo(detectCanvas, now);

    // Build JSON khớp với landmarks_json_to_array trong extractor.py
    const landmarks = { pose: null, left_hand: null, right_hand: null, face: null };

    if (poseResult.landmarks.length > 0) {
        landmarks.pose = poseResult.landmarks[0].map(lm => [
            lm.x, lm.y, lm.z, lm.visibility ?? 1.0,
        ]);
    }

    handResult.landmarks.forEach((lms, i) => {
        const label  = handResult.handednesses[i]?.[0]?.categoryName ?? "";
        const coords = lms.map(lm => [lm.x, lm.y, lm.z]);
        if (label === "Left")       landmarks.left_hand  = coords;
        else if (label === "Right") landmarks.right_hand = coords;
    });

    if (faceResult.faceLandmarks.length > 0) {
        landmarks.face = faceResult.faceLandmarks[0].map(lm => [lm.x, lm.y, lm.z]);
    }

    ws.send(JSON.stringify({ frame_id: frameCount, timestamp: Date.now() / 1000, landmarks }));

    drawSkeleton(poseResult, handResult);

    frameCount++;
    fpsCount++;
    statFrames.textContent = "Frames: " + frameCount;
    const elapsed = now - lastFpsTime;
    if (elapsed >= 1000) {
        statFps.textContent = "FPS: " + Math.round(fpsCount * 1000 / elapsed);
        fpsCount    = 0;
        lastFpsTime = now;
    }
}

// ── Vẽ skeleton lên overlay ───────────────────────────────────────────────────
function drawSkeleton(poseResult, handResult) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!DrawingUtils) return;

    // Landmarks detect trên canvas đã flip → mirror lại khi vẽ
    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);

    const drawUtils = new DrawingUtils(ctx);

    if (poseResult.landmarks.length > 0) {
        drawUtils.drawConnectors(
            poseResult.landmarks[0],
            PoseLandmarker.POSE_CONNECTIONS,
            { color: "#00FF00", lineWidth: 1 },
        );
        drawUtils.drawLandmarks(poseResult.landmarks[0], { color: "#FF0000", radius: 2 });
    }

    handResult.landmarks.forEach(lms => {
        drawUtils.drawConnectors(lms, HandLandmarker.HAND_CONNECTIONS,
            { color: "#00BFFF", lineWidth: 2 });
        drawUtils.drawLandmarks(lms, { color: "#FF69B4", radius: 3 });
    });

    ctx.restore();
}

// ── Output helpers ────────────────────────────────────────────────────────────
function appendText(text) {
    const placeholder = textOutput.querySelector(".placeholder");
    if (placeholder) placeholder.remove();
    const span = document.createElement("span");
    span.className   = "detected-text";
    span.textContent = text + " ";
    textOutput.appendChild(span);
    textOutput.scrollTop = textOutput.scrollHeight;
}

function playAudio(base64mp3) {
    audioPlayer.src = "data:audio/mp3;base64," + base64mp3;
    audioPlayer.play().catch(() => {});
}

// ── Button handlers ───────────────────────────────────────────────────────────
btnStart.addEventListener("click", async () => {
    btnStart.disabled = true;
    btnStop.disabled  = false;
    statusBadge.textContent = "Đang khởi động...";
    statusBadge.className   = "badge loading";

    try {
        await startWebcam();
        connectWS();
        running    = true;
        frameCount = 0;
        frameTimer = setInterval(sendFrame, FRAME_INTERVAL);
    } catch (err) {
        console.error("Lỗi khởi động:", err);
        statusBadge.textContent = "Lỗi camera";
        statusBadge.className   = "badge error";
        btnStart.disabled = false;
        btnStop.disabled  = true;
    }
});

btnStop.addEventListener("click", () => {
    running = false;
    if (frameTimer) { clearInterval(frameTimer); frameTimer = null; }
    btnStart.disabled = false;
    btnStop.disabled  = true;
    disconnectWS();
    stopWebcam();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    statusBadge.textContent = "Đã dừng";
    statusBadge.className   = "badge disconnected";
});

// ── Init ──────────────────────────────────────────────────────────────────────
btnStart.disabled = true;
initMediaPipe().catch(err => {
    console.error("[MP] Không tải được MediaPipe:", err);
    btnStart.disabled = false;
    statusBadge.textContent = "MediaPipe lỗi";
    statusBadge.className   = "badge error";
});
