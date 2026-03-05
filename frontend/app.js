/**
 * SL2Text — Frontend Application
 *
 * Sử dụng MediaPipe Vision Tasks (JS) để trích xuất landmarks
 * trên browser, sau đó gửi qua WebSocket tới backend.
 */

import { FilesetResolver, PoseLandmarker, HandLandmarker, FaceLandmarker }
    from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

// ── DOM Elements ────────────────────────────────────────────
const video        = document.getElementById("webcam");
const canvas       = document.getElementById("overlay");
const ctx          = canvas.getContext("2d");
const btnStart     = document.getElementById("btn-start");
const btnStop      = document.getElementById("btn-stop");
const statusBadge  = document.getElementById("status-badge");
const textOutput   = document.getElementById("text-output");
const audioPlayer  = document.getElementById("audio-player");
const statFps      = document.getElementById("stat-fps");
const statLatency  = document.getElementById("stat-latency");
const statFrames   = document.getElementById("stat-frames");

// ── State ───────────────────────────────────────────────────
let poseLandmarker  = null;
let handLandmarker  = null;
let faceLandmarker  = null;
let ws              = null;
let running         = false;
let frameCount      = 0;
let lastFpsTime     = performance.now();
let fpsCount        = 0;

// ── MediaPipe Connections (để vẽ) ───────────────────────────
const POSE_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],
    [11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
    [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
    [11,23],[12,24],[23,24],[23,25],[24,26],
    [25,27],[26,28],[27,29],[28,30],[29,31],[30,32],[27,31],[28,32],
];
const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],[0,17],
];

// ── Khởi tạo MediaPipe ─────────────────────────────────────
async function initMediaPipe() {
    statusBadge.textContent = "Đang tải model...";
    statusBadge.className = "badge loading";

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm"
    );

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            delegate: "GPU",
        },
        runningMode: "VIDEO",
        numPoses: 1,
    });

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
    });

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            delegate: "GPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: false,
    });

    statusBadge.textContent = "Sẵn sàng";
    statusBadge.className = "badge ready";
    console.log("[SL2Text] MediaPipe loaded");
}

// ── Webcam ──────────────────────────────────────────────────
async function startWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
    });
    video.srcObject = stream;
    await video.play();
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
}

function stopWebcam() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }
}

// ── WebSocket ───────────────────────────────────────────────
function connectWS() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${location.host}/ws/recognize`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        statusBadge.textContent = "Đang nhận diện";
        statusBadge.className = "badge connected";
        console.log("[WS] Connected");
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.text) {
            appendText(data.text);
        }
        if (data.audio) {
            playAudio(data.audio);
        }
        if (data.latency_ms) {
            statLatency.textContent = `Latency: ${data.latency_ms} ms`;
        }
    };

    ws.onclose = () => {
        statusBadge.textContent = "Mất kết nối";
        statusBadge.className = "badge disconnected";
    };

    ws.onerror = (err) => {
        console.error("[WS] Error:", err);
    };
}

function disconnectWS() {
    if (ws) { ws.close(); ws = null; }
}

// ── Output helpers ──────────────────────────────────────────
function appendText(text) {
    const placeholder = textOutput.querySelector(".placeholder");
    if (placeholder) placeholder.remove();

    const span = document.createElement("span");
    span.className = "detected-text";
    span.textContent = text + " ";
    textOutput.appendChild(span);
    textOutput.scrollTop = textOutput.scrollHeight;
}

function playAudio(base64mp3) {
    audioPlayer.src = "data:audio/mp3;base64," + base64mp3;
    audioPlayer.play().catch(() => {});
}

// ── Vẽ landmarks lên canvas ─────────────────────────────────
function drawResults(poseResult, handResult, faceResult) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const w = canvas.width, h = canvas.height;

    // Pose
    if (poseResult?.landmarks?.length) {
        for (const landmarks of poseResult.landmarks) {
            drawConnectors(landmarks, POSE_CONNECTIONS, "#00FF00", 2);
            drawDots(landmarks, "#00FF00", 3);
        }
    }
    // Hands
    if (handResult?.landmarks?.length) {
        for (const landmarks of handResult.landmarks) {
            drawConnectors(landmarks, HAND_CONNECTIONS, "#FFFFFF", 2);
            drawDots(landmarks, "#FF6600", 4);
        }
    }
    // Face (chỉ vẽ điểm, không vẽ đường nối vì quá nhiều)
    if (faceResult?.faceLandmarks?.length) {
        for (const landmarks of faceResult.faceLandmarks) {
            drawDots(landmarks, "#00CCFF", 1);
        }
    }

    function drawDots(landmarks, color, radius) {
        ctx.fillStyle = color;
        for (const lm of landmarks) {
            ctx.beginPath();
            ctx.arc(lm.x * w, lm.y * h, radius, 0, 2 * Math.PI);
            ctx.fill();
        }
    }

    function drawConnectors(landmarks, connections, color, lineWidth) {
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        for (const [s, e] of connections) {
            if (s < landmarks.length && e < landmarks.length) {
                ctx.beginPath();
                ctx.moveTo(landmarks[s].x * w, landmarks[s].y * h);
                ctx.lineTo(landmarks[e].x * w, landmarks[e].y * h);
                ctx.stroke();
            }
        }
    }
}

// ── Frame loop ──────────────────────────────────────────────
function processFrame() {
    if (!running) return;

    const now = performance.now();
    const timestamp = Math.round(now);

    // MediaPipe detection
    const poseResult = poseLandmarker.detectForVideo(video, timestamp);
    const handResult = handLandmarker.detectForVideo(video, timestamp);
    const faceResult = faceLandmarker.detectForVideo(video, timestamp);

    drawResults(poseResult, handResult, faceResult);

    // Xây dựng payload landmarks
    const payload = buildPayload(poseResult, handResult, faceResult, timestamp);

    // Gửi qua WebSocket
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
    }

    // Stats
    frameCount++;
    fpsCount++;
    statFrames.textContent = `Frames: ${frameCount}`;

    if (now - lastFpsTime >= 1000) {
        statFps.textContent = `FPS: ${fpsCount}`;
        fpsCount = 0;
        lastFpsTime = now;
    }

    requestAnimationFrame(processFrame);
}

function buildPayload(poseResult, handResult, faceResult, timestamp) {
    // Pose: 33 landmarks → [[x,y,z,visibility], ...]
    let pose = null;
    if (poseResult?.landmarks?.length) {
        pose = poseResult.landmarks[0].map(lm => [lm.x, lm.y, lm.z, lm.visibility ?? 1.0]);
    }

    // Hands: phân loại Left/Right theo handedness
    let left_hand = null, right_hand = null;
    if (handResult?.landmarks?.length) {
        for (let i = 0; i < handResult.landmarks.length; i++) {
            const label = handResult.handednesses[i]?.[0]?.categoryName || "Right";
            const coords = handResult.landmarks[i].map(lm => [lm.x, lm.y, lm.z]);
            if (label === "Left") left_hand = coords;
            else right_hand = coords;
        }
    }

    // Face: 478 landmarks → [[x,y,z], ...]
    let face = null;
    if (faceResult?.faceLandmarks?.length) {
        face = faceResult.faceLandmarks[0].map(lm => [lm.x, lm.y, lm.z]);
    }

    return {
        frame_id: frameCount,
        timestamp: timestamp,
        landmarks: { pose, left_hand, right_hand, face },
    };
}

// ── Button handlers ─────────────────────────────────────────
btnStart.addEventListener("click", async () => {
    btnStart.disabled = true;
    btnStop.disabled  = false;

    await startWebcam();
    connectWS();
    running = true;
    frameCount = 0;
    requestAnimationFrame(processFrame);
});

btnStop.addEventListener("click", () => {
    running = false;
    btnStart.disabled = false;
    btnStop.disabled  = true;

    disconnectWS();
    stopWebcam();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    statusBadge.textContent = "Đã dừng";
    statusBadge.className = "badge disconnected";
});

// ── Init ────────────────────────────────────────────────────
initMediaPipe().catch(err => {
    console.error("Lỗi khởi tạo MediaPipe:", err);
    statusBadge.textContent = "Lỗi tải model";
    statusBadge.className = "badge error";
});
