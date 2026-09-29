const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const POSE_INDICES = [11, 12, 13, 14, 15, 16];
const POSE_CONNECTIONS = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
const FEATURE_COUNT = 146;

const WASM_PATH =
  "/public/vendor/mediapipe/tasks-vision/wasm";
const MODULE_PATH =
  "/public/vendor/mediapipe/tasks-vision/vision_bundle.mjs";
const MODEL_PATH = "/public/models/mediapipe/hand_landmarker.task";
const POSE_MODEL_PATH = "/public/models/mediapipe/pose_landmarker_lite.task";

function withTimeout(promise, timeoutMs, stage) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`${stage} 단계가 ${timeoutMs / 1000}초 안에 완료되지 않았습니다.`)),
      timeoutMs,
    );
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

export class BrowserHandLandmarker {
  constructor(video, canvas, onResult, onLog) {
    this.video = video;
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.onResult = onResult;
    this.onLog = onLog;
    this.landmarker = null;
    this.poseLandmarker = null;
    this.animationFrame = null;
    this.lastVideoTime = -1;
    this.frameTimes = [];
    this.lastHandCount = null;
    this.lastPoseDetected = null;
    this.lastInferenceTime = 0;
    this.sampleIntervalMs = 1000 / 15;
  }

  async initialize() {
    if (this.landmarker) return;

    this.log("MediaPipe JavaScript 모듈 로딩 시작");
    const { FilesetResolver, HandLandmarker, PoseLandmarker } = await import(
      MODULE_PATH
    );
    this.log("MediaPipe JavaScript 모듈 로딩 완료");
    this.log("MediaPipe WASM 초기화 시작", WASM_PATH);
    const vision = await withTimeout(
      FilesetResolver.forVisionTasks(WASM_PATH),
      30000,
      "WASM 초기화",
    );
    this.log("MediaPipe WASM 초기화 완료");
    const options = {
      baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    };

    try {
      this.log("Hand Landmarker GPU 초기화 시작");
      this.landmarker = await withTimeout(
        HandLandmarker.createFromOptions(vision, options),
        30000,
        "GPU 모델 초기화",
      );
      this.log("Hand Landmarker GPU 초기화 완료");
    } catch (gpuError) {
      console.warn("MediaPipe GPU 초기화 실패, CPU로 재시도합니다.", gpuError);
      this.log("GPU 초기화 실패, CPU로 재시도", gpuError.message);
      options.baseOptions.delegate = "CPU";
      this.landmarker = await withTimeout(
        HandLandmarker.createFromOptions(vision, options),
        30000,
        "CPU 모델 초기화",
      );
      this.log("Hand Landmarker CPU 초기화 완료");
    }

    const poseOptions = {
      baseOptions: { modelAssetPath: POSE_MODEL_PATH, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    };

    try {
      this.log("Pose Landmarker GPU 초기화 시작");
      this.poseLandmarker = await withTimeout(
        PoseLandmarker.createFromOptions(vision, poseOptions),
        30000,
        "Pose GPU 모델 초기화",
      );
      this.log("Pose Landmarker GPU 초기화 완료");
    } catch (gpuError) {
      this.log("Pose GPU 초기화 실패, CPU로 재시도", gpuError.message);
      poseOptions.baseOptions.delegate = "CPU";
      this.poseLandmarker = await withTimeout(
        PoseLandmarker.createFromOptions(vision, poseOptions),
        30000,
        "Pose CPU 모델 초기화",
      );
      this.log("Pose Landmarker CPU 초기화 완료");
    }
  }

  log(message, detail) {
    this.onLog?.(message, detail);
  }

  start() {
    if (!this.landmarker || !this.poseLandmarker || this.animationFrame) return;
    this.lastVideoTime = -1;
    this.frameTimes = [];
    this.lastHandCount = null;
    this.lastPoseDetected = null;
    this.lastInferenceTime = 0;
    this.predict();
  }

  stop() {
    if (this.animationFrame) cancelAnimationFrame(this.animationFrame);
    this.animationFrame = null;
    this.lastVideoTime = -1;
    this.clear();
    this.onResult?.({ handCount: 0, fps: 0 });
  }

  predict = () => {
    if (this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      this.resizeCanvas();

      const now = performance.now();
      if (
        this.video.currentTime !== this.lastVideoTime &&
        now - this.lastInferenceTime >= this.sampleIntervalMs
      ) {
        const startedAt = performance.now();
        const result = this.landmarker.detectForVideo(this.video, startedAt);
        const poseResult = this.poseLandmarker.detectForVideo(this.video, startedAt);
        this.lastVideoTime = this.video.currentTime;
        this.lastInferenceTime = now;
        const hands = result.landmarks ?? [];
        const poses = poseResult.landmarks ?? [];
        const features = featuresForFrame(result, poseResult);
        this.draw(hands, poses[0] ?? null);
        this.recordFrame(performance.now());

        if (hands.length !== this.lastHandCount) {
          this.log("손 검출 개수 변경", {
            handCount: hands.length,
            handedness: (result.handedness ?? []).map(
              (categories) => categories[0]?.categoryName ?? "unknown",
            ),
          });
          this.lastHandCount = hands.length;
        }

        const poseDetected = poses.length > 0;
        if (poseDetected !== this.lastPoseDetected) {
          this.log("상체 Pose 검출 상태 변경", { poseDetected });
          this.lastPoseDetected = poseDetected;
        }

        this.onResult?.({
          handCount: hands.length,
          poseDetected,
          features,
          fps: this.calculateFps(),
          handResult: result,
          poseResult,
          timestamp: now,
        });
      }
    }

    this.animationFrame = requestAnimationFrame(this.predict);
  };

  resizeCanvas() {
    const width = this.video.videoWidth;
    const height = this.video.videoHeight;
    if (!width || !height) return;

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
  }

  draw(hands, pose) {
    this.clear();
    const { width, height } = this.canvas;

    for (const landmarks of hands) {
      this.context.lineWidth = 4;
      this.context.strokeStyle = "#69e697";
      this.context.lineCap = "round";

      for (const [start, end] of HAND_CONNECTIONS) {
        this.context.beginPath();
        this.context.moveTo(landmarks[start].x * width, landmarks[start].y * height);
        this.context.lineTo(landmarks[end].x * width, landmarks[end].y * height);
        this.context.stroke();
      }

      for (const landmark of landmarks) {
        this.context.beginPath();
        this.context.arc(landmark.x * width, landmark.y * height, 6, 0, Math.PI * 2);
        this.context.fillStyle = "#ff6b86";
        this.context.fill();
        this.context.lineWidth = 2;
        this.context.strokeStyle = "#ffffff";
        this.context.stroke();
      }
    }

    if (pose) {
      this.context.lineWidth = 5;
      this.context.strokeStyle = "#fbbf24";
      this.context.lineCap = "round";
      for (const [start, end] of POSE_CONNECTIONS) {
        this.context.beginPath();
        this.context.moveTo(pose[start].x * width, pose[start].y * height);
        this.context.lineTo(pose[end].x * width, pose[end].y * height);
        this.context.stroke();
      }
      for (const index of POSE_INDICES) {
        this.context.beginPath();
        this.context.arc(pose[index].x * width, pose[index].y * height, 7, 0, Math.PI * 2);
        this.context.fillStyle = "#fbbf24";
        this.context.fill();
        this.context.lineWidth = 2;
        this.context.strokeStyle = "#ffffff";
        this.context.stroke();
      }
    }
  }

  clear() {
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  recordFrame(timestamp) {
    this.frameTimes.push(timestamp);
    const cutoff = timestamp - 1000;
    while (this.frameTimes[0] < cutoff) this.frameTimes.shift();
  }

  calculateFps() {
    return this.frameTimes.length > 1 ? this.frameTimes.length : 0;
  }
}

function featuresForFrame(handResult, poseResult) {
  const poses = poseResult.landmarks ?? [];
  const poseDetected = poses.length > 0;
  let center = [0, 0, 0];
  let scale = 1;
  let poseValues = new Array(POSE_INDICES.length * 3).fill(0);

  if (poseDetected) {
    const pose = poses[0];
    const leftShoulder = pose[11];
    const rightShoulder = pose[12];
    center = [
      (leftShoulder.x + rightShoulder.x) / 2,
      (leftShoulder.y + rightShoulder.y) / 2,
      (leftShoulder.z + rightShoulder.z) / 2,
    ];
    scale = Math.hypot(
      leftShoulder.x - rightShoulder.x,
      leftShoulder.y - rightShoulder.y,
      leftShoulder.z - rightShoulder.z,
    );
    if (scale < 1e-8) scale = 1;
    poseValues = POSE_INDICES.flatMap((index) => vector(pose[index], center, scale));
  }

  const hands = { Left: null, Right: null };
  const handLandmarks = handResult.landmarks ?? [];
  const handedness = handResult.handedness ?? [];
  handLandmarks.forEach((landmarks, index) => {
    const name = handedness[index]?.[0]?.categoryName;
    if ((name === "Left" || name === "Right") && hands[name] == null) {
      hands[name] = landmarks;
    }
  });

  const values = [];
  for (const name of ["Left", "Right"]) {
    const landmarks = hands[name];
    if (!poseDetected || landmarks == null) {
      values.push(...new Array(21 * 3).fill(0));
    } else {
      for (const point of landmarks) values.push(...vector(point, center, scale));
    }
  }

  const leftDetected = poseDetected && hands.Left != null;
  const rightDetected = poseDetected && hands.Right != null;
  values.push(...poseValues, Number(leftDetected), Number(rightDetected));

  if (values.length !== FEATURE_COUNT) {
    throw new Error(`특징 길이가 ${FEATURE_COUNT}이 아닙니다: ${values.length}`);
  }
  return values;
}

function vector(point, center, scale) {
  return [
    (point.x - center[0]) / scale,
    (point.y - center[1]) / scale,
    (point.z - center[2]) / scale,
  ];
}

export { FEATURE_COUNT, POSE_INDICES, featuresForFrame };
