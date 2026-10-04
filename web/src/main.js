import { CameraController, cameraErrorMessage } from "./camera.js";
import { BrowserHandLandmarker } from "./hand-landmarker.js";
import { DebugLogger } from "./debug-log.js";
import { WordSignInference } from "./inference.js";
import { IdleSentenceCollector, RecognitionStabilizer } from "./recognition.js";

const elements = {
  video: document.querySelector("#camera"),
  landmarkCanvas: document.querySelector("#landmarkCanvas"),
  cameraStage: document.querySelector("#cameraStage"),
  emptyState: document.querySelector("#emptyState"),
  cameraLabel: document.querySelector("#cameraLabel"),
  statusDot: document.querySelector("#statusDot"),
  statusMessage: document.querySelector("#statusMessage"),
  startButton: document.querySelector("#startButton"),
  switchButton: document.querySelector("#switchButton"),
  stopButton: document.querySelector("#stopButton"),
  modelState: document.querySelector("#modelState"),
  handCount: document.querySelector("#handCount"),
  inferenceFps: document.querySelector("#inferenceFps"),
  poseState: document.querySelector("#poseState"),
  featureState: document.querySelector("#featureState"),
  debugLog: document.querySelector("#debugLog"),
  clearLogButton: document.querySelector("#clearLogButton"),
  predictionLabel: document.querySelector("#predictionLabel"),
  predictionConfidence: document.querySelector("#predictionConfidence"),
  bufferState: document.querySelector("#bufferState"),
  bufferProgress: document.querySelector("#bufferProgress"),
  topPredictions: document.querySelector("#topPredictions"),
  stableState: document.querySelector("#stableState"),
  recognizedTokens: document.querySelector("#recognizedTokens"),
  undoTokenButton: document.querySelector("#undoTokenButton"),
  clearTokensButton: document.querySelector("#clearTokensButton"),
  idleState: document.querySelector("#idleState"),
  pendingPayload: document.querySelector("#pendingPayload"),
};

const camera = new CameraController(elements.video);
const logger = new DebugLogger(elements.debugLog);
const inference = new WordSignInference((message, detail) => logger.info(message, detail));
const recognizer = new RecognitionStabilizer();
const sentenceCollector = new IdleSentenceCollector({ idleDurationMs: 3000 });
let predictionBusy = false;
let sentenceReadyForServer = false;
const handLandmarker = new BrowserHandLandmarker(
  elements.video,
  elements.landmarkCanvas,
  async ({ handCount, poseDetected = false, features = [], fps, timestamp }) => {
    elements.handCount.textContent = String(handCount);
    elements.poseState.textContent = poseDetected ? "검출" : "미검출";
    elements.featureState.textContent = `${features.length} / 146`;
    elements.inferenceFps.textContent = `${fps.toFixed(1)} FPS`;
    if (!features.length || predictionBusy) return;
    predictionBusy = true;
    try {
      const state = await inference.addFrame(features, timestamp);
      renderPrediction(state);
    } catch (error) {
      logger.error("모델 추론 실패", error);
    } finally {
      predictionBusy = false;
    }
  },
  (message, detail) => logger.info(message, detail),
);
let modelReady = false;

function renderPrediction(state) {
  const size = state.size ?? 0;
  elements.bufferState.textContent = `${size} / 60`;
  elements.bufferProgress.style.width = `${Math.min(100, (state.progress ?? 0) * 100)}%`;
  if (!state.prediction) {
    elements.predictionLabel.textContent = `특징 수집 중 ${size}/60`;
    elements.predictionConfidence.textContent = "신뢰도 0.0%";
    return;
  }

  elements.predictionLabel.textContent = state.prediction.label;
  elements.predictionConfidence.textContent =
    `신뢰도 ${(state.prediction.confidence * 100).toFixed(1)}%`;
  elements.topPredictions.replaceChildren(
    ...state.top.map(({ label, confidence }) => {
      const item = document.createElement("li");
      item.textContent = `${label} · ${(confidence * 100).toFixed(1)}%`;
      return item;
    }),
  );

  const recognitionState = recognizer.push(state.prediction);
  renderRecognition(recognitionState);
  if (recognitionState.added) {
    sentenceReadyForServer = false;
    logger.info("단어 확정", {
      label: recognitionState.added,
      tokens: recognitionState.tokens,
    });
  }

  const sentenceState = sentenceCollector.update(recognitionState);
  renderSentenceState(sentenceState);
  if (sentenceState.ready) finalizeSentence(sentenceState.payload);
}

function finalizeSentence(payload) {
  // 서버 API가 정해지면 이 함수 안에서 payload를 fetch로 전송합니다.
  elements.pendingPayload.textContent = JSON.stringify(payload, null, 2);
  logger.info("문장 전송 준비 완료", payload);
  window.dispatchEvent(new CustomEvent("sentence-ready", { detail: payload }));
  sentenceReadyForServer = true;

  renderRecognition(recognizer.clear());
  renderSentenceState(sentenceCollector.reset(), "전송 대기 데이터 생성 완료");
}

function renderSentenceState(state, message = null) {
  if (message) {
    elements.idleState.textContent = message;
    elements.idleState.dataset.state = "ready";
    return;
  }
  if (!state.active) {
    if (sentenceReadyForServer) {
      elements.idleState.textContent = "전송 대기 데이터 생성 완료";
      elements.idleState.dataset.state = "ready";
      return;
    }
    elements.idleState.textContent = "단어 입력 후 IDLE 3초를 기다립니다.";
    elements.idleState.dataset.state = "waiting";
    return;
  }

  elements.idleState.textContent =
    `IDLE 유지 중 · ${(state.remainingMs / 1000).toFixed(1)}초 후 자동 확정`;
  elements.idleState.dataset.state = "counting";
}

function renderRecognition(state) {
  const stableText = state.stableLabel
    ? `${state.stableLabel} 안정화 · ${state.votes} / ${state.minVotes}표`
    : `확정 대기 · ${state.votes ?? 0} / ${state.minVotes}표`;
  elements.stableState.textContent = state.armed
    ? stableText
    : `${stableText} · 다음 단어 전환 대기`;

  if (state.tokens.length === 0) {
    const empty = document.createElement("span");
    empty.className = "token-empty";
    empty.textContent = "아직 확정된 단어가 없습니다.";
    elements.recognizedTokens.replaceChildren(empty);
  } else {
    elements.recognizedTokens.replaceChildren(
      ...state.tokens.map((token, index) => {
        const chip = document.createElement("span");
        chip.className = "recognized-token";
        chip.textContent = `${index + 1}. ${token}`;
        return chip;
      }),
    );
  }

  const hasTokens = state.tokens.length > 0;
  elements.undoTokenButton.disabled = !hasTokens;
  elements.clearTokensButton.disabled = !hasTokens;
}

function setStatus(message, state = "idle") {
  elements.statusMessage.textContent = message;
  elements.statusDot.dataset.state = state;
}

function renderCameraState() {
  const running = camera.isRunning;

  elements.cameraStage.classList.toggle("is-empty", !running);
  elements.cameraStage.classList.toggle("is-mirrored", running && camera.isFrontCamera);
  elements.emptyState.hidden = running;
  elements.cameraLabel.textContent = running
    ? camera.isFrontCamera
      ? "전면 카메라"
      : "후면 카메라"
    : "대기 중";

  elements.startButton.disabled = running;
  elements.switchButton.disabled = !running;
  elements.stopButton.disabled = !running;
}

function renderSecurityState() {
  const isLocalhost = ["localhost", "127.0.0.1", "::1"].includes(location.hostname);
  const isSafe = window.isSecureContext || isLocalhost;

  if (!isSafe) {
    setStatus("카메라를 사용하려면 HTTPS 또는 localhost로 접속해야 합니다.", "error");
    elements.startButton.disabled = true;
  }
}

async function startCamera() {
  logger.info("카메라 시작 버튼 클릭");
  setStatus("카메라 권한을 요청하고 있습니다…", "loading");
  elements.startButton.disabled = true;

  try {
    await camera.start();
    logger.info("카메라 스트림 연결 완료", {
      width: elements.video.videoWidth,
      height: elements.video.videoHeight,
    });
    renderCameraState();
    setStatus("카메라가 정상적으로 연결되었습니다.", "success");
  } catch (error) {
    logger.error("카메라 연결 실패", error);
    console.error(error);
    camera.stop();
    handLandmarker.stop();
    if (!modelReady) elements.modelState.textContent = "로드 실패";
    renderCameraState();
    setStatus(cameraErrorMessage(error), "error");
    return;
  }

  if (!modelReady) {
    elements.modelState.textContent = "불러오는 중";
    setStatus("카메라 연결 완료 · MediaPipe 모델을 불러오는 중입니다…", "loading");
    try {
      // 모바일 브라우저의 순간 메모리 사용량을 줄이기 위해 큰 런타임을 순차 로드합니다.
      await handLandmarker.initialize();
      await inference.initialize();
      modelReady = true;
      elements.modelState.textContent = "준비 완료";
      handLandmarker.start();
      setStatus("카메라·MediaPipe·24클래스 모델이 준비되었습니다.", "success");
    } catch (error) {
      logger.error("MediaPipe 초기화 실패", error);
      console.error("MediaPipe initialization failed:", error);
      elements.modelState.textContent = "로드 실패";
      setStatus(
        `카메라는 연결됐지만 MediaPipe를 불러오지 못했습니다: ${error.message}`,
        "error",
      );
    }
  } else {
    handLandmarker.start();
  }
}

async function switchCamera() {
  setStatus("카메라를 전환하고 있습니다…", "loading");
  elements.switchButton.disabled = true;

  try {
    handLandmarker.stop();
    await camera.switchFacingMode();
    handLandmarker.start();
    renderCameraState();
    setStatus(
      `${camera.isFrontCamera ? "전면" : "후면"} 카메라로 전환했습니다.`,
      "success",
    );
  } catch (error) {
    console.error(error);
    camera.stop();
    handLandmarker.stop();
    renderCameraState();
    setStatus(cameraErrorMessage(error), "error");
  }
}

function stopCamera() {
  logger.info("카메라 종료");
  handLandmarker.stop();
  camera.stop();
  inference.reset();
  sentenceCollector.reset();
  sentenceReadyForServer = false;
  renderRecognition(recognizer.resetWindow());
  renderSentenceState(sentenceCollector.state());
  renderPrediction({ size: 0, progress: 0 });
  renderCameraState();
  setStatus("카메라를 종료했습니다.", "idle");
}

elements.startButton.addEventListener("click", startCamera);
elements.switchButton.addEventListener("click", switchCamera);
elements.stopButton.addEventListener("click", stopCamera);
elements.clearLogButton.addEventListener("click", () => logger.clear());
elements.undoTokenButton.addEventListener("click", () => {
  sentenceCollector.reset();
  sentenceReadyForServer = false;
  const state = recognizer.undo();
  renderRecognition(state);
  renderSentenceState(sentenceCollector.state());
  logger.info("마지막 확정 단어 취소", { removed: state.removed, tokens: state.tokens });
});
elements.clearTokensButton.addEventListener("click", () => {
  sentenceCollector.reset();
  sentenceReadyForServer = false;
  renderRecognition(recognizer.clear());
  renderSentenceState(sentenceCollector.state());
  logger.info("확정 단어 전체 지우기");
});
window.addEventListener("pagehide", () => {
  handLandmarker.dispose();
  camera.stop();
});

renderCameraState();
renderSecurityState();
renderRecognition(recognizer.state());
renderSentenceState(sentenceCollector.state());
logger.info("페이지 초기화 완료", {
  secureContext: window.isSecureContext,
  userAgent: navigator.userAgent,
});
