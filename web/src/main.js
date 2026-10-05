import { CameraController, cameraErrorMessage } from "./camera.js";
import { BrowserHandLandmarker } from "./hand-landmarker.js";
import { DebugLogger } from "./debug-log.js";
import { WordSignInference } from "./inference.js";
import {
  KOBART_WEBSOCKET_URL,
  requestKoreanSentence,
} from "./kobart-client.js";
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
  generationState: document.querySelector("#generationState"),
  generatedSentence: document.querySelector("#generatedSentence"),
};

const camera = new CameraController(elements.video);
const logger = new DebugLogger(elements.debugLog);
const inference = new WordSignInference((message, detail) => logger.info(message, detail));
const recognizer = new RecognitionStabilizer();
const sentenceCollector = new IdleSentenceCollector({ idleDurationMs: 3000 });
let predictionBusy = false;
let requestInFlight = false;
let idleSegmentConsumed = false;
let lastFeatureTimestamp = null;
let scrollPaused = false;
let scrollResumeTimer = null;
const handLandmarker = new BrowserHandLandmarker(
  elements.video,
  elements.landmarkCanvas,
  async ({ handCount, poseDetected = false, features = [], fps, timestamp }) => {
    elements.handCount.textContent = String(handCount);
    elements.poseState.textContent = poseDetected ? "검출" : "미검출";
    elements.featureState.textContent = `${poseDetected ? features.length : 0} / 146`;
    elements.inferenceFps.textContent = `${fps.toFixed(1)} FPS`;
    if (!features.length || !poseDetected || scrollPaused || predictionBusy) return;

    if (lastFeatureTimestamp !== null && timestamp - lastFeatureTimestamp > 500) {
      resetTemporalInference("프레임 처리 간격이 길어져 추론 버퍼를 초기화했습니다.");
    }
    lastFeatureTimestamp = timestamp;
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

function resetTemporalInference(logMessage = null) {
  inference.reset();
  lastFeatureTimestamp = null;
  renderRecognition(recognizer.resetWindow());
  renderSentenceState(sentenceCollector.reset());
  renderPrediction({ size: 0, progress: 0 });
  if (logMessage) logger.info(logMessage);
}

function pauseInferenceForScroll() {
  if (!camera.isRunning || !modelReady) return;

  if (!scrollPaused) {
    scrollPaused = true;
    handLandmarker.stop();
    resetTemporalInference("화면 스크롤 감지 · 추론을 잠시 중지합니다.");
  }

  clearTimeout(scrollResumeTimer);
  scrollResumeTimer = setTimeout(async () => {
    if (!camera.isRunning || document.hidden) return;
    try {
      await elements.video.play();
      scrollPaused = false;
      handLandmarker.start();
      setStatus("스크롤 종료 · 특징을 처음부터 다시 수집합니다.", "success");
    } catch (error) {
      logger.error("스크롤 후 카메라 재개 실패", error);
      setStatus("카메라 재개에 실패했습니다. 카메라를 다시 시작해주세요.", "error");
    }
  }, 350);
}

async function handleVisibilityChange() {
  if (!camera.isRunning || !modelReady) return;

  if (document.hidden) {
    clearTimeout(scrollResumeTimer);
    scrollPaused = true;
    handLandmarker.stop();
    resetTemporalInference("페이지가 가려져 추론을 중지했습니다.");
    return;
  }

  try {
    await elements.video.play();
    scrollPaused = false;
    handLandmarker.start();
    setStatus("카메라 추론을 다시 시작했습니다.", "success");
  } catch (error) {
    logger.error("페이지 복귀 후 카메라 재개 실패", error);
    setStatus("카메라를 다시 시작해주세요.", "error");
  }
}

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
    idleSegmentConsumed = false;
    sentenceCollector.reset();
    if (!requestInFlight) {
      renderGenerationState("collecting", "새 단어를 확정했습니다.");
    }
    logger.info("단어 확정", {
      label: recognitionState.added,
      tokens: recognitionState.tokens,
    });
  }

  if (recognitionState.stableLabel && recognitionState.stableLabel !== "IDLE") {
    idleSegmentConsumed = false;
  }

  if (requestInFlight) return;
  if (idleSegmentConsumed) {
    sentenceCollector.reset();
    renderSentenceState(sentenceCollector.state(), "새 수어 동작을 기다립니다.");
    return;
  }

  const sentenceState = sentenceCollector.update(recognitionState);
  renderSentenceState(sentenceState);
  if (sentenceState.ready) sendSentenceRequest(sentenceState.payload.words);
}

async function sendSentenceRequest(words) {
  if (requestInFlight || idleSegmentConsumed || words.length === 0) return;

  const sentWords = [...words];
  requestInFlight = true;
  idleSegmentConsumed = true;
  sentenceCollector.reset();
  renderRecognition(recognizer.state());
  logger.info("KoBART 문장 생성 요청 시작", {
    url: KOBART_WEBSOCKET_URL,
    words: sentWords,
  });

  try {
    const response = await requestKoreanSentence(sentWords, {
      onStateChange(state, detail) {
        if (state === "connecting") {
          renderGenerationState("connecting", "KoBART 서버에 연결 중입니다…");
          renderSentenceState(sentenceCollector.state(), "서버 연결 중");
        } else if (state === "processing") {
          renderGenerationState("processing", "서버에서 한국어 문장을 생성 중입니다…");
          renderSentenceState(sentenceCollector.state(), "문장 생성 처리 중");
          logger.info("KoBART 요청 전송 완료", {
            session_id: detail.session_id,
            words: detail.words,
          });
        }
      },
    });

    const recognitionState = recognizer.consumePrefix(sentWords);
    if (!recognitionState.consumed) {
      throw new Error("전송한 단어와 현재 단어 목록이 달라 결과를 적용하지 않았습니다.");
    }

    elements.generatedSentence.textContent = response.sentence;
    renderGenerationState("success", "한국어 문장 생성이 완료되었습니다.");
    renderRecognition(recognitionState);
    renderSentenceState(sentenceCollector.state(), "전송 완료 · 새 수어 동작을 기다립니다.");
    logger.info("KoBART 문장 생성 성공", {
      session_id: response.sessionId,
      sentence: response.sentence,
      remainingTokens: recognitionState.tokens,
    });
  } catch (error) {
    renderGenerationState("error", `문장 생성 실패 · ${error.message}`);
    renderRecognition(recognizer.state());
    renderSentenceState(sentenceCollector.state(), "전송 실패 · 새 수어 동작 후 다시 시도합니다.");
    logger.error("KoBART 문장 생성 실패", error);
  } finally {
    requestInFlight = false;
    renderRecognition(recognizer.state());
  }
}

function renderSentenceState(state, message = null) {
  if (message) {
    elements.idleState.textContent = message;
    elements.idleState.dataset.state = "ready";
    return;
  }
  if (!state.active) {
    elements.idleState.textContent = "단어 입력 후 IDLE 3초를 기다립니다.";
    elements.idleState.dataset.state = "waiting";
    if (!requestInFlight && !idleSegmentConsumed) {
      renderGenerationState("collecting", "수어 단어를 모으는 중입니다.");
    }
    return;
  }

  elements.idleState.textContent =
    `IDLE 유지 중 · ${(state.remainingMs / 1000).toFixed(1)}초 후 자동 전송`;
  elements.idleState.dataset.state = "counting";
  renderGenerationState("idle-countdown", "IDLE 3초를 확인하고 있습니다.");
}

function renderGenerationState(state, message) {
  elements.generationState.dataset.state = state;
  elements.generationState.textContent = message;
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
  elements.undoTokenButton.disabled = !hasTokens || requestInFlight;
  elements.clearTokensButton.disabled = !hasTokens || requestInFlight;
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
  clearTimeout(scrollResumeTimer);
  scrollPaused = false;
  idleSegmentConsumed = false;
  resetTemporalInference();
  renderCameraState();
  setStatus("카메라를 종료했습니다.", "idle");
}

elements.startButton.addEventListener("click", startCamera);
elements.switchButton.addEventListener("click", switchCamera);
elements.stopButton.addEventListener("click", stopCamera);
elements.clearLogButton.addEventListener("click", () => logger.clear());
elements.undoTokenButton.addEventListener("click", () => {
  if (requestInFlight) return;
  sentenceCollector.reset();
  idleSegmentConsumed = false;
  const state = recognizer.undo();
  renderRecognition(state);
  renderSentenceState(sentenceCollector.state());
  logger.info("마지막 확정 단어 취소", { removed: state.removed, tokens: state.tokens });
});
elements.clearTokensButton.addEventListener("click", () => {
  if (requestInFlight) return;
  sentenceCollector.reset();
  idleSegmentConsumed = false;
  renderRecognition(recognizer.clear());
  renderSentenceState(sentenceCollector.state());
  elements.generatedSentence.textContent = "아직 생성된 문장이 없습니다.";
  logger.info("확정 단어 전체 지우기");
});
window.addEventListener("scroll", pauseInferenceForScroll, { passive: true });
document.addEventListener("visibilitychange", handleVisibilityChange);
window.addEventListener("pagehide", () => {
  clearTimeout(scrollResumeTimer);
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
