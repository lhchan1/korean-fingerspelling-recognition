export class CameraController {
  constructor(videoElement) {
    this.video = videoElement;
    this.stream = null;
    this.facingMode = "user";
    this.settings = null;
  }

  get isRunning() {
    return Boolean(this.stream);
  }

  get isFrontCamera() {
    return this.facingMode === "user";
  }

  async start({ exactFacingMode = false } = {}) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("UNSUPPORTED_CAMERA_API");
    }

    this.stop();

    const isMobile = navigator.userAgentData?.mobile === true ||
      /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
    const constraints = {
      audio: false,
      video: {
        facingMode: exactFacingMode
          ? { exact: this.facingMode }
          : { ideal: this.facingMode },
        width: { ideal: isMobile ? 640 : 1280 },
        height: { ideal: isMobile ? 480 : 720 },
        frameRate: { ideal: 30, max: 30 },
      },
    };

    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.video.srcObject = this.stream;
    try {
      await this.video.play();
      await this.waitUntilVideoReady();
      this.updateActualSettings();
      return this.stream;
    } catch (error) {
      this.stop();
      throw error;
    }
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
    }

    this.stream = null;
    this.settings = null;
    this.video.pause();
    this.video.srcObject = null;
  }

  async waitUntilVideoReady(timeoutMs = 10000) {
    const ready = () =>
      this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
      this.video.videoWidth > 0 &&
      this.video.videoHeight > 0;

    if (!ready()) {
      await new Promise((resolve, reject) => {
        let timeoutId;
        const cleanup = () => {
          clearTimeout(timeoutId);
          this.video.removeEventListener("loadedmetadata", checkReady);
          this.video.removeEventListener("loadeddata", checkReady);
          this.video.removeEventListener("canplay", checkReady);
          this.video.removeEventListener("error", handleError);
        };
        const checkReady = () => {
          if (!ready()) return;
          cleanup();
          resolve();
        };
        const handleError = () => {
          cleanup();
          reject(new Error("VIDEO_STREAM_ERROR"));
        };

        this.video.addEventListener("loadedmetadata", checkReady);
        this.video.addEventListener("loadeddata", checkReady);
        this.video.addEventListener("canplay", checkReady);
        this.video.addEventListener("error", handleError);
        timeoutId = setTimeout(() => {
          cleanup();
          reject(new Error("VIDEO_READY_TIMEOUT"));
        }, timeoutMs);
        checkReady();
      });
    }

    // 모바일 브라우저가 새 카메라의 첫 프레임과 실제 크기를 반영할 시간을 준다.
    await new Promise((resolve) => requestAnimationFrame(resolve));
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }

  updateActualSettings() {
    const track = this.stream?.getVideoTracks?.()[0];
    this.settings = track?.getSettings?.() ?? null;
    const actualFacingMode = this.settings?.facingMode;
    if (actualFacingMode === "user" || actualFacingMode === "environment") {
      this.facingMode = actualFacingMode;
    }
  }

  async switchFacingMode() {
    const previousFacingMode = this.facingMode;
    this.facingMode = this.isFrontCamera ? "environment" : "user";
    try {
      try {
        return await this.start({ exactFacingMode: true });
      } catch (error) {
        if (
          ![
            "OverconstrainedError",
            "ConstraintNotSatisfiedError",
            "NotFoundError",
            "TypeError",
          ].includes(error.name)
        ) {
          throw error;
        }
        return await this.start({ exactFacingMode: false });
      }
    } catch (error) {
      this.facingMode = previousFacingMode;
      throw error;
    }
  }
}

export function cameraErrorMessage(error) {
  if (error.message === "UNSUPPORTED_CAMERA_API") {
    return "이 브라우저는 카메라 API를 지원하지 않거나 안전한 접속 환경이 아닙니다.";
  }

  if (error.message === "VIDEO_READY_TIMEOUT") {
    return "카메라 영상 준비가 지연되고 있습니다. 카메라를 종료한 뒤 다시 시작해주세요.";
  }

  if (error.message === "VIDEO_STREAM_ERROR") {
    return "카메라 영상 스트림을 재생하지 못했습니다.";
  }

  if (error.message === "CAMERA_SWITCH_UNAVAILABLE") {
    return "전환할 다른 카메라를 찾지 못했습니다.";
  }

  const messages = {
    NotAllowedError:
      "카메라 권한이 거부되었습니다. 브라우저의 사이트 설정에서 카메라를 허용해주세요.",
    NotFoundError: "사용 가능한 카메라를 찾지 못했습니다.",
    NotReadableError:
      "카메라를 열 수 없습니다. 다른 앱이 카메라를 사용 중인지 확인해주세요.",
    OverconstrainedError:
      "현재 기기에서 요청한 카메라 설정을 사용할 수 없습니다.",
    AbortError: "카메라 시작이 중단되었습니다. 다시 시도해주세요.",
  };

  return messages[error.name] ?? `카메라 오류가 발생했습니다: ${error.message}`;
}
