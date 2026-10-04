export class CameraController {
  constructor(videoElement) {
    this.video = videoElement;
    this.stream = null;
    this.facingMode = "user";
  }

  get isRunning() {
    return Boolean(this.stream);
  }

  get isFrontCamera() {
    return this.facingMode === "user";
  }

  async start() {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("UNSUPPORTED_CAMERA_API");
    }

    this.stop();

    const isMobile = navigator.userAgentData?.mobile === true ||
      /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: this.facingMode },
        width: { ideal: isMobile ? 640 : 1280 },
        height: { ideal: isMobile ? 480 : 720 },
        frameRate: { ideal: 30, max: 30 },
      },
    };

    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.video.srcObject = this.stream;
    try {
      await this.video.play();
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
    this.video.pause();
    this.video.srcObject = null;
  }

  async switchFacingMode() {
    const previousFacingMode = this.facingMode;
    this.facingMode = this.isFrontCamera ? "environment" : "user";
    try {
      return await this.start();
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
