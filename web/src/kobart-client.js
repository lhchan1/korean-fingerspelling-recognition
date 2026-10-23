export const KOBART_WEBSOCKET_URL =
  "wss://sign-kobart.duckdns.org/ws/generate";

export const KOBART_REQUEST_TIMEOUT_MS = 30_000;

function createSessionId() {
  if (globalThis.crypto?.randomUUID) {
    return `web-${globalThis.crypto.randomUUID()}`;
  }
  return `web-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function requestKoreanSentence(
  words,
  {
    url = KOBART_WEBSOCKET_URL,
    timeoutMs = KOBART_REQUEST_TIMEOUT_MS,
    onStateChange = () => {},
    onTiming = () => {},
  } = {},
) {
  if (!Array.isArray(words) || words.length === 0) {
    return Promise.reject(new Error("전송할 단어가 없습니다."));
  }
  if (words.some((word) => typeof word !== "string" || !word.trim())) {
    return Promise.reject(new Error("words는 비어 있지 않은 문자열 배열이어야 합니다."));
  }

  const requestWords = words.map((word) => word.trim());

  const sessionId = createSessionId();

  return new Promise((resolve, reject) => {
    let socket;
    let settled = false;
    let startedAt = null;
    let openedAt = null;
    let sentAt = null;
    let receivedAt = null;

    const finish = (error, result = null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeoutId);
      const endedAt = performance.now();
      const ms = value => value == null ? null : Math.round(value * 100) / 100;
      const clientTiming = {
        timingVersion: "browser-timing-v1",
        sessionId, wordCount: requestWords.length,
        status: error ? "error" : "success",
        connectMs: ms(openedAt == null ? null : openedAt - startedAt),
        sendToResponseMs: ms(receivedAt == null || sentAt == null ? null : receivedAt - sentAt),
        requestResponseMs: ms(receivedAt == null ? null : receivedAt - startedAt),
        elapsedMs: ms(startedAt == null ? null : endedAt - startedAt),
        serverGenerationMs: null,
      };

      if (socket) {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
          socket.close(1000, "request complete");
        }
      }

      try { onTiming(clientTiming); } catch (logError) { console.warn("시간 로그 출력 실패", logError); }
      if (error) reject(error);
      else resolve({ ...result, clientTiming });
    };

    const timeoutId = setTimeout(() => {
      finish(new Error("문장 생성 요청이 30초 안에 완료되지 않았습니다."));
    }, timeoutMs);

    onStateChange("connecting", { sessionId, words: [...requestWords] });

    try {
      startedAt = performance.now();
      socket = new WebSocket(url);
    } catch (error) {
      finish(new Error(`WebSocket 연결을 시작하지 못했습니다: ${error.message}`));
      return;
    }

    socket.onopen = () => {
      if (settled) return;
      openedAt = performance.now();
      const payload = { session_id: sessionId, words: requestWords };
      sentAt = performance.now();
      socket.send(JSON.stringify(payload));
      onStateChange("processing", payload);
    };

    socket.onmessage = (event) => {
      if (settled) return;
      const messageReceivedAt = performance.now();

      let response;
      try {
        response = JSON.parse(event.data);
      } catch {
        finish(new Error("서버가 올바른 JSON 응답을 보내지 않았습니다."));
        return;
      }

      if (response?.type === "error") {
        const message =
          typeof response.message === "string" && response.message.trim()
            ? response.message.trim()
            : "서버에서 문장 생성 오류가 발생했습니다.";
        finish(new Error(message));
        return;
      }

      if (response?.type !== "result") return;
      if (response.session_id !== sessionId) return;
      if (typeof response.sentence !== "string" || !response.sentence.trim()) {
        finish(new Error("서버 응답에 생성된 문장이 없습니다."));
        return;
      }

      receivedAt = messageReceivedAt;
      finish(null, {
        ...response,
        sentence: response.sentence.trim(),
        sessionId,
        sentWords: [...requestWords],
      });
    };

    socket.onerror = () => {
      finish(new Error("KoBART 서버에 연결하지 못했습니다."));
    };

    socket.onclose = (event) => {
      if (settled) return;
      const suffix = event.reason ? `: ${event.reason}` : "";
      finish(new Error(`서버 연결이 응답 전에 종료되었습니다${suffix}`));
    };
  });
}
