import {
  IdleSentenceCollector,
  RecognitionStabilizer,
} from "../src/recognition.js?v=20261013-camera-switch-fix1";

const result = document.querySelector("#result");
try {
  const recognizer = new RecognitionStabilizer();
  for (let index = 0; index < 10; index += 1) {
    recognizer.push({ label: "나", confidence: 0.9 });
  }
  if (recognizer.state().tokens.join(",") !== "나") {
    throw new Error("held pose duplicated");
  }

  for (let index = 0; index < 5; index += 1) {
    recognizer.push({ label: "IDLE", confidence: 0.9 });
  }
  for (let index = 0; index < 10; index += 1) {
    recognizer.push({ label: "아프다", confidence: 0.9 });
  }
  if (recognizer.state().tokens.join(",") !== "나,아프다") {
    throw new Error(`unexpected tokens: ${recognizer.state().tokens}`);
  }

  const armedBeforeSwitch = recognizer.state().armed;
  recognizer.resetWindow({ preserveArmed: true });
  if (recognizer.state().armed !== armedBeforeSwitch) {
    throw new Error("camera switch changed recognition arm state");
  }

  recognizer.undo();
  if (recognizer.state().tokens.join(",") !== "나") throw new Error("undo failed");
  recognizer.clear();
  if (recognizer.state().tokens.length !== 0) throw new Error("clear failed");

  const expectedSentence = ["어제", "기침", "콧물", "열", "가래", "있다"];
  for (const word of expectedSentence) {
    for (let vote = 0; vote < 5; vote += 1) {
      recognizer.push({ label: word, confidence: 0.9 });
    }
    for (let vote = 0; vote < 5; vote += 1) {
      recognizer.push({ label: "IDLE", confidence: 0.9 });
    }
  }
  if (recognizer.state().tokens.join(",") !== expectedSentence.join(",")) {
    throw new Error(`multiword accumulation failed: ${recognizer.state().tokens}`);
  }

  const collector = new IdleSentenceCollector({ idleDurationMs: 3000 });
  const idleState = { stableLabel: "IDLE", tokens: ["나", "아프다"] };
  if (collector.update(idleState, 1000).ready) throw new Error("idle completed early");
  const counting = collector.update(idleState, 3000);
  if (!counting.active || counting.remainingMs !== 1000) {
    throw new Error("idle countdown failed");
  }
  const cancelled = collector.update(
    { stableLabel: "나", tokens: ["나", "아프다"] },
    3500,
  );
  if (cancelled.active || cancelled.ready) throw new Error("idle cancellation failed");
  collector.update(idleState, 4000);
  const ready = collector.update(idleState, 7000);
  if (!ready.ready || ready.payload.words.join(",") !== "나,아프다") {
    throw new Error("sentence payload failed");
  }
  if (ready.payload.source !== "web-on-device-v9") {
    throw new Error(`unexpected source: ${ready.payload.source}`);
  }

  recognizer.clear();
  recognizer.tokens.push("어제", "기침", "새단어");
  const consumed = recognizer.consumePrefix(["어제", "기침"]);
  if (!consumed.consumed || consumed.tokens.join(",") !== "새단어") {
    throw new Error("sent prefix consumption failed");
  }
  const mismatch = recognizer.consumePrefix(["없는단어"]);
  if (mismatch.consumed || mismatch.tokens.join(",") !== "새단어") {
    throw new Error("mismatched prefix was removed");
  }

  result.textContent = JSON.stringify({ status: "PASS", payload: ready.payload });
} catch (error) {
  result.textContent = JSON.stringify({ status: "FAIL", error: error.message });
}
