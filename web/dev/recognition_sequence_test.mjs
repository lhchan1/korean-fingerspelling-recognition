import assert from "node:assert/strict";
import { IdleSentenceCollector, RecognitionStabilizer } from "../src/recognition.js";
import { requestKoreanSentence } from "../src/kobart-client.js";

const STEP_MS = 250;

function simulate(words, { signSeconds = 3 } = {}) {
  const recognizer = new RecognitionStabilizer();
  const collector = new IdleSentenceCollector({ idleDurationMs: 3000 });
  const requests = [];
  let now = 0;

  function feed(label, count, confidence = 0.95) {
    for (let index = 0; index < count; index += 1) {
      const state = recognizer.push({ label, confidence });
      const sentence = collector.update(state, now);
      if (sentence.ready) requests.push(sentence.payload.words);
      now += STEP_MS;
    }
  }

  for (const word of words) feed(word, (signSeconds * 1000) / STEP_MS);
  assert.deepEqual(recognizer.state().tokens, words, "continuous signs must each add one token");
  assert.equal(requests.length, 0, "no request before terminal IDLE");
  feed("IDLE", 20);
  assert.equal(requests.length, 1, "terminal IDLE must send exactly once");
  assert.deepEqual(requests[0], words, "request must contain the complete variable-length array");
}

simulate(["나", "지금", "조금", "아프다"]);
simulate(["나", "지금", "배", "조금", "아프다"]);
simulate(["어제", "기침", "콧물", "열", "가래", "있다"]);

const earlyRecognizer = new RecognitionStabilizer();
earlyRecognizer.push({ label: "나", confidence: 0.98, early: true });
assert.deepEqual(earlyRecognizer.state().tokens, [], "one early guess must not commit");
earlyRecognizer.push({ label: "나", confidence: 0.96, early: true });
assert.deepEqual(earlyRecognizer.state().tokens, ["나"], "two strong early guesses should capture the first sign");
earlyRecognizer.push({ label: "나", confidence: 0.99, early: true });
assert.deepEqual(earlyRecognizer.state().tokens, ["나"], "early guesses must not duplicate the held first sign");

const weakEarlyRecognizer = new RecognitionStabilizer();
weakEarlyRecognizer.push({ label: "가슴", confidence: 0.91, early: true });
weakEarlyRecognizer.push({ label: "가슴", confidence: 0.8, early: true });
assert.deepEqual(weakEarlyRecognizer.state().tokens, [], "weak early disagreement must not commit");

const recognizer = new RecognitionStabilizer();
function feed(label, count, confidence = 0.95) {
  for (let index = 0; index < count; index += 1) {
    recognizer.push({ label, confidence });
  }
}
feed("나", 10);
feed("지금", 1);
feed("나", 5);
assert.deepEqual(recognizer.state().tokens, ["나"], "brief false label or held sign must not duplicate");
feed("지금", 10);
assert.deepEqual(recognizer.state().tokens, ["나", "지금"], "stable direct transition must append");
feed("OTHER", 5);
feed("지금", 5);
assert.deepEqual(recognizer.state().tokens, ["나", "지금", "지금"], "neutral gap permits repeated word");

const collector = new IdleSentenceCollector({ idleDurationMs: 3000 });
const idleState = { stableLabel: "IDLE", tokens: ["나", "아프다"] };
assert.equal(collector.update(idleState, 1000).ready, false);
assert.equal(collector.update({ stableLabel: "나", tokens: idleState.tokens }, 2500).active, false);
assert.equal(collector.update(idleState, 4000).ready, false);
assert.deepEqual(collector.update(idleState, 7000).payload.words, idleState.tokens);
assert.equal(collector.update({ stableLabel: "IDLE", tokens: [] }, 10000).ready, false);

recognizer.clear();
feed("나", 5);
feed("지금", 5);
assert.equal(recognizer.consumePrefix(["나"]).consumed, true);
assert.deepEqual(recognizer.state().tokens, ["지금"]);

let wirePayload = null;
globalThis.WebSocket = class MockWebSocket {
  static OPEN = 1;
  static CONNECTING = 0;

  constructor() {
    this.readyState = MockWebSocket.CONNECTING;
    queueMicrotask(() => {
      this.readyState = MockWebSocket.OPEN;
      this.onopen?.();
    });
  }

  send(message) {
    wirePayload = JSON.parse(message);
    queueMicrotask(() => this.onmessage?.({
      data: JSON.stringify({
        type: "result",
        session_id: wirePayload.session_id,
        sentence: "테스트 문장",
      }),
    }));
  }

  close() {
    this.readyState = 3;
  }
};
const sixWords = ["어제", "기침", "콧물", "열", "가래", "있다"];
await requestKoreanSentence(sixWords, { url: "wss://mock.invalid/test" });
assert.deepEqual(wirePayload.words, sixWords, "WebSocket payload must preserve all words and their order");
assert.match(wirePayload.session_id, /^web-/);

console.log(JSON.stringify({ status: "PASS", scenarios: [4, 5, 6], test: "continuous-words-idle" }));
