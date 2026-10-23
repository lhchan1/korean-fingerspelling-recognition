import test from 'node:test';
import assert from 'node:assert/strict';
import { TimingSummary } from '../src/timing-metrics.js';
import { WordSignInference } from '../src/inference.js';
import { BrowserHandLandmarker } from '../src/hand-landmarker.js';
import { requestKoreanSentence } from '../src/kobart-client.js';

let clock = 0;
Object.defineProperty(globalThis, 'performance', { value: { now: () => clock }, configurable: true });

test('summary excludes warmup and reports exact median/p95 with bounded windows', () => {
  const logs = [];
  const metric = new TimingSummary(row => logs.push(row), { warmupSamples: 1, intervalMs: 100 });
  metric.add({ ms: 999 }, 0);
  metric.add({ ms: 2 }, 1);
  metric.add({ ms: 4 }, 51);
  metric.add({ ms: 6 }, 101);
  assert.deepEqual(logs[0].ms, { mean: 4, median: 4, p95: 6, min: 2, max: 6 });
  assert.equal(logs[0].samples, 3);
  assert.equal(logs[0].warmupSamplesExcluded, 1);
  metric.flush(110);
  assert.equal(logs.length, 1);
});

test('model timing includes blocking result read and preserves prediction', async () => {
  clock = 0;
  globalThis.window = { tf: {
    getBackend: () => 'mock', tidy: fn => fn(),
    tensor3d: () => { clock += 2; return {}; },
  } };
  const logs = [];
  const inference = new WordSignInference((message, detail) => logs.push({ message, detail }));
  inference.sequenceLength = 60; inference.featureCount = 146;
  inference.labels = Array.from({length: 24}, (_, i) => `label-${i}`);
  inference.model = { predict: () => {
    clock += 5;
    return { dataSync: () => { clock += 7; return Float32Array.from({ length: 24 }, (_, i) => i === 3 ? 1 : 0); } };
  } };
  let result;
  for (let i = 0; i < 63; i++) result = await inference.addFrame(Array(146).fill(0), (i + 1) * 300);
  inference.timings.flush();
  assert.equal(result.prediction.label, 'label-3');
  assert.equal(inference.buffer.length, 60);
  const report = logs.at(-1).detail;
  assert.equal(report.samples, 1);
  assert.equal(report.modelMs.mean, 12);
  assert.equal(report.inputPrepMs.mean, 2);
  assert.equal(report.totalMs.mean, 14);
});

test('MediaPipe measures hand + pose separately from drawing and callbacks', () => {
  clock = 0;
  globalThis.HTMLMediaElement = { HAVE_CURRENT_DATA: 2 };
  globalThis.requestAnimationFrame = () => 1;
  const logs = [];
  const video = { readyState: 2, currentTime: 0 };
  const detector = new BrowserHandLandmarker(video, { getContext: () => ({}) }, () => { clock += 50; }, (m,d) => logs.push({m,d}));
  detector.landmarker = { detectForVideo: () => { clock += 4; return {landmarks: []}; } };
  detector.poseLandmarker = { detectForVideo: () => { clock += 6; return {landmarks: []}; } };
  detector.resizeCanvas = () => {};
  detector.draw = () => { clock += 20; };
  for (let i = 0; i < 6; i++) { clock += 100; video.currentTime++; detector.predict(); }
  detector.timings.flush();
  const report = logs.find(row => row.m === 'MediaPipe 처리 시간 요약').d;
  assert.equal(report.samples, 1);
  assert.equal(report.handMs.mean, 4);
  assert.equal(report.poseMs.mean, 6);
  assert.equal(report.landmarkMs.mean, 10);
});

class MockSocket {
  static OPEN = 1; static CONNECTING = 0;
  constructor() { MockSocket.last = this; this.readyState = 0; }
  send(payload) { this.payload = JSON.parse(payload); }
  close() { this.readyState = 3; this.closed = true; }
  open() { this.readyState = 1; this.onopen(); }
  message(payload) { this.onmessage({data: JSON.stringify(payload)}); }
}
globalThis.WebSocket = MockSocket;

test('WSS separates connect and send-to-response, ignores unmatched responses', async () => {
  clock = 100;
  const logs = [];
  const promise = requestKoreanSentence(['머리', '아프다'], { onTiming: r => logs.push(r) });
  const socket = MockSocket.last;
  clock = 120; socket.open();
  clock = 140; socket.message({ type: 'result', session_id: 'wrong', sentence: 'ignored' });
  assert.equal(logs.length, 0);
  clock = 220; socket.message({ type: 'result', session_id: socket.payload.session_id, sentence: ' 머리가 아파요. ' });
  const response = await promise;
  assert.equal(response.sentence, '머리가 아파요.');
  assert.equal(logs[0].connectMs, 20);
  assert.equal(logs[0].sendToResponseMs, 100);
  assert.equal(logs[0].requestResponseMs, 120);
  assert.equal(logs[0].serverGenerationMs, null);
  assert.equal(logs[0].wordCount, 2);
  assert.equal(socket.closed, true);
});

test('failed request emits failure timing without a fabricated response time', async () => {
  clock = 0;
  let timing;
  const promise = requestKoreanSentence(['약'], { timeoutMs: 5, onTiming: r => timing = r });
  await assert.rejects(promise, /완료되지/);
  assert.equal(timing.status, 'error');
  assert.equal(timing.requestResponseMs, null);
  assert.equal(timing.sendToResponseMs, null);
  assert.equal(MockSocket.last.closed, true);
});
