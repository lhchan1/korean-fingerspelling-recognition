const SUPPRESSED_LABELS = new Set(["IDLE", "OTHER"]);
const DEFAULT_IDLE_DURATION_MS = 3000;

export class RecognitionStabilizer {
  constructor({ threshold = 0.75, windowSize = 5, minVotes = 3 } = {}) {
    this.threshold = threshold;
    this.windowSize = windowSize;
    this.minVotes = minVotes;
    this.recent = [];
    this.tokens = [];
    this.armed = true;
    this.lastCommittedLabel = null;
    this.previousEarlyStrongLabel = null;
  }

  push(prediction) {
    const accepted =
      prediction && prediction.confidence >= this.threshold ? prediction.label : null;
    this.recent.push(accepted);
    if (this.recent.length > this.windowSize) this.recent.shift();

    const { label, votes } = this.mostCommon();
    let stableLabel = label !== null && votes >= this.minVotes ? label : null;
    const earlyStrongLabel =
      prediction?.early &&
      this.tokens.length === 0 &&
      prediction.confidence >= 0.9 &&
      !SUPPRESSED_LABELS.has(prediction.label)
        ? prediction.label
        : null;
    const earlyConfirmed =
      !stableLabel &&
      earlyStrongLabel !== null &&
      earlyStrongLabel === this.previousEarlyStrongLabel;
    if (earlyConfirmed) {
      stableLabel = earlyStrongLabel;
    }
    this.previousEarlyStrongLabel = earlyStrongLabel;
    let added = null;

    if (stableLabel && SUPPRESSED_LABELS.has(stableLabel)) {
      this.armed = true;
    } else if (
      stableLabel &&
      (this.armed ||
        (stableLabel !== this.lastCommittedLabel &&
          votes >= Math.min(this.windowSize, this.minVotes + 1)))
    ) {
      // A confirmed change of sign can start the next word without an IDLE/OTHER gap.
      // Require one extra vote on direct changes to reduce transition flicker.
      this.tokens.push(stableLabel);
      this.armed = false;
      this.lastCommittedLabel = stableLabel;
      added = stableLabel;
    }

    return this.state({ accepted, stableLabel, votes, added, requiredVotes: earlyConfirmed ? 2 : this.minVotes });
  }

  mostCommon() {
    const counts = new Map();
    for (const label of this.recent) {
      if (label === null) continue;
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }

    let winner = null;
    let votes = 0;
    for (const [label, count] of counts) {
      if (count > votes) {
        winner = label;
        votes = count;
      }
    }
    return { label: winner, votes };
  }

  undo() {
    const removed = this.tokens.pop() ?? null;
    return this.state({ removed });
  }

  clear() {
    this.recent = [];
    this.tokens = [];
    this.armed = true;
    this.lastCommittedLabel = null;
    this.previousEarlyStrongLabel = null;
    return this.state();
  }

  consumePrefix(expectedTokens) {
    const expected = Array.isArray(expectedTokens) ? expectedTokens : [];
    const matches =
      expected.length > 0 &&
      expected.length <= this.tokens.length &&
      expected.every((token, index) => this.tokens[index] === token);

    if (!matches) return this.state({ consumed: false });
    this.tokens.splice(0, expected.length);
    return this.state({ consumed: true, consumedTokens: [...expected] });
  }

  resetWindow() {
    this.recent = [];
    this.armed = true;
    this.lastCommittedLabel = null;
    this.previousEarlyStrongLabel = null;
    return this.state();
  }

  state(extra = {}) {
    return {
      threshold: this.threshold,
      windowSize: this.windowSize,
      minVotes: this.minVotes,
      recent: [...this.recent],
      tokens: [...this.tokens],
      armed: this.armed,
      ...extra,
    };
  }
}

export class IdleSentenceCollector {
  constructor({ idleDurationMs = DEFAULT_IDLE_DURATION_MS } = {}) {
    this.idleDurationMs = idleDurationMs;
    this.idleStartedAt = null;
  }

  update(recognitionState, timestamp = performance.now()) {
    const tokens = recognitionState?.tokens ?? [];
    const isIdle = recognitionState?.stableLabel === "IDLE";

    if (!isIdle || tokens.length === 0) {
      this.idleStartedAt = null;
      return this.state();
    }

    if (this.idleStartedAt === null) this.idleStartedAt = timestamp;
    const elapsedMs = Math.max(0, timestamp - this.idleStartedAt);
    const remainingMs = Math.max(0, this.idleDurationMs - elapsedMs);

    if (remainingMs > 0) return this.state({ active: true, elapsedMs, remainingMs });

    const payload = {
      words: [...tokens],
      clientCreatedAt: new Date().toISOString(),
      source: "web-on-device",
    };
    this.reset();
    return this.state({ ready: true, payload, elapsedMs, remainingMs: 0 });
  }

  reset() {
    this.idleStartedAt = null;
    return this.state();
  }

  state(extra = {}) {
    return {
      active: this.idleStartedAt !== null,
      ready: false,
      idleDurationMs: this.idleDurationMs,
      elapsedMs: 0,
      remainingMs: this.idleDurationMs,
      ...extra,
    };
  }
}
