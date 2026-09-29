const SUPPRESSED_LABELS = new Set(["IDLE", "OTHER"]);

export class RecognitionStabilizer {
  constructor({ threshold = 0.75, windowSize = 5, minVotes = 3, maxTokens = 12 } = {}) {
    this.threshold = threshold;
    this.windowSize = windowSize;
    this.minVotes = minVotes;
    this.maxTokens = maxTokens;
    this.recent = [];
    this.tokens = [];
    this.armed = true;
  }

  push(prediction) {
    const accepted =
      prediction && prediction.confidence >= this.threshold ? prediction.label : null;
    this.recent.push(accepted);
    if (this.recent.length > this.windowSize) this.recent.shift();

    const { label, votes } = this.mostCommon();
    const stableLabel = label !== null && votes >= this.minVotes ? label : null;
    let added = null;

    if (stableLabel && SUPPRESSED_LABELS.has(stableLabel)) {
      this.armed = true;
    } else if (stableLabel && this.armed) {
      this.tokens.push(stableLabel);
      if (this.tokens.length > this.maxTokens) this.tokens.shift();
      this.armed = false;
      added = stableLabel;
    }

    return this.state({ accepted, stableLabel, votes, added });
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
    return this.state();
  }

  resetWindow() {
    this.recent = [];
    this.armed = true;
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

export { SUPPRESSED_LABELS };
