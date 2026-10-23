// Browser wall-clock timings, not device-independent benchmark results.
export const TIMING_VERSION = "browser-timing-v1";

export class TimingSummary {
  constructor(onReport, { warmupSamples = 0, intervalMs = 5000 } = {}) {
    this.onReport = onReport;
    this.warmupSamples = warmupSamples;
    this.intervalMs = intervalMs;
    this.reset();
  }

  reset() {
    this.seen = 0;
    this.rows = [];
    this.startedAt = null;
  }

  add(row, now = performance.now()) {
    if (!Object.values(row).every(value => Number.isFinite(value) && value >= 0)) return;
    this.seen += 1;
    if (this.seen <= this.warmupSamples) return;
    if (this.startedAt === null) this.startedAt = now;
    this.rows.push(row);
    if (now - this.startedAt >= this.intervalMs || this.rows.length >= 512) this.flush(now);
  }

  flush(now = performance.now()) {
    if (!this.rows.length) return;
    const report = {
      timingVersion: TIMING_VERSION,
      samples: this.rows.length,
      warmupSamplesExcluded: Math.min(this.seen, this.warmupSamples),
      windowMs: Math.round(now - this.startedAt),
    };
    for (const key of Object.keys(this.rows[0])) {
      const values = this.rows.map(row => row[key]).sort((a, b) => a - b);
      const n = values.length;
      const round = value => Math.round(value * 100) / 100;
      report[key] = {
        mean: round(values.reduce((a, b) => a + b, 0) / n),
        median: round((values[Math.floor((n - 1) / 2)] + values[Math.floor(n / 2)]) / 2),
        p95: round(values[Math.ceil(n * 0.95) - 1]),
        min: round(values[0]), max: round(values[n - 1]),
      };
    }
    this.rows = [];
    this.startedAt = null;
    this.onReport(report);
  }
}
