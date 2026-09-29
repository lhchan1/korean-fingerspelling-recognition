const MODEL_VERSION = "word-lstm-v7";
const MODEL_URL = `/public/models/word-sign/model.json?v=${MODEL_VERSION}`;
const CONFIG_URL = `/public/models/word-sign/config.json?v=${MODEL_VERSION}`;
const LABELS_URL = `/public/models/word-sign/labels.json?v=${MODEL_VERSION}`;

export class WordSignInference {
  constructor(onLog) {
    this.onLog = onLog;
    this.model = null;
    this.labels = [];
    this.sequenceLength = 0;
    this.featureCount = 0;
    this.buffer = [];
    this.lastPredictionAt = 0;
    this.predictIntervalMs = 250;
  }

  async initialize() {
    if (!window.tf) throw new Error("TensorFlow.js 런타임을 찾을 수 없습니다.");
    this.log("TensorFlow.js 초기화", { version: window.tf.version.tfjs });

    const [configResponse, labelsResponse] = await Promise.all([
      fetch(CONFIG_URL),
      fetch(LABELS_URL),
    ]);
    if (!configResponse.ok || !labelsResponse.ok) {
      throw new Error("모델 config 또는 labels 파일을 불러오지 못했습니다.");
    }

    const config = await configResponse.json();
    this.labels = await labelsResponse.json();
    this.sequenceLength = Number(config.sequence_length);
    this.featureCount = Number(config.feature_count);

    if (this.sequenceLength !== 60 || this.featureCount !== 146) {
      throw new Error(
        `지원하지 않는 모델 입력: ${this.sequenceLength}×${this.featureCount}`,
      );
    }
    if (this.labels.length !== 24) {
      throw new Error(`라벨 개수가 24개가 아닙니다: ${this.labels.length}`);
    }

    this.log("24클래스 모델 파일 로딩 시작");
    this.model = await window.tf.loadLayersModel(MODEL_URL);
    const inputShape = this.model.inputs[0].shape;
    const outputShape = this.model.outputs[0].shape;

    if (inputShape[1] !== 60 || inputShape[2] !== 146 || outputShape[1] !== 24) {
      throw new Error(
        `모델 shape 불일치: input=${inputShape}, output=${outputShape}`,
      );
    }

    // First inference compiles the selected browser backend.
    window.tf.tidy(() => {
      const warmup = window.tf.zeros([1, 60, 146], "float32");
      this.model.predict(warmup);
    });
    this.log("24클래스 모델 준비 완료", { inputShape, outputShape });
  }

  reset() {
    this.buffer = [];
    this.lastPredictionAt = 0;
  }

  async addFrame(features, timestamp = performance.now()) {
    if (!this.model) return { progress: 0, ready: false };
    if (features.length !== this.featureCount) {
      throw new Error(`특징 개수 불일치: ${features.length}/${this.featureCount}`);
    }
    if (!features.every(Number.isFinite)) {
      throw new Error("특징에 NaN 또는 Infinity가 포함되어 있습니다.");
    }

    this.buffer.push(Float32Array.from(features));
    if (this.buffer.length > this.sequenceLength) this.buffer.shift();

    const state = {
      progress: this.buffer.length / this.sequenceLength,
      size: this.buffer.length,
      ready: this.buffer.length === this.sequenceLength,
    };
    if (!state.ready || timestamp - this.lastPredictionAt < this.predictIntervalMs) {
      return state;
    }

    this.lastPredictionAt = timestamp;
    const probabilities = window.tf.tidy(() => {
      const flattened = new Float32Array(this.sequenceLength * this.featureCount);
      this.buffer.forEach((frame, index) => {
        flattened.set(frame, index * this.featureCount);
      });
      const input = window.tf.tensor3d(
        flattened,
        [1, this.sequenceLength, this.featureCount],
        "float32",
      );
      return Array.from(this.model.predict(input).dataSync());
    });

    const ranked = probabilities
      .map((confidence, id) => ({ label: this.labels[id], confidence, id }))
      .sort((a, b) => b.confidence - a.confidence);
    return { ...state, prediction: ranked[0], top: ranked.slice(0, 3) };
  }

  log(message, detail) {
    this.onLog?.(message, detail);
  }
}
