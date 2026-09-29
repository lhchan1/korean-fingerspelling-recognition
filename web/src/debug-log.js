const MAX_LINES = 120;

export class DebugLogger {
  constructor(outputElement) {
    this.output = outputElement;
    this.lines = [];
  }

  info(message, detail) {
    this.write("INFO", message, detail);
  }

  warn(message, detail) {
    this.write("WARN", message, detail);
  }

  error(message, detail) {
    this.write("ERROR", message, detail);
  }

  clear() {
    this.lines = [];
    this.render();
  }

  write(level, message, detail) {
    const timestamp = new Date().toLocaleTimeString("ko-KR", { hour12: false });
    const suffix = detail == null ? "" : ` · ${this.stringify(detail)}`;
    const line = `[${timestamp}] ${level.padEnd(5)} ${message}${suffix}`;

    this.lines.push(line);
    if (this.lines.length > MAX_LINES) this.lines.shift();
    this.render();

    const consoleMethod = level === "ERROR" ? "error" : level === "WARN" ? "warn" : "log";
    console[consoleMethod](line);
    this.sendToTerminal({ timestamp, level, message, detail: this.stringify(detail) });
  }

  stringify(value) {
    if (value == null) return "";
    if (value instanceof Error) return `${value.name}: ${value.message}`;
    if (typeof value === "string") return value;
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }

  render() {
    this.output.textContent = this.lines.join("\n") || "아직 기록된 로그가 없습니다.";
    this.output.scrollTop = this.output.scrollHeight;
  }

  sendToTerminal(payload) {
    const isLocalDevelopment = ["localhost", "127.0.0.1", "::1"].includes(
      window.location.hostname,
    );
    if (!isLocalDevelopment) return;

    fetch("/api/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      keepalive: true,
    }).catch(() => {
      // dev_server.py가 없으면 화면과 브라우저 콘솔 로그만 사용합니다.
    });
  }
}
