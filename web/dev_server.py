#!/usr/bin/env python3
"""Static development server that mirrors browser logs to the terminal."""

from __future__ import annotations

import json
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


HOST = "127.0.0.1"
PORT = 5173


class DevelopmentHandler(SimpleHTTPRequestHandler):
    def copyfile(self, source, outputfile) -> None:
        try:
            shutil.copyfileobj(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            # A browser can cancel a large WASM/model download during refresh.
            pass

    def do_POST(self) -> None:
        if self.path != "/api/log":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            timestamp = payload.get("timestamp", "--:--:--")
            level = str(payload.get("level", "INFO")).ljust(5)
            message = payload.get("message", "")
            detail = payload.get("detail", "")
            suffix = f" · {detail}" if detail else ""
            print(f"[BROWSER {timestamp}] {level} {message}{suffix}", flush=True)
            self.send_response(204)
            self.end_headers()
        except (ValueError, json.JSONDecodeError) as error:
            self.send_error(400, str(error))

    def log_message(self, format: str, *args: object) -> None:
        if self.path != "/api/log":
            super().log_message(format, *args)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), DevelopmentHandler)
    print(f"Capstone web development server: http://localhost:{PORT}", flush=True)
    print("Browser logs will appear below. Stop with Control+C.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDevelopment server stopped.")
    finally:
        server.server_close()
