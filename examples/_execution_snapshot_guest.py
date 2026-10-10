"""Synthetic, credential-free state for the execution-snapshot substrate probe."""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path("/workspace/cayu_snapshot_probe")
PORT = 8087


def serve() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    # This value is generated here and is never written into the guest filesystem.
    token = secrets.token_hex(32)
    counter = 0
    paused = False
    payload = ROOT / "payload.bin"
    payload.write_bytes(secrets.token_bytes(4 * 1024 * 1024))
    (ROOT / "action.json").write_text(json.dumps({"call": "fixture-write", "count": 1}))

    def persist() -> None:
        temporary = ROOT / "counter.tmp"
        with temporary.open("w") as stream:
            json.dump({"counter": counter}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(ROOT / "counter.json")

    persist()

    def write_background() -> None:
        nonlocal counter
        while True:
            with lock:
                if not paused:
                    counter += 1
                    persist()
            time.sleep(0.02)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            with lock:
                body = json.dumps(
                    {"token": token, "counter": counter, "quiescent": paused}
                ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            nonlocal paused
            with lock:
                paused = self.path == "/quiesce"
                # Lock acquisition proves the previous writer has finished and fsynced.
            self.do_GET()

        def log_message(self, format: str, *_args: object) -> None:
            pass

    threading.Thread(target=write_background, daemon=True).start()
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    serve()
