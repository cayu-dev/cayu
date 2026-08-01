"""Exercise ``cayu serve`` through an installed wheel and scaffolded project."""

from __future__ import annotations

import base64
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cayu-serve-wheel-") as temporary:
        root = Path(temporary)
        scaffold = subprocess.run(
            [sys.executable, "-m", "cayu", "new", "proof", "--dir", str(root)],
            check=False,
            capture_output=True,
            text=True,
        )
        if scaffold.returncode != 0:
            raise RuntimeError(scaffold.stdout + scaffold.stderr)
        project = root / "proof"
        with (project / "pyproject.toml").open("a", encoding="utf-8") as config:
            config.write('\n[tool.cayu.serve]\nauth = "server_auth:AUTH"\n')
        (project / "server_auth.py").write_text(
            """import os

from cayu.server import BasicAuth

AUTH = BasicAuth(
    username="wheel-operator",
    password=os.environ["CAYU_WHEEL_SMOKE_PASSWORD"],
)
""",
            encoding="utf-8",
        )
        nested = project / "nested" / "directory"
        nested.mkdir(parents=True)
        environment = os.environ.copy()
        environment["CAYU_WHEEL_SMOKE_PASSWORD"] = "wheel-secret-password"
        process, port = _start_server(nested, environment)
        try:
            _require_status(f"http://127.0.0.1:{port}/api/sessions", expected=401)
            token = base64.b64encode(b"wheel-operator:wheel-secret-password").decode()
            _require_status(
                f"http://127.0.0.1:{port}/api/sessions",
                expected=200,
                headers={"Authorization": f"Basic {token}"},
            )
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)
            allowed_exits = {0, -signal.SIGTERM, 128 + signal.SIGTERM}
            shutdown_markers = (
                "Application shutdown complete",
                "Finished server process",
            )
            if (
                process.returncode not in allowed_exits
                or any(marker not in stderr for marker in shutdown_markers)
                or "Traceback (most recent call last)" in stderr
            ):
                raise RuntimeError(f"cayu serve exited {process.returncode}\n{stdout}{stderr}")
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
    print("built-wheel cayu serve smoke passed")
    return 0


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _start_server(
    nested: Path,
    environment: dict[str, str],
) -> tuple[subprocess.Popen[str], int]:
    for attempt in range(5):
        port = _available_port()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "cayu",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=nested,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_health(process, port)
        except RuntimeError as exc:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
            if "address already in use" in str(exc).lower() and attempt < 4:
                continue
            raise
        return process, port
    raise AssertionError("server startup retry loop exhausted")


def _wait_for_health(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + 10
    url = f"http://127.0.0.1:{port}/api/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"cayu serve exited before health check\n{stdout}{stderr}")
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200 and response.read():
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    raise TimeoutError(f"cayu serve did not become healthy at {url}")


def _require_status(
    url: str,
    *,
    expected: int,
    headers: dict[str, str] | None = None,
) -> None:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            status = response.status
            response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.read()
    if status != expected:
        raise AssertionError(f"{url} returned {status}, expected {expected}")


if __name__ == "__main__":
    raise SystemExit(main())
