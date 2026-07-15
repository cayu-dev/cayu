from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading

from tests.egress_e2e_support import (
    bloom_maybe_contains,
    connect_probe_script,
    recursive_window_bloom_script,
)


def _serve_tcp_responses(responses: tuple[bytes, ...]) -> tuple[int, list[bool], threading.Thread]:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    accepted: list[bool] = []

    def serve() -> None:
        try:
            for response in responses:
                connection, _address = server.accept()
                accepted.append(True)
                with connection:
                    connection.recv(8192)
                    connection.sendall(response)
        finally:
            server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return server.getsockname()[1], accepted, thread


def test_guest_bloom_scan_detects_secret_without_receiving_it(tmp_path) -> None:
    secret = b"sk_test_secret_only_known_to_host"
    absent = b"sk_test_different_secret_value"
    (tmp_path / "credentials.txt").write_bytes(b"prefix:" + secret + b":suffix")
    script = recursive_window_bloom_script(
        roots=(str(tmp_path),),
        window_size=len(secret),
    )

    assert secret.decode() not in script
    assert secret.hex() not in script

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["files_scanned"] == 1
    assert bloom_maybe_contains(result["bloom"], secret)
    assert not bloom_maybe_contains(result["bloom"], absent)


def test_connect_probe_reports_tcp_connection_when_tls_handshake_fails() -> None:
    port, accepted, thread = _serve_tcp_responses((b"not tls",))
    script = connect_probe_script("127.0.0.1", port, probe_kind="tls")

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    thread.join(timeout=3)
    result = json.loads(completed.stdout)

    assert accepted == [True]
    assert result["tcp_connected"] is True
    assert result["tls_completed"] is False


def test_connect_probe_reports_tcp_connections_when_metadata_returns_403() -> None:
    forbidden = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n"
    port, accepted, thread = _serve_tcp_responses((forbidden, forbidden))

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            connect_probe_script("127.0.0.1", port, probe_kind="metadata"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    thread.join(timeout=3)
    result = json.loads(completed.stdout)

    assert accepted == [True, True]
    assert result["tcp_connected"] is True
    assert result["http_statuses"] == [403, 403]
