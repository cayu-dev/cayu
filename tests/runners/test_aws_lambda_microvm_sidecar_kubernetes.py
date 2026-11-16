"""Sidecar Kubernetes mode: a workspace supervised inside a sandbox pod."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import errno
import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from examples.aws.lambda_microvm_sidecar import supervisor as supervisor_module
from examples.aws.lambda_microvm_sidecar import token_auth
from examples.aws.lambda_microvm_sidecar.remote_processes import RemoteProcesses
from examples.aws.lambda_microvm_sidecar.supervisor import LocalProcesses
from tests.runners.lambda_microvm_harness import write_sidecar_tls
from tests.runners.sidecar_executor_harness import InProcessExecutor, agent_namespace_unavailable

from cayu.runners.aws_lambda_microvm import LAMBDA_MICROVM_PROTOCOL_VERSION

_APP = "examples.aws.lambda_microvm_sidecar.app"
_ENTRYPOINT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "aws"
    / "lambda_microvm_sidecar"
    / "entrypoint.sh"
)
_TOKEN = "t" * 40
_NAMESPACE_UNAVAILABLE = agent_namespace_unavailable()
_needs_agent_namespace = pytest.mark.skipif(
    _NAMESPACE_UNAVAILABLE is not None, reason=str(_NAMESPACE_UNAVAILABLE)
)
_LAMBDA_ONLY_ROUTES = {
    "/aws/lambda-microvms/runtime/v1/ready",
    "/aws/lambda-microvms/runtime/v1/run",
    "/aws/lambda-microvms/runtime/v1/resume",
    "/aws/lambda-microvms/runtime/v1/suspend",
    "/aws/lambda-microvms/runtime/v1/terminate",
    "/v1/owner/lifecycle",
    "/v1/owner/lifecycle/release",
}


def _write_token(path: Path, token: str = _TOKEN, mode: int = 0o600) -> Path:
    path.write_text(f"{token}\n", encoding="ascii")
    path.chmod(mode)
    return path


@pytest.fixture
def token_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Owned by the test user, as by the sidecar's own non-root user.
    path = _write_token(tmp_path / "token")
    monkeypatch.setenv("CAYU_SIDECAR_TOKEN_FILE", str(path))
    certificate, key = write_sidecar_tls(tmp_path / "tls")
    monkeypatch.setenv("CAYU_SIDECAR_TLS_CERT_FILE", str(certificate))
    monkeypatch.setenv("CAYU_SIDECAR_TLS_KEY_FILE", str(key))
    return path


@pytest.fixture
def executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[InProcessExecutor]:
    """The agent container's executor, which a Kubernetes-mode sidecar runs commands through."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("CAYU_MICROVM_WORKSPACE_ROOT", str(workspace))
    served = InProcessExecutor(workspace)
    monkeypatch.setenv("CAYU_EXECUTOR_SOCKET", served.address)
    try:
        yield served
    finally:
        served.close()


def _boot_id(module: ModuleType, executor: InProcessExecutor) -> str:
    return str(uuid.uuid5(uuid.UUID(module.BOOT_ID), executor.executor.boot_id))


@pytest.fixture
def load_app(
    monkeypatch: pytest.MonkeyPatch, token_file: Path, executor: InProcessExecutor
) -> Iterator:
    loaded: list[ModuleType] = []
    # A Kubernetes-mode sidecar holds its egress relay port from the start; tests
    # take a free one, so they can run beside anything else on the host.
    monkeypatch.setattr(supervisor_module, "_AGENT_PROXY_RELAY_PORT", 0)

    def load(mode: str | None) -> ModuleType:
        if mode is None:
            monkeypatch.delenv("CAYU_SIDECAR_MODE", raising=False)
        else:
            monkeypatch.setenv("CAYU_SIDECAR_MODE", mode)
        sys.modules.pop(_APP, None)
        module = importlib.import_module(_APP)
        loaded.append(module)
        return module

    try:
        yield load
    finally:
        for module in loaded:
            module.SUPERVISOR.cancel_all()
            if module.EXECUTION_BOUNDARY.loopback_relay:
                module.EXECUTION_BOUNDARY._relay.close()
        sys.modules.pop(_APP, None)


def _paths(module: ModuleType) -> set[str]:
    return {route.path for route in module.app.routes}


def _get(
    module: ModuleType, path: str, *, authorization: str | None = None, scheme: str = "https"
) -> httpx.Response:
    async def call() -> httpx.Response:
        headers = {} if authorization is None else {"Authorization": authorization}
        transport = httpx.ASGITransport(app=module.app)
        base_url = f"{scheme}://sidecar"
        async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
            return await client.get(path, headers=headers)

    return asyncio.run(call())


def _post(module: ModuleType, path: str, payload: dict) -> httpx.Response:
    async def call() -> httpx.Response:
        transport = httpx.ASGITransport(app=module.app)
        async with httpx.AsyncClient(transport=transport, base_url="https://sidecar") as client:
            return await client.post(
                path, json=payload, headers={"Authorization": f"Bearer {_TOKEN}"}
            )

    return asyncio.run(call())


def test_lambda_mode_is_the_default_and_keeps_its_boundary(
    load_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAYU_MICROVM_AGENT_NETNS", "cayu-agent")
    module = load_app(None)

    assert module.SIDECAR_MODE == "lambda"
    assert module.PROCESSES is None
    assert type(module.SUPERVISOR.processes) is LocalProcesses
    assert (module.EXECUTION_BOUNDARY.agent_uid, module.EXECUTION_BOUNDARY.agent_gid) == (
        1000,
        1000,
    )
    assert module.EXECUTION_BOUNDARY.agent_netns == "cayu-agent"
    assert module.EXECUTION_BOUNDARY.isolates_process_tree(execution_profile="agent")
    assert module.FORWARDS.listen_host == "192.0.2.1"
    assert _paths(module) >= _LAMBDA_ONLY_ROUTES
    assert asyncio.run(module.health()) == {
        "status": "ok",
        "protocol_version": LAMBDA_MICROVM_PROTOCOL_VERSION,
        "capabilities": ["owner_epoch", "carry_commands"],
    }
    assert load_app("lambda").SIDECAR_MODE == "lambda"


def test_kubernetes_mode_runs_commands_through_the_executor(
    load_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The image sets the Lambda namespace; Kubernetes mode must ignore it.
    monkeypatch.setenv("CAYU_MICROVM_AGENT_NETNS", "cayu-agent")
    module = load_app("kubernetes")

    assert module.SIDECAR_MODE == "kubernetes"
    assert module.SUPERVISOR.processes is module.PROCESSES
    assert isinstance(module.PROCESSES, RemoteProcesses)
    # No setpriv: the executor already runs as the agent container's user.
    assert module.EXECUTION_BOUNDARY.agent_uid is None
    assert module.EXECUTION_BOUNDARY.agent_gid is None
    assert module.EXECUTION_BOUNDARY.agent_netns is None
    assert not module.EXECUTION_BOUNDARY.isolates_process_tree(execution_profile="agent")
    assert module.FORWARDS.listen_host == "127.0.0.1"


def test_kubernetes_mode_omits_lambda_hooks_and_lifecycle_leases(load_app) -> None:
    lambda_paths = _paths(load_app("lambda"))
    kubernetes_paths = _paths(load_app("kubernetes"))

    assert not _LAMBDA_ONLY_ROUTES & kubernetes_paths
    assert kubernetes_paths == lambda_paths - _LAMBDA_ONLY_ROUTES - {"/openapi.json"}
    assert {"/health", "/v1/owner", "/v1/owner/check", "/v1/commands"} <= kubernetes_paths


@pytest.mark.parametrize("mode", ["", "Lambda", "k8s", "kubernetes "])
def test_unknown_sidecar_mode_refuses_to_start(load_app, mode: str) -> None:
    with pytest.raises(RuntimeError, match="CAYU_SIDECAR_MODE"):
        load_app(mode)


def test_lambda_mode_serves_without_a_token(load_app, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CAYU_SIDECAR_TOKEN_FILE",
        "CAYU_SIDECAR_TLS_CERT_FILE",
        "CAYU_SIDECAR_TLS_KEY_FILE",
    ):
        monkeypatch.delenv(name)
    module = load_app("lambda")

    assert module.HOST_TOKEN is None
    assert module.HOST_TLS is None
    response = _get(module, "/health", scheme="http")
    assert response.status_code == 200
    assert response.json()["capabilities"] == ["owner_epoch", "carry_commands"]
    assert _get(module, "/openapi.json", scheme="http").status_code == 200


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Bearer", "Bearer ", f"Basic {_TOKEN}", "Bearer " + "x" * 40, f"Bearer {_TOKEN}x"],
)
def test_kubernetes_mode_rejects_requests_without_the_host_token(
    load_app, authorization: str | None
) -> None:
    module = load_app("kubernetes")

    for path in ("/health", "/v1/commands/missing"):
        response = _get(module, path, authorization=authorization)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"


def test_kubernetes_mode_serves_the_host_token(load_app, executor: InProcessExecutor) -> None:
    module = load_app("kubernetes")

    for scheme in ("Bearer", "bearer"):
        response = _get(module, "/health", authorization=f"{scheme} {_TOKEN}")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "protocol_version": LAMBDA_MICROVM_PROTOCOL_VERSION,
            "capabilities": [
                "owner_epoch",
                "carry_commands",
                "bearer_auth",
                "boot_id",
                "egress_relay",
            ],
            "boot_id": _boot_id(module, executor),
        }
    assert _get(module, "/v1/commands/missing", authorization=f"Bearer {_TOKEN}").status_code == 404
    # The unauthenticated schema route is not served.
    assert _get(module, "/openapi.json", authorization=f"Bearer {_TOKEN}").status_code == 404


def test_kubernetes_mode_refuses_a_request_that_did_not_arrive_over_tls(load_app) -> None:
    module = load_app("kubernetes")

    response = _get(module, "/health", authorization=f"Bearer {_TOKEN}", scheme="http")
    assert response.status_code == 401


@pytest.mark.parametrize("unset", ["CAYU_SIDECAR_TLS_CERT_FILE", "CAYU_SIDECAR_TLS_KEY_FILE"])
def test_kubernetes_mode_requires_a_tls_certificate_and_key(
    load_app, monkeypatch: pytest.MonkeyPatch, unset: str
) -> None:
    monkeypatch.delenv(unset)

    with pytest.raises(
        RuntimeError, match="CAYU_SIDECAR_TLS_CERT_FILE and CAYU_SIDECAR_TLS_KEY_FILE"
    ):
        load_app("kubernetes")


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        (0o604, "accessible to others"),
        (0o644, "accessible to others"),
        (0o601, "accessible to others"),
        (0o620, "writable by its group"),
    ],
)
def test_kubernetes_mode_refuses_a_tls_key_others_can_reach(
    load_app, mode: int, message: str
) -> None:
    Path(os.environ["CAYU_SIDECAR_TLS_KEY_FILE"]).chmod(mode)

    with pytest.raises(token_auth.TokenFileError, match=f"TLS key file .* {message}"):
        load_app("kubernetes")


@pytest.mark.parametrize("mode", [0o400, 0o440, 0o640])
def test_kubernetes_mode_accepts_a_key_and_token_its_group_may_read(
    load_app, token_file: Path, mode: int
) -> None:
    # A Secret volume with an fsGroup makes them group-readable; the agent is in another container.
    Path(os.environ["CAYU_SIDECAR_TLS_KEY_FILE"]).chmod(mode)
    token_file.chmod(mode)

    assert load_app("kubernetes").HOST_TOKEN is not None


@pytest.mark.skipif(os.geteuid() == 0, reason="root owns the files it writes")
def test_tls_key_file_must_be_owned_by_root_or_the_sidecar_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _certificate, key = write_sidecar_tls(tmp_path)
    token_auth.check_private_key_file(str(key))
    other = os.geteuid() + 1
    monkeypatch.setattr(token_auth.os, "geteuid", lambda: other)

    with pytest.raises(
        token_auth.TokenFileError,
        match=f"TLS key file .* owned by uid 0 or by the sidecar's uid {other}",
    ):
        token_auth.check_private_key_file(str(key))


def test_kubernetes_mode_refuses_a_certificate_that_is_not_its_keys(
    load_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    other, _other_key = write_sidecar_tls(tmp_path / "other")
    monkeypatch.setenv("CAYU_SIDECAR_TLS_CERT_FILE", str(other))

    with pytest.raises(token_auth.TokenFileError, match="do not load as a pair"):
        load_app("kubernetes")


def test_kubernetes_mode_picks_up_a_token_rotated_on_disk(load_app, token_file: Path) -> None:
    module = load_app("kubernetes")
    original = token_file.stat()
    rotated = "r" * 40

    # Unchanged modification time: the file is not read again.
    _write_token(token_file, rotated)
    os.utime(token_file, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert _get(module, "/health", authorization=f"Bearer {rotated}").status_code == 401
    assert _get(module, "/health", authorization=f"Bearer {_TOKEN}").status_code == 200

    # A changed modification time revokes the earlier token before anyone presents the new one.
    os.utime(token_file, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
    assert _get(module, "/health", authorization=f"Bearer {_TOKEN}").status_code == 401
    assert _get(module, "/health", authorization=f"Bearer {rotated}").status_code == 200
    assert _get(module, "/health", authorization=f"Bearer {_TOKEN}").status_code == 401


def test_kubernetes_mode_keeps_its_token_when_a_rotated_file_is_unsafe(
    load_app, token_file: Path
) -> None:
    module = load_app("kubernetes")
    original = token_file.stat()

    _write_token(token_file, "short")
    os.utime(token_file, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
    assert _get(module, "/health", authorization="Bearer short").status_code == 401
    assert _get(module, "/health", authorization=f"Bearer {_TOKEN}").status_code == 200


def test_kubernetes_mode_requires_a_token_file(
    load_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("CAYU_SIDECAR_TOKEN_FILE")
    monkeypatch.setenv("CAYU_SIDECAR_TOKEN", _TOKEN)
    with pytest.raises(RuntimeError, match="CAYU_SIDECAR_TOKEN_FILE"):
        load_app("kubernetes")

    monkeypatch.setenv("CAYU_SIDECAR_TOKEN_FILE", str(tmp_path / "absent"))
    with pytest.raises(token_auth.TokenFileError, match="cannot open"):
        load_app("kubernetes")

    monkeypatch.setenv("CAYU_SIDECAR_TOKEN_FILE", str(tmp_path))
    with pytest.raises(token_auth.TokenFileError, match="not a regular file"):
        load_app("kubernetes")


@pytest.mark.parametrize(
    ("token", "mode", "message"),
    [
        ("t" * 31, 0o600, "at least 32 bytes"),
        ("", 0o600, "at least 32 bytes"),
        ("t" * (token_auth.MAX_TOKEN_FILE_BYTES + 1), 0o600, "too large"),
        (_TOKEN, 0o660, "writable by its group"),
        (_TOKEN, 0o604, "accessible to others"),
        (_TOKEN, 0o602, "accessible to others"),
    ],
)
def test_kubernetes_mode_refuses_an_unsafe_token_file(
    load_app, token_file: Path, token: str, mode: int, message: str
) -> None:
    _write_token(token_file, token, mode)

    with pytest.raises(token_auth.TokenFileError, match=message):
        load_app("kubernetes")


@pytest.mark.skipif(os.geteuid() == 0, reason="root owns the files it writes")
def test_kubernetes_mode_refuses_a_token_file_another_user_owns(
    load_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = os.geteuid() + 1
    monkeypatch.setattr(token_auth.os, "geteuid", lambda: other)

    with pytest.raises(token_auth.TokenFileError, match="owned by uid 0 or by the sidecar's uid"):
        load_app("kubernetes")


def test_kubernetes_mode_accepts_token_and_key_files_root_owns(
    load_app, monkeypatch: pytest.MonkeyPatch, token_file: Path
) -> None:
    # A Secret volume's files belong to root, not to the sidecar's own user.
    real_fstat = os.fstat

    def root_owned(descriptor: int) -> os.stat_result:
        status = real_fstat(descriptor)
        fields = list(status[:10])
        fields[4] = token_auth.ROOT_UID
        return os.stat_result(fields, {"st_mtime_ns": status.st_mtime_ns})

    monkeypatch.setattr(token_auth.os, "fstat", root_owned)
    monkeypatch.setattr(token_auth.os, "geteuid", lambda: os.getuid() + 1)

    load_app("kubernetes")
    assert token_auth.BearerToken(str(token_file)).accepts(_TOKEN.encode("ascii"))


def test_kubernetes_boot_id_combines_the_supervisor_and_executor_boots(
    load_app, executor: InProcessExecutor
) -> None:
    module = load_app("kubernetes")
    boot_id = asyncio.run(module.health())["boot_id"]

    assert str(uuid.UUID(boot_id)) == boot_id == _boot_id(module, executor)
    assert boot_id not in {module.BOOT_ID, executor.executor.boot_id}
    assert asyncio.run(module.health())["boot_id"] == boot_id
    assert _get(module, "/health", authorization=f"Bearer {_TOKEN}").json()["boot_id"] == boot_id

    # A supervisor restart changes it.
    restarted = load_app("kubernetes")
    assert asyncio.run(restarted.health())["boot_id"] not in {boot_id, restarted.BOOT_ID}

    # So does an executor restart, or a renewed agent namespace, under the same supervisor.
    executor.executor.boot_id = str(uuid.uuid4())
    assert asyncio.run(module.health())["boot_id"] not in {boot_id, _boot_id(restarted, executor)}
    assert asyncio.run(module.health())["boot_id"] == _boot_id(module, executor)


def test_kubernetes_health_fails_while_the_executor_is_unreachable(
    load_app, executor: InProcessExecutor
) -> None:
    module = load_app("kubernetes")
    executor.close()

    response = _get(module, "/health", authorization=f"Bearer {_TOKEN}")
    assert response.status_code == 503
    assert "command executor" in response.json()["detail"]


def test_lambda_health_has_no_boot_id(load_app) -> None:
    module = load_app("lambda")

    assert module.BOOT_ID is None
    assert asyncio.run(module.health()) == {
        "status": "ok",
        "protocol_version": LAMBDA_MICROVM_PROTOCOL_VERSION,
        "capabilities": ["owner_epoch", "carry_commands"],
    }


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_kubernetes_entrypoint_starts_only_the_sidecar(tmp_path: Path) -> None:
    stubs = tmp_path / "bin"
    stubs.mkdir()
    calls = tmp_path / "calls"
    for name in ("ip", "iptables", "python3.11"):
        stub = stubs / name
        stub.write_text(f'#!/bin/sh\necho "{name} $*" >> "{calls}"\n', encoding="utf-8")
        stub.chmod(0o755)

    environment = {
        "PATH": f"{stubs}:/usr/bin:/bin",
        "CAYU_SIDECAR_MODE": "kubernetes",
        "CAYU_MICROVM_AGENT_NETNS": "cayu-agent",
        "CAYU_SIDECAR_TLS_CERT_FILE": "/var/run/cayu-sidecar/tls.crt",
        "CAYU_SIDECAR_TLS_KEY_FILE": "/var/run/cayu-sidecar/tls.key",
    }
    subprocess.run(["bash", str(_ENTRYPOINT)], env=environment, timeout=30, check=True)

    # No namespace, veth or firewall setup before the sidecar starts, and only HTTPS.
    assert calls.read_text(encoding="utf-8").splitlines() == [
        "python3.11 -m uvicorn lambda_microvm_sidecar.app:app --host 0.0.0.0 --port 8080 "
        "--ssl-certfile /var/run/cayu-sidecar/tls.crt --ssl-keyfile /var/run/cayu-sidecar/tls.key"
    ]

    for unset in ("CAYU_SIDECAR_TLS_CERT_FILE", "CAYU_SIDECAR_TLS_KEY_FILE"):
        calls.unlink(missing_ok=True)
        refused = subprocess.run(
            ["bash", str(_ENTRYPOINT)],
            env={key: value for key, value in environment.items() if key != unset},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert refused.returncode != 0
        assert f"Kubernetes mode requires {unset}" in refused.stderr
        assert not calls.exists()


def test_kubernetes_mode_runs_a_command_in_the_executor(load_app) -> None:
    module = load_app("kubernetes")
    claim = "c" * 40
    assert _post(module, "/v1/owner", {"claim_id": claim}).status_code == 200
    started = _post(
        module,
        "/v1/commands",
        {
            "command_id": "echo",
            "owner_claim": claim,
            "kind": "shell",
            "shell": "printf ran",
            "cwd": module.ROOT,
        },
    )
    assert started.status_code == 202
    deadline = time.monotonic() + 10
    while (result := module.SUPERVISOR.get("echo"))["state"] in {"accepted", "running"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert (result["exit_code"], base64.b64decode(result["stdout_base64"])) == (0, b"ran")


def test_kubernetes_mode_refuses_commands_while_the_executor_is_unreachable(
    load_app, executor: InProcessExecutor
) -> None:
    module = load_app("kubernetes")
    module.PROCESSES.client.reconnect_s = 0
    claim = "c" * 40
    assert _post(module, "/v1/owner", {"claim_id": claim}).status_code == 200
    executor.close()

    response = _post(
        module,
        "/v1/commands",
        {
            "command_id": "refused",
            "owner_claim": claim,
            "kind": "process",
            "argv": ["true"],
            "cwd": module.ROOT,
        },
    )
    assert response.status_code == 503
    assert "command executor" in response.json()["detail"]
    assert module.SUPERVISOR.get("refused")["state"] == "not_found"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_kubernetes_entrypoint_refuses_to_run_as_root(tmp_path: Path) -> None:
    stubs = tmp_path / "bin"
    stubs.mkdir()
    calls = tmp_path / "calls"
    for name, script in (
        ("id", "echo 0"),
        ("python3.11", f'echo "python3.11 $*" >> "{calls}"'),
    ):
        stub = stubs / name
        stub.write_text(f"#!/bin/sh\n{script}\n", encoding="utf-8")
        stub.chmod(0o755)

    refused = subprocess.run(
        ["bash", str(_ENTRYPOINT)],
        env={
            "PATH": f"{stubs}:/usr/bin:/bin",
            "CAYU_SIDECAR_MODE": "kubernetes",
            "CAYU_SIDECAR_TLS_CERT_FILE": "/var/run/cayu-sidecar/tls.crt",
            "CAYU_SIDECAR_TLS_KEY_FILE": "/var/run/cayu-sidecar/tls.key",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert refused.returncode != 0
    assert "Kubernetes mode runs the sidecar as a non-root user" in refused.stderr
    assert not calls.exists()


# ------------------------------------------------------------------ egress relay

_TRANSPORT_TOKEN = "ab" * 32
_BROKER_PROXY = "http://10.42.3.20:23145"
# What an agent command does with HTTPS_PROXY: one CONNECT, then tunneled bytes.
_PROXY_CLIENT = """
import json, os, socket, urllib.parse
proxy = urllib.parse.urlsplit(os.environ["HTTPS_PROXY"])
with socket.create_connection((proxy.hostname, proxy.port), timeout=10) as connection:
    connection.sendall(b"CONNECT example.com:443 HTTP/1.1\\r\\nHost: example.com:443\\r\\n\\r\\n")
    head = b""
    while not head.endswith(b"\\r\\n\\r\\n"):
        chunk = connection.recv(1)
        if not chunk:
            break
        head += chunk
    connection.sendall(b"ping")
    echoed = connection.recv(4)
print(json.dumps({
    "environment": dict(os.environ),
    "status": head.split(b"\\r\\n")[0].decode(),
    "echoed": echoed.decode(),
}))
"""


def _read_head(connection: socket.socket) -> bytes:
    head = b""
    while not head.endswith(b"\r\n\r\n"):
        chunk = connection.recv(1)
        if not chunk:
            break
        head += chunk
    return head


class _FakeBrokerProxy:
    """The broker's egress proxy as the relay meets it: the transport tunnel, then a CONNECT.

    A connection that does not first present ``token`` is closed unanswered, as
    the real proxy does; an authenticated one is answered, and its tunnel echoes.
    """

    def __init__(self, token: str = _TRANSPORT_TOKEN) -> None:
        self.authorization = "Basic " + base64.b64encode(f"cayu:{token}".encode()).decode()
        self.preambles: list[bytes] = []
        self.connects: list[bytes] = []
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self) -> None:
        self._listener.close()

    def _accept(self) -> None:
        while True:
            try:
                connection, _address = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(connection,), daemon=True).start()

    def _serve(self, connection: socket.socket) -> None:
        with connection:
            preamble = _read_head(connection)
            self.preambles.append(preamble)
            if preamble != (
                b"CONNECT cayu-transport.invalid:443 HTTP/1.1\r\n"
                b"Host: cayu-transport.invalid:443\r\n"
                b"Proxy-Authorization: " + self.authorization.encode() + b"\r\n\r\n"
            ):
                return
            connection.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            self.connects.append(_read_head(connection))
            connection.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            with contextlib.suppress(OSError):
                while data := connection.recv(65536):
                    connection.sendall(data)


@pytest.fixture
def broker_proxy() -> Iterator[_FakeBrokerProxy]:
    proxy = _FakeBrokerProxy()
    try:
        yield proxy
    finally:
        proxy.close()


def _redirected_relays(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    proxies: dict[tuple[str, int], _FakeBrokerProxy],
) -> list[tuple[str, int]]:
    """Point the sidecar's loopback relay at fake proxies in place of the configured addresses.

    Returns the targets the relay was retargeted to; everything else is the sidecar's own relay.
    """
    requested: list[tuple[str, int]] = []
    relay = module.EXECUTION_BOUNDARY._relay
    retarget = relay.retarget

    def redirected(host: str, port: int, *, transport_token: bytes) -> None:
        requested.append((host, port))
        fake = proxies.get((host, port), next(iter(proxies.values())))
        retarget("127.0.0.1", fake.port, transport_token=transport_token)

    monkeypatch.setattr(relay, "retarget", redirected)
    return requested


def _spawned_environments(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record every environment the supervisor sends to the agent container's executor."""
    spawned: list[dict] = []
    spawn = module.PROCESSES.spawn

    def recording(record: Any, argv: list[str], *, cwd: str, env: dict[str, str]) -> Any:
        spawned.append(dict(env))
        return spawn(record, argv, cwd=cwd, env=env)

    monkeypatch.setattr(module.PROCESSES, "spawn", recording)
    return spawned


def _finished(module: ModuleType, command_id: str) -> dict:
    deadline = time.monotonic() + 20
    while (result := module.SUPERVISOR.get(command_id))["state"] in {"accepted", "running"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return result


def _stdout(result: dict) -> str:
    assert result["exit_code"] == 0, base64.b64decode(result["stderr_base64"])
    return base64.b64decode(result["stdout_base64"]).decode()


def _egress_environment(**extra: str) -> dict[str, str]:
    return {
        "HTTPS_PROXY": _BROKER_PROXY,
        "https_proxy": _BROKER_PROXY,
        "NO_PROXY": "localhost,127.0.0.1,::1",
        "no_proxy": "localhost,127.0.0.1,::1",
        supervisor_module.TRANSPORT_TOKEN_ENV: _TRANSPORT_TOKEN,
        **extra,
    }


def _proxy_environment(url: str, token: str = _TRANSPORT_TOKEN) -> dict[str, str]:
    return _egress_environment(
        HTTPS_PROXY=url, https_proxy=url, **{supervisor_module.TRANSPORT_TOKEN_ENV: token}
    )


def _command(
    module: ModuleType,
    claim: str,
    command_id: str,
    argv: list[str],
    env: dict[str, str],
    profile: str = "agent",
) -> httpx.Response:
    return _post(
        module,
        "/v1/commands",
        {
            "command_id": command_id,
            "owner_claim": claim,
            "kind": "process",
            "argv": argv,
            "cwd": module.ROOT,
            "env": env,
            "execution_profile": profile,
        },
    )


def _run_proxy_client(
    module: ModuleType,
    claim: str,
    command_id: str,
    profile: str = "agent",
    env: dict[str, str] | None = None,
) -> dict:
    started = _command(
        module,
        claim,
        command_id,
        [sys.executable, "-I", "-c", _PROXY_CLIENT],
        _egress_environment() if env is None else env,
        profile,
    )
    assert started.status_code == 202, started.text
    return json.loads(_stdout(_finished(module, command_id)))


def _relay_port(module: ModuleType) -> int:
    return int(module.EXECUTION_BOUNDARY._relay.proxy_url.rsplit(":", 1)[1])


def _assert_relayed_without_the_token(
    module: ModuleType,
    broker_proxy: _FakeBrokerProxy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _redirected_relays(module, monkeypatch, {("10.42.3.20", 23145): broker_proxy})
    spawned = _spawned_environments(module, monkeypatch)
    claim = "c" * 40
    assert _post(module, "/v1/owner", {"claim_id": claim}).status_code == 200

    # The sandbox has no trusted lane: a system command runs as the agent too.
    for profile in ("agent", "trusted"):
        report = _run_proxy_client(module, claim, f"egress-{profile}", profile)
        relay_url = module.EXECUTION_BOUNDARY._relay.proxy_url
        assert relay_url.startswith("http://127.0.0.1:")
        assert report["status"] == "HTTP/1.1 200 Connection Established"
        assert report["echoed"] == "ping"
        environment = report["environment"]
        assert environment["HTTPS_PROXY"] == environment["https_proxy"] == relay_url
        assert environment["NO_PROXY"] == environment["no_proxy"] == "localhost,127.0.0.1,::1"
        assert supervisor_module.TRANSPORT_TOKEN_ENV not in environment
        assert _TRANSPORT_TOKEN not in json.dumps(environment)

    # One target for the owner, the configured broker address only.
    assert requested == [("10.42.3.20", 23145)]
    # The relay presented the token before each command's own CONNECT.
    assert len(broker_proxy.connects) == 2
    assert all(head.startswith(b"CONNECT example.com:443 ") for head in broker_proxy.connects)
    # The executor never received the token.
    assert len(spawned) == 2
    assert all(_TRANSPORT_TOKEN not in json.dumps(environment) for environment in spawned)


def test_kubernetes_relay_carries_commands_to_the_proxy_without_showing_them_the_token(
    load_app, broker_proxy: _FakeBrokerProxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_relayed_without_the_token(load_app("kubernetes"), broker_proxy, monkeypatch)


@_needs_agent_namespace
def test_kubernetes_relay_serves_commands_in_the_agent_namespace(
    load_app,
    broker_proxy: _FakeBrokerProxy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The agent namespace keeps the pod's network, so its commands reach the loopback relay.
    isolated = InProcessExecutor(tmp_path / "workspace", isolate=True)
    monkeypatch.setenv("CAYU_EXECUTOR_SOCKET", isolated.address)
    try:
        _assert_relayed_without_the_token(load_app("kubernetes"), broker_proxy, monkeypatch)
    finally:
        isolated.close()


def _start(module: ModuleType, claim: str, command_id: str, env: dict[str, str]) -> httpx.Response:
    return _command(module, claim, command_id, ["env"], env)


def _assert_relay_closes_at_once(port: int) -> None:
    with (
        socket.create_connection(("127.0.0.1", port), timeout=10) as connection,
        contextlib.suppress(ConnectionResetError),
    ):
        assert connection.recv(1) == b""


@pytest.mark.parametrize(
    ("token", "message"),
    [
        (None, "agent proxy requires a transport token in Kubernetes mode"),
        ("AB" * 32, "agent proxy transport token is malformed"),
        ("ab" * 31, "agent proxy transport token is malformed"),
    ],
)
def test_kubernetes_mode_refuses_a_proxy_without_a_valid_transport_token(
    load_app, token: str | None, message: str
) -> None:
    module = load_app("kubernetes")
    claim = "c" * 40
    assert _post(module, "/v1/owner", {"claim_id": claim}).status_code == 200
    env = _egress_environment()
    if token is None:
        del env[supervisor_module.TRANSPORT_TOKEN_ENV]
    else:
        env[supervisor_module.TRANSPORT_TOKEN_ENV] = token

    response = _start(module, claim, "refused", env)

    assert response.status_code == 400
    assert response.json()["detail"] == message
    assert not module.EXECUTION_BOUNDARY._relay.routed
    assert module.SUPERVISOR.get("refused")["state"] == "not_found"


@pytest.mark.parametrize(
    ("proxy_url", "message"),
    [
        # The pod itself: the supervisor's port, or the relay's own.
        ("http://127.0.0.1:8080", "agent proxy host must be a non-public IPv4 literal"),
        ("http://127.0.0.1:18080", "agent proxy host must be a non-public IPv4 literal"),
        # Link-local instance metadata servers.
        ("http://169.254.169.254:80", "agent proxy host must be a non-public IPv4 literal"),
        ("http://169.254.169.252:988", "agent proxy host must be a non-public IPv4 literal"),
        # Public addresses.
        ("http://8.8.8.8:3128", "agent proxy host must be a non-public IPv4 literal"),
        ("http://100.128.0.1:3128", "agent proxy host must be a non-public IPv4 literal"),
        ("http://0.0.0.0:3128", "agent proxy host must be a non-public IPv4 literal"),
        ("http://224.0.0.1:3128", "agent proxy host must be a non-public IPv4 literal"),
        ("http://255.255.255.255:3128", "agent proxy host must be a non-public IPv4 literal"),
        ("http://proxy.example.test:3128", "agent proxy host must be a non-public IPv4 literal"),
        ("https://10.0.0.1:1", "agent proxy must be an unauthenticated HTTP origin"),
        ("http://user:secret@10.0.0.1:1", "agent proxy must be an unauthenticated HTTP origin"),
    ],
)
def test_kubernetes_mode_refuses_a_public_or_non_http_proxy(
    load_app, proxy_url: str, message: str
) -> None:
    module = load_app("kubernetes")
    claim = "c" * 40
    assert _post(module, "/v1/owner", {"claim_id": claim}).status_code == 200

    response = _start(module, claim, "refused", _proxy_environment(proxy_url))

    assert response.status_code == 400
    assert response.json()["detail"] == message
    assert not module.EXECUTION_BOUNDARY._relay.routed


@pytest.mark.parametrize(
    "host",
    [
        "10.42.3.20",
        "172.16.0.1",
        "192.168.1.1",
        # The shared address space, which GKE can assign to pods.
        "100.64.0.10",
        "100.127.255.254",
        # Class E, which GKE can also assign to pods.
        "240.10.0.5",
    ],
)
def test_kubernetes_mode_relays_to_any_non_public_proxy(
    load_app, broker_proxy: _FakeBrokerProxy, monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    module = load_app("kubernetes")
    requested = _redirected_relays(module, monkeypatch, {(host, 3128): broker_proxy})
    claim = "c" * 40
    assert _post(module, "/v1/owner", {"claim_id": claim}).status_code == 200

    report = _run_proxy_client(
        module, claim, "relayed", env=_proxy_environment(f"http://{host}:3128")
    )

    assert report["echoed"] == "ping"
    assert requested == [(host, 3128)]


def test_lambda_mode_keeps_its_rfc_1918_proxy_rule() -> None:
    boundary = supervisor_module.CommandExecutionBoundary(agent_netns="agent-netns")
    with pytest.raises(
        supervisor_module.CommandRequestError, match="agent proxy host must be a private IPv4"
    ):
        boundary.environment_for(
            _proxy_environment("http://100.64.0.10:3128"), execution_profile="agent"
        )
    assert boundary._relay is None


def test_kubernetes_relay_without_egress_settings_closes_its_connections(
    load_app, broker_proxy: _FakeBrokerProxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_app("kubernetes")
    spawned = _spawned_environments(module, monkeypatch)
    claim = "c" * 40
    assert _post(module, "/v1/owner", {"claim_id": claim}).status_code == 200

    # A stray token without proxy settings is still removed.
    env = {"A": "1", supervisor_module.TRANSPORT_TOKEN_ENV: _TRANSPORT_TOKEN}
    assert _start(module, claim, "plain", env).status_code == 202
    output = _stdout(_finished(module, "plain"))

    assert "A=1" in output.splitlines()
    assert "PROXY" not in output.upper() and _TRANSPORT_TOKEN not in output
    assert spawned[0]["A"] == "1" and supervisor_module.TRANSPORT_TOKEN_ENV not in spawned[0]
    # The port is held from the start, but without a target nothing is forwarded.
    assert not module.EXECUTION_BOUNDARY._relay.routed
    _assert_relay_closes_at_once(_relay_port(module))
    assert not broker_proxy.preambles


# An agent process trying to take the relay's port, as a squatter would.
_SQUATTER = """
import json, socket, sys
listener = socket.socket()
try:
    listener.bind(("127.0.0.1", int(sys.argv[1])))
    listener.listen()
    print(json.dumps({"bound": True}))
except OSError as exc:
    print(json.dumps({"bound": False, "errno": exc.errno}))
"""


def _squat(module: ModuleType, claim: str, command_id: str) -> dict:
    port = str(_relay_port(module))
    started = _command(module, claim, command_id, [sys.executable, "-I", "-c", _SQUATTER, port], {})
    assert started.status_code == 202, started.text
    return json.loads(_stdout(_finished(module, command_id)))


def test_kubernetes_sidecar_holds_the_relay_port_before_any_command(
    load_app, broker_proxy: _FakeBrokerProxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_app("kubernetes")
    requested = _redirected_relays(module, monkeypatch, {("10.42.3.20", 23145): broker_proxy})
    first, second = "c" * 40, "d" * 40
    assert _post(module, "/v1/owner", {"claim_id": first}).status_code == 200

    # Before any proxied command, and again after a superseding claim, the port is taken.
    assert _squat(module, first, "squat-early") == {"bound": False, "errno": errno.EADDRINUSE}
    assert _post(module, "/v1/owner", {"claim_id": second}).status_code == 200
    assert _squat(module, second, "squat-after-claim") == {
        "bound": False,
        "errno": errno.EADDRINUSE,
    }

    # So the new owner's egress still works.
    assert _run_proxy_client(module, second, "relayed")["echoed"] == "ping"
    assert requested == [("10.42.3.20", 23145)]


def test_kubernetes_sidecar_refuses_to_start_when_the_relay_port_is_taken(
    load_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    with socket.create_server(("127.0.0.1", 0)) as squatter:
        port = squatter.getsockname()[1]
        monkeypatch.setattr(supervisor_module, "_AGENT_PROXY_RELAY_PORT", port)
        with pytest.raises(
            RuntimeError, match=f"agent proxy relay cannot listen on 127.0.0.1:{port}"
        ):
            load_app("kubernetes")


# A command that opens one tunnel each time the host drops a numbered go file in its directory.
_CARRIED_CLIENT = """
import os, socket, sys, time, urllib.parse
proxy = urllib.parse.urlsplit(os.environ["HTTPS_PROXY"])
for step in range(int(sys.argv[1])):
    while not os.path.exists(f"go-{step}"):
        time.sleep(0.01)
    with socket.create_connection((proxy.hostname, proxy.port), timeout=10) as connection:
        connection.sendall(b"CONNECT example.com:443 HTTP/1.1\\r\\nHost: example.com:443\\r\\n\\r\\n")
        head = b""
        while not head.endswith(b"\\r\\n\\r\\n"):
            chunk = connection.recv(1)
            if not chunk:
                break
            head += chunk
        print(step, head.split(b"\\r\\n")[0].decode(), flush=True)
"""


def _wait_for(condition: Any) -> None:
    deadline = time.monotonic() + 20
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_kubernetes_relay_is_retargeted_for_a_new_owner_and_keeps_carried_commands(
    load_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_proxy, second_proxy = _FakeBrokerProxy(), _FakeBrokerProxy(token="cd" * 32)
    try:
        module = load_app("kubernetes")
        relay = module.EXECUTION_BOUNDARY._relay
        port = _relay_port(module)
        requested = _redirected_relays(
            module,
            monkeypatch,
            {("10.42.3.20", 23145): first_proxy, ("10.42.3.21", 40000): second_proxy},
        )
        first, second = "c" * 40, "d" * 40
        assert _post(module, "/v1/owner", {"claim_id": first, "owner_epoch": 1}).status_code == 200
        argv = [sys.executable, "-I", "-c", _CARRIED_CLIENT, "3"]
        assert _command(module, first, "carried", argv, _egress_environment()).status_code == 202
        workspace = Path(module.ROOT)
        (workspace / "go-0").touch()
        _wait_for(lambda: len(first_proxy.connects) == 1)

        # The same owner cannot point the relay elsewhere, with another address or token.
        moved = _start(module, first, "moved", _proxy_environment("http://10.42.3.21:40000"))
        assert moved.status_code == 400
        assert moved.json()["detail"] == "agent proxy endpoint changed within one sandbox pod"
        other_token = {supervisor_module.TRANSPORT_TOKEN_ENV: "cd" * 32}
        assert _start(module, first, "token", _egress_environment(**other_token)).status_code == 400
        split = _start(
            module, first, "split", _egress_environment(HTTPS_PROXY="http://10.42.3.21:40000")
        )
        assert split.json()["detail"] == "agent proxy environment must name one private endpoint"

        # A new owner takes the command over. The relay keeps its port and, until that owner
        # names its own proxy, its target, so the carried command keeps reaching the proxy.
        claimed = _post(
            module,
            "/v1/owner",
            {"claim_id": second, "owner_epoch": 2, "carry_commands": ["carried"]},
        )
        assert claimed.json()["carried_commands"] == ["carried"]
        assert module.EXECUTION_BOUNDARY._relay is relay
        (workspace / "go-1").touch()
        _wait_for(lambda: len(first_proxy.connects) == 2)

        # The new owner's first proxied command retargets the relay to its proxy and token.
        new_session = _proxy_environment("http://10.42.3.21:40000", token="cd" * 32)
        report = _run_proxy_client(module, second, "new-owner", env=new_session)
        assert report["environment"]["HTTPS_PROXY"] == f"http://127.0.0.1:{port}"
        assert report["echoed"] == "ping"
        (workspace / "go-2").touch()
        result = _finished(module, "carried")
        assert _stdout(result).splitlines() == [
            "0 HTTP/1.1 200 Connection Established",
            "1 HTTP/1.1 200 Connection Established",
            "2 HTTP/1.1 200 Connection Established",
        ]
        assert len(first_proxy.connects) == 2
        assert len(second_proxy.connects) == 2
        assert requested == [("10.42.3.20", 23145), ("10.42.3.21", 40000)]
        assert module.EXECUTION_BOUNDARY._relay is relay
    finally:
        first_proxy.close()
        second_proxy.close()


def test_kubernetes_relay_outlives_an_executor_restart(
    load_app, broker_proxy: _FakeBrokerProxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_app("kubernetes")
    requested = _redirected_relays(module, monkeypatch, {("10.42.3.20", 23145): broker_proxy})
    relay = module.EXECUTION_BOUNDARY._relay
    first, second = "c" * 40, "d" * 40
    assert _post(module, "/v1/owner", {"claim_id": first}).status_code == 200
    assert _run_proxy_client(module, first, "before")["echoed"] == "ping"

    module._executor_restarted()

    assert module.EXECUTION_BOUNDARY._relay is relay
    assert module.EXECUTION_BOUNDARY._relay_target is None
    assert _post(module, "/v1/owner", {"claim_id": second}).status_code == 200
    report = _run_proxy_client(module, second, "after", env=_proxy_environment(_BROKER_PROXY))
    assert report["echoed"] == "ping"
    assert requested == [("10.42.3.20", 23145), ("10.42.3.20", 23145)]


def _tunnel(port: int) -> socket.socket:
    connection = socket.create_connection(("127.0.0.1", port), timeout=10)
    connection.sendall(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n")
    assert _read_head(connection).startswith(b"HTTP/1.1 200")
    connection.sendall(b"ping")
    assert connection.recv(4) == b"ping"
    return connection


@pytest.fixture
def loopback_relay(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.setattr(supervisor_module, "_AGENT_PROXY_RELAY_PORT", 0)
    relay = supervisor_module._LoopbackRelay()
    try:
        yield relay
    finally:
        relay.close()


def _loopback_port(relay: Any) -> int:
    return int(relay.proxy_url.rsplit(":", 1)[1])


def test_loopback_relay_bounds_its_connections(
    loopback_relay: Any, broker_proxy: _FakeBrokerProxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisor_module, "MAX_LOOPBACK_RELAY_CONNECTIONS", 2)
    loopback_relay.retarget(
        "127.0.0.1", broker_proxy.port, transport_token=_TRANSPORT_TOKEN.encode()
    )
    port = _loopback_port(loopback_relay)
    held = [_tunnel(port), _tunnel(port)]
    # Closed as soon as it is accepted, before anything reaches the proxy.
    _assert_relay_closes_at_once(port)
    assert len(broker_proxy.preambles) == 2

    held.pop().close()
    deadline = time.monotonic() + 10
    while True:
        try:
            held.append(_tunnel(port))
            break
        except (AssertionError, OSError):
            assert time.monotonic() < deadline
            time.sleep(0.05)
    for connection in held:
        connection.close()


def test_loopback_relay_closes_a_connection_the_proxy_does_not_admit(loopback_relay: Any) -> None:
    proxy = _FakeBrokerProxy(token="cd" * 32)
    loopback_relay.retarget("127.0.0.1", proxy.port, transport_token=_TRANSPORT_TOKEN.encode())
    try:
        with socket.create_connection(("127.0.0.1", _loopback_port(loopback_relay))) as client:
            client.settimeout(10)
            client.sendall(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n")
            with contextlib.suppress(ConnectionResetError):
                assert client.recv(1) == b""
        assert len(proxy.preambles) == 1 and not proxy.connects
    finally:
        proxy.close()


def test_loopback_relay_keeps_open_tunnels_when_retargeted(
    loopback_relay: Any, broker_proxy: _FakeBrokerProxy
) -> None:
    other = _FakeBrokerProxy(token="cd" * 32)
    try:
        loopback_relay.retarget(
            "127.0.0.1", broker_proxy.port, transport_token=_TRANSPORT_TOKEN.encode()
        )
        port = _loopback_port(loopback_relay)
        before = _tunnel(port)
        loopback_relay.retarget("127.0.0.1", other.port, transport_token=b"cd" * 32)
        after = _tunnel(port)

        for connection in (before, after):
            connection.sendall(b"pong")
            assert connection.recv(4) == b"pong"
            connection.close()
        assert len(broker_proxy.connects) == 1 and len(other.connects) == 1
    finally:
        other.close()


def test_loopback_relay_replaces_the_agent_namespace_relay() -> None:
    with pytest.raises(ValueError, match="loopback_relay"):
        supervisor_module.CommandExecutionBoundary(agent_netns="agent-netns", loopback_relay=True)
