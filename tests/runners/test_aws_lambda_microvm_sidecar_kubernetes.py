"""Sidecar Kubernetes mode: a workspace supervised inside a sandbox pod."""

from __future__ import annotations

import asyncio
import base64
import importlib
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from examples.aws.lambda_microvm_sidecar import token_auth
from examples.aws.lambda_microvm_sidecar.remote_processes import RemoteProcesses
from examples.aws.lambda_microvm_sidecar.supervisor import LocalProcesses
from tests.runners.lambda_microvm_harness import write_sidecar_tls
from tests.runners.sidecar_executor_harness import InProcessExecutor

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
            "capabilities": ["owner_epoch", "carry_commands", "bearer_auth", "boot_id"],
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
