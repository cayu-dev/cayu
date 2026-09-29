"""Browser workload admission and sidecar transfer bounds on Lambda MicroVM."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fcntl")

import examples.aws.lambda_microvm_sidecar.supervisor as supervisor_module
from examples.aws.lambda_microvm_sidecar.supervisor import (
    MAX_OUTPUT_LIMIT_BYTES,
    MAX_STDIN_BYTES,
    TRANSPORT_TOKEN_ENV,
    CommandConflictError,
    CommandExecutionBoundary,
    CommandRequestError,
    CommandSupervisor,
    _validated_payload,
)
from tests.egress.test_aws_lambda_microvm_adapter import _broker_and_grant
from tests.runners.lambda_microvm_harness import (
    ClientTokenLambdaModel,
    SupervisorTransport,
    _terminal_result,
)

import cayu.runners.aws_lambda_microvm as runner_module
from cayu import ExecCommand, LambdaMicroVMRunner
from cayu.cli.lambda_microvm import _export_sidecar
from cayu.egress._remote_adapter import PROXY_TRANSPORT_TOKEN_ENV
from cayu.egress.proxy_server import TransparentEgressProxyServer
from cayu.runners import LambdaMicroVMBrowserWorkloadError
from cayu.runners.aws_lambda_microvm import (
    LAMBDA_MICROVM_MAX_OUTPUT_BYTES,
    LAMBDA_MICROVM_MAX_STDIN_BYTES,
)
from cayu.runners.workloads import (
    BROWSER_FETCH_WORKLOAD_NAME,
    BROWSER_SESSION_WORKLOAD_NAME,
    BROWSER_WORKER_FILES,
    PINNED_BROWSER_FETCH_WORKLOAD,
    PINNED_BROWSER_SESSION_WORKLOAD,
    browser_worker_source_digests,
    browser_worker_sources,
)

SIDECAR_ROOT = Path(__file__).resolve().parents[2] / "examples" / "aws" / "lambda_microvm_sidecar"


def _report(**overrides: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "directory": True,
        "interpreter": True,
        "files": browser_worker_source_digests(),
        "versions": {"playwright": "1.62.0", "websockets": "17.0.1"},
        "headless_shell": True,
        "certutil": True,
    }
    report.update(overrides)
    return report


class _BrowserImageTransport(SupervisorTransport):
    """Answer the trusted browser-workload probe as a browser image would."""

    def __init__(
        self,
        root: Path,
        *,
        exit_code: int = 0,
        stdout: str | None = None,
        launch_exit_code: int = 0,
    ) -> None:
        super().__init__(root)
        self.probe_exit_code = exit_code
        self.probe_stdout = json.dumps(_report()) if stdout is None else stdout
        self.launch_exit_code = launch_exit_code
        self.probes: list[dict[str, Any]] = []
        self.launches: list[dict[str, Any]] = []

    async def start_command(
        self, *, command_id: str, payload: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        argv = payload.get("argv") or []
        if runner_module._BROWSER_WORKLOAD_PROBE_SCRIPT in argv:
            admitted = self.admit_start(payload)
            self.probes.append(admitted)
            self._scripted_results[command_id] = _terminal_result(
                command_id, exit_code=self.probe_exit_code, stdout=self.probe_stdout
            )
            return {"command_id": command_id, "state": "accepted"}
        if runner_module._BROWSER_LAUNCH_PROBE_SCRIPT in argv:
            self.launches.append(self.admit_start(payload))
            self._scripted_results[command_id] = _terminal_result(
                command_id, exit_code=self.launch_exit_code
            )
            return {"command_id": command_id, "state": "accepted"}
        return await super().start_command(command_id=command_id, payload=payload, **kwargs)


def _runner(transport: SupervisorTransport, root: Path) -> LambdaMicroVMRunner:
    return LambdaMicroVMRunner(
        ClientTokenLambdaModel(),
        microvm_id="mvm-browser",
        endpoint="mvm-browser.lambda-microvm.invalid",
        endpoint_transport=transport,
        default_cwd=str(root),
        poll_interval_s=0,
    )


def test_browser_workload_is_declared_only_after_the_trusted_probe_matches(
    tmp_path: Path,
) -> None:
    transport = _BrowserImageTransport(tmp_path)
    runner = _runner(transport, tmp_path)

    assert runner.workload_authority(BROWSER_SESSION_WORKLOAD_NAME) is None
    assert runner.workload_authority(BROWSER_FETCH_WORKLOAD_NAME) is None

    asyncio.run(runner.verify_browser_workload())

    (probe,) = transport.probes
    assert probe["execution_profile"] == "trusted"
    assert probe["argv"][:3] == ["/usr/local/bin/python", "-I", "-c"]
    assert probe["argv"][4:6] == ["/opt/cayu-browser", "/ms-playwright"]
    assert probe["argv"][6:] == [name for name, _source in BROWSER_WORKER_FILES]
    (launch,) = transport.launches
    assert launch["execution_profile"] == "agent"
    assert launch["argv"][:2] == ["/usr/local/bin/python", "-I"]
    assert launch["env"]["PLAYWRIGHT_BROWSERS_PATH"] == "/ms-playwright"
    assert launch["timeout_s"] == runner_module.LAMBDA_MICROVM_BROWSER_LAUNCH_TIMEOUT_SECONDS
    assert "chromium_sandbox=True" in runner_module._BROWSER_LAUNCH_PROBE_SCRIPT
    assert "no-sandbox" not in runner_module._BROWSER_LAUNCH_PROBE_SCRIPT
    assert runner.workload_authority(BROWSER_SESSION_WORKLOAD_NAME) is (
        PINNED_BROWSER_SESSION_WORKLOAD
    )
    assert runner.workload_authority(BROWSER_FETCH_WORKLOAD_NAME) is (PINNED_BROWSER_FETCH_WORKLOAD)
    assert runner.workload_authority("cayu.other") is None


def _drifted_worker() -> dict[str, Any]:
    files = browser_worker_source_digests()
    files["worker.py"] = "0" * 64
    return {"files": files}


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (_drifted_worker(), "worker file worker.py"),
        ({"files": {**browser_worker_source_digests(), "worker.py": None}}, "worker.py"),
        ({"versions": {"playwright": "1.61.0", "websockets": "17.0.1"}}, "playwright"),
        ({"versions": {"playwright": "1.62.0", "websockets": None}}, "websockets"),
        ({"directory": False}, "/opt/cayu-browser"),
        ({"interpreter": False}, "interpreter"),
        ({"headless_shell": False}, "headless shell"),
        ({"certutil": False}, "certutil"),
    ],
)
def test_browser_workload_mismatch_refuses_without_declaring(
    tmp_path: Path, overrides: dict[str, Any], reason: str
) -> None:
    transport = _BrowserImageTransport(tmp_path, stdout=json.dumps(_report(**overrides)))
    runner = _runner(transport, tmp_path)

    with pytest.raises(LambdaMicroVMBrowserWorkloadError, match=reason):
        asyncio.run(runner.verify_browser_workload())

    assert runner.workload_authority(BROWSER_SESSION_WORKLOAD_NAME) is None


def test_sandboxed_launch_failure_refuses_after_a_matching_worker(tmp_path: Path) -> None:
    transport = _BrowserImageTransport(tmp_path, launch_exit_code=1)
    runner = _runner(transport, tmp_path)

    with pytest.raises(LambdaMicroVMBrowserWorkloadError, match="sandbox as the agent user"):
        asyncio.run(runner.verify_browser_workload())

    assert len(transport.probes) == 1
    assert runner.workload_authority(BROWSER_SESSION_WORKLOAD_NAME) is None


def test_mismatched_worker_is_refused_before_any_chromium_launch(tmp_path: Path) -> None:
    transport = _BrowserImageTransport(tmp_path, stdout=json.dumps(_report(certutil=False)))
    runner = _runner(transport, tmp_path)

    with pytest.raises(LambdaMicroVMBrowserWorkloadError, match="certutil"):
        asyncio.run(runner.verify_browser_workload())

    assert transport.launches == []


@pytest.mark.parametrize(("exit_code", "stdout"), [(127, ""), (0, "not json"), (0, "[]")])
def test_image_without_browser_worker_refuses(tmp_path: Path, exit_code: int, stdout: str) -> None:
    transport = _BrowserImageTransport(tmp_path, exit_code=exit_code, stdout=stdout)
    runner = _runner(transport, tmp_path)

    with pytest.raises(LambdaMicroVMBrowserWorkloadError):
        asyncio.run(runner.verify_browser_workload())

    assert runner.workload_authority(BROWSER_FETCH_WORKLOAD_NAME) is None


def test_probe_script_reports_owned_files_versions_and_browser(tmp_path: Path) -> None:
    # The probe runs as root in the guest; here it runs against a scratch tree
    # owned by the test user, so ownership checks report False rather than
    # accepting files the agent could have written.
    root = tmp_path / "cayu-browser"
    root.mkdir()
    for name, content in browser_worker_sources().items():
        (root / name).write_bytes(content)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            runner_module._BROWSER_WORKLOAD_PROBE_SCRIPT,
            str(root),
            str(tmp_path / "browsers"),
            *(name for name, _source in BROWSER_WORKER_FILES),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    owned_by_root = os.getuid() == 0

    assert report["directory"] is owned_by_root
    assert set(report["files"]) == {name for name, _source in BROWSER_WORKER_FILES}
    assert report["headless_shell"] is False
    assert set(report["versions"]) == {"playwright", "websockets"}
    if not owned_by_root:
        assert set(report["files"].values()) == {None}
        assert runner_module._browser_workload_mismatch(report, browser_worker_source_digests())


def test_stdin_beyond_the_sidecar_ceiling_is_refused_before_dispatch(tmp_path: Path) -> None:
    transport = _BrowserImageTransport(tmp_path)
    runner = _runner(transport, tmp_path)
    oversized = "x" * (LAMBDA_MICROVM_MAX_STDIN_BYTES + 1)

    with pytest.raises(ValueError, match="stdin exceeds"):
        runner.preflight_exec(ExecCommand.process("cat"), stdin=oversized)
    with pytest.raises(ValueError, match="stdin exceeds"):
        asyncio.run(runner.exec(ExecCommand.process("cat"), stdin=oversized))

    assert transport.payloads == []
    runner.preflight_exec(ExecCommand.process("cat"), stdin="x" * LAMBDA_MICROVM_MAX_STDIN_BYTES)


def test_sidecar_and_runner_share_transfer_ceilings() -> None:
    assert MAX_STDIN_BYTES == LAMBDA_MICROVM_MAX_STDIN_BYTES
    assert MAX_OUTPUT_LIMIT_BYTES == LAMBDA_MICROVM_MAX_OUTPUT_BYTES
    # A default browser upload batch (16 MiB) travels base64-encoded in JSON.
    assert MAX_STDIN_BYTES >= 4 * ((16 * 1024 * 1024 + 2) // 3) + 64 * 1024


def test_sidecar_refuses_stdin_beyond_its_ceiling(tmp_path: Path) -> None:
    boundary = CommandExecutionBoundary()
    oversized = base64.b64encode(b"x" * (MAX_STDIN_BYTES + 1)).decode("ascii")

    with pytest.raises(CommandRequestError, match="stdin exceeds"):
        _validated_payload(
            {"kind": "process", "argv": ["cat"], "cwd": str(tmp_path), "stdin_base64": oversized},
            root=tmp_path.resolve(),
            execution_boundary=boundary,
        )


def _wait(supervisor: CommandSupervisor, command_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        snapshot = supervisor.get(command_id)
        if snapshot["state"] in {"completed", "cancelled", "failed"}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"command {command_id} did not finish")


def test_sidecar_keeps_only_a_payload_digest_and_drops_released_output(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path)
    secret = "cookie=session-secret"
    payload = {
        "kind": "process",
        "argv": [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
        "cwd": str(tmp_path),
        "env": {"PROFILE_TOKEN": secret},
        "stdin_base64": base64.b64encode(secret.encode()).decode("ascii"),
    }

    supervisor.start("cmd-private", dict(payload))
    delivered = _wait(supervisor, "cmd-private")
    record = supervisor._records["cmd-private"]

    assert base64.b64decode(delivered["stdout_base64"]).decode() == secret
    assert len(record.payload_fingerprint) == 64
    assert secret not in record.payload_fingerprint

    released = supervisor.release("cmd-private")

    assert released == {"command_id": "cmd-private", "state": "released"}
    assert supervisor.get("cmd-private") == {"command_id": "cmd-private", "state": "released"}
    assert record.result is None
    # A replayed start stays idempotent and never re-runs or re-exposes output.
    assert supervisor.start("cmd-private", dict(payload))["state"] == "released"
    with pytest.raises(CommandConflictError):
        supervisor.start("cmd-private", {**payload, "env": {}})
    assert supervisor.release("cmd-unknown")["state"] == "not_found"


def test_sidecar_release_does_not_touch_a_running_command(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path, cancel_timeout_s=2)
    supervisor.start(
        "cmd-running",
        {"kind": "process", "argv": ["sleep", "30"], "cwd": str(tmp_path)},
    )
    deadline = time.monotonic() + 5
    while supervisor.get("cmd-running")["state"] != "running" and time.monotonic() < deadline:
        time.sleep(0.01)

    assert supervisor.release("cmd-running")["state"] == "running"
    assert supervisor.cancel("cmd-running")["state"] == "cancelled"


def test_runner_releases_each_terminal_result_after_reading_it(tmp_path: Path) -> None:
    transport = _BrowserImageTransport(tmp_path)
    runner = _runner(transport, tmp_path)

    result = asyncio.run(runner.exec(ExecCommand.process("printf", "delivered")))

    assert result.stdout == "delivered"
    command_ids = [
        command_id
        for command_id in transport.released
        if command_id in transport.supervisor._records
    ]
    assert command_ids
    assert all(
        transport.supervisor.get(command_id)["state"] == "released" for command_id in command_ids
    )


def test_runner_tolerates_a_transport_without_release(tmp_path: Path) -> None:
    class _NoRelease(SupervisorTransport):
        release_command = None  # type: ignore[assignment]

    runner = _runner(_NoRelease(tmp_path), tmp_path)

    assert asyncio.run(runner.exec(ExecCommand.process("printf", "ok"))).stdout == "ok"


_DAEMON = """
import os, subprocess, sys
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    close_fds=True, start_new_session=True,
)
open(sys.argv[1], "w").write(str(child.pid))
if sys.argv[2] == "hang":
    import time; time.sleep(60)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_detached_browser_daemon_survives_completion_timeout_and_cancel_all(
    tmp_path: Path,
) -> None:
    # The browser worker detaches its daemon into a new session with no inherited
    # pipes, as _start_interactive_daemon does. The sidecar stops only the
    # command's own process group, so the daemon outlives every command ending.
    supervisor = CommandSupervisor(root=tmp_path, cancel_timeout_s=2)
    pids: list[int] = []
    try:
        for command_id, mode, timeout_s in (
            ("cmd-complete", "exit", None),
            ("cmd-timeout", "hang", 1),
        ):
            pid_file = tmp_path / f"{command_id}.pid"
            supervisor.start(
                command_id,
                {
                    "kind": "process",
                    "argv": [sys.executable, "-c", _DAEMON, str(pid_file), mode],
                    "cwd": str(tmp_path),
                    "timeout_s": timeout_s,
                },
            )
            finished = _wait(supervisor, command_id)
            assert finished["timed_out"] is (mode == "hang")
            assert finished["stdout_truncated"] is False
            pids.append(int(pid_file.read_text()))

        supervisor.cancel_all(reason="suspend")

        assert all(_alive(pid) for pid in pids)
    finally:
        for pid in pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def test_agent_proxy_relay_is_recreated_after_cancel_all(tmp_path: Path) -> None:
    relays: list[Any] = []

    class _Relay:
        proxy_url = "http://192.0.2.1:18080"

        def __init__(self, host: str, port: int) -> None:
            self.target = (host, port)
            self.closed = False
            relays.append(self)

        def close(self) -> None:
            self.closed = True

    boundary = CommandExecutionBoundary(
        agent_uid=1000, agent_gid=1000, agent_netns="cayu-agent", relay_factory=_Relay
    )
    supervisor = CommandSupervisor(root=tmp_path, execution_boundary=boundary)
    proxy = {"HTTPS_PROXY": "http://10.0.1.20:9443", "https_proxy": "http://10.0.1.20:9443"}

    first = boundary.environment_for(proxy, execution_profile="agent")
    supervisor.cancel_all(reason="suspend")
    second = boundary.environment_for(proxy, execution_profile="agent")

    assert first["HTTPS_PROXY"] == second["HTTPS_PROXY"] == "http://192.0.2.1:18080"
    assert [relay.closed for relay in relays] == [True, False]
    assert relays[1].target == ("10.0.1.20", 9443)


def test_browser_export_composes_sidecar_worker_and_dockerfile(tmp_path: Path) -> None:
    destination = tmp_path / "browser-context"

    _export_sidecar(destination, replace=False, browser=True)

    dockerfile = (destination / "Dockerfile").read_text()
    assert dockerfile == (SIDECAR_ROOT / "browser" / "Dockerfile").read_text()
    for name, content in browser_worker_sources().items():
        assert (destination / "browser-worker" / name).read_bytes() == content
    assert (destination / "lambda_microvm_sidecar" / "supervisor.py").read_bytes() == (
        SIDECAR_ROOT / "supervisor.py"
    ).read_bytes()
    assert (
        destination / "lambda_microvm_sidecar" / "cayu-lambda-microvm-sidecar-manifest.json"
    ).is_file()
    assert "COPY browser-worker/ /opt/cayu-browser/" in dockerfile
    assert "COPY lambda_microvm_sidecar /opt/cayu/lambda_microvm_sidecar" in dockerfile
    assert "PLAYWRIGHT_BROWSERS_PATH=/ms-playwright" in dockerfile
    assert '"playwright==1.62.0" "websockets==17.0.1"' in dockerfile
    assert "ln -sf /usr/bin/python3.11 /usr/local/bin/python" in dockerfile
    assert "nss-tools" in dockerfile
    assert "--no-sandbox" not in dockerfile


def test_browser_dockerfile_keeps_every_base_image_layer() -> None:
    base = (SIDECAR_ROOT / "Dockerfile").read_text()
    browser = (SIDECAR_ROOT / "browser" / "Dockerfile").read_text()

    base_layers = base.split("\nENV CAYU_MICROVM_AGENT_UID")[0]
    assert base_layers in browser
    sidecar_layers = base.split("WORKDIR /opt/cayu\n")[1].split("EXPOSE 8080")[0]
    assert (
        sidecar_layers.replace(
            "COPY . /opt/cayu/lambda_microvm_sidecar",
            "COPY lambda_microvm_sidecar /opt/cayu/lambda_microvm_sidecar",
        )
        in browser
    )
    # The browser image reads its files before the ready hook, then execs the
    # same PID 1 entrypoint.
    assert browser.rstrip().endswith(
        'CMD ["bash", "/opt/cayu/lambda_microvm_sidecar/browser/start.sh"]'
    )
    start = (SIDECAR_ROOT / "browser" / "start.sh").read_text()
    assert start.rstrip().endswith("exec bash /opt/cayu/lambda_microvm_sidecar/entrypoint.sh")
    assert "/ms-playwright" in start


_TOKEN = "ab" * 32


class _RecordingRelay:
    proxy_url = "http://192.0.2.1:18080"
    instances: list[_RecordingRelay] = []

    def __init__(self, host: str, port: int, *, transport_token: bytes | None = None) -> None:
        self.target = (host, port)
        self.transport_token = transport_token
        self.closed = False
        self.instances.append(self)

    def close(self) -> None:
        self.closed = True


def _boundary() -> CommandExecutionBoundary:
    _RecordingRelay.instances = []
    return CommandExecutionBoundary(
        agent_uid=1000, agent_gid=1000, agent_netns="cayu-agent", relay_factory=_RecordingRelay
    )


def test_agent_commands_never_see_the_proxy_transport_token() -> None:
    boundary = _boundary()
    environment = {
        "HTTPS_PROXY": "http://10.0.1.20:9443",
        "https_proxy": "http://10.0.1.20:9443",
        TRANSPORT_TOKEN_ENV: _TOKEN,
    }

    agent = boundary.environment_for(environment, execution_profile="agent")
    trusted = boundary.environment_for(environment, execution_profile="trusted")

    assert TRANSPORT_TOKEN_ENV not in agent
    assert agent["HTTPS_PROXY"] == "http://192.0.2.1:18080"
    assert trusted[TRANSPORT_TOKEN_ENV] == _TOKEN
    (relay,) = _RecordingRelay.instances
    assert relay.transport_token == _TOKEN.encode("ascii")
    # Commands without a proxy still never receive the token.
    assert TRANSPORT_TOKEN_ENV not in boundary.environment_for(
        {TRANSPORT_TOKEN_ENV: _TOKEN}, execution_profile="agent"
    )
    unfenced = CommandExecutionBoundary(agent_uid=1000, agent_gid=1000)
    assert TRANSPORT_TOKEN_ENV not in unfenced.environment_for(
        environment, execution_profile="agent"
    )


def test_relay_refuses_a_changed_or_malformed_transport_token() -> None:
    boundary = _boundary()
    proxy = {"HTTPS_PROXY": "http://10.0.1.20:9443"}
    boundary.environment_for({**proxy, TRANSPORT_TOKEN_ENV: _TOKEN}, execution_profile="agent")

    with pytest.raises(CommandRequestError, match="changed within one MicroVM"):
        boundary.environment_for(
            {**proxy, TRANSPORT_TOKEN_ENV: "cd" * 32}, execution_profile="agent"
        )
    with pytest.raises(CommandRequestError, match="changed within one MicroVM"):
        boundary.environment_for(proxy, execution_profile="agent")

    boundary.close()
    with pytest.raises(CommandRequestError, match="malformed"):
        boundary.environment_for(
            {**proxy, TRANSPORT_TOKEN_ENV: "not-hex"}, execution_profile="agent"
        )
    # A new owner's token is accepted once the previous relay is reset.
    boundary.environment_for({**proxy, TRANSPORT_TOKEN_ENV: "cd" * 32}, execution_profile="agent")
    assert _RecordingRelay.instances[-1].transport_token == ("cd" * 32).encode("ascii")


def test_binding_env_key_matches_the_sidecar() -> None:
    assert PROXY_TRANSPORT_TOKEN_ENV == TRANSPORT_TOKEN_ENV


def _relay_exchange(relay_url: str, request: bytes) -> bytes:
    host, port = relay_url.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=5) as client:
        client.sendall(request)
        response = bytearray()
        while not response.endswith(b"\r\n\r\n"):
            try:
                chunk = client.recv(1)
            except ConnectionResetError:
                break
            if not chunk:
                break
            response.extend(chunk)
        return bytes(response)


def test_relay_authenticates_each_connection_to_the_real_cayu_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "_AGENT_PROXY_RELAY_PORT", 0)
    broker, _grant = _broker_and_grant()
    token = _TOKEN.encode("ascii")
    connect = b"CONNECT receiver.internal:443 HTTP/1.1\r\nHost: receiver.internal:443\r\n\r\n"

    async def scenario() -> tuple[bytes, bytes, bytes]:
        server = TransparentEgressProxyServer(
            broker,
            loop=asyncio.get_running_loop(),
            host="127.0.0.1",
            transport_auth_token=token,
        )
        port = await server.start()
        relays = [
            supervisor_module._TcpRelay(
                "127.0.0.1", port, transport_token=value, listen_host="127.0.0.1"
            )
            for value in (token, ("cd" * 32).encode("ascii"), None)
        ]
        try:
            return tuple(
                await asyncio.gather(
                    *(
                        asyncio.to_thread(_relay_exchange, relay.proxy_url, connect)
                        for relay in relays
                    )
                )
            )
        finally:
            for relay in relays:
                relay.close()
            await server.close()

    authenticated, wrong_token, unauthenticated = asyncio.run(scenario())

    # The authenticated tunnel reaches the proxy's own CONNECT handling.
    assert authenticated.startswith(b"HTTP/1.1 200 Connection Established")
    assert wrong_token == b""
    assert unauthenticated == b""
