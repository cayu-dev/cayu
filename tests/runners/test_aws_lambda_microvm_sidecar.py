from __future__ import annotations

import asyncio
import base64
import importlib.util
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from cayu.cli.lambda_microvm import _export_sidecar
from cayu.runners.aws_lambda_microvm import LAMBDA_MICROVM_PROTOCOL_VERSION

SUPERVISOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "aws"
    / "lambda_microvm_sidecar"
    / "supervisor.py"
)
SPEC = importlib.util.spec_from_file_location("cayu_lambda_microvm_supervisor", SUPERVISOR_PATH)
assert SPEC is not None and SPEC.loader is not None
SUPERVISOR_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUPERVISOR_MODULE
SPEC.loader.exec_module(SUPERVISOR_MODULE)
CommandSupervisor = SUPERVISOR_MODULE.CommandSupervisor
CommandConflictError = SUPERVISOR_MODULE.CommandConflictError
CommandExecutionBoundary = SUPERVISOR_MODULE.CommandExecutionBoundary


def test_sidecar_dockerfile_pins_supported_python() -> None:
    dockerfile = (SUPERVISOR_PATH.parent / "Dockerfile").read_text()

    assert "iptables-nft" in dockerfile
    assert "shadow-utils" in dockerfile
    assert "util-linux" in dockerfile
    assert "ln -sf /usr/bin/python3.11 /usr/bin/python3" in dockerfile
    assert "useradd --uid 1000" in dockerfile
    assert "CAYU_MICROVM_AGENT_UID=1000" in dockerfile
    assert 'CMD ["bash", "/opt/cayu/lambda_microvm_sidecar/entrypoint.sh"]' in dockerfile
    assert "COPY . /opt/cayu/lambda_microvm_sidecar" in dockerfile
    assert "amazon-efs-utils" in dockerfile
    assert '"botocore>=1.43.44,<2"' in dockerfile


def test_sidecar_entrypoint_forwards_signals_and_reaps_adopted_children() -> None:
    entrypoint = (SUPERVISOR_PATH.parent / "entrypoint.sh").read_text()

    assert 'ip netns add "$CAYU_MICROVM_AGENT_NETNS"' in entrypoint
    assert "ip link add cayu-root type veth peer name cayu-agent" in entrypoint
    assert "-i cayu-root" in entrypoint
    assert "--dport 18080 -j ACCEPT" in entrypoint
    assert "-i cayu-root -j REJECT" in entrypoint
    assert "trap 'forward_signal TERM' TERM" in entrypoint
    assert "amazon-efs-mount-watchdog" in entrypoint
    assert "wait -n || true" in entrypoint
    assert 'wait "$sidecar_pid"' in entrypoint


def test_agent_execution_boundary_drops_identity_and_capabilities() -> None:
    boundary = CommandExecutionBoundary(
        agent_uid=1000,
        agent_gid=1000,
        agent_netns="cayu-agent",
    )

    assert boundary.argv_for(
        ["python3", "-V"],
        execution_profile="agent",
    ) == [
        "/usr/sbin/ip",
        "netns",
        "exec",
        "cayu-agent",
        "/usr/bin/setpriv",
        "--reuid=1000",
        "--regid=1000",
        "--clear-groups",
        "--no-new-privs",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--bounding-set=-all",
        "--",
        "python3",
        "-V",
    ]


def test_agent_execution_boundary_relays_only_configured_private_proxy() -> None:
    relays: list[tuple[str, int]] = []

    class _Relay:
        proxy_url = "http://192.0.2.1:18080"

        def close(self) -> None:
            return None

    def relay_factory(host: str, port: int) -> _Relay:
        relays.append((host, port))
        return _Relay()

    boundary = CommandExecutionBoundary(
        agent_uid=1000,
        agent_gid=1000,
        agent_netns="cayu-agent",
        relay_factory=relay_factory,
    )

    environment = boundary.environment_for(
        {
            "HTTPS_PROXY": "http://10.42.3.20:23145",
            "HTTP_PROXY": "http://10.42.3.20:23145",
            "NO_PROXY": "localhost",
        },
        execution_profile="agent",
    )

    assert relays == [("10.42.3.20", 23145)]
    assert environment == {
        "HTTPS_PROXY": "http://192.0.2.1:18080",
        "HTTP_PROXY": "http://192.0.2.1:18080",
        "NO_PROXY": "localhost",
    }


@pytest.mark.parametrize(
    "proxy_url",
    [
        "http://127.0.0.1:8080",
        "http://169.254.169.254:80",
        "http://192.0.2.10:8080",
        "http://8.8.8.8:8080",
    ],
)
def test_agent_execution_boundary_rejects_non_rfc1918_proxy(proxy_url: str) -> None:
    boundary = CommandExecutionBoundary(
        agent_uid=1000,
        agent_gid=1000,
        agent_netns="cayu-agent",
    )

    with pytest.raises(ValueError, match="private IPv4 literal"):
        boundary.environment_for(
            {"HTTPS_PROXY": proxy_url},
            execution_profile="agent",
        )


def test_trusted_execution_stays_in_supervisor_profile(tmp_path: Path) -> None:
    boundary = CommandExecutionBoundary(agent_uid=1000, agent_gid=1000)

    assert boundary.argv_for(["mount", "-a"], execution_profile="trusted") == [
        "mount",
        "-a",
    ]

    supervisor = CommandSupervisor(root=tmp_path, execution_boundary=boundary)
    with pytest.raises(ValueError, match="execution_profile"):
        supervisor.start(
            "invalid-profile",
            {
                "kind": "process",
                "argv": ["python3", "-V"],
                "cwd": str(tmp_path),
                "execution_profile": "unknown",
            },
        )


def test_sidecar_health_reports_protocol_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAYU_MICROVM_WORKSPACE_ROOT", str(tmp_path))
    sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)
    app_module = importlib.import_module("examples.aws.lambda_microvm_sidecar.app")
    try:
        assert asyncio.run(app_module.health()) == {
            "status": "ok",
            "protocol_version": LAMBDA_MICROVM_PROTOCOL_VERSION,
        }
        assert asyncio.run(app_module.ready_hook()) == {"status": "ok"}
    finally:
        sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)


def test_exported_sidecar_preserves_guest_runtime_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = tmp_path / "exported_sidecar"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _export_sidecar(exported, replace=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("CAYU_MICROVM_WORKSPACE_ROOT", str(workspace))
    app_name = "exported_sidecar.app"
    supervisor_name = "exported_sidecar.supervisor"
    sys.modules.pop(app_name, None)
    sys.modules.pop(supervisor_name, None)
    sys.modules.pop("exported_sidecar", None)
    try:
        app_module = importlib.import_module(app_name)
        assert asyncio.run(app_module.health()) == {
            "status": "ok",
            "protocol_version": LAMBDA_MICROVM_PROTOCOL_VERSION,
        }
        assert asyncio.run(app_module.ready_hook()) == {"status": "ok"}
        assert asyncio.run(app_module.run_hook()) == {"status": "ok"}
        assert asyncio.run(app_module.resume_hook()) == {"status": "ok"}
        assert asyncio.run(app_module.suspend_hook()) == {"status": "ok"}
        assert asyncio.run(app_module.terminate_hook()) == {"status": "ok"}

        exported_supervisor = importlib.import_module(supervisor_name)
        supervisor = exported_supervisor.CommandSupervisor(root=workspace)
        supervisor.start(
            "exported-command",
            {
                "kind": "process",
                "argv": [sys.executable, "-c", "print('exported-sidecar-ready')"],
                "cwd": str(workspace),
                "env": {},
                "stdin_base64": None,
                "timeout_s": 2,
                "output_limit_bytes": 8,
            },
        )
        result = wait_for_terminal(supervisor, "exported-command")
        assert base64.b64decode(result["stdout_base64"]) == b"exported"
        assert result["stdout_truncated"] is True
        assert result["exit_code"] == 0
        assert "wait -n || true" in (exported / "entrypoint.sh").read_text(encoding="utf-8")
    finally:
        sys.modules.pop(app_name, None)
        sys.modules.pop(supervisor_name, None)
        sys.modules.pop("exported_sidecar", None)


def wait_for_terminal(
    supervisor: CommandSupervisor,
    command_id: str,
    *,
    timeout_s: float = 3,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = supervisor.get(command_id)
        if result["state"] in {"completed", "cancelled", "failed"}:
            return result
        time.sleep(0.01)
    raise AssertionError(f"command {command_id} did not finish")


def wait_for_process_exit(pid: int, *, timeout_s: float = 3) -> None:
    """Wait until a process has exited, treating Linux zombies as terminated."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return

        # A killed orphan can remain visible to kill(2) as a zombie until the
        # host's PID 1 reaps it. It is no longer executing at that point.
        if Path("/proc").is_dir():
            stat_path = Path(f"/proc/{pid}/stat")
            try:
                stat = stat_path.read_text()
            except FileNotFoundError:
                return
            except OSError:
                pass
            else:
                _prefix, separator, fields = stat.rpartition(") ")
                if separator and fields.split(maxsplit=1)[0] in {"X", "Z"}:
                    return
        time.sleep(0.01)
    raise AssertionError(f"process {pid} did not exit")


def test_sidecar_executes_process_form_without_host_env_and_bounds_output(
    tmp_path: Path,
) -> None:
    os.environ["CAYU_HOST_SECRET_SHOULD_NOT_LEAK"] = "must-not-appear"
    supervisor = CommandSupervisor(root=tmp_path)
    script = (
        "import os,sys; "
        "sys.stdout.write(os.environ.get('VISIBLE', 'missing') + 'abcdef'); "
        "sys.stderr.write(os.environ.get('CAYU_HOST_SECRET_SHOULD_NOT_LEAK', 'absent'))"
    )

    started = supervisor.start(
        "cmd-process",
        {
            "kind": "process",
            "argv": [sys.executable, "-c", script],
            "cwd": str(tmp_path),
            "env": {"VISIBLE": "yes"},
            "stdin_base64": None,
            "timeout_s": 2,
            "output_limit_bytes": 5,
        },
    )
    result = wait_for_terminal(supervisor, "cmd-process")

    assert started == {"command_id": "cmd-process", "state": "accepted"}
    assert base64.b64decode(result["stdout_base64"]) == b"yesab"
    assert base64.b64decode(result["stderr_base64"]) == b"absen"
    assert result["stdout_bytes"] == 9
    assert result["stderr_bytes"] == 6
    assert result["exit_code"] == 0
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is True
    assert result["timed_out"] is False
    assert result["cancelled"] is False


def test_sidecar_times_out_and_stops_process_group(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path)
    supervisor.start(
        "cmd-timeout",
        {
            "kind": "shell",
            "shell": "printf before; sleep 30",
            "cwd": str(tmp_path),
            "env": {},
            "stdin_base64": None,
            "timeout_s": 0.05,
            "output_limit_bytes": 100,
        },
    )

    result = wait_for_terminal(supervisor, "cmd-timeout")

    assert result["state"] == "completed"
    assert result["timed_out"] is True
    assert result["cancelled"] is False
    assert base64.b64decode(result["stdout_base64"]) == b"before"


def test_sidecar_timeout_kills_sigterm_ignoring_descendant(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path)
    pid_file = tmp_path / "descendant.pid"
    child = "import time; time.sleep(30)"
    shell = (
        "trap '' TERM; "
        f"{sys.executable} -c {shlex.quote(child)} & "
        "descendant_pid=$!; "
        "trap - TERM; "
        'kill -0 "$descendant_pid"; '
        f"printf '%s' \"$descendant_pid\" > {shlex.quote(str(pid_file))}; "
        "wait"
    )
    supervisor.start(
        "cmd-stubborn-descendant",
        {
            "kind": "shell",
            "shell": shell,
            "cwd": str(tmp_path),
            "env": {},
            "stdin_base64": None,
            "timeout_s": 0.2,
            "output_limit_bytes": 100,
        },
    )

    result = wait_for_terminal(supervisor, "cmd-stubborn-descendant")
    descendant_pid = int(pid_file.read_text())

    assert result["timed_out"] is True
    wait_for_process_exit(descendant_pid)


def test_sidecar_cancels_running_command_idempotently(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path)
    supervisor.start(
        "cmd-cancel",
        {
            "kind": "shell",
            "shell": "sleep 30",
            "cwd": str(tmp_path),
            "env": {},
            "stdin_base64": None,
            "timeout_s": None,
            "output_limit_bytes": 100,
        },
    )

    first = supervisor.cancel("cmd-cancel")
    second = supervisor.cancel("cmd-cancel")
    result = wait_for_terminal(supervisor, "cmd-cancel")

    assert first["state"] == "cancelled"
    assert second["state"] == "cancelled"
    assert result["cancelled"] is True
    assert result["timed_out"] is False


def test_sidecar_tombstones_cancel_before_start_without_executing(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path)
    marker = tmp_path / "must-not-exist"
    payload = {
        "kind": "process",
        "argv": [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
        "cwd": str(tmp_path),
        "env": {},
        "stdin_base64": None,
        "timeout_s": 1,
        "output_limit_bytes": 100,
    }

    cancelled = supervisor.cancel("cmd-cancel-before-start")
    first_start = supervisor.start("cmd-cancel-before-start", payload)
    retry = supervisor.start("cmd-cancel-before-start", payload)

    assert cancelled["command_id"] == "cmd-cancel-before-start"
    assert cancelled["state"] == "cancelled"
    assert cancelled["cancelled"] is True
    assert cancelled["timed_out"] is False
    assert first_start == cancelled
    assert retry == cancelled
    assert marker.exists() is False
    with pytest.raises(CommandConflictError):
        supervisor.start("cmd-cancel-before-start", {**payload, "timeout_s": 2})


def test_sidecar_expires_cancel_tombstones_after_result_ttl(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path, result_ttl_s=0.1)

    assert supervisor.cancel("cmd-tombstone")["state"] == "cancelled"
    time.sleep(0.15)

    assert supervisor.get("cmd-tombstone")["state"] == "not_found"


def test_sidecar_reports_bounded_spawn_failure_in_stderr(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path)
    supervisor.start(
        "cmd-missing-binary",
        {
            "kind": "process",
            "argv": ["/definitely/missing-cayu-binary"],
            "cwd": str(tmp_path),
            "env": {},
            "stdin_base64": None,
            "timeout_s": 1,
            "output_limit_bytes": 20,
        },
    )

    result = wait_for_terminal(supervisor, "cmd-missing-binary")

    assert result["state"] == "failed"
    assert base64.b64decode(result["stderr_base64"]) == b"FileNotFoundError: ["
    assert result["stderr_truncated"] is True


def test_sidecar_expires_completed_command_results(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path, result_ttl_s=0.2)
    supervisor.start(
        "cmd-expiring",
        {
            "kind": "process",
            "argv": [sys.executable, "-c", "pass"],
            "cwd": str(tmp_path),
            "env": {},
            "stdin_base64": None,
            "timeout_s": 1,
            "output_limit_bytes": 100,
        },
    )
    wait_for_terminal(supervisor, "cmd-expiring")
    time.sleep(0.25)

    assert supervisor.get("cmd-expiring") == {
        "command_id": "cmd-expiring",
        "state": "not_found",
    }
