from __future__ import annotations

import asyncio
import base64
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

SUPERVISOR_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "lambda_microvm_sidecar" / "supervisor.py"
)
SPEC = importlib.util.spec_from_file_location("cayu_lambda_microvm_supervisor", SUPERVISOR_PATH)
assert SPEC is not None and SPEC.loader is not None
SUPERVISOR_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUPERVISOR_MODULE
SPEC.loader.exec_module(SUPERVISOR_MODULE)
CommandSupervisor = SUPERVISOR_MODULE.CommandSupervisor
CommandConflictError = SUPERVISOR_MODULE.CommandConflictError


def test_sidecar_dockerfile_pins_supported_python() -> None:
    dockerfile = (SUPERVISOR_PATH.parent / "Dockerfile").read_text()

    assert "dnf install -y python3.11 python3.11-pip bash" in dockerfile
    assert "ln -sf /usr/bin/python3.11 /usr/bin/python3" in dockerfile
    assert 'CMD ["bash", "/opt/cayu/entrypoint.sh"]' in dockerfile


def test_sidecar_entrypoint_forwards_signals_and_reaps_adopted_children() -> None:
    entrypoint = (SUPERVISOR_PATH.parent / "entrypoint.sh").read_text()

    assert "trap 'forward_signal TERM' TERM" in entrypoint
    assert "wait -n || true" in entrypoint
    assert 'wait "$child_pid"' in entrypoint


def test_sidecar_health_reports_protocol_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAYU_MICROVM_WORKSPACE_ROOT", str(tmp_path))
    sys.modules.pop("examples.lambda_microvm_sidecar.app", None)
    app_module = importlib.import_module("examples.lambda_microvm_sidecar.app")
    try:
        assert asyncio.run(app_module.health()) == {
            "status": "ok",
            "protocol_version": "1",
        }
    finally:
        sys.modules.pop("examples.lambda_microvm_sidecar.app", None)


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
    child = (
        "import os,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    supervisor.start(
        "cmd-stubborn-descendant",
        {
            "kind": "shell",
            "shell": f"{sys.executable} -c {child!r} & wait",
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
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)


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
