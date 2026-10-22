from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Any

import pytest

from cayu.runners._cleanup import (
    cleanup_runner_command_with_diagnostic,
    runner_cleanup_observation_timeout,
    validate_runner_cleanup_policy,
)
from cayu.runners.base import ExecCommand, ExecResult, runner_workspace_mutation_settlement
from cayu.runners.docker import DockerRunner, _kill_supervised_command_script
from cayu.tools.base import ToolContext
from cayu.tools.commands import ExecCommandTool


def _artifact(action: str, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "cayu.runner_cleanup.v1",
        "adapter": "test",
        "action": action,
        "status": status,
        "timeout_s": 5.0,
        **extra,
    }


class _Handle:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls = 0

    async def kill(self) -> bool:
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome  # type: ignore[return-value]


class _Sandbox:
    def __init__(self) -> None:
        self.calls = 0

    async def kill(self) -> bool:
        self.calls += 1
        return True


def _cleanup(handle: _Handle | None, sandbox: _Sandbox):
    return asyncio.run(
        cleanup_runner_command_with_diagnostic(
            sandbox,
            handle=handle,
            adapter="test",
            timeout_s=5.0,
            policy="command_or_sandbox",
        )
    )


def test_confirmed_command_kill_keeps_the_sandbox() -> None:
    handle, sandbox = _Handle(True), _Sandbox()

    result = _cleanup(handle, sandbox)

    assert result.artifacts == [_artifact("kill_command", "completed")]
    assert result.close_runner is False
    assert (handle.calls, sandbox.calls) == (1, 0)


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (False, _artifact("kill_command", "failed", error="kill returned false")),
        (RuntimeError("docker stream broke"), _artifact("kill_command", "failed")),
        (None, _artifact("kill_command", "unsupported", error="command handle is not available")),
    ],
)
def test_unconfirmed_command_kill_falls_back_to_the_sandbox(
    outcome: object, expected: dict[str, Any]
) -> None:
    handle = None if outcome is None else _Handle(outcome)
    sandbox = _Sandbox()

    result = _cleanup(handle, sandbox)

    command_artifact = dict(result.artifacts[0])
    command_artifact.pop("error_type", None)
    assert command_artifact == expected
    assert result.artifacts[1] == _artifact("kill_sandbox", "completed")
    assert result.artifact == _artifact("kill_sandbox", "completed")
    assert result.close_runner is True
    assert sandbox.calls == 1


def test_sandbox_fallback_policy_is_opt_in_per_runner() -> None:
    with pytest.raises(ValueError, match="command, sandbox, none"):
        validate_runner_cleanup_policy("command_or_sandbox", "timeout_cleanup")
    assert (
        validate_runner_cleanup_policy(
            "command_or_sandbox", "timeout_cleanup", allow_sandbox_fallback=True
        )
        == "command_or_sandbox"
    )
    assert runner_cleanup_observation_timeout("command_or_sandbox", 5.0) == 10.0
    assert runner_cleanup_observation_timeout("sandbox", 5.0) == 5.0


def test_completed_sandbox_fallback_is_quiescent_after_unconfirmed_command_kill() -> None:
    result = ExecResult(
        timed_out=True,
        exit_code=-9,
        artifacts=[_artifact("kill_command", "failed"), _artifact("kill_sandbox", "completed")],
    )

    assert runner_workspace_mutation_settlement(result=result, error=None) == "runner_quiescent"
    failed_fallback = result.model_copy(
        update={
            "artifacts": [
                _artifact("kill_command", "failed"),
                _artifact("kill_sandbox", "timeout"),
            ]
        }
    )
    assert runner_workspace_mutation_settlement(result=failed_fallback, error=None) == "uncertain"


def _docker_runner(**kwargs: Any) -> DockerRunner:
    return DockerRunner(
        "a1",
        docker_path="/usr/bin/docker",
        close_action="none",
        timeout_cleanup="command_or_sandbox",
        cancellation_cleanup="command_or_sandbox",
        **kwargs,
    )


def test_docker_timeout_with_confirmed_tree_keeps_container(monkeypatch) -> None:
    issued: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs):
        del kwargs
        issued.append(command.argv)
        if "kill -TERM" in command.argv[-1]:
            return ExecResult()
        return ExecResult(timed_out=True, exit_code=-9)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = _docker_runner()

    result = asyncio.run(runner.exec(ExecCommand.bash("sleep 999"), timeout_s=1))

    assert result.timed_out is True
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "docker",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]
    assert runner.lifecycle_state == "reusable"
    assert not any(argv[1:3] == ["rm", "-f"] for argv in issued)
    # A shell supervisor cannot see descendants that left its process group,
    # so this policy asks the guest helper to report such cleanup unconfirmed.
    assert issued[1][-1].endswith("exit 1")


def test_docker_unconfirmed_tree_removes_container(monkeypatch) -> None:
    issued: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs):
        del kwargs
        issued.append(command.argv)
        if "kill -TERM" in command.argv[-1]:
            return ExecResult(exit_code=1)
        if command.argv[1:3] == ["rm", "-f"]:
            return ExecResult()
        return ExecResult(timed_out=True, exit_code=-9)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = _docker_runner()

    result = asyncio.run(runner.exec(ExecCommand.process("sleep", "999"), timeout_s=1))

    assert [(item["action"], item["status"]) for item in result.artifacts] == [
        ("kill_command", "failed"),
        ("kill_sandbox", "completed"),
    ]
    assert runner.lifecycle_state == "closed"
    assert ["/usr/bin/docker", "rm", "-f", "a1"] in issued
    assert runner_workspace_mutation_settlement(result=result, error=None) == "runner_quiescent"


def test_docker_cancellation_with_confirmed_tree_keeps_container(monkeypatch) -> None:
    async def fake_run_subprocess(command, **kwargs):
        del kwargs
        if "kill -TERM" in command.argv[-1]:
            return ExecResult()
        raise asyncio.CancelledError

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = _docker_runner()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        asyncio.run(runner.exec(ExecCommand.process("sleep", "999")))

    assert [
        (item["action"], item["status"])
        for item in cancelled.value.artifacts  # type: ignore[attr-defined]
    ] == [("kill_command", "completed")]
    assert runner.lifecycle_state == "reusable"


def test_docker_durable_receipt_cleanup_uses_container_removal(monkeypatch) -> None:
    issued: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs):
        del kwargs
        issued.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = _docker_runner()

    async def cleanup():
        from cayu.runners.docker import _DockerCommandHandle

        handle = _DockerCommandHandle(
            docker_path="/usr/bin/docker",
            name="a1",
            pid_file="/tmp/cayu-docker-commands/unused.pid",
            docker_cli_env_allowlist=(),
        )
        return await runner._cleanup_command(
            handle=handle, policy="command_or_sandbox", durable=True
        )

    result = asyncio.run(cleanup())

    assert [(item["action"], item["status"]) for item in result.artifacts] == [
        ("kill_sandbox", "completed")
    ]
    assert issued == [["/usr/bin/docker", "rm", "-f", "a1"]]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell process groups")
@pytest.mark.parametrize("require_confirmed_tree", [False, True])
def test_shell_supervisor_cleanup_is_unconfirmed_when_tree_confirmation_is_required(
    tmp_path, require_confirmed_tree: bool
) -> None:
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    pid_file = tmp_path / "command.pid"
    pid_file.write_text(f"{process.pid} 1\n", encoding="ascii")
    try:
        completed = subprocess.run(
            [
                "sh",
                "-c",
                _kill_supervised_command_script(
                    str(pid_file), require_confirmed_tree=require_confirmed_tree
                ),
            ],
            capture_output=True,
            timeout=10,
        )
        assert process.wait(timeout=5) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert completed.returncode == (1 if require_confirmed_tree else 0)
    assert not pid_file.exists()


_FAKE_PYTHON_SUPERVISOR = """\
import os, signal, sys, time
state, acknowledge = sys.argv[1], sys.argv[2] == "1"
def stop(_signum, _frame):
    if acknowledge:
        os.close(os.open(state + ".stopped", os.O_WRONLY | os.O_CREAT | os.O_EXCL))
    os.unlink(state)
    os._exit(0)
signal.signal(signal.SIGTERM, stop)
with open(state + ".tmp", "w") as handle:
    handle.write(f"{os.getpid()} 2\\n")
os.rename(state + ".tmp", state)
time.sleep(30)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("acknowledged", [False, True])
def test_python_supervisor_stop_needs_its_acknowledgement_not_just_a_removed_state_file(
    tmp_path, acknowledged: bool
) -> None:
    # A normal exit also removes the state file, and in direct mode it can
    # leave detached descendants running. Only the handler's acknowledgement,
    # written after it saw no remaining child, confirms the stop.
    pid_file = tmp_path / "command.pid"
    supervisor = subprocess.Popen(
        [sys.executable, "-c", _FAKE_PYTHON_SUPERVISOR, str(pid_file), str(int(acknowledged))]
    )
    try:
        completed = subprocess.run(
            [
                "sh",
                "-c",
                _kill_supervised_command_script(str(pid_file), require_confirmed_tree=True),
            ],
            capture_output=True,
            timeout=20,
        )
        supervisor.wait(timeout=5)
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()

    assert completed.returncode == (0 if acknowledged else 1)
    assert not pid_file.exists()
    assert not (tmp_path / "command.pid.stopped").exists()


class _TimedOutRunner(DockerRunner):
    def __init__(self, artifacts: list[dict[str, Any]]) -> None:
        super().__init__("a1", docker_path="/usr/bin/docker")
        self._artifacts = artifacts

    async def exec(self, command, **kwargs):  # type: ignore[override]
        del command, kwargs
        return ExecResult(timed_out=True, exit_code=-9, artifacts=self._artifacts)


@pytest.mark.parametrize(
    ("artifacts", "expected"),
    [
        (
            [_artifact("kill_command", "completed")],
            "Command timed out after 3 seconds. The command was stopped.",
        ),
        (
            [_artifact("kill_command", "failed"), _artifact("kill_sandbox", "completed")],
            "Command timed out after 3 seconds. The command was stopped by removing its "
            "sandbox, so this environment cannot run more commands.",
        ),
        ([_artifact("kill_command", "failed")], "Command timed out after 3 seconds."),
    ],
)
def test_exec_command_result_reports_only_a_confirmed_stop(
    artifacts: list[dict[str, Any]], expected: str
) -> None:
    result = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="sess_1", runner=_TimedOutRunner(artifacts)),
            {"argv": ["sleep", "30"], "timeout_s": 3},
        )
    )

    assert result.is_error is True
    assert result.content == expected
