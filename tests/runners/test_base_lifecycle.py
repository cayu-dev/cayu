"""Shared Runner lifecycle contract (base class) tests."""

from __future__ import annotations

import asyncio

import pytest

from cayu.runners._cleanup import RunnerCleanupResult
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
    RunnerCancelledError,
    attach_cancellation_artifacts,
    is_same_or_child,
)


class StubRunner(Runner):
    isolation = "stub"

    def __init__(self, default_cwd: str = "/workspace") -> None:
        self.default_cwd = default_cwd

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        self._ensure_exec_open()
        return ExecResult(stdout="ok")


def _artifact(action: str, status: str) -> dict:
    return {
        "type": "cayu.runner_cleanup.v1",
        "adapter": "stub",
        "action": action,
        "status": status,
        "timeout_s": 5.0,
    }


def test_is_same_or_child_edges():
    assert is_same_or_child("/workspace", "/workspace") is True
    assert is_same_or_child("/workspace/sub", "/workspace") is True
    assert is_same_or_child("/workspace2", "/workspace") is False
    assert is_same_or_child("/etc", "/workspace") is False
    assert is_same_or_child("/anything", "/") is True
    assert is_same_or_child("relative", "/") is False


def test_resolve_cwd_shared_implementation():
    runner = StubRunner()
    assert runner.resolve_cwd() == "/workspace"
    assert runner.resolve_cwd("sub/dir") == "/workspace/sub/dir"
    with pytest.raises(ValueError, match="must be relative"):
        runner.resolve_cwd("/etc")
    with pytest.raises(ValueError, match="escapes"):
        runner.resolve_cwd("../../etc")


def test_default_close_and_context_manager():
    async def run() -> StubRunner:
        async with StubRunner() as runner:
            assert (await runner.exec(ExecCommand.process("true"))).stdout == "ok"
        return runner

    runner = asyncio.run(run())
    assert runner._closed is True
    with pytest.raises(RuntimeError, match="StubRunner is closed."):
        asyncio.run(runner.exec(ExecCommand.process("true")))


def test_exec_closed_latch_message_and_reopen():
    runner = StubRunner()
    runner._close_exec("stub command cleanup did not complete; command state is unknown")
    with pytest.raises(RuntimeError, match="StubRunner is closed: stub command cleanup"):
        asyncio.run(runner.exec(ExecCommand.process("true")))
    runner.reopen_exec()
    assert runner._exec_closed is False
    assert runner._exec_closed_reason is None
    assert asyncio.run(runner.exec(ExecCommand.process("true"))).stdout == "ok"


def test_reopen_exec_rejects_closed_runner():
    runner = StubRunner()
    asyncio.run(runner.close())
    with pytest.raises(RuntimeError, match="StubRunner is closed."):
        runner.reopen_exec()


def test_apply_cleanup_result_latches_failed_command_kill():
    runner = StubRunner()
    runner._apply_cleanup_result(
        RunnerCleanupResult(artifact=_artifact("kill_command", "failed"), close_runner=False)
    )
    assert runner._exec_closed is True
    assert runner._exec_closed_reason == (
        "stub command cleanup did not complete; command state is unknown"
    )
    assert runner._closed is False


def test_apply_cleanup_result_keeps_completed_command_kill_open():
    runner = StubRunner()
    runner._apply_cleanup_result(
        RunnerCleanupResult(artifact=_artifact("kill_command", "completed"), close_runner=False)
    )
    assert runner._exec_closed is False
    assert runner._closed is False


def test_apply_cleanup_result_marks_closed_after_sandbox_kill():
    runner = StubRunner()
    runner._apply_cleanup_result(
        RunnerCleanupResult(artifact=_artifact("kill_sandbox", "completed"), close_runner=True)
    )
    assert runner._closed is True


def test_attach_cancellation_artifacts_sets_and_appends():
    exc = asyncio.CancelledError()
    first = _artifact("kill_command", "completed")
    attach_cancellation_artifacts(exc, [first])
    assert exc.artifacts == [first]
    assert exc.artifacts[0] is not first  # copied, not aliased

    second = _artifact("kill_sandbox", "completed")
    attach_cancellation_artifacts(exc, [second])
    assert exc.artifacts == [first, second]


def test_runner_cancelled_error_stays_compatible():
    # Third-party runners may still raise the subclass; the runtime reads the
    # same out-of-band `artifacts` attribute in both cases.
    artifact = _artifact("kill_command", "completed")
    error = RunnerCancelledError(artifacts=[artifact])
    assert isinstance(error, asyncio.CancelledError)
    assert getattr(error, "artifacts", None) == [artifact]


def test_exec_result_exposes_nonnegative_total_output_bytes():
    result = ExecResult(
        stdout="abc",
        stderr="warning",
        stdout_bytes=9,
        stderr_bytes=12,
        stdout_truncated=True,
    )

    assert result.stdout_bytes == 9
    assert result.stderr_bytes == 12

    with pytest.raises(ValueError, match="stdout_bytes"):
        ExecResult(stdout_bytes=-1)
    with pytest.raises(ValueError, match="stderr_bytes"):
        ExecResult(stderr_bytes=-1)

    properties = ExecResult.model_json_schema()["properties"]
    for field_name in ("stdout_bytes", "stderr_bytes"):
        assert {"minimum": 0, "type": "integer"} in properties[field_name]["anyOf"]
