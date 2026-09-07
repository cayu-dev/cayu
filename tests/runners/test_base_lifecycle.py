"""Shared Runner lifecycle contract (base class) tests."""

from __future__ import annotations

import asyncio

import pytest
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu.runners._cleanup import RunnerCleanupResult, runner_cancellation_failure
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
    attach_cancellation_artifacts,
    is_same_or_child,
)
from cayu.vaults import SecretRedactor


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


class _PreflightRejectingRunner(Runner):
    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        del command, cwd, env, env_remove, timeout_s, stdin, output_limit_bytes
        raise ValueError("Runner preflight rejected the request.")


def _artifact(action: str, status: str) -> dict:
    return {
        "type": "cayu.runner_cleanup.v1",
        "adapter": "stub",
        "action": action,
        "status": status,
        "timeout_s": 5.0,
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("policy", "action"),
    [("command", "kill_command"), ("sandbox", "kill_sandbox"), ("none", "none")],
)
async def test_cleanup_deadline_reports_the_selected_boundary(policy, action):
    runner = StubRunner()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def cleanup():
        await release.wait()
        finished.set()
        return RunnerCleanupResult(artifact=_artifact(action, "completed"), close_runner=False)

    try:
        result = await runner._settle_command_cleanup(
            cleanup, adapter="stub", timeout_s=0.01, policy=policy
        )
        assert result.artifact["action"] == action
        assert result.artifact["status"] == "timeout"
        assert runner.lifecycle_state == "poisoned"
        assert not finished.is_set()
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 1)


@pytest.mark.anyio
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("fatal", [False, True])
async def test_cleanup_group_preserves_current_cancellation_and_fatal_leaves(terminal, fatal):
    runner = StubRunner()
    entered = asyncio.Event()
    release = asyncio.Event()
    signal = SystemExit(7) if fatal else RuntimeError("cleanup failed")
    failure = BaseExceptionGroup("cleanup group", [asyncio.CancelledError("child"), signal])

    async def cleanup():
        entered.set()
        await release.wait()
        raise failure

    async def run():
        if terminal:
            await runner._settle_terminal_lifecycle(cleanup, action="close", timeout_s=1)
        else:
            await runner._settle_command_cleanup(cleanup, adapter="stub", timeout_s=1)

    task = asyncio.create_task(run())
    await entered.wait()
    task.cancel("owner")
    await asyncio.sleep(0)
    release.set()
    if fatal:
        with pytest.raises(BaseExceptionGroup) as captured:
            await task
        assert captured.value is failure
        assert not task.cancelled()
    else:
        with pytest.raises(asyncio.CancelledError) as captured:
            await task
        assert captured.value.args == ("owner",)
        assert runner_cancellation_failure(captured.value) is failure
        assert task.cancelled()
    assert task.cancelling() == 1
    assert runner.lifecycle_state == "poisoned"


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
    assert runner.resolve_cwd(" spaced ") == "/workspace/ spaced "
    assert runner.resolve_cwd("sub/../tests") == "/workspace/tests"
    assert runner.resolve_cwd("/workspace") == "/workspace"
    assert runner.resolve_cwd("/workspace/sub/../tests") == "/workspace/tests"
    with pytest.raises(ValueError, match="outside the runner root"):
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


def test_default_redacted_forwarder_drops_request_inputs_on_rejection() -> None:
    secret = "default-redacted-forwarder-secret-canary-ABCDEFGHIJKLMNOP"

    with pytest.raises(ValueError) as raised:
        asyncio.run(
            _PreflightRejectingRunner().exec_redacted(
                ExecCommand.process("curl", f"Authorization: Bearer {secret}"),
                redactor=SecretRedactor(secret),
                env={"TOKEN": secret},
                stdin=secret,
            )
        )

    current = raised.value.__traceback__
    while current is not None:
        frame = current.tb_frame
        if is_cayu_source_filename(frame.f_code.co_filename):
            assert secret not in repr(frame.f_locals)
        current = current.tb_next


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
    assert runner.lifecycle_state == "poisoned"
    with pytest.raises(RuntimeError, match="permanently poisoned"):
        runner.reopen_exec()
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
