from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from pathlib import Path

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    ExecCommand,
    ExecCommandTool,
    ExecResult,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
)
from cayu.runners import DEFAULT_EXEC_OUTPUT_LIMIT_BYTES, Runner
from cayu.runners._cleanup import cleanup_runner_command_with_diagnostic
from cayu.runtime.context import validate_context_messages


class _FailedCommandKill:
    async def kill(self) -> bool:
        return False


class UnknownStateProcessRunner(Runner):
    """Real subprocess runner whose command cleanup deterministically fails."""

    isolation = "fault-injection"

    def __init__(self, root: Path) -> None:
        self.default_cwd = str(root)
        self._closed = False
        self._exec_closed = False
        self._exec_closed_reason = None
        self.process: asyncio.subprocess.Process | None = None
        self.exec_attempts = 0

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
        self.exec_attempts += 1
        if command.kind != "process" or command.argv is None:
            raise AssertionError("The fault runner only supports process commands.")
        if self.process is not None:
            raise AssertionError("The closed exec path must reject a second process launch.")

        child_env = os.environ.copy()
        if env is not None:
            child_env.update(env)
        self.process = await asyncio.create_subprocess_exec(
            *command.argv,
            cwd=self.resolve_cwd(cwd),
            env=child_env,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        if stdin is not None and self.process.stdin is not None:
            self.process.stdin.write(stdin.encode())
            await self.process.stdin.drain()
            self.process.stdin.close()

        try:
            await asyncio.wait_for(self.process.wait(), timeout=timeout_s)
        except TimeoutError:
            cleanup = await cleanup_runner_command_with_diagnostic(
                self,
                handle=_FailedCommandKill(),
                adapter=self.isolation,
                timeout_s=0.1,
                policy="command",
            )
            self._apply_cleanup_result(cleanup)
            return ExecResult(
                exit_code=-1,
                timed_out=True,
                artifacts=[cleanup.artifact],
            )

        return ExecResult(exit_code=self.process.returncode or 0)

    async def force_reap(self) -> None:
        process = self.process
        if process is None:
            return
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        await asyncio.wait_for(process.communicate(), timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process-group cleanup")
def test_runner_cleanup_failure_latches_exec_and_preserves_a_coherent_session(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "child.pid"
    child_script = (
        "import os, pathlib, sys, time; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8'); "
        "time.sleep(60)"
    )
    runner = UnknownStateProcessRunner(tmp_path)
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_timeout",
                    name="exec_command",
                    arguments={
                        "argv": [sys.executable, "-c", child_script, str(pid_path)],
                        "timeout_s": 1,
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_after_unknown_cleanup",
                    name="exec_command",
                    arguments={"argv": [sys.executable, "-c", "print('must not run')"]},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Stopped because command state is unknown."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ],
        name="runner-cleanup-fault-provider",
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="fault-runner"), runner=runner),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="runner-cleanup-assistant",
            model="scripted-model",
            provider_name="runner-cleanup-fault-provider",
        ),
        tools=[ExecCommandTool()],
    )

    async def exercise_contract():
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="runner-cleanup-assistant",
                        session_id="runner-cleanup-failure",
                        messages=[
                            Message.text("user", "Run the command and stop if cleanup fails.")
                        ],
                    )
                )
            ]
            transcript = await app.session_store.load_transcript("runner-cleanup-failure")
            assert runner.process is not None
            child_pid = int(pid_path.read_text(encoding="utf-8"))
            child_was_alive_with_unknown_state = runner.process.returncode is None
            return events, transcript, child_pid, child_was_alive_with_unknown_state
        finally:
            await runner.force_reap()

    events, transcript, child_pid, child_was_alive_with_unknown_state = asyncio.run(
        exercise_contract()
    )

    assert child_was_alive_with_unknown_state is True
    assert runner.process is not None
    assert runner.process.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)

    failed_calls = [event for event in events if event.type == EventType.TOOL_CALL_FAILED]
    assert len(failed_calls) == 2
    timeout_result = failed_calls[0].payload["result"]
    cleanup_artifact = {
        "type": "cayu.runner_cleanup.v1",
        "adapter": "fault-injection",
        "action": "kill_command",
        "status": "failed",
        "timeout_s": 0.1,
        "error": "kill returned false",
    }
    assert timeout_result["artifacts"] == [cleanup_artifact]
    assert timeout_result["structured"]["artifacts"] == [cleanup_artifact]
    assert "command state is unknown" in failed_calls[1].payload["result"]["content"]
    assert runner.exec_attempts == 1
    assert events[-1].type == EventType.SESSION_COMPLETED

    validate_context_messages(transcript)
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert transcript[2].content[0].artifacts == [cleanup_artifact]
