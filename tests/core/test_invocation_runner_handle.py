from __future__ import annotations

import asyncio
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError
from tests.core._workload_secret_support import FakeProvider, collect_events
from tests.provider_traceback_assertions import (
    assert_cayu_traceback_does_not_retain,
    is_cayu_source_filename,
)

import cayu.runners.local as local_runner_module
import cayu.runtime._invocation_secrets as invocation_secrets_module
import cayu.tools._operation_boundary as operation_boundary_module
import cayu.tools._resources as resources_module
import cayu.tools._runner as runner_module
from cayu._exception_groups import iter_exception_tree
from cayu._task_wait import capture_awaitable_outcome
from cayu._validation import compact_json_utf8_size
from cayu._workspace_mutation import (
    WorkspaceMutationProcessFence,
    workspace_mutation_task_settlement_probe,
)
from cayu.core import AgentSpec, EventType, Message
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import ModelStreamEvent
from cayu.proxies import PassthroughProxy
from cayu.runners import (
    ExecCommand,
    ExecResult,
    LocalRunner,
    Runner,
    RunnerCancelledError,
    RunnerExecutionError,
    RunnerUnavailableError,
    attach_cancellation_artifacts,
)
from cayu.runners._cleanup import runner_cancellation_failure, sanitize_runner_artifacts
from cayu.runtime import CayuApp, InMemorySessionStore, RunRequest, SessionStatus
from cayu.runtime._invocation_secrets import InvocationSecretTracker
from cayu.runtime._tool_execution import run_tool
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._resources import (
    InvocationWorkspaceMutationOwner,
    WorkspaceMutationSettlementError,
)
from cayu.tools._runner import (
    InvocationRunnerHandle,
    is_current_runner_cancellation_group,
    sanitize_runner_failure_group,
)
from cayu.vaults import (
    REDACTED_SECRET,
    ResolvedSecret,
    SecretRedactor,
    SecretRef,
    StaticVault,
    Vault,
)


class _BlockingRunner(Runner):
    def __init__(self) -> None:
        self.started: asyncio.Event | None = None
        self.cancelled = False

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        del command, cwd, env, timeout_s, stdin, output_limit_bytes
        assert self.started is not None
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return ExecResult()


class _BlockingCancellationProjectionStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self._armed = False
        self._blocked = False
        self.projection_started: asyncio.Event | None = None
        self.release_projection: asyncio.Event | None = None

    def arm(self) -> None:
        self._armed = True
        self.projection_started = asyncio.Event()
        self.release_projection = asyncio.Event()

    async def transform_checkpoint(self, session_id, checkpoint_transform) -> None:
        if self._armed and not self._blocked:
            self._blocked = True
            assert self.projection_started is not None
            assert self.release_projection is not None
            self.projection_started.set()
            await self.release_projection.wait()
        await super().transform_checkpoint(session_id, checkpoint_transform)


class _FatalCancellationProjectionStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self._armed = False
        self._failed = False

    def arm(self) -> None:
        self._armed = True

    async def transform_checkpoint(self, session_id, checkpoint_transform) -> None:
        if self._armed and not self._failed:
            self._failed = True
            raise GeneratorExit("checkpoint projection terminated")
        await super().transform_checkpoint(session_id, checkpoint_transform)


class _ConcurrentRunner(Runner):
    def __init__(self) -> None:
        self.arrivals = 0
        self.all_started: asyncio.Event | None = None

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        del cwd, env, timeout_s, stdin, output_limit_bytes
        assert command.argv is not None
        assert self.all_started is not None
        self.arrivals += 1
        if self.arrivals == 2:
            self.all_started.set()
        await self.all_started.wait()
        return ExecResult(stdout=command.argv[-1])


class _RevisionRunner(Runner):
    def __init__(self, result: ExecResult) -> None:
        self.result = result
        self.started: asyncio.Event | None = None
        self.release: asyncio.Event | None = None
        self.last_kwargs: dict = {}

    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        del command, kwargs
        raise AssertionError("Revision test must use exec_redacted.")

    async def exec_redacted(
        self,
        command: ExecCommand,
        *,
        redactor: SecretRedactor,
        **kwargs,
    ) -> ExecResult:
        del command, redactor
        self.last_kwargs = dict(kwargs)
        assert self.started is not None
        assert self.release is not None
        self.started.set()
        await self.release.wait()
        return self.result


class _DeferredSettlementRunner(Runner):
    pending_command_settlement_cancellation_safe = True

    def __init__(self, *, cancelled: bool) -> None:
        self.cancelled = cancelled
        self.started: asyncio.Event | None = None
        self.settlement_started: asyncio.Event | None = None
        self.release_settlement: asyncio.Event | None = None
        self.settlement_calls = 0

    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        del command, kwargs
        assert self.started is not None
        self.started.set()
        artifact = {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_command",
            "status": "deferred",
            "timeout_s": 5.0,
        }
        if not self.cancelled:
            return ExecResult(timed_out=True, artifacts=[artifact])
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancellation:
            attach_cancellation_artifacts(cancellation, [artifact])
            raise
        raise AssertionError("Cancellation runner unexpectedly resumed.")

    async def await_pending_command_settlement(self) -> bool:
        self.settlement_calls += 1
        assert self.settlement_started is not None
        assert self.release_settlement is not None
        self.settlement_started.set()
        await self.release_settlement.wait()
        return True


class _PostReturnMutatingSettlementRunner(_DeferredSettlementRunner):
    """Mutate extension-owned evidence after the dispatch boundary receives it."""

    pending_command_settlement_cancellation_safe = True

    def __init__(self) -> None:
        super().__init__(cancelled=False)
        self.result_mutated: asyncio.Event | None = None
        self.returned_result: ExecResult | None = None

    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        result = await super().exec(command, **kwargs)
        self.returned_result = result
        asyncio.get_running_loop().call_soon(self._mutate_returned_result)
        return result

    def _mutate_returned_result(self) -> None:
        assert self.returned_result is not None
        assert self.result_mutated is not None
        self.returned_result.timed_out = False
        self.returned_result.artifacts.clear()
        self.result_mutated.set()


class _PostRaiseMutatingSettlementRunner(_DeferredSettlementRunner):
    """Mutate exception cleanup evidence after the dispatch boundary receives it."""

    pending_command_settlement_cancellation_safe = True

    def __init__(self) -> None:
        super().__init__(cancelled=False)
        self.error_mutated: asyncio.Event | None = None
        self.returned_error: RuntimeError | None = None

    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        del command, kwargs
        assert self.started is not None
        self.started.set()
        error = RuntimeError("runner dispatch failed")
        error.artifacts = [
            {
                "type": "cayu.runner_cleanup.v1",
                "adapter": "e2b",
                "action": "kill_command",
                "status": "deferred",
                "timeout_s": 5.0,
            }
        ]
        self.returned_error = error
        asyncio.get_running_loop().call_soon(self._mutate_returned_error)
        raise error

    def _mutate_returned_error(self) -> None:
        assert self.returned_error is not None
        assert self.error_mutated is not None
        self.returned_error.artifacts.clear()
        self.error_mutated.set()


class _UncertainTimeoutRunner(Runner):
    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        del command, kwargs
        return ExecResult(
            timed_out=True,
            artifacts=[
                {
                    "type": "cayu.runner_cleanup.v1",
                    "adapter": "e2b",
                    "action": "kill_command",
                    "status": "timeout",
                    "timeout_s": 5.0,
                }
            ],
        )


class _OpaqueFailureRunner(Runner):
    isolation = "microsandbox"

    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        del command, kwargs
        raise RuntimeError("workload-secret-")


class _InjectedFailureRunner(Runner):
    isolation = "microsandbox"

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        del command, kwargs
        raise self.error


class _HostileAdapterValue:
    def __hash__(self) -> int:
        raise RuntimeError("adapter-secret-canary-ABCDEFGHIJKLMNOP")


class _HostileInstanceAdapterRunner(Runner):
    def __init__(self) -> None:
        self.isolation = _HostileAdapterValue()

    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        del command, kwargs
        raise RuntimeError("runner-secret-canary-ABCDEFGHIJKLMNOP")


class _SynchronousFailureRunner(Runner):
    isolation = "microsandbox"

    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        del command, kwargs
        raise AssertionError("Synchronous failure test must use exec_redacted.")

    def exec_redacted(  # type: ignore[override]
        self,
        command: ExecCommand,
        *,
        redactor: SecretRedactor,
        **kwargs,
    ) -> ExecResult:
        del command, redactor, kwargs
        raise RuntimeError("workload-secret-canary-ABCDEFGHIJKLMNOP")


class _LegacyCancelledRunner(Runner):
    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        del command, kwargs
        raise RunnerCancelledError(
            "legacy runner cancellation",
            artifacts=[
                {
                    "type": "cayu.runner_cleanup.v1",
                    "adapter": "microsandbox",
                    "action": "kill_command",
                    "status": "completed",
                    "timeout_s": 5.0,
                }
            ],
        )


class _TimeoutLegacyCancellationRunner(_BlockingRunner):
    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        try:
            return await super().exec(command, **kwargs)
        except asyncio.CancelledError:
            raise RunnerCancelledError("legacy timeout cancellation") from None


class _RewritingCancellationRunner(_BlockingRunner):
    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        try:
            return await super().exec(command, **kwargs)
        except asyncio.CancelledError:
            raise asyncio.CancelledError(
                "joined-secret-",
                "canary-ABCDEFGHIJKLMNOP",
            ) from None


class _CleanupFailureRunner(_BlockingRunner):
    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        try:
            return await super().exec(command, **kwargs)
        except asyncio.CancelledError as exc:
            attach_cancellation_artifacts(
                exc,
                [
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "microsandbox",
                        "action": "kill_command",
                        "status": "failed",
                        "timeout_s": 5.0,
                        "error_type": "RuntimeError",
                        "error": "workload-secret-",
                    }
                ],
            )
            raise


class _CleanupReplacingCancellationRunner(_BlockingRunner):
    isolation = "microsandbox"

    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        try:
            return await super().exec(command, **kwargs)
        finally:
            raise RuntimeError("runner-cleanup-secret-canary-ABCDEFGHIJKLMNOP")


class _CancellationSuppressingRunner(Runner):
    def __init__(self) -> None:
        self.called = False

    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        del command, kwargs
        self.called = True
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            return ExecResult(stdout="backend suppressed cancellation")
        raise AssertionError("Cancellation was not delivered to the runner.")


class _CancellationUncancellingRunner(_BlockingRunner):
    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        try:
            return await super().exec(command, **kwargs)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            assert task is not None
            task.uncancel()
            return ExecResult(stdout="backend consumed cancellation")


class _SynchronousCallerTaskCapturingRunner(Runner):
    def __init__(self, forged_reason: str) -> None:
        self.forged_reason = forged_reason
        self.started: asyncio.Event | None = None
        self.dispatch_task: asyncio.Task | None = None

    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        del command, kwargs
        raise AssertionError("This runner uses its synchronous redacted dispatcher.")

    def exec_redacted(
        self,
        command: ExecCommand,
        *,
        redactor: SecretRedactor,
        **kwargs,
    ):
        del command, redactor, kwargs
        dispatch_task = asyncio.current_task()
        assert dispatch_task is not None
        self.dispatch_task = dispatch_task

        async def operation() -> ExecResult:
            assert self.started is not None
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                dispatch_task.uncancel()
                dispatch_task.cancel(self.forged_reason)
                await asyncio.Event().wait()
            raise AssertionError("Cancellation was not delivered to the runner.")

        return operation()


class _CausalCancellationReasonRunner(_BlockingRunner):
    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        try:
            return await super().exec(command, **kwargs)
        except asyncio.CancelledError:
            raise RuntimeError("runner cleanup failed") from asyncio.CancelledError(
                "backend-controlled-secret-reason"
            )


class _FatalRunner(Runner):
    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        del command, kwargs
        raise GeneratorExit("runner terminated")


class _ImmediateRunner(Runner):
    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        del command, kwargs
        return ExecResult(stdout="completed")


class _OrdinaryGroupedFailureRunner(Runner):
    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        del command, kwargs
        raise ExceptionGroup("runner failed", [RuntimeError("ordinary failure")])


class _GroupedCleanupFailureRunner(_BlockingRunner):
    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        try:
            return await super().exec(command, **kwargs)
        except asyncio.CancelledError as cancellation:
            attach_cancellation_artifacts(
                cancellation,
                [
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "microsandbox",
                        "action": "kill_command",
                        "status": "failed",
                        "timeout_s": 5.0,
                        "error": "workload-secret-canary-ABCDEFGHIJKLMNOP",
                    }
                ],
            )
            cleanup = RuntimeError("workload-secret-canary-ABCDEFGHIJKLMNOP")
            raise BaseExceptionGroup(
                "workload-secret-canary-ABCDEFGHIJKLMNOP",
                [cancellation, cleanup],
            ) from cleanup


class _DelayedGroupedCleanupFailureRunner(_BlockingRunner):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_started: asyncio.Event | None = None

    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        try:
            return await super().exec(command, **kwargs)
        except asyncio.CancelledError:
            assert self.cleanup_started is not None
            self.cleanup_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as caller_cancellation:
                cleanup = RuntimeError("runner cleanup failed")
                raise BaseExceptionGroup(
                    "runner cleanup reported caller cancellation",
                    [caller_cancellation, cleanup],
                ) from cleanup


class _FatalCleanupGroupRunner(_BlockingRunner):
    isolation = "microsandbox"

    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        try:
            return await super().exec(command, **kwargs)
        except asyncio.CancelledError:
            raise BaseExceptionGroup(
                "runner-cleanup-secret-canary-ABCDEFGHIJKLMNOP",
                [GeneratorExit("runner-cleanup-secret-canary-ABCDEFGHIJKLMNOP")],
            ) from None


class _LateSecretRunner(Runner):
    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        del command, kwargs
        return ExecResult(
            stdout="late-registered-",
            stdout_truncated=True,
            stdout_bytes=64,
        )


class _MisreportedBoundedRunner(Runner):
    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        del command, kwargs
        raise AssertionError("Misreported capture test must use exec_redacted.")

    async def exec_redacted(
        self,
        command: ExecCommand,
        *,
        redactor: SecretRedactor,
        **kwargs,
    ) -> ExecResult:
        del command, redactor, kwargs
        return ExecResult(
            stdout="late-registered-",
            stdout_truncated=False,
            stdout_bytes=64,
        )


def test_runner_execution_observer_brackets_dispatch_with_bounded_evidence() -> None:
    observations: list[tuple[str, dict[str, Any]]] = []

    async def observe(
        phase: str,
        payload: dict[str, Any],
        command_evidence_revision: int,
    ) -> None:
        assert command_evidence_revision == 0
        observations.append((phase, payload))

    async def scenario() -> ExecResult:
        handle = InvocationRunnerHandle(
            _ImmediateRunner(),
            redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                revision=0,
                redactor=SecretRedactor(),
            ),
            execution_observer=observe,
        )
        return await handle.exec(ExecCommand.process("echo", "hello"))

    result = asyncio.run(scenario())

    assert result.stdout == "completed"
    assert [phase for phase, _payload in observations] == ["started", "completed"]
    started = observations[0][1]
    completed = observations[1][1]
    assert started == {
        "adapter": "unknown",
        "command": {
            "kind": "process",
            "arguments_state": "available",
            "preview_format": "exec_command_json_prefix",
            "preview": '{"argv":["echo","hello"],"kind":"process","shell":null}',
            "truncated": False,
        },
    }
    assert completed["command"] == started["command"]
    assert completed["exit_code"] == 0
    assert completed["timed_out"] is False
    assert type(completed["duration_ms"]) is int
    assert "stdout" not in completed
    assert "stderr" not in completed


def test_runner_execution_observer_bounds_and_redacts_command_evidence() -> None:
    secret = "runner-command-secret-canary-ABCDEFGHIJKLMNOP"
    observations: list[tuple[str, dict[str, Any]]] = []

    async def observe(
        phase: str,
        payload: dict[str, Any],
        command_evidence_revision: int,
    ) -> None:
        assert command_evidence_revision == 0
        observations.append((phase, payload))

    async def scenario() -> None:
        handle = InvocationRunnerHandle(
            _ImmediateRunner(),
            redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                revision=0,
                redactor=SecretRedactor(secret),
            ),
            execution_observer=observe,
        )
        await handle.exec(
            ExecCommand.process(
                "echo",
                secret,
                "retained-prefix-" + "x" * 16_000 + "-unretained-tail",
            )
        )

    asyncio.run(scenario())

    assert [phase for phase, _payload in observations] == ["started", "completed"]
    started = observations[0][1]["command"]
    completed = observations[1][1]["command"]
    assert started == completed
    assert started["arguments_state"] == "redacted_and_truncated"
    assert started["truncated"] is True
    assert REDACTED_SECRET in started["preview"]
    assert secret not in started["preview"]
    assert "unretained-tail" not in started["preview"]
    assert compact_json_utf8_size(started) <= runner_module._RUNNER_COMMAND_EVIDENCE_MAX_BYTES


@pytest.mark.parametrize(
    "command",
    [
        ExecCommand.process("echo", "hello"),
        ExecCommand.bash("printf 'hello'"),
    ],
)
def test_runner_command_json_prefix_matches_complete_compact_serialization(command) -> None:
    prefix, complete = runner_module._runner_command_json_prefix(command, max_bytes=4096)

    assert complete is True
    assert prefix.decode("utf-8") == json.dumps(
        command.model_dump(mode="json", warnings=False),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_runner_command_json_prefix_bounds_escaped_source_without_serializing_all_of_it() -> None:
    command = ExecCommand.process("echo", '"' * 100_000)
    prefix, complete = runner_module._runner_command_json_prefix(
        command,
        max_bytes=97,
    )

    assert complete is False
    assert len(prefix) == 97
    assert (
        prefix
        == json.dumps(
            command.model_dump(mode="json", warnings=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")[:97]
    )


def test_runner_execution_observer_quarantines_unpublished_arguments() -> None:
    observations: list[tuple[str, dict[str, Any]]] = []

    async def observe(
        phase: str,
        payload: dict[str, Any],
        command_evidence_revision: int,
    ) -> None:
        assert command_evidence_revision == 0
        observations.append((phase, payload))

    async def scenario() -> None:
        handle = InvocationRunnerHandle(
            _ImmediateRunner(),
            redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                revision=0,
                redactor=SecretRedactor(),
            ),
            execution_observer=observe,
            publish_execution_arguments=False,
        )
        await handle.exec(ExecCommand.process("echo", "private-command-material"))

    asyncio.run(scenario())

    assert [phase for phase, _payload in observations] == ["started", "completed"]
    assert observations[0][1]["command"] == {
        "kind": "process",
        "arguments_state": "unavailable",
    }
    assert observations[1][1]["command"] == observations[0][1]["command"]


def test_runtime_runner_events_honor_tool_argument_quarantine() -> None:
    private_material = "private-runner-command-material"

    class PrivateRunnerTool(Tool):
        spec = ToolSpec(
            name="private_runner",
            description="Run a command without publishing its arguments.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )

        @property
        def _publish_arguments(self) -> bool:
            return False

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.runner is not None
            result = await ctx.runner.exec(ExecCommand.process("echo", args["value"]))
            return ToolResult(content=result.stdout)

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call-private-runner",
                    name="private_runner",
                    arguments={"value": private_material},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="runner"),
            runner=_ImmediateRunner(),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[PrivateRunnerTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "run privately")],
            ),
        )
    )
    runner_events = [
        event
        for event in events
        if event.type in {EventType.RUNNER_EXEC_STARTED, EventType.RUNNER_EXEC_COMPLETED}
    ]

    assert len(runner_events) == 2
    assert all(
        event.payload["command"] == {"kind": "process", "arguments_state": "unavailable"}
        for event in runner_events
    )
    assert private_material not in repr(runner_events)


def test_runtime_persists_runner_start_before_dispatch_and_seals_completion() -> None:
    class BlockingAfterRunnerTool(Tool):
        spec = ToolSpec(
            name="runner_evidence_fence",
            description="Keep the invocation open after a runner command.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self, command_completed: asyncio.Event, release: asyncio.Event) -> None:
            self.command_completed = command_completed
            self.release = release

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            await ctx.runner.exec(ExecCommand.process("echo", "public-command-material"))
            self.command_completed.set()
            await self.release.wait()
            return ToolResult(content="done")

    async def scenario() -> tuple[list[Any], list[Any]]:
        store = InMemorySessionStore()
        command_completed = asyncio.Event()
        release = asyncio.Event()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call-runner-evidence-fence",
                        name="runner_evidence_fence",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="runner"),
                runner=_ImmediateRunner(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[BlockingAfterRunnerTool(command_completed, release)],
        )

        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="runner-evidence-fence",
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        await command_completed.wait()
        events_during_invocation = await store.load_events("runner-evidence-fence")
        release.set()
        await run_task
        durable_events = await store.load_events("runner-evidence-fence")
        return events_during_invocation, durable_events

    events_during_invocation, durable_events = asyncio.run(scenario())
    runner_events_during_invocation = [
        event
        for event in events_during_invocation
        if event.type in {EventType.RUNNER_EXEC_STARTED, EventType.RUNNER_EXEC_COMPLETED}
    ]
    assert [event.type for event in runner_events_during_invocation] == [
        EventType.RUNNER_EXEC_STARTED
    ]
    assert runner_events_during_invocation[0].payload["command"] == {
        "kind": "process",
        "arguments_state": "unavailable",
    }
    runner_events = [
        event
        for event in durable_events
        if event.type in {EventType.RUNNER_EXEC_STARTED, EventType.RUNNER_EXEC_COMPLETED}
    ]
    assert [event.type for event in runner_events] == [
        EventType.RUNNER_EXEC_STARTED,
        EventType.RUNNER_EXEC_COMPLETED,
    ]
    assert runner_events[0].payload["command"] == {
        "kind": "process",
        "arguments_state": "unavailable",
    }
    assert runner_events[1].payload["command"]["arguments_state"] == "available"
    assert "public-command-material" in runner_events[1].payload["command"]["preview"]


def test_runtime_runner_start_persistence_failure_prevents_dispatch() -> None:
    class RejectRunnerStartStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.runner_start_attempts = 0

        async def append_event(self, session_id: str, event) -> None:
            if event.type is EventType.RUNNER_EXEC_STARTED:
                self.runner_start_attempts += 1
                raise RuntimeError("runner start evidence unavailable")
            await super().append_event(session_id, event)

    class CountingRunner(_ImmediateRunner):
        def __init__(self) -> None:
            self.calls = 0

        async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
            self.calls += 1
            return await super().exec(command, **kwargs)

    class RunnerTool(Tool):
        spec = ToolSpec(
            name="runner_start_failure",
            description="Run one command.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            result = await ctx.runner.exec(ExecCommand.process("echo", "blocked"))
            return ToolResult(content=result.stdout)

    store = RejectRunnerStartStore()
    runner = CountingRunner()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call-runner-start-failure",
                    name="runner_start_failure",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="runner"), runner=runner),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[RunnerTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "do not dispatch")],
            ),
        )
    )

    assert store.runner_start_attempts == 1
    assert runner.calls == 0
    assert not any(
        event.type in {EventType.RUNNER_EXEC_STARTED, EventType.RUNNER_EXEC_COMPLETED}
        for event in events
    )


def test_runner_start_evidence_failure_prevents_dispatch() -> None:
    class CountingRunner(_ImmediateRunner):
        def __init__(self) -> None:
            self.calls = 0

        async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
            self.calls += 1
            return await super().exec(command, **kwargs)

    runner = CountingRunner()

    async def reject_start(
        phase: str,
        payload: dict[str, Any],
        command_evidence_revision: int,
    ) -> None:
        del phase, payload, command_evidence_revision
        raise RuntimeError("runner start evidence unavailable")

    async def scenario() -> None:
        handle = InvocationRunnerHandle(
            runner,
            redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                revision=0,
                redactor=SecretRedactor(),
            ),
            execution_observer=reject_start,
        )
        with pytest.raises(RunnerExecutionError):
            await handle.exec(ExecCommand.process("echo", "blocked"))

    asyncio.run(scenario())
    assert runner.calls == 0


def test_invocation_runner_handle_hides_invalid_command_input_before_snapshot() -> None:
    secret = "invocation-command-secret-canary-ABCDEFGHIJKLMNOP"
    snapshot_calls = 0

    def snapshot_provider() -> InvocationRedactorSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        )

    handle = InvocationRunnerHandle(
        _BlockingRunner(),
        redactor_snapshot_provider=snapshot_provider,
    )
    command = ExecCommand.process(
        "curl",
        "-H",
        f"Authorization: Bearer {secret}",
        "valid",
    )
    assert command.argv is not None
    command.argv[-1] = "invalid\x00argument"

    with pytest.raises(ValueError) as raised:
        asyncio.run(handle.exec(command))

    assert snapshot_calls == 0
    rendered = f"{raised.value!s} {raised.value!r}"
    assert secret not in rendered
    assert "Authorization: Bearer" not in rendered
    assert type(raised.value) is ValidationError
    assert secret not in repr(raised.value.errors())
    assert secret not in raised.value.json()
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    current = raised.value.__traceback__
    while current is not None:
        frame = current.tb_frame
        if is_cayu_source_filename(frame.f_code.co_filename):
            assert secret not in repr(frame.f_locals)
        current = current.tb_next


@pytest.mark.parametrize(
    "invalid_input",
    (
        "command_type",
        "cwd",
        "environment",
        "env_remove",
        "timeout",
        "stdin",
        "output_limit",
    ),
)
def test_invocation_runner_handle_validates_portable_request_before_snapshot(
    invalid_input: str,
) -> None:
    secret = "INVOCATION_PREFLIGHT_SECRET_CANARY_ABCDEFGHIJKLMNOP"
    snapshot_calls = 0

    def snapshot_provider() -> InvocationRedactorSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        )

    handle = InvocationRunnerHandle(
        _BlockingRunner(),
        redactor_snapshot_provider=snapshot_provider,
    )
    command: Any = ExecCommand.process("true")
    kwargs: dict[str, Any] = {}
    if invalid_input == "command_type":
        command = {"secret": secret}
    elif invalid_input == "cwd":
        kwargs["cwd"] = f"{secret}\ud800"
    elif invalid_input == "environment":
        kwargs["env"] = {"TOKEN": secret, "INVALID": "invalid\x00value"}
    elif invalid_input == "env_remove":
        kwargs["env_remove"] = (secret, "invalid\x00name")
    elif invalid_input == "timeout":
        kwargs["timeout_s"] = secret
    elif invalid_input == "stdin":
        kwargs["stdin"] = f"{secret}\ud800"
    else:
        kwargs["output_limit_bytes"] = secret

    with pytest.raises(RunnerExecutionError) as raised:
        asyncio.run(handle.exec(command, **kwargs))

    assert snapshot_calls == 0
    assert raised.value.diagnostic["error_type"] in {"Exception", "TypeError", "ValueError"}
    assert secret not in f"{raised.value!s} {raised.value!r}"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    current = raised.value.__traceback__
    while current is not None:
        frame = current.tb_frame
        if is_cayu_source_filename(frame.f_code.co_filename):
            assert secret not in repr(frame.f_locals)
        current = current.tb_next


def test_invocation_preflight_rejects_mutated_local_root_before_secret_state(
    tmp_path,
) -> None:
    class CountingVault(StaticVault):
        def __init__(self) -> None:
            super().__init__({"runner-token": "secret-value"})
            self.resolve_calls = 0

        async def resolve(self, ref, *, scope=None):  # type: ignore[no-untyped-def]
            self.resolve_calls += 1
            return await super().resolve(ref, scope=scope)

    vault = CountingVault()
    runner = LocalRunner(
        tmp_path,
        secret_env={"TOKEN": SecretRef(name="runner-token")},
        secret_resolver=vault,
    )
    runner.root = Path(f"{tmp_path}/invalid\x00root")
    snapshot_calls = 0

    def snapshot_provider() -> InvocationRedactorSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return InvocationRedactorSnapshot(revision=0, redactor=SecretRedactor())

    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=snapshot_provider,
    )

    with pytest.raises(RunnerExecutionError):
        asyncio.run(handle.exec(ExecCommand.process("true")))

    assert snapshot_calls == 0
    assert vault.resolve_calls == 0


def test_invocation_preflight_does_not_emit_mutated_command_serializer_diagnostics(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "mutated-command-serializer-secret-canary-ABCDEFGHIJKLMNOP"

    class SecretBearingValue:
        def __repr__(self) -> str:
            return secret

    runner = _CancellationSuppressingRunner()
    snapshot_calls = 0

    def snapshot_provider() -> InvocationRedactorSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return InvocationRedactorSnapshot(revision=0, redactor=SecretRedactor())

    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=snapshot_provider,
    )
    command = ExecCommand.process("true")
    assert command.argv is not None
    command.argv[0] = SecretBearingValue()  # type: ignore[list-item]

    with (
        warnings.catch_warnings(record=True) as emitted,
        caplog.at_level(logging.WARNING),
        pytest.raises(ValidationError) as raised,
    ):
        warnings.simplefilter("always")
        asyncio.run(handle.exec(command))

    captured = capsys.readouterr()
    diagnostic_output = " ".join(
        (
            str(raised.value),
            repr(raised.value),
            captured.out,
            captured.err,
            *(record.getMessage() for record in caplog.records),
            *(str(record.message) for record in emitted),
        )
    )
    assert emitted == []
    assert secret not in diagnostic_output
    assert snapshot_calls == 0
    assert runner.called is False


@pytest.mark.parametrize(
    ("case_sensitive", "override_name"),
    ((True, "API_TOKEN"), (False, "api_token")),
)
def test_invocation_preflight_rejects_local_secret_collision_before_secret_state(
    case_sensitive: bool,
    override_name: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingVault(StaticVault):
        def __init__(self) -> None:
            super().__init__({"runner-token": "secret-value"})
            self.resolve_calls = 0

        async def resolve(self, ref, *, scope=None):  # type: ignore[no-untyped-def]
            self.resolve_calls += 1
            return await super().resolve(ref, scope=scope)

    monkeypatch.setattr(
        local_runner_module,
        "_local_environment_names_case_sensitive",
        lambda: case_sensitive,
    )
    vault = CountingVault()
    runner = LocalRunner(
        tmp_path,
        secret_env={"API_TOKEN": SecretRef(name="runner-token")},
        secret_resolver=vault,
    )
    snapshot_calls = 0

    def snapshot_provider() -> InvocationRedactorSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return InvocationRedactorSnapshot(revision=0, redactor=SecretRedactor())

    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=snapshot_provider,
    )

    with pytest.raises(RunnerExecutionError):
        asyncio.run(
            handle.exec(
                ExecCommand.process("true"),
                env={override_name: "override"},
            )
        )

    assert snapshot_calls == 0
    assert vault.resolve_calls == 0


def test_invocation_preflight_failure_preserves_pending_caller_cancellation() -> None:
    secret = "pending-preflight-cancellation-secret-canary-ABCDEFGHIJKLMNOP"
    snapshot_calls = 0
    runner = _CancellationSuppressingRunner()

    def snapshot_provider() -> InvocationRedactorSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        )

    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=snapshot_provider,
    )

    async def invoke() -> ExecResult:
        task = asyncio.current_task()
        assert task is not None
        task.cancel(secret)
        return await handle.exec(
            ExecCommand.process("never-dispatched"),
            env={"TOKEN": secret, "INVALID": "invalid\x00value"},
        )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        task = asyncio.create_task(invoke())
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, task.cancelling(), task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert runner.called is False
    assert snapshot_calls == 1
    assert cancellation.args == (REDACTED_SECRET,)
    assert type(cancellation.__cause__) is RunnerExecutionError
    assert cancellation.__cause__.diagnostic["error_type"] == "Exception"
    assert cancellation.__context__ is None
    assert secret not in repr(cancellation)
    assert secret not in repr(cancellation.__cause__)


@pytest.mark.parametrize("operation", ("preflight_exec", "exec"))
def test_invocation_preflight_does_not_trust_runner_cancellation(operation: str) -> None:
    class ForgingPreflightRunner(Runner):
        isolation = "docker"

        def preflight_exec(self, command: ExecCommand, **kwargs) -> None:
            del command, kwargs
            raise asyncio.CancelledError("runner-forged cancellation")

        async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
            del command, kwargs
            raise AssertionError("Forged preflight cancellation must prevent dispatch.")

    handle = InvocationRunnerHandle(
        ForgingPreflightRunner(),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    async def scenario() -> tuple[RunnerExecutionError, int]:
        method = getattr(handle, operation)
        with pytest.raises(RunnerExecutionError) as raised:
            await method(ExecCommand.process("never-dispatched"))
        task = asyncio.current_task()
        assert task is not None
        return raised.value, task.cancelling()

    failure, cancelling = asyncio.run(scenario())

    assert cancelling == 0
    assert failure.diagnostic["adapter"] == "docker"
    assert failure.diagnostic["error_type"] == "CancelledError"
    assert failure.__cause__ is None
    assert failure.__context__ is None


def test_invocation_runner_handle_preserves_real_caller_cancellation() -> None:
    runner = _BlockingRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    async def scenario() -> tuple[int, bool]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(
            handle.exec(ExecCommand.process("blocked")),
        )
        await runner.started.wait()
        task.cancel("caller cancelled")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
            await task
        return cancelling, task.cancelled()

    cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert runner.cancelled is True


def test_runner_cleanup_failure_cannot_replace_real_task_cancellation() -> None:
    secret = "runner-cleanup-secret-canary-ABCDEFGHIJKLMNOP"
    runner = _CleanupReplacingCancellationRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        runner.started = asyncio.Event()

        async def invoke() -> ExecResult:
            return await handle.exec(ExecCommand.process("blocked"))

        task = asyncio.create_task(invoke())
        await runner.started.wait()
        task.cancel(secret)
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, task.cancelling(), task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == (REDACTED_SECRET,)
    assert type(cancellation.__cause__) is RunnerExecutionError
    assert cancellation.__cause__.diagnostic == {
        "type": "cayu.runner_execution_error.v1",
        "adapter": "microsandbox",
        "status": "failed",
        "error_type": "RuntimeError",
        "timed_out": False,
        "cancelled": False,
    }
    assert cancellation.__context__ is None
    assert secret not in repr(cancellation)
    assert secret not in repr(cancellation.__cause__)


def test_runner_preserves_pending_cancellation_suppressed_by_delegate() -> None:
    secret = "runner-pending-cancellation-secret-canary-ABCDEFGHIJKLMNOP"
    runner = _CancellationSuppressingRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool, bool]:
        async def invoke() -> tuple[asyncio.CancelledError, int]:
            task = asyncio.current_task()
            assert task is not None
            task.cancel(secret)
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await handle.exec(ExecCommand.process("suppressed"))
            cancelling = task.cancelling()
            # Catching the facade's sanitized signal must consume the original
            # delivery too. A subsequent cleanup await cannot receive the raw
            # Task.cancel() reason a second time.
            await asyncio.sleep(0)
            task.uncancel()
            return exc_info.value, cancelling

        task = asyncio.create_task(invoke())
        cancellation, cancelling = await task
        return cancellation, cancelling, task.cancelled(), runner.called

    cancellation, cancelling, cancelled, dispatched = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is False
    assert dispatched is False
    assert cancellation.args == (REDACTED_SECRET,)
    assert cancellation.__cause__ is None
    assert cancellation.__context__ is None
    assert secret not in repr(cancellation)


def test_runner_delegate_cannot_uncancel_the_callers_task() -> None:
    runner = _CancellationUncancellingRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(handle.exec(ExecCommand.process("blocked")))
        await runner.started.wait()
        task.cancel("caller cancelled")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, task.cancelling(), task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancellation.args == ("caller cancelled",)
    assert cancelling == 1
    assert cancelled is True


def test_synchronous_runner_dispatch_cannot_replace_caller_cancellation() -> None:
    secret = "forged-cancellation-secret-canary-ABCDEFGHIJKLMNOP"
    runner = _SynchronousCallerTaskCapturingRunner(secret)
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    async def scenario() -> tuple[asyncio.CancelledError, asyncio.Task, int, bool]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(handle.exec(ExecCommand.process("blocked")))
        await runner.started.wait()
        task.cancel("authentic caller reason")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, task, task.cancelling(), task.cancelled()

    cancellation, caller_task, cancelling, cancelled = asyncio.run(scenario())

    assert runner.dispatch_task is not caller_task
    assert cancellation.args == ("authentic caller reason",)
    assert cancellation.__cause__ is None
    assert cancellation.__context__ is None
    assert secret not in repr(cancellation)
    assert cancelling == 1
    assert cancelled is True


def test_runner_failure_cause_cannot_forge_the_caller_cancellation_reason() -> None:
    secret = "backend-controlled-secret-reason"
    runner = _CausalCancellationReasonRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(handle.exec(ExecCommand.process("blocked")))
        await runner.started.wait()
        task.cancel("authentic caller reason")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, task.cancelling(), task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancellation.args == ("authentic caller reason",)
    assert type(cancellation.__cause__) is RunnerExecutionError
    assert secret not in repr(cancellation)
    assert secret not in repr(cancellation.__cause__)
    assert cancelling == 1
    assert cancelled is True


def test_runner_preserves_child_generator_exit() -> None:
    handle = InvocationRunnerHandle(
        _FatalRunner(),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    with pytest.raises(GeneratorExit, match="runner terminated"):
        asyncio.run(handle.exec(ExecCommand.process("fatal")))


def test_runner_does_not_reclassify_historical_task_cancellation() -> None:
    handle = InvocationRunnerHandle(
        _ImmediateRunner(),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    async def scenario() -> tuple[ExecResult, int]:
        task = asyncio.current_task()
        assert task is not None
        task.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        historical_count = task.cancelling()
        try:
            return await handle.exec(ExecCommand.process("complete")), historical_count
        finally:
            task.uncancel()

    result, historical_count = asyncio.run(scenario())

    assert historical_count == 1
    assert result.stdout == "completed"


def test_ordinary_runner_exception_group_skips_cancellation_tree_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = InvocationRunnerHandle(
        _OrdinaryGroupedFailureRunner(),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    def unexpected_group_rebuild(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Ordinary groups must not use cancellation sanitization.")

    monkeypatch.setattr(
        runner_module,
        "sanitize_runner_failure_group",
        unexpected_group_rebuild,
    )

    with pytest.raises(RunnerExecutionError) as exc_info:
        asyncio.run(handle.exec(ExecCommand.process("grouped-failure")))

    assert exc_info.value.diagnostic["error_type"] == "ExceptionGroup"


@pytest.mark.parametrize(
    "runner_type",
    [
        _CleanupReplacingCancellationRunner,
        _GroupedCleanupFailureRunner,
        _FatalCleanupGroupRunner,
    ],
    ids=[
        "ordinary-cleanup-failure",
        "mixed-cancellation-cleanup-group",
        "fatal-cleanup-group",
    ],
)
def test_runner_cleanup_failure_cancellation_stops_the_runtime_turn(
    runner_type: type[_BlockingRunner],
) -> None:
    secret = "runner-cleanup-secret-canary-ABCDEFGHIJKLMNOP"
    session_id = f"runner-cleanup-cancellation-{runner_type.__name__}"
    runner = runner_type()

    class RunCommandTool(Tool):
        spec = ToolSpec(
            name="run_during_cancellation",
            description="Run through the invocation runner.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            assert ctx.vault is not None
            assert ctx.proxy is not None
            await ctx.vault.resolve(SecretRef(name="token"))
            await ctx.proxy.authorize_request(
                destination="https://example.test/operation",
                credential=SecretRef(name="token"),
            )
            await ctx.runner.exec(ExecCommand.process("blocked"))
            return ToolResult(content="unexpected")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call-runner",
                    name="run_during_cancellation",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("unexpected"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    store = _BlockingCancellationProjectionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            runner=runner,
            vault=StaticVault({"token": secret}),
            proxy=PassthroughProxy(StaticVault({"token": secret})),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[RunCommandTool()],
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, int, bool]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        await runner.started.wait()
        store.arm()
        task.cancel(secret)
        initial_cancelling = task.cancelling()
        assert store.projection_started is not None
        assert store.release_projection is not None
        await store.projection_started.wait()
        task.cancel("caller cancellation during publication")
        final_cancelling = task.cancelling()
        store.release_projection.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return (
            exc_info.value,
            initial_cancelling,
            final_cancelling,
            task.cancelled(),
        )

    cancellation, initial_cancelling, final_cancelling, cancelled = asyncio.run(scenario())
    events = asyncio.run(store.load_events(session_id))
    session = asyncio.run(store.load(session_id))

    assert initial_cancelling == 1
    assert final_cancelling == 2
    assert cancelled is True
    assert len(provider.requests) == 1
    assert session is not None
    assert session.status is SessionStatus.INTERRUPTED
    assert [event.type for event in events].count(EventType.SESSION_INTERRUPTED) == 1
    assert all(event.type is not EventType.CREDENTIAL_PROXY_CHECKED for event in events)
    assert all(event.type is not EventType.SESSION_COMPLETED for event in events)
    assert secret not in repr(cancellation)
    assert secret not in repr(events)
    if runner_type is _GroupedCleanupFailureRunner:
        assert cancellation.artifacts == [
            {
                "type": "cayu.runner_cleanup.v1",
                "adapter": "microsandbox",
                "action": "kill_command",
                "status": "failed",
                "timeout_s": 5.0,
            }
        ]


def test_fatal_snapshot_failure_does_not_replace_runner_cancellation() -> None:
    runner = _CleanupReplacingCancellationRunner()

    class RunCommandTool(Tool):
        spec = ToolSpec(
            name="run_during_fatal_projection",
            description="Run through the invocation runner.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            await ctx.runner.exec(ExecCommand.process("blocked"))
            return ToolResult(content="unexpected")

    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call-fatal-projection",
                name="run_during_fatal_projection",
                arguments={},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    store = _FatalCancellationProjectionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), runner=runner),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[RunCommandTool()],
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="fatal-cancellation-projection",
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        await runner.started.wait()
        store.arm()
        task.cancel("caller cancellation")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())
    events = asyncio.run(store.load_events("fatal-cancellation-projection"))
    session = asyncio.run(store.load("fatal-cancellation-projection"))

    assert cancelling == 1
    assert cancelled is True
    assert session is not None
    assert session.status is SessionStatus.INTERRUPTED
    assert [event.type for event in events].count(EventType.SESSION_INTERRUPTED) == 1
    assert all(event.type is not EventType.SESSION_COMPLETED for event in events)
    notes = getattr(cancellation, "__notes__", ())
    assert notes == [
        "Assistant publication projection terminated with GeneratorExit while "
        "preserving cancellation."
    ]
    assert "checkpoint projection terminated" not in repr(cancellation)
    assert "checkpoint projection terminated" not in repr(events)


def test_parallel_runner_cleanup_failures_cross_task_group_cancellation() -> None:
    secret = "parallel-runner-cleanup-secret-canary-ABCDEFGHIJKLMNOP"

    class ParallelGroupedCleanupRunner(Runner):
        isolation = "microsandbox"

        def __init__(self) -> None:
            self.arrivals = 0
            self.all_started: asyncio.Event | None = None

        async def exec(
            self,
            command: ExecCommand,
            **kwargs,
        ) -> ExecResult:
            del command, kwargs
            assert self.all_started is not None
            self.arrivals += 1
            if self.arrivals == 2:
                self.all_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise BaseExceptionGroup(
                    "runner cleanup",
                    [cancellation, RuntimeError(secret)],
                ) from None
            return ExecResult()

    class RunCommandTool(Tool):
        spec = ToolSpec(
            name="parallel_runner_cleanup",
            description="Block in a parallel runner command.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            assert ctx.vault is not None
            await ctx.vault.resolve(SecretRef(name="token"))
            await ctx.runner.exec(ExecCommand.process("blocked"))
            return ToolResult(content="unexpected")

    runner = ParallelGroupedCleanupRunner()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call-runner-a",
                name="parallel_runner_cleanup",
                arguments={},
            ),
            ModelStreamEvent.tool_call(
                id="call-runner-b",
                name="parallel_runner_cleanup",
                arguments={},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        max_parallel_tool_calls=2,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            runner=runner,
            vault=StaticVault({"token": secret}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[RunCommandTool()],
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        runner.all_started = asyncio.Event()
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="parallel-runner-cleanup",
                    messages=[Message.text("user", "run both")],
                ),
            )
        )
        await runner.all_started.wait()
        task.cancel("caller cancellation")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())
    events = asyncio.run(store.load_events("parallel-runner-cleanup"))
    session = asyncio.run(store.load("parallel-runner-cleanup"))
    assert isinstance(cancellation.__cause__, BaseExceptionGroup)
    cleanup_failures = [
        candidate
        for candidate in iter_exception_tree(cancellation.__cause__)
        if isinstance(candidate, RunnerExecutionError)
    ]

    assert cancelling == 1
    assert cancelled is True
    assert len(provider.requests) == 1
    assert session is not None
    assert session.status is SessionStatus.INTERRUPTED
    assert [event.type for event in events].count(EventType.SESSION_INTERRUPTED) == 1
    assert all(event.type is not EventType.SESSION_COMPLETED for event in events)
    assert len(cleanup_failures) == 2
    assert all(failure.diagnostic["error_type"] == "RuntimeError" for failure in cleanup_failures)
    assert secret not in repr(cancellation)
    assert secret not in repr(events)


def test_external_cancellation_resanitizes_mutated_authenticated_runner_failure() -> None:
    secret = "mutated-runner-cause-secret-canary-ABCDEFGHIJKLMNOP"
    runner = _CleanupReplacingCancellationRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    async def scenario() -> tuple[
        asyncio.CancelledError,
        BaseException,
    ]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(handle.exec(ExecCommand.process("blocked")))
        await runner.started.wait()
        task.cancel("caller cancellation")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        cancellation = exc_info.value
        original_cause = cancellation.__cause__
        assert isinstance(original_cause, RunnerExecutionError)
        original_cause.add_note(secret)
        try:
            raise RuntimeError(secret)
        except RuntimeError as injected_failure:
            original_cause.__cause__ = injected_failure
            original_cause.__context__ = injected_failure
            original_cause.__traceback__ = injected_failure.__traceback__
        invocation_secrets_module.initialize_cancellation_evidence(cancellation)
        invocation_secrets_module.set_cancellation_redactor(
            cancellation,
            SecretRedactor(secret),
        )
        invocation_secrets_module.sanitize_external_cancellation(cancellation)
        return cancellation, original_cause

    cancellation, original_cause = asyncio.run(scenario())
    published_cause = cancellation.__cause__

    assert isinstance(published_cause, RunnerExecutionError)
    assert published_cause is not original_cause
    assert published_cause.__traceback__ is None
    assert published_cause.__cause__ is None
    assert published_cause.__context__ is None
    assert getattr(published_cause, "__notes__", None) is None
    assert secret not in repr(cancellation)
    assert secret not in repr(published_cause)
    assert secret not in repr(cancellation.__dict__)
    assert cancellation.__dict__ == {}


def test_parallel_grouped_child_cancellation_preserves_runner_cleanup_failure() -> None:
    secret = "parallel-sibling-runner-cleanup-secret-canary-ABCDEFGHIJKLMNOP"
    runner = _CleanupReplacingCancellationRunner()
    sibling_started: asyncio.Event | None = None

    class RunCommandTool(Tool):
        spec = ToolSpec(
            name="run_until_sibling_fails",
            description="Block in a runner command.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            assert ctx.vault is not None
            await ctx.vault.resolve(SecretRef(name="token"))
            try:
                await ctx.runner.exec(ExecCommand.process("blocked"))
            except asyncio.CancelledError as cancellation:
                raise BaseExceptionGroup(
                    "tool grouped runner cancellation",
                    [cancellation],
                ) from None
            return ToolResult(content="unexpected")

    class FatalSiblingTool(Tool):
        spec = ToolSpec(
            name="fail_parallel_sibling",
            description="Fail after the runner command is dispatched.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            assert sibling_started is not None
            assert runner.started is not None
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise GeneratorExit("parallel sibling terminated") from None
            return ToolResult(content="unexpected")

    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call-runner",
                name="run_until_sibling_fails",
                arguments={},
            ),
            ModelStreamEvent.tool_call(
                id="call-fatal",
                name="fail_parallel_sibling",
                arguments={},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    store = _BlockingCancellationProjectionStore()
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        max_parallel_tool_calls=2,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            runner=runner,
            vault=StaticVault({"token": secret}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[RunCommandTool(), FatalSiblingTool()],
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, int, bool]:
        nonlocal sibling_started
        runner.started = asyncio.Event()
        sibling_started = asyncio.Event()
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="parallel-fatal-sibling",
                    messages=[Message.text("user", "run both")],
                ),
            )
        )
        await runner.started.wait()
        await sibling_started.wait()
        store.arm()
        task.cancel("caller cancellation")
        initial_cancelling = task.cancelling()
        assert store.projection_started is not None
        assert store.release_projection is not None
        await store.projection_started.wait()
        task.cancel("caller cancellation during publication")
        final_cancelling = task.cancelling()
        store.release_projection.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return (
            exc_info.value,
            initial_cancelling,
            final_cancelling,
            task.cancelled(),
        )

    cancellation, initial_cancelling, final_cancelling, cancelled = asyncio.run(scenario())
    session = asyncio.run(store.load("parallel-fatal-sibling"))
    events = asyncio.run(store.load_events("parallel-fatal-sibling"))
    assert isinstance(cancellation.__cause__, BaseExceptionGroup)
    cleanup_failures: list[RunnerExecutionError] = []
    cause = cancellation.__cause__
    while cause is not None:
        cleanup_failures.extend(
            candidate
            for candidate in iter_exception_tree(cause)
            if isinstance(candidate, RunnerExecutionError)
            and all(candidate is not existing for existing in cleanup_failures)
        )
        cause = cause.__cause__
    assert initial_cancelling == 1
    assert final_cancelling == 2
    assert cancelled is True
    assert session is not None
    assert session.status is SessionStatus.INTERRUPTED
    assert [event.type for event in events].count(EventType.SESSION_INTERRUPTED) == 1
    assert all(event.type is not EventType.SESSION_COMPLETED for event in events)
    assert {failure.diagnostic["error_type"] for failure in cleanup_failures} == {
        "GeneratorExit",
        "RuntimeError",
    }
    assert all(failure.__cause__ is None for failure in cleanup_failures)
    assert all(failure.__context__ is None for failure in cleanup_failures)
    assert secret not in repr(cancellation)
    assert secret not in repr(cleanup_failures)


@pytest.mark.parametrize("secret_input", ["command", "env", "stdin"])
def test_invocation_runner_handle_detaches_secret_bearing_cancellation_traceback(
    tmp_path,
    secret_input: str,
) -> None:
    secret = f"runner-{secret_input}-cancellation-secret-canary-ABCDEFGHIJKLMNOP"
    runner = LocalRunner(tmp_path)
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        kwargs = {
            "env": {"WORKLOAD_TOKEN": secret} if secret_input == "env" else None,
            "stdin": secret if secret_input == "stdin" else None,
        }
        command = ExecCommand.process(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            *([secret] if secret_input == "command" else []),
        )
        task = asyncio.create_task(
            handle.exec(
                command,
                **kwargs,
            )
        )
        await asyncio.sleep(0.1)
        task.cancel("caller cancelled")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert type(cancellation) is asyncio.CancelledError
    assert cancellation.args == ("caller cancelled",)
    assert cancellation.__cause__ is None
    assert cancellation.__context__ is None
    cayu_frames: list[str] = []
    traceback = cancellation.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if is_cayu_source_filename(frame.f_code.co_filename):
            cayu_frames.append(frame.f_code.co_name)
            assert all(
                value is not handle and value is not runner for value in frame.f_locals.values()
            )
            for name, value in frame.f_locals.items():
                assert secret not in repr(value), (
                    frame.f_code.co_filename,
                    frame.f_code.co_name,
                    name,
                )
        traceback = traceback.tb_next
    assert cayu_frames == ["exec", "_raise_clean_runner_cancellation"]


def test_invocation_runner_handle_preserves_cancellation_before_dispatch() -> None:
    runner = _BlockingRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(handle.exec(ExecCommand.process("blocked")))
        task.cancel("cancelled before dispatch")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancellation.args == ("cancelled before dispatch",)
    assert cancelling == 1
    assert cancelled is True
    assert runner.started is not None
    assert runner.started.is_set() is False


def test_runner_cancellation_before_owned_dispatch_does_not_quarantine_reuse() -> None:
    class PreDispatchCancellingRunner(Runner):
        def __init__(self) -> None:
            self.cancel_preflight = True
            self.exec_calls = 0

        def preflight_exec(self, command: ExecCommand, **kwargs) -> None:
            del command, kwargs
            if self.cancel_preflight:
                self.cancel_preflight = False
                current = asyncio.current_task()
                assert current is not None
                current.cancel("cancel before owned runner dispatch")

        async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
            del command, kwargs
            self.exec_calls += 1
            return ExecResult()

    runner = PreDispatchCancellingRunner()
    fence = WorkspaceMutationProcessFence()

    def handle(owner: InvocationWorkspaceMutationOwner) -> InvocationRunnerHandle:
        return InvocationRunnerHandle(
            runner,
            redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                revision=0,
                redactor=SecretRedactor(),
            ),
            mutation_owner=owner,
        )

    async def scenario() -> tuple[asyncio.Task[ExecResult], ExecResult]:
        first_owner = InvocationWorkspaceMutationOwner(
            on_settlement_unproven=fence.fail_closed,
        )
        first = asyncio.create_task(handle(first_owner).exec(ExecCommand.process("not-dispatched")))
        with pytest.raises(
            asyncio.CancelledError,
            match="cancel before owned runner dispatch",
        ):
            await first
        await first_owner.seal_and_wait()
        fence.require_available_nowait()

        second_owner = InvocationWorkspaceMutationOwner(
            on_settlement_unproven=fence.fail_closed,
        )
        result = await handle(second_owner).exec(ExecCommand.process("dispatched"))
        await second_owner.seal_and_wait()
        fence.require_available_nowait()
        return first, result

    first, result = asyncio.run(scenario())

    assert first.cancelling() == 1
    assert first.cancelled() is True
    assert runner.exec_calls == 1
    assert result.exit_code == 0


def test_invocation_runner_handle_redacts_caller_cancellation_reason() -> None:
    secret = "caller-cancellation-secret-canary-ABCDEFGHIJKLMNOP"
    runner = _BlockingRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    async def scenario() -> asyncio.CancelledError:
        runner.started = asyncio.Event()
        task = asyncio.create_task(handle.exec(ExecCommand.process("blocked")))
        await runner.started.wait()
        task.cancel(f"operator stop: {secret}")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        assert task.cancelled() is True
        return exc_info.value

    cancellation = asyncio.run(scenario())

    assert cancellation.args == (f"operator stop: {REDACTED_SECRET}",)
    assert secret not in repr(cancellation)


def test_invocation_runner_handle_redacts_numeric_caller_cancellation_reason() -> None:
    secret = "1234567890123456"
    runner = _BlockingRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    async def scenario() -> asyncio.CancelledError:
        runner.started = asyncio.Event()
        task = asyncio.create_task(handle.exec(ExecCommand.process("blocked")))
        await runner.started.wait()
        task.cancel(1234567890123456)
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value

    cancellation = asyncio.run(scenario())

    assert cancellation.args == (REDACTED_SECRET,)
    assert secret not in repr(cancellation)


def test_invocation_runner_handle_rejects_reason_that_reconstructs_secret() -> None:
    secret = "joined-secret-', 'canary-ABCDEFGHIJKLMNOP"
    runner = _RewritingCancellationRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    async def scenario() -> asyncio.CancelledError:
        runner.started = asyncio.Event()
        task = asyncio.create_task(handle.exec(ExecCommand.process("blocked")))
        await runner.started.wait()
        task.cancel("caller cancelled")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value

    cancellation = asyncio.run(scenario())

    assert cancellation.args == ("caller cancelled",)
    assert secret not in repr(cancellation)


def test_invocation_runner_handle_preserves_legacy_cancellation_subtype() -> None:
    handle = InvocationRunnerHandle(
        _LegacyCancelledRunner(),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    async def scenario() -> tuple[RunnerCancelledError, bool]:
        task = asyncio.create_task(handle.exec(ExecCommand.process("cancel")))
        with pytest.raises(RunnerCancelledError) as exc_info:
            await task
        return exc_info.value, task.cancelled()

    cancellation, cancelled = asyncio.run(scenario())

    assert type(cancellation) is RunnerCancelledError
    assert cancellation.args == ("Runner command was cancelled.",)
    assert cancellation.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]
    assert cancelled is True


def test_invocation_runner_handle_downgrades_unsafe_legacy_cancellation_type() -> None:
    handle = InvocationRunnerHandle(
        _LegacyCancelledRunner(),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor("RunnerCancelledError"),
        ),
    )

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(handle.exec(ExecCommand.process("cancel")))

    cancellation = exc_info.value
    assert type(cancellation) is asyncio.CancelledError
    assert "RunnerCancelledError" not in repr(cancellation)


def test_caller_cancellation_downgrades_legacy_runner_replacement() -> None:
    runner = _TimeoutLegacyCancellationRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(handle.exec(ExecCommand.process("blocked")))
        await runner.started.wait()
        task.cancel("caller cancelled")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert type(cancellation) is asyncio.CancelledError
    assert cancelling == 1
    assert cancelled is True


def test_invocation_runner_handle_preserves_mixed_cancellation_cleanup_group() -> None:
    runner = _GroupedCleanupFailureRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor("workload-secret-canary-ABCDEFGHIJKLMNOP"),
        ),
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(handle.exec(ExecCommand.process("blocked")))
        await runner.started.wait()
        task.cancel("caller cancellation")
        cancelling = task.cancelling()
        try:
            await task
        except asyncio.CancelledError as cancellation:
            return cancellation, cancelling, task.cancelled()
        raise AssertionError("Authentic caller cancellation was not propagated.")

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == ("caller cancellation",)
    assert cancellation.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_command",
            "status": "failed",
            "timeout_s": 5.0,
        }
    ]
    assert isinstance(cancellation.__cause__, ExceptionGroup)
    assert len(cancellation.__cause__.exceptions) == 1
    assert isinstance(cancellation.__cause__.exceptions[0], RunnerExecutionError)
    assert "workload-secret" not in repr(cancellation)
    assert "workload-secret" not in repr(cancellation.__cause__)
    assert cancellation.__context__ is None


def test_invocation_runner_handle_iteratively_partitions_deep_cancellation_group() -> None:
    class DeepGroupedCleanupFailureRunner(_BlockingRunner):
        async def exec(
            self,
            command: ExecCommand,
            **kwargs,
        ) -> ExecResult:
            try:
                return await super().exec(command, **kwargs)
            except asyncio.CancelledError as cancellation:
                nested: BaseException = cancellation
                for _ in range(10_000):
                    nested = BaseExceptionGroup("nested cancellation", [nested])
                raise BaseExceptionGroup(
                    "runner cleanup",
                    [nested, RuntimeError("cleanup failed")],
                ) from None

    runner = DeepGroupedCleanupFailureRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(handle.exec(ExecCommand.process("blocked")))
        await runner.started.wait()
        task.cancel("caller cancellation")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancellation.args == ("caller cancellation",)
    assert isinstance(cancellation.__cause__, ExceptionGroup)
    assert len(cancellation.__cause__.exceptions) == 1
    assert isinstance(cancellation.__cause__.exceptions[0], RunnerExecutionError)
    assert cancellation.__context__ is None
    assert cancelling == 1
    assert cancelled is True


def test_invocation_runner_handle_keeps_cancellation_for_group_without_cancel_leaf() -> None:
    secret = "runner-cleanup-secret-canary-ABCDEFGHIJKLMNOP"
    runner = _FatalCleanupGroupRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(handle.exec(ExecCommand.process("blocked")))
        await runner.started.wait()
        task.cancel("authentic caller cancellation")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, task.cancelling(), task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancellation.args == ("authentic caller cancellation",)
    assert isinstance(cancellation.__cause__, ExceptionGroup)
    assert len(cancellation.__cause__.exceptions) == 1
    assert isinstance(cancellation.__cause__.exceptions[0], RunnerExecutionError)
    assert cancellation.__context__ is None
    assert secret not in repr(cancellation)
    assert secret not in repr(cancellation.__cause__)
    assert cancelling == 1
    assert cancelled is True


def test_parallel_group_resanitization_removes_raw_sibling_failure() -> None:
    secret = "parallel-sibling-secret-canary-ABCDEFGHIJKLMNOP"

    class FatalToolFailure(BaseException):
        pass

    inner = sanitize_runner_failure_group(
        BaseExceptionGroup(
            "raw runner failure",
            [asyncio.CancelledError("caller cancellation"), RuntimeError(secret)],
        ),
        caller_cancelled=True,
    )
    outer = BaseExceptionGroup(
        "raw task group",
        [inner, FatalToolFailure(secret)],
    )

    sanitized = sanitize_runner_failure_group(
        outer,
        caller_cancelled=True,
    )

    assert is_current_runner_cancellation_group(sanitized) is True
    assert isinstance(sanitized.exceptions[0], BaseExceptionGroup)
    assert type(sanitized.exceptions[0].exceptions[0]) is asyncio.CancelledError
    assert isinstance(sanitized.exceptions[0].exceptions[1], RunnerExecutionError)
    assert isinstance(sanitized.exceptions[1], RunnerExecutionError)
    assert secret not in repr(sanitized)
    assert sanitized.__cause__ is None
    assert sanitized.__context__ is None


@pytest.mark.parametrize("hostile_value", [[], {}, True])
def test_runner_failure_sanitizer_ignores_wrong_type_membership_values(
    hostile_value: object,
) -> None:
    raw = RunnerUnavailableError(
        "secret-bearing runner failure",
        diagnostic={
            "adapter": hostile_value,
            "reason": hostile_value,
            "probe": {
                "method": "Sandbox.ping",
                "status": hostile_value,
            },
        },
    )

    sanitized = sanitize_runner_failure_group(
        ExceptionGroup("runner failure", [raw]),
    )

    failure = sanitized.exceptions[0]
    assert isinstance(failure, RunnerUnavailableError)
    assert failure.diagnostic["adapter"] == "unknown"
    assert "reason" not in failure.diagnostic
    assert failure.diagnostic["probe"] == {"method": "Sandbox.ping"}
    assert failure.__cause__ is None
    assert failure.__context__ is None


def test_cleanup_artifact_omits_overflowing_numeric_fields() -> None:
    artifacts = sanitize_runner_artifacts(
        [
            {
                "type": "cayu.runner_cleanup.v1",
                "adapter": "docker",
                "action": "kill_command",
                "status": "completed",
                "timeout_s": 1.0,
                "late_start_cleanup_timeout_s": 10**400,
            }
        ]
    )

    assert artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "docker",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 1.0,
        }
    ]


def test_tool_timeout_classifies_grouped_runner_cleanup_as_timeout() -> None:
    runner = _GroupedCleanupFailureRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor("workload-secret-canary-ABCDEFGHIJKLMNOP"),
        ),
    )

    class BlockingTool(Tool):
        spec = ToolSpec(
            name="blocking",
            description="Block in a runner command.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            await ctx.runner.exec(ExecCommand.process("blocked"))
            return ToolResult(content="unexpected")

    async def scenario():
        runner.started = asyncio.Event()
        return await run_tool(
            tool=BlockingTool(),
            effect=ToolEffect.NONE,
            ctx=ToolContext(session_id="session", runner=handle),
            arguments={},
            redactor=lambda: SecretRedactor(),
            timeout_seconds=0.01,
        )

    outcome = asyncio.run(scenario())

    assert outcome.result.is_error is True
    assert outcome.result.structured == {
        "terminal_outcome": "tool_execution_timeout",
        "tool_effect": "none",
        "outcome_unknown": False,
        "manual_reconciliation_required": False,
        "tool_execution_boundary": "in_process",
        "tool_timeout_strength": "cooperative_in_process",
    }
    assert "workload-secret" not in repr(outcome)
    assert runner.cancelled is True


def test_caller_cancellation_wins_during_grouped_tool_timeout_cleanup() -> None:
    runner = _DelayedGroupedCleanupFailureRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    class BlockingTool(Tool):
        spec = ToolSpec(
            name="grouped_cleanup_cancellation",
            description="Block in grouped runner cleanup.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            await ctx.runner.exec(ExecCommand.process("blocked"))
            return ToolResult(content="unexpected")

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        runner.started = asyncio.Event()
        runner.cleanup_started = asyncio.Event()
        execution = asyncio.create_task(
            run_tool(
                tool=BlockingTool(),
                effect=ToolEffect.NONE,
                ctx=ToolContext(session_id="grouped-timeout-caller-cancellation", runner=handle),
                arguments={},
                redactor=SecretRedactor,
                timeout_seconds=0.01,
            )
        )
        await asyncio.wait_for(runner.cleanup_started.wait(), timeout=1)
        execution.cancel("caller cancellation during grouped cleanup")
        with pytest.raises(asyncio.CancelledError) as raised:
            await execution
        return raised.value, execution.cancelling(), execution.cancelled()

    cancellation, cancellation_requests, cancelled = asyncio.run(scenario())

    assert cancellation.args == ("Runner command was cancelled.",)
    failure = runner_cancellation_failure(cancellation)
    assert failure is not None
    leaves = [
        candidate
        for candidate in iter_exception_tree(failure)
        if not isinstance(candidate, BaseExceptionGroup)
    ]
    assert len(leaves) == 1
    assert type(leaves[0]) is RunnerExecutionError
    assert str(leaves[0]) == "Runner command execution failed."
    assert cancellation_requests == 1
    assert cancelled is True


@pytest.mark.parametrize(
    "runner_type",
    [_BlockingRunner, _CleanupReplacingCancellationRunner],
    ids=["plain", "cleanup-failure"],
)
def test_clean_runner_cancellation_remains_compatible_with_asyncio_timeout(
    runner_type: type[_BlockingRunner],
) -> None:
    runner = runner_type()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    class BlockingTool(Tool):
        spec = ToolSpec(
            name="blocking_scalar",
            description="Block in a runner command.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            await ctx.runner.exec(ExecCommand.process("blocked"))
            return ToolResult(content="unexpected")

    async def scenario():
        runner.started = asyncio.Event()
        outcome = await run_tool(
            tool=BlockingTool(),
            effect=ToolEffect.NONE,
            ctx=ToolContext(session_id="session", runner=handle),
            arguments={},
            redactor=lambda: SecretRedactor(),
            timeout_seconds=0.01,
        )
        current_task = asyncio.current_task()
        assert current_task is not None
        return outcome, current_task.cancelling()

    outcome, cancelling = asyncio.run(scenario())

    assert outcome.result.is_error is True
    assert outcome.result.structured == {
        "terminal_outcome": "tool_execution_timeout",
        "tool_effect": "none",
        "outcome_unknown": False,
        "manual_reconciliation_required": False,
        "tool_execution_boundary": "in_process",
        "tool_timeout_strength": "cooperative_in_process",
    }
    assert cancelling == 0
    assert runner.cancelled is True


def test_legacy_runner_cancellation_remains_compatible_with_asyncio_timeout() -> None:
    runner = _TimeoutLegacyCancellationRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    class BlockingTool(Tool):
        spec = ToolSpec(
            name="blocking_legacy_cancellation",
            description="Block in a legacy runner command.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            await ctx.runner.exec(ExecCommand.process("blocked"))
            return ToolResult(content="unexpected")

    async def scenario():
        runner.started = asyncio.Event()
        return await run_tool(
            tool=BlockingTool(),
            effect=ToolEffect.NONE,
            ctx=ToolContext(session_id="session", runner=handle),
            arguments={},
            redactor=lambda: SecretRedactor(),
            timeout_seconds=0.01,
        )

    outcome = asyncio.run(scenario())

    assert outcome.result.is_error is True
    assert outcome.result.structured == {
        "terminal_outcome": "tool_execution_timeout",
        "tool_effect": "none",
        "outcome_unknown": False,
        "manual_reconciliation_required": False,
        "tool_execution_boundary": "in_process",
        "tool_timeout_strength": "cooperative_in_process",
    }
    assert runner.cancelled is True


@pytest.mark.parametrize(
    "runner_type",
    [_LateSecretRunner, _MisreportedBoundedRunner],
    ids=["truncated-flag", "byte-count-over-limit"],
)
def test_late_secret_revision_fails_closed_before_tool_result_publication(
    runner_type: type[Runner],
) -> None:
    secret = "late-registered-secret-canary-ABCDEFGHIJKLMNOP"
    tracker = InvocationSecretTracker(SecretRedactor())
    handle = InvocationRunnerHandle(
        runner_type(),
        redactor_snapshot_provider=tracker.snapshot,
        ambiguous_capture_observer=tracker.record_ambiguous_runner_capture,
    )

    class LateSecretTool(Tool):
        spec = ToolSpec(
            name="late_secret",
            description="Resolve a secret after bounded runner capture.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            result = await ctx.runner.exec(
                ExecCommand.process("emit"),
                output_limit_bytes=16,
            )
            tracker.record(
                ResolvedSecret(
                    name="token",
                    value=SecretStr(secret),
                )
            )
            return ToolResult(content=result.stdout)

    outcome = asyncio.run(
        run_tool(
            tool=LateSecretTool(),
            effect=ToolEffect.NONE,
            ctx=ToolContext(session_id="session", runner=handle),
            arguments={},
            redactor=lambda: tracker.redactor,
            finalize_publication=tracker.seal_for_publication,
        )
    )

    rendered = repr(outcome)
    assert outcome.result.is_error is True
    assert outcome.result.structured == {
        "terminal_outcome": "invalid_tool_output",
        "tool_effect": "none",
        "outcome_unknown": False,
        "manual_reconciliation_required": False,
    }
    assert "late-registered-" not in rendered
    assert secret not in rendered


def test_pending_secret_resolution_seals_runner_output_and_cannot_publish_late_secret() -> None:
    secret = ResolvedSecret(
        name="token",
        value=SecretStr("pending-secret-canary-ABCDEFGHIJKLMNOP"),
    )
    tracker = InvocationSecretTracker(SecretRedactor())
    token = tracker.begin_resolution()
    tracker.record_ambiguous_runner_capture(0)

    publication = tracker.seal_for_publication()

    assert publication.unsafe_output is True
    assert publication.redactor.has_values is False
    assert tracker.complete_resolution(token, secret) is False
    with pytest.raises(RuntimeError, match="unavailable after tool publication"):
        tracker.begin_resolution()
    assert tracker.redactor.has_values is False


def test_duplicate_secret_resolution_does_not_stale_ambiguous_capture() -> None:
    secret_value = "duplicate-secret-canary-ABCDEFGHIJKLMNOP"
    initial = SecretRedactor(secret_value)
    tracker = InvocationSecretTracker(initial)

    tracker.record(
        ResolvedSecret(
            name="first_alias",
            value=SecretStr(secret_value),
        )
    )
    assert tracker.snapshot().revision == 0
    assert tracker.snapshot().redactor is initial

    tracker.record_ambiguous_runner_capture(0)
    tracker.record(
        ResolvedSecret(
            name="second_alias",
            value=SecretStr(secret_value),
        )
    )

    publication = tracker.seal_for_publication()

    assert tracker.snapshot().revision == 0
    assert publication.unsafe_output is False
    assert publication.redactor is initial


def test_only_genuinely_new_secret_stales_ambiguous_capture() -> None:
    first_value = "first-secret-canary-ABCDEFGHIJKLMNOP"
    second_value = "second-secret-canary-ABCDEFGHIJKLMNOP"
    tracker = InvocationSecretTracker(SecretRedactor())

    tracker.record(
        ResolvedSecret(
            name="first",
            value=SecretStr(first_value),
        )
    )
    assert tracker.snapshot().revision == 1
    tracker.record_ambiguous_runner_capture(1)
    tracker.record(
        ResolvedSecret(
            name="first_alias",
            value=SecretStr(first_value),
        )
    )
    assert tracker.snapshot().revision == 1
    tracker.record(
        ResolvedSecret(
            name="second",
            value=SecretStr(second_value),
        )
    )

    publication = tracker.seal_for_publication()

    assert tracker.snapshot().revision == 2
    assert publication.unsafe_output is True


def test_duplicate_secret_after_bounded_capture_preserves_safe_tool_result() -> None:
    secret_value = "duplicate-publication-secret-canary-ABCDEFGHIJKLMNOP"
    tracker = InvocationSecretTracker(SecretRedactor(secret_value))
    handle = InvocationRunnerHandle(
        _LateSecretRunner(),
        redactor_snapshot_provider=tracker.snapshot,
        ambiguous_capture_observer=tracker.record_ambiguous_runner_capture,
    )

    class DuplicateSecretTool(Tool):
        spec = ToolSpec(
            name="duplicate_secret",
            description="Resolve an already-active secret after bounded capture.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            result = await ctx.runner.exec(
                ExecCommand.process("emit"),
                output_limit_bytes=16,
            )
            tracker.record(
                ResolvedSecret(
                    name="duplicate_alias",
                    value=SecretStr(secret_value),
                )
            )
            return ToolResult(content=result.stdout)

    outcome = asyncio.run(
        run_tool(
            tool=DuplicateSecretTool(),
            effect=ToolEffect.NONE,
            ctx=ToolContext(session_id="session", runner=handle),
            arguments={},
            redactor=lambda: tracker.redactor,
            finalize_publication=tracker.seal_for_publication,
        )
    )

    assert outcome.result.is_error is False
    # The adapter's already-truncated channel is conservatively omitted, but
    # resolving an identical value must not reclassify the completed operation
    # itself as invalid or outcome-unknown.
    assert outcome.result.content == ""
    assert tracker.snapshot().revision == 0


def test_late_vault_resolution_error_traceback_does_not_retain_secret() -> None:
    secret_value = "late-vault-secret-canary-ABCDEFGHIJKLMNOP"

    class DelayedVault(Vault):
        def __init__(self) -> None:
            self.started: asyncio.Event | None = None
            self.release: asyncio.Event | None = None

        async def get(
            self,
            name: str,
            *,
            scope: dict | None = None,
        ) -> SecretRef:
            del scope
            return SecretRef(name=name)

        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict | None = None,
        ) -> ResolvedSecret:
            del scope
            assert self.started is not None
            assert self.release is not None
            self.started.set()
            await self.release.wait()
            return ResolvedSecret(name=ref.name, value=SecretStr(secret_value))

    async def scenario() -> RuntimeError:
        tracker = InvocationSecretTracker(SecretRedactor())
        vault = DelayedVault()
        vault.started = asyncio.Event()
        vault.release = asyncio.Event()
        tracking_vault = invocation_secrets_module._TrackingVault(vault, tracker)
        resolution = asyncio.create_task(tracking_vault.resolve(SecretRef(name="api_key")))
        await vault.started.wait()
        publication = tracker.seal_for_publication()
        assert publication.unsafe_output is True
        vault.release.set()
        with pytest.raises(RuntimeError, match="completed after tool publication") as exc_info:
            await resolution
        return exc_info.value

    error = asyncio.run(scenario())

    assert secret_value not in repr(error)
    traceback = error.__traceback__
    while traceback is not None:
        assert all(
            secret_value not in repr(value) for value in traceback.tb_frame.f_locals.values()
        )
        traceback = traceback.tb_next


def test_cancellation_during_secret_projection_preserves_cancellation_and_clean_traceback() -> None:
    secret_value = "projection-cancellation-secret-canary-ABCDEFGHIJKLMNOP"

    async def scenario() -> tuple[asyncio.CancelledError, asyncio.Task[ResolvedSecret]]:
        tracker = InvocationSecretTracker(SecretRedactor())
        projection_started = asyncio.Event()
        release_projection = asyncio.Event()

        async def persist_projection(_snapshot: InvocationRedactorSnapshot) -> None:
            projection_started.set()
            await release_projection.wait()

        tracking_vault = invocation_secrets_module._TrackingVault(
            StaticVault({"api_key": secret_value}),
            tracker,
            persist_projection,
        )
        resolution = asyncio.create_task(tracking_vault.resolve(SecretRef(name="api_key")))
        await projection_started.wait()
        resolution.cancel("cancel secret projection")
        assert resolution.cancelling() == 1
        release_projection.set()
        try:
            await resolution
        except asyncio.CancelledError as cancellation:
            return cancellation, resolution
        raise AssertionError("Secret resolution did not preserve caller cancellation.")

    cancellation, resolution = asyncio.run(scenario())

    assert cancellation.args == ("cancel secret projection",)
    assert resolution.cancelled() is True
    assert resolution.cancelling() == 1
    traceback = cancellation.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(
                secret_value not in repr(value) for value in traceback.tb_frame.f_locals.values()
            )
        traceback = traceback.tb_next


@pytest.mark.parametrize("surface", ["vault", "proxy"])
def test_pending_cancellation_before_secret_resolution_prevents_dispatch(surface: str) -> None:
    secret_value = "pending-projection-cancellation-secret-canary-ABCDEFGHIJKLMNOP"

    class CountingVault(Vault):
        def __init__(self) -> None:
            self.dispatches = 0

        async def get(
            self,
            name: str,
            *,
            scope: dict | None = None,
        ) -> SecretRef:
            del scope
            return SecretRef(name=name)

        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict | None = None,
        ) -> ResolvedSecret:
            del scope
            self.dispatches += 1
            return ResolvedSecret(name=ref.name, value=SecretStr(secret_value))

    async def scenario() -> tuple[
        asyncio.CancelledError,
        asyncio.Task[ResolvedSecret],
        list[int],
        int,
        int,
    ]:
        tracker = InvocationSecretTracker(SecretRedactor())
        persisted_revisions: list[int] = []
        vault = CountingVault()

        async def persist_projection(snapshot: InvocationRedactorSnapshot) -> None:
            persisted_revisions.append(snapshot.revision)

        resolver = (
            invocation_secrets_module._TrackingVault(
                vault,
                tracker,
                persist_projection,
            )
            if surface == "vault"
            else invocation_secrets_module._TrackingCredentialProxy(
                PassthroughProxy(vault),
                tracker,
                lambda _record: None,
                persist_projection,
            )
        )

        async def resolve_after_pending_cancellation() -> ResolvedSecret:
            current = asyncio.current_task()
            assert current is not None
            current.cancel("cancel before secret projection")
            return await resolver.resolve(SecretRef(name="api_key"))

        resolution = asyncio.create_task(resolve_after_pending_cancellation())
        try:
            await resolution
        except asyncio.CancelledError as cancellation:
            return (
                cancellation,
                resolution,
                persisted_revisions,
                vault.dispatches,
                tracker.snapshot().revision,
            )
        raise AssertionError("Pending cancellation did not remain authoritative.")

    cancellation, resolution, persisted_revisions, dispatches, revision = asyncio.run(scenario())

    assert cancellation.args == ("cancel before secret projection",)
    assert resolution.cancelled() is True
    assert resolution.cancelling() == 1
    assert dispatches == 0
    assert revision == 0
    assert persisted_revisions == []
    traceback = cancellation.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(
                secret_value not in repr(value) for value in traceback.tb_frame.f_locals.values()
            )
        traceback = traceback.tb_next


@pytest.mark.parametrize("surface", ["vault", "proxy"])
def test_cancellation_fences_a_nonresponsive_secret_resolver(surface: str) -> None:
    secret_value = "detached-resolution-secret-canary-ABCDEFGHIJKLMNOP"

    class CancellationIgnoringVault(Vault):
        def __init__(self) -> None:
            self.started: asyncio.Event | None = None
            self.release: asyncio.Event | None = None
            self.finished: asyncio.Event | None = None

        async def get(
            self,
            name: str,
            *,
            scope: dict | None = None,
        ) -> SecretRef:
            del scope
            return SecretRef(name=name)

        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict | None = None,
        ) -> ResolvedSecret:
            del scope
            assert self.started is not None
            assert self.release is not None
            assert self.finished is not None
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await self.release.wait()
            self.finished.set()
            return ResolvedSecret(name=ref.name, value=SecretStr(secret_value))

    async def scenario() -> tuple[
        asyncio.CancelledError,
        asyncio.Task[ResolvedSecret],
        list[int],
        int,
        bool,
        bool,
    ]:
        tracker = InvocationSecretTracker(SecretRedactor())
        persisted_revisions: list[int] = []
        vault = CancellationIgnoringVault()
        vault.started = asyncio.Event()
        vault.release = asyncio.Event()
        vault.finished = asyncio.Event()

        async def persist_projection(snapshot: InvocationRedactorSnapshot) -> None:
            persisted_revisions.append(snapshot.revision)

        resolver = (
            invocation_secrets_module._TrackingVault(
                vault,
                tracker,
                persist_projection,
            )
            if surface == "vault"
            else invocation_secrets_module._TrackingCredentialProxy(
                PassthroughProxy(vault),
                tracker,
                lambda _record: None,
                persist_projection,
            )
        )
        resolution = asyncio.create_task(resolver.resolve(SecretRef(name="api_key")))
        await vault.started.wait()
        resolution.cancel("operator cancelled secret resolution")
        completed, _pending = await asyncio.wait({resolution}, timeout=1)
        assert completed == {resolution}
        try:
            await resolution
        except asyncio.CancelledError as cancellation:
            assert vault.finished.is_set() is False
            vault.release.set()
            await asyncio.wait_for(vault.finished.wait(), timeout=1)
            await asyncio.sleep(0)
            publication = tracker.seal_for_publication()
            return (
                cancellation,
                resolution,
                persisted_revisions,
                tracker.snapshot().revision,
                publication.secret_scope_incomplete,
                publication.unsafe_output,
            )
        raise AssertionError("Secret resolution did not preserve caller cancellation.")

    cancellation, resolution, persisted_revisions, revision, incomplete, unsafe_output = (
        asyncio.run(scenario())
    )

    assert cancellation.args == ("operator cancelled secret resolution",)
    assert resolution.cancelled() is True
    assert resolution.cancelling() == 1
    assert persisted_revisions == []
    assert revision == 0
    assert incomplete is True
    assert unsafe_output is True
    traceback = cancellation.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(
                secret_value not in repr(value) for value in traceback.tb_frame.f_locals.values()
            )
        traceback = traceback.tb_next


def test_invocation_runner_handle_rejects_invalid_redactor_before_dispatch() -> None:
    runner = _BlockingRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: object(),
    )

    with pytest.raises(TypeError, match="must return InvocationRedactorSnapshot"):
        asyncio.run(handle.exec(ExecCommand.process("never-dispatched")))

    assert runner.started is None


def test_invocation_runner_handle_rejects_dispatch_after_mutation_window_seals() -> None:
    runner = _RevisionRunner(ExecResult())
    owner = InvocationWorkspaceMutationOwner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
        mutation_owner=owner,
    )

    async def scenario() -> RunnerExecutionError:
        await owner.seal_and_wait()
        with pytest.raises(RunnerExecutionError) as raised:
            await handle.exec(ExecCommand.process("must-not-dispatch"))
        return raised.value

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        failure = asyncio.run(scenario())

    assert runner.last_kwargs == {}
    assert failure.diagnostic["error_type"] == "RuntimeError"
    assert failure.__cause__ is None
    assert failure.__context__ is None
    assert not any("never awaited" in str(record.message) for record in emitted)


def test_supervisory_exit_during_owner_settlement_transfers_pending_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[SystemExit, SystemExit, bool]:
        fence = WorkspaceMutationProcessFence()
        owner = InvocationWorkspaceMutationOwner(
            on_settlement_unproven=fence.fail_closed,
        )
        started = asyncio.Event()
        release = asyncio.Event()

        async def mutation() -> str:
            started.set()
            await release.wait()
            return "settled"

        operation = asyncio.create_task(owner.run(mutation))
        await started.wait()
        original_shield = asyncio.shield
        signal = SystemExit(17)

        async def supervisory_shield(awaitable):  # type: ignore[no-untyped-def]
            del awaitable
            raise signal

        monkeypatch.setattr(resources_module.asyncio, "shield", supervisory_shield)
        try:
            with pytest.raises(SystemExit) as raised:
                await owner.seal_and_wait()
        finally:
            monkeypatch.setattr(resources_module.asyncio, "shield", original_shield)

        settlement = asyncio.create_task(fence.wait_until_available())
        await asyncio.sleep(0)
        assert settlement.done() is False

        release.set()
        assert await operation == "settled"
        await settlement
        return raised.value, signal, operation.done()

    raised_signal, original_signal, operation_done = asyncio.run(scenario())

    assert raised_signal is original_signal
    assert raised_signal.code == 17
    assert operation_done is True


@pytest.mark.parametrize("cancelled", [False, True])
def test_invocation_runner_handle_retains_deferred_mutation_until_positive_settlement(
    cancelled: bool,
) -> None:
    runner = _DeferredSettlementRunner(cancelled=cancelled)
    fence = WorkspaceMutationProcessFence()
    owner = InvocationWorkspaceMutationOwner(
        on_settlement_unproven=fence.fail_closed,
    )
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
        mutation_owner=owner,
    )

    async def scenario() -> tuple[asyncio.Task[ExecResult], ExecResult | None]:
        runner.started = asyncio.Event()
        runner.settlement_started = asyncio.Event()
        runner.release_settlement = asyncio.Event()
        operation = asyncio.create_task(handle.exec(ExecCommand.process("mutate")))
        await runner.started.wait()
        if cancelled:
            operation.cancel("stop mutating command")
        if cancelled:
            with pytest.raises(asyncio.CancelledError, match="stop mutating command"):
                await operation
            with pytest.raises(WorkspaceMutationSettlementError):
                await owner.seal_and_wait()
            fence_waiter = asyncio.create_task(fence.wait_until_available())
            await runner.settlement_started.wait()
            assert fence_waiter.done() is False
            runner.release_settlement.set()
            await fence_waiter
            result = None
        else:
            await runner.settlement_started.wait()
            await asyncio.sleep(0)
            assert operation.done() is False
            runner.release_settlement.set()
            result = await operation
            await owner.seal_and_wait()
        fence.require_available_nowait()
        return operation, result

    operation, result = asyncio.run(scenario())

    assert operation.cancelled() is cancelled
    assert operation.cancelling() == int(cancelled)
    assert runner.settlement_calls == 1
    if result is not None:
        assert result.timed_out is True


def test_invocation_runner_handle_fails_closed_without_mutation_settlement_proof() -> None:
    owner = InvocationWorkspaceMutationOwner()
    handle = InvocationRunnerHandle(
        _UncertainTimeoutRunner(),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
        mutation_owner=owner,
    )

    async def scenario() -> ExecResult:
        result = await handle.exec(ExecCommand.process("mutate"))
        with pytest.raises(WorkspaceMutationSettlementError):
            await owner.seal_and_wait()
        return result

    result = asyncio.run(scenario())

    assert result.timed_out is True


@pytest.mark.parametrize("declaration", ("inherited", "wrong_type"))
def test_invocation_runner_handle_does_not_start_unsafe_settlement_waiter(
    declaration: str,
) -> None:
    class DeclaredSettlementRunner(Runner):
        pending_command_settlement_cancellation_safe = True

    class UnsafeSettlementRunner(DeclaredSettlementRunner):
        def __init__(self) -> None:
            self.settlement_calls = 0

        async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
            del command, kwargs
            return ExecResult(
                timed_out=True,
                artifacts=[
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "test",
                        "action": "kill_command",
                        "status": "deferred",
                    }
                ],
            )

        async def await_pending_command_settlement(self) -> bool:
            self.settlement_calls += 1
            await asyncio.Event().wait()
            return True

    if declaration == "wrong_type":
        # Truthiness is not authority; only the exact bool on the concrete
        # implementation class may opt into caller-loop task ownership.
        UnsafeSettlementRunner.pending_command_settlement_cancellation_safe = 1  # type: ignore[assignment]

    runner = UnsafeSettlementRunner()
    fence = WorkspaceMutationProcessFence()
    owner = InvocationWorkspaceMutationOwner(on_settlement_unproven=fence.fail_closed)
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
        mutation_owner=owner,
    )

    async def scenario() -> ExecResult:
        result = await handle.exec(ExecCommand.process("mutate"))
        with pytest.raises(WorkspaceMutationSettlementError):
            await owner.seal_and_wait()
        with pytest.raises(WorkspaceMutationSettlementError):
            fence.require_available_nowait()
        return result

    result = asyncio.run(scenario())

    assert result.timed_out is True
    assert runner.settlement_calls == 0


@pytest.mark.parametrize(
    ("settlement_failure", "cancellation_reasons"),
    [
        (
            RuntimeError("PRIVATE_SETTLEMENT_RACE_CANARY"),
            ("caller cancelled during settlement",),
        ),
        (
            RuntimeError("PRIVATE_REPEATED_SETTLEMENT_RACE_CANARY"),
            ("first settlement cancellation", "second settlement cancellation"),
        ),
        (
            SystemExit("PRIVATE_SETTLEMENT_RACE_CANARY"),
            ("caller cancelled during settlement",),
        ),
    ],
)
def test_invocation_runner_handle_preserves_cancellation_racing_settlement_failure(
    settlement_failure: BaseException,
    cancellation_reasons: tuple[str, ...],
) -> None:
    class RacingSettlementRunner(Runner):
        pending_command_settlement_cancellation_safe = True

        def __init__(self) -> None:
            self.parent: asyncio.Task[ExecResult] | None = None

        async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
            del command, kwargs
            return ExecResult(
                timed_out=True,
                artifacts=[
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "test",
                        "action": "kill_command",
                        "status": "deferred",
                    }
                ],
            )

        async def await_pending_command_settlement(self) -> bool:
            assert self.parent is not None
            for reason in cancellation_reasons:
                self.parent.cancel(reason)
            raise settlement_failure

    runner = RacingSettlementRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
        mutation_owner=InvocationWorkspaceMutationOwner(),
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        operation = asyncio.create_task(handle.exec(ExecCommand.process("mutate")))
        runner.parent = operation
        with pytest.raises(asyncio.CancelledError) as raised:
            await operation
        return raised.value, operation.cancelling(), operation.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancellation.args == (cancellation_reasons[-1],)
    assert cancelling == len(cancellation_reasons)
    assert cancelled is True
    failure = runner_cancellation_failure(cancellation)
    assert failure is not None
    leaves = [
        candidate
        for candidate in iter_exception_tree(failure)
        if not isinstance(candidate, BaseExceptionGroup)
    ]
    assert len(leaves) == 1
    if isinstance(settlement_failure, SystemExit):
        assert isinstance(leaves[0], SystemExit)
        assert leaves[0].code == 1
    else:
        assert type(leaves[0]) is RuntimeError
        assert str(leaves[0]) == "Runner mutation settlement failed."
    assert "PRIVATE_" not in repr(cancellation)
    assert "PRIVATE_" not in repr(failure)


def test_invocation_runner_handle_transfers_exact_settlement_after_foreground_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _DeferredSettlementRunner(cancelled=False)
    fence = WorkspaceMutationProcessFence()
    owner = InvocationWorkspaceMutationOwner(
        on_settlement_unproven=fence.fail_closed,
    )
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
        mutation_owner=owner,
    )
    monkeypatch.setattr(
        runner_module,
        "_RUNNER_MUTATION_SETTLEMENT_FOREGROUND_TIMEOUT_SECONDS",
        0.01,
    )

    async def scenario() -> ExecResult:
        runner.started = asyncio.Event()
        runner.settlement_started = asyncio.Event()
        runner.release_settlement = asyncio.Event()
        operation = asyncio.create_task(handle.exec(ExecCommand.process("mutate")))
        await runner.started.wait()
        await runner.settlement_started.wait()
        result = await asyncio.wait_for(operation, timeout=1)
        with pytest.raises(WorkspaceMutationSettlementError):
            await owner.seal_and_wait()

        fence_waiter = asyncio.create_task(fence.wait_until_available())
        await asyncio.sleep(0)
        assert fence_waiter.done() is False
        assert runner.settlement_calls == 1
        runner.release_settlement.set()
        await fence_waiter
        fence.require_available_nowait()
        return result

    result = asyncio.run(scenario())

    assert result.timed_out is True
    assert runner.settlement_calls == 1


def test_supervisory_generator_exit_transfers_active_runner_settlement_owner() -> None:
    runner = _DeferredSettlementRunner(cancelled=False)
    fence = WorkspaceMutationProcessFence()
    owner = InvocationWorkspaceMutationOwner(
        on_settlement_unproven=fence.fail_closed,
    )

    async def scenario() -> tuple[GeneratorExit, int]:
        runner.started = asyncio.Event()
        runner.settlement_started = asyncio.Event()
        runner.release_settlement = asyncio.Event()
        settlement = runner_module._settle_invocation_runner_mutation(
            runner=runner,
            owner=owner,
            settlement="deferred",
            settlement_error=None,
            cancellation=None,
        )
        iterator = settlement.__await__()
        assert iterator.send(None) is None
        assert runner.settlement_started is not None
        await runner.settlement_started.wait()

        supervisory_signal = GeneratorExit("supervising invocation was abandoned")
        with pytest.raises(GeneratorExit) as raised:
            iterator.throw(supervisory_signal)
        assert raised.value is supervisory_signal
        with pytest.raises(WorkspaceMutationSettlementError):
            await owner.seal_and_wait()

        later = asyncio.create_task(fence.wait_until_available())
        await asyncio.sleep(0)
        assert later.done() is False
        assert runner.settlement_calls == 1
        assert runner.release_settlement is not None
        runner.release_settlement.set()
        await later
        fence.require_available_nowait()
        return supervisory_signal, runner.settlement_calls

    signal, calls = asyncio.run(scenario())

    assert signal.args == ("supervising invocation was abandoned",)
    assert calls == 1


@pytest.mark.parametrize(
    "runner_type",
    [_PostReturnMutatingSettlementRunner, _PostRaiseMutatingSettlementRunner],
    ids=("mutated-result", "mutated-error"),
)
def test_supervisory_exit_after_runner_child_completion_retains_deferred_settlement(
    runner_type: type[_PostReturnMutatingSettlementRunner]
    | type[_PostRaiseMutatingSettlementRunner],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = runner_type()
    fence = WorkspaceMutationProcessFence()
    owner = InvocationWorkspaceMutationOwner(
        on_settlement_unproven=fence.fail_closed,
    )
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
        mutation_owner=owner,
    )
    original_sleep = operation_boundary_module.asyncio.sleep
    delivered = False

    async def supervisory_sleep(delay):
        nonlocal delivered
        result = await original_sleep(delay)
        if runner.started is not None and runner.started.is_set() and not delivered:
            delivered = True
            raise GeneratorExit("supervisory exit after command completion")
        return result

    monkeypatch.setattr(operation_boundary_module.asyncio, "sleep", supervisory_sleep)

    async def scenario() -> int:
        runner.started = asyncio.Event()
        runner.settlement_started = asyncio.Event()
        runner.release_settlement = asyncio.Event()
        if isinstance(runner, _PostReturnMutatingSettlementRunner):
            runner.result_mutated = asyncio.Event()
        else:
            runner.error_mutated = asyncio.Event()
        with pytest.raises(
            GeneratorExit,
            match="supervisory exit after command completion",
        ):
            await handle.exec(ExecCommand.process("mutate"))
        assert delivered is True
        if isinstance(runner, _PostReturnMutatingSettlementRunner):
            assert runner.result_mutated is not None
            await runner.result_mutated.wait()
            assert runner.returned_result is not None
            assert runner.returned_result.timed_out is False
            assert runner.returned_result.artifacts == []
        else:
            assert runner.error_mutated is not None
            await runner.error_mutated.wait()
            assert runner.returned_error is not None
            assert runner.returned_error.artifacts == []
        with pytest.raises(WorkspaceMutationSettlementError):
            await owner.seal_and_wait()

        later = asyncio.create_task(fence.wait_until_available())
        assert runner.settlement_started is not None
        await asyncio.wait_for(runner.settlement_started.wait(), timeout=1)
        assert later.done() is False
        assert runner.settlement_calls == 1
        assert runner.release_settlement is not None
        runner.release_settlement.set()
        await asyncio.wait_for(later, timeout=1)
        fence.require_available_nowait()
        return runner.settlement_calls

    calls = asyncio.run(scenario())

    assert calls == 1


def test_grouped_supervisory_exit_during_owner_settlement_transfers_pending_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[BaseExceptionGroup, BaseExceptionGroup]:
        fence = WorkspaceMutationProcessFence()
        owner = InvocationWorkspaceMutationOwner(
            on_settlement_unproven=fence.fail_closed,
        )
        started = asyncio.Event()
        release = asyncio.Event()

        async def mutation() -> str:
            started.set()
            await release.wait()
            return "settled"

        operation = asyncio.create_task(owner.run(mutation))
        await started.wait()
        original_shield = asyncio.shield
        signal = BaseExceptionGroup(
            "supervisory owner failures",
            [SystemExit(17), RuntimeError("secondary cleanup failure")],
        )

        async def supervisory_shield(awaitable):  # type: ignore[no-untyped-def]
            del awaitable
            raise signal

        monkeypatch.setattr(resources_module.asyncio, "shield", supervisory_shield)
        try:
            with pytest.raises(BaseExceptionGroup) as raised:
                await owner.seal_and_wait()
        finally:
            monkeypatch.setattr(resources_module.asyncio, "shield", original_shield)

        settlement = asyncio.create_task(fence.wait_until_available())
        await asyncio.sleep(0)
        assert settlement.done() is False
        release.set()
        assert await operation == "settled"
        await settlement
        return raised.value, signal

    raised_signal, original_signal = asyncio.run(scenario())

    assert raised_signal is original_signal


def test_supervisory_signal_after_foreground_wait_still_transfers_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _DeferredSettlementRunner(cancelled=False)
    fence = WorkspaceMutationProcessFence()
    owner = InvocationWorkspaceMutationOwner(
        on_settlement_unproven=fence.fail_closed,
    )

    class InterruptedOutcome:
        @property
        def timed_out(self) -> bool:
            raise GeneratorExit("supervisory signal during settlement classification")

    async def interrupted_wait(task, **kwargs):
        del task, kwargs
        assert runner.settlement_started is not None
        await runner.settlement_started.wait()
        return InterruptedOutcome()

    monkeypatch.setattr(runner_module, "await_shielded_task_outcome", interrupted_wait)

    async def scenario() -> int:
        runner.started = asyncio.Event()
        runner.settlement_started = asyncio.Event()
        runner.release_settlement = asyncio.Event()
        with pytest.raises(
            GeneratorExit,
            match="supervisory signal during settlement classification",
        ):
            await runner_module._settle_invocation_runner_mutation(
                runner=runner,
                owner=owner,
                settlement="deferred",
                settlement_error=None,
                cancellation=None,
            )
        with pytest.raises(WorkspaceMutationSettlementError):
            await owner.seal_and_wait()

        later = asyncio.create_task(fence.wait_until_available())
        await asyncio.sleep(0)
        assert later.done() is False
        assert runner.settlement_calls == 1
        assert runner.release_settlement is not None
        runner.release_settlement.set()
        await later
        fence.require_available_nowait()
        return runner.settlement_calls

    assert asyncio.run(scenario()) == 1


@pytest.mark.parametrize(
    ("settlement_failure", "expected_exit_code"),
    [
        (SystemExit(17), 17),
        (SystemExit("PRIVATE_SETTLEMENT_SIGNAL_CANARY"), 1),
    ],
)
def test_invocation_runner_handle_sanitizes_settlement_system_exit(
    settlement_failure: SystemExit,
    expected_exit_code: int,
) -> None:
    class SignalRunner(Runner):
        pending_command_settlement_cancellation_safe = True

        async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
            del command, kwargs
            return ExecResult(
                timed_out=True,
                artifacts=[
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "e2b",
                        "action": "kill_command",
                        "status": "deferred",
                    }
                ],
            )

        async def await_pending_command_settlement(self) -> bool:
            raise settlement_failure

    runner = SignalRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
        mutation_owner=InvocationWorkspaceMutationOwner(),
    )

    with pytest.raises(SystemExit) as raised:
        asyncio.run(handle.exec(ExecCommand.process("mutate")))

    assert raised.value.code == expected_exit_code
    assert "PRIVATE_SETTLEMENT_SIGNAL_CANARY" not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert_cayu_traceback_does_not_retain(raised.value, runner)


def test_invocation_runner_handle_sanitizes_settlement_signal_group() -> None:
    class SignalGroupRunner(Runner):
        pending_command_settlement_cancellation_safe = True

        async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
            del command, kwargs
            return ExecResult(
                timed_out=True,
                artifacts=[
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "e2b",
                        "action": "kill_command",
                        "status": "deferred",
                    }
                ],
            )

        async def await_pending_command_settlement(self) -> bool:
            raise BaseExceptionGroup(
                "PRIVATE_SETTLEMENT_GROUP_CANARY",
                [
                    SystemExit(17),
                    RuntimeError("PRIVATE_SETTLEMENT_FAILURE_CANARY"),
                ],
            )

    handle = InvocationRunnerHandle(
        SignalGroupRunner(),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
        mutation_owner=InvocationWorkspaceMutationOwner(),
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        asyncio.run(handle.exec(ExecCommand.process("mutate")))

    leaves = [
        candidate
        for candidate in iter_exception_tree(raised.value)
        if not isinstance(candidate, BaseExceptionGroup)
    ]
    assert len(leaves) == 2
    assert any(isinstance(candidate, SystemExit) and candidate.code == 17 for candidate in leaves)
    assert sum(isinstance(candidate, RuntimeError) for candidate in leaves) == 1
    assert "PRIVATE_SETTLEMENT" not in repr(raised.value)
    assert raised.value.__cause__ is None


def test_invocation_runner_handle_fails_closed_when_settlement_classifier_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained_probes: list[Any] = []
    owner = InvocationWorkspaceMutationOwner(
        on_settlement_unproven=retained_probes.append,
    )
    handle = InvocationRunnerHandle(
        _UncertainTimeoutRunner(),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
        mutation_owner=owner,
    )

    def fail_classifier(**kwargs):
        del kwargs
        raise RuntimeError("PRIVATE_SETTLEMENT_CLASSIFIER_CANARY")

    monkeypatch.setattr(
        runner_module,
        "runner_workspace_mutation_settlement",
        fail_classifier,
    )

    async def scenario() -> tuple[ExecResult, WorkspaceMutationSettlementError]:
        result = await handle.exec(ExecCommand.process("mutate"))
        with pytest.raises(WorkspaceMutationSettlementError) as raised:
            await owner.seal_and_wait()
        return result, raised.value

    result, failure = asyncio.run(scenario())

    assert result.timed_out is True
    assert len(retained_probes) == 1
    assert "PRIVATE_SETTLEMENT_CLASSIFIER_CANARY" not in str(failure)


def test_workspace_mutation_fence_shares_one_positive_probe_between_waiters() -> None:
    async def scenario() -> int:
        fence = WorkspaceMutationProcessFence()
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def probe() -> bool:
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return True

        fence.fail_closed(probe)
        first = asyncio.create_task(fence.wait_until_available())
        second = asyncio.create_task(fence.wait_until_available())
        await started.wait()
        release.set()
        await asyncio.gather(first, second)
        fence.require_available_nowait()
        return calls

    assert asyncio.run(scenario()) == 1


def test_workspace_mutation_fence_preserves_supervisory_generator_exit_and_probe() -> None:
    async def scenario() -> tuple[GeneratorExit, int]:
        fence = WorkspaceMutationProcessFence()
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def probe() -> bool:
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return True

        fence.fail_closed(probe)
        first = fence.wait_until_available()
        iterator = first.__await__()
        iterator.send(None)
        await started.wait()

        supervisory_signal = GeneratorExit("supervising fence waiter was abandoned")
        with pytest.raises(GeneratorExit) as raised:
            iterator.throw(supervisory_signal)
        assert raised.value is supervisory_signal

        later = asyncio.create_task(fence.wait_until_available())
        await asyncio.sleep(0)
        assert later.done() is False
        assert calls == 1
        release.set()
        await later
        fence.require_available_nowait()
        return supervisory_signal, calls

    signal, calls = asyncio.run(scenario())

    assert signal.args == ("supervising fence waiter was abandoned",)
    assert calls == 1


def test_workspace_mutation_fence_waits_for_every_factory_child_owner() -> None:
    async def scenario() -> tuple[int, int]:
        registration_fence = WorkspaceMutationProcessFence()
        first_child = registration_fence.child_fence()
        second_child = registration_fence.child_fence()
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        release_first = asyncio.Event()
        release_second = asyncio.Event()
        first_calls = 0
        second_calls = 0

        async def first_probe() -> bool:
            nonlocal first_calls
            first_calls += 1
            first_started.set()
            await release_first.wait()
            return True

        async def second_probe() -> bool:
            nonlocal second_calls
            second_calls += 1
            second_started.set()
            await release_second.wait()
            return True

        first_child.fail_closed(first_probe)
        second_child.fail_closed(second_probe)
        waiter = asyncio.create_task(registration_fence.wait_until_available())
        await asyncio.gather(first_started.wait(), second_started.wait())

        release_first.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        first_child.require_available_nowait()
        with pytest.raises(WorkspaceMutationSettlementError):
            second_child.require_available_nowait()
        assert waiter.done() is False

        release_second.set()
        await waiter
        registration_fence.require_available_nowait()
        return first_calls, second_calls

    assert asyncio.run(scenario()) == (1, 1)


def test_workspace_mutation_fence_preserves_active_system_exit_code() -> None:
    async def scenario() -> int:
        fence = WorkspaceMutationProcessFence()
        calls = 0

        async def probe() -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise SystemExit(17)
            return True

        fence.fail_closed(probe)
        with pytest.raises(SystemExit) as raised:
            await fence.wait_until_available()
        assert raised.value.code == 17
        await fence.wait_until_available()
        fence.require_available_nowait()
        return calls

    assert asyncio.run(scenario()) == 1


def test_workspace_mutation_task_probe_rejects_terminal_child_cancellation() -> None:
    async def scenario() -> None:
        fence = WorkspaceMutationProcessFence()
        observer_started = asyncio.Event()
        cancellation_delivered = asyncio.Event()
        release = asyncio.Event()

        async def observer() -> None:
            observer_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_delivered.set()
                await release.wait()
                raise

        observation_task = asyncio.create_task(observer())
        await observer_started.wait()
        observation_task.cancel("observer deadline")
        await cancellation_delivered.wait()
        fence.fail_closed(workspace_mutation_task_settlement_probe(observation_task))
        waiter = asyncio.create_task(fence.wait_until_available())
        await asyncio.sleep(0)
        assert waiter.done() is False
        release.set()
        with pytest.raises(
            WorkspaceMutationSettlementError,
            match="settlement could not be proven",
        ):
            await waiter
        assert observation_task.cancelled() is True
        with pytest.raises(
            WorkspaceMutationSettlementError,
            match="settlement could not be proven",
        ):
            fence.require_available_nowait()

    asyncio.run(scenario())


def test_workspace_mutation_task_probe_preserves_late_process_signal_once() -> None:
    async def scenario() -> int:
        fence = WorkspaceMutationProcessFence()
        release = asyncio.Event()

        async def observer() -> None:
            await release.wait()
            raise SystemExit(17)

        observation_task = asyncio.create_task(capture_awaitable_outcome(observer))
        fence.fail_closed(workspace_mutation_task_settlement_probe(observation_task))
        release.set()
        with pytest.raises(SystemExit) as raised:
            await fence.wait_until_available()
        assert raised.value.code == 17
        await fence.wait_until_available()
        fence.require_available_nowait()
        return 1

    assert asyncio.run(scenario()) == 1


def test_workspace_mutation_task_probe_preserves_late_generator_exit_once() -> None:
    async def scenario() -> int:
        fence = WorkspaceMutationProcessFence()
        release = asyncio.Event()

        async def observer() -> None:
            await release.wait()
            raise GeneratorExit("PRIVATE_LATE_OBSERVER_SIGNAL_CANARY")

        observation_task = asyncio.create_task(capture_awaitable_outcome(observer))
        fence.fail_closed(workspace_mutation_task_settlement_probe(observation_task))
        release.set()
        with pytest.raises(GeneratorExit) as raised:
            await fence.wait_until_available()
        assert raised.value.args == ()
        await fence.wait_until_available()
        fence.require_available_nowait()
        return 1

    assert asyncio.run(scenario()) == 1


def test_workspace_mutation_fence_delivers_active_signal_group_once() -> None:
    async def scenario() -> tuple[list[BaseException], int]:
        fence = WorkspaceMutationProcessFence()
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def probe() -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                await release.wait()
                raise BaseExceptionGroup(
                    "PRIVATE_SETTLEMENT_GROUP_CANARY",
                    [
                        SystemExit(17),
                        RuntimeError("PRIVATE_SETTLEMENT_FAILURE_CANARY"),
                    ],
                )
            return True

        async def observe() -> BaseException:
            try:
                await fence.wait_until_available()
            except BaseException as exc:
                return exc
            raise AssertionError("Unsettled fence unexpectedly became available.")

        fence.fail_closed(probe)
        first = asyncio.create_task(observe())
        second = asyncio.create_task(observe())
        await started.wait()
        release.set()
        failures = await asyncio.gather(first, second)
        await fence.wait_until_available()
        return failures, calls

    failures, calls = asyncio.run(scenario())

    groups = [failure for failure in failures if isinstance(failure, BaseExceptionGroup)]
    quarantines = [
        failure for failure in failures if isinstance(failure, WorkspaceMutationSettlementError)
    ]
    assert len(groups) == 1
    assert len(quarantines) == 1
    signals = [
        candidate
        for candidate in iter_exception_tree(groups[0])
        if isinstance(candidate, SystemExit)
    ]
    assert len(signals) == 1
    assert signals[0].code == 17
    assert "PRIVATE_SETTLEMENT" not in repr(failures)
    assert calls == 1


def test_workspace_mutation_fence_preserves_signal_and_sibling_failure() -> None:
    async def scenario() -> BaseException:
        fence = WorkspaceMutationProcessFence()
        signal_child = fence.child_fence()
        failed_child = fence.child_fence()

        async def signal_probe() -> bool:
            raise SystemExit(17)

        async def failed_probe() -> bool:
            return False

        signal_child.fail_closed(signal_probe)
        failed_child.fail_closed(failed_probe)
        try:
            await fence.wait_until_available()
        except BaseException as exc:
            return exc
        raise AssertionError("Unsettled fence unexpectedly became available.")

    failure = asyncio.run(scenario())

    assert isinstance(failure, BaseExceptionGroup)
    leaves = [
        candidate
        for candidate in iter_exception_tree(failure)
        if not isinstance(candidate, BaseExceptionGroup)
    ]
    assert len(leaves) == 2
    signals = [candidate for candidate in leaves if isinstance(candidate, SystemExit)]
    assert len(signals) == 1
    assert signals[0].code == 17
    assert sum(isinstance(candidate, RuntimeError) for candidate in leaves) == 1


@pytest.mark.parametrize(
    ("signal_type", "signal_arg", "expected_args"),
    [
        (SystemExit, 17, (17,)),
        (GeneratorExit, "PRIVATE_LATE_SETTLEMENT_SIGNAL", ()),
    ],
)
def test_workspace_mutation_fence_delivers_late_process_signal_once_after_waiter_cancel(
    signal_type: type[BaseException],
    signal_arg: object,
    expected_args: tuple[object, ...],
) -> None:
    async def scenario() -> tuple[asyncio.Task[None], BaseException, int]:
        fence = WorkspaceMutationProcessFence()
        started = asyncio.Event()
        release = asyncio.Event()
        first_finished = asyncio.Event()
        calls = 0

        async def probe() -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                await release.wait()
                first_finished.set()
                raise signal_type(signal_arg)
            return True

        fence.fail_closed(probe)
        cancelled_waiter = asyncio.create_task(fence.wait_until_available())
        await started.wait()
        cancelled_waiter.cancel("caller stopped waiting")
        with pytest.raises(asyncio.CancelledError, match="caller stopped waiting"):
            await cancelled_waiter
        release.set()
        await first_finished.wait()
        await asyncio.sleep(0)
        try:
            await fence.wait_until_available()
        except BaseException as signal:
            delivered = signal
        else:  # pragma: no cover - paired invariant
            raise AssertionError("Late settlement signal was discarded.")
        await fence.wait_until_available()
        fence.require_available_nowait()
        return cancelled_waiter, delivered, calls

    cancelled_waiter, delivered, calls = asyncio.run(scenario())

    assert cancelled_waiter.cancelling() == 1
    assert cancelled_waiter.cancelled()
    assert type(delivered) is signal_type
    assert delivered.args == expected_args
    assert calls == 1


def test_concurrent_invocation_runner_handles_keep_secret_scopes_separate() -> None:
    secrets = ("parallel-runner-secret-a", "parallel-runner-secret-b")
    runner = _ConcurrentRunner()
    handles = tuple(
        InvocationRunnerHandle(
            runner,
            redactor_snapshot_provider=lambda secret=secret: InvocationRedactorSnapshot(
                revision=0,
                redactor=SecretRedactor(secret),
            ),
        )
        for secret in secrets
    )

    async def scenario() -> tuple[ExecResult, ExecResult]:
        runner.all_started = asyncio.Event()
        first, second = await asyncio.gather(
            handles[0].exec(ExecCommand.process("emit", secrets[0])),
            handles[1].exec(ExecCommand.process("emit", secrets[1])),
        )
        return first, second

    results = asyncio.run(scenario())

    assert tuple(result.stdout for result in results) == (
        "[REDACTED_SECRET]",
        "[REDACTED_SECRET]",
    )


@pytest.mark.parametrize(
    ("result", "expected_stdout", "expected_truncated"),
    [
        (
            ExecResult(
                stdout="workload-secret-canary-ABCDEFGHIJKLMNOP",
                stdout_bytes=41,
            ),
            REDACTED_SECRET,
            False,
        ),
        (
            ExecResult(
                stdout="workload-secret-",
                stdout_bytes=41,
                stdout_truncated=True,
            ),
            "",
            True,
        ),
    ],
    ids=["complete", "already-truncated"],
)
def test_invocation_runner_handle_reprojects_when_secret_revision_changes(
    result: ExecResult,
    expected_stdout: str,
    expected_truncated: bool,
) -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    runner = _RevisionRunner(result)
    snapshot = InvocationRedactorSnapshot(
        revision=0,
        redactor=SecretRedactor(),
    )
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: snapshot,
    )

    async def scenario() -> ExecResult:
        nonlocal snapshot
        runner.started = asyncio.Event()
        runner.release = asyncio.Event()
        task = asyncio.create_task(
            handle.exec(
                ExecCommand.process("emit"),
                output_limit_bytes=64,
            )
        )
        await runner.started.wait()
        snapshot = InvocationRedactorSnapshot(
            revision=1,
            redactor=SecretRedactor(secret),
        )
        runner.release.set()
        return await task

    projected = asyncio.run(scenario())

    assert projected.stdout == expected_stdout
    assert projected.stdout_truncated is expected_truncated
    assert projected.stdout_bytes == 41


def test_invocation_runner_handle_detaches_opaque_failure_text() -> None:
    handle = InvocationRunnerHandle(
        _OpaqueFailureRunner(),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor("workload-secret-canary-ABCDEFGHIJKLMNOP"),
        ),
    )

    with pytest.raises(RunnerExecutionError) as exc_info:
        asyncio.run(handle.exec(ExecCommand.process("fail")))

    assert str(exc_info.value) == "Runner command execution failed."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert exc_info.value.diagnostic["error_type"] == "RuntimeError"
    assert "workload-secret-" not in repr(exc_info.value.diagnostic)


def test_invocation_runner_handle_preserves_builtin_runner_error_classification(
    tmp_path,
) -> None:
    handle = InvocationRunnerHandle(
        LocalRunner(tmp_path),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    with pytest.raises(RunnerExecutionError) as exc_info:
        asyncio.run(
            handle.exec(
                ExecCommand.process(sys.executable, "-c", "pass"),
                cwd="missing",
            )
        )

    error = exc_info.value
    assert error.diagnostic["adapter"] == "local"
    assert error.diagnostic["error_type"] == "FileNotFoundError"
    assert error.__cause__ is None
    assert error.__context__ is None


def test_invocation_runner_handle_rejects_secret_authored_exception_type() -> None:
    secret = "RunnerSecretCanaryABC123"
    secret_error_type = type(secret, (RuntimeError,), {})
    handle = InvocationRunnerHandle(
        _InjectedFailureRunner(secret_error_type("opaque runner failure")),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    with pytest.raises(RunnerExecutionError) as exc_info:
        asyncio.run(handle.exec(ExecCommand.process("fail")))

    error = exc_info.value
    assert error.diagnostic["error_type"] == "Exception"
    assert secret not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize(
    ("adapter", "error_type"),
    [
        ("e2b", "SandboxException"),
        ("microsandbox", "ExecFailedError"),
    ],
)
def test_invocation_runner_handle_preserves_supported_sdk_error_classification(
    adapter: str,
    error_type: str,
) -> None:
    sdk_error_type = type(error_type, (Exception,), {})
    sdk_runner_type = type(
        f"{adapter.title()}SdkFailureRunner",
        (_InjectedFailureRunner,),
        {"isolation": adapter},
    )
    runner = sdk_runner_type(sdk_error_type("opaque SDK failure"))
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    with pytest.raises(RunnerExecutionError) as exc_info:
        asyncio.run(handle.exec(ExecCommand.process("fail")))

    assert exc_info.value.diagnostic == {
        "type": "cayu.runner_execution_error.v1",
        "adapter": adapter,
        "status": "failed",
        "error_type": error_type,
        "timed_out": False,
        "cancelled": False,
    }


def test_invocation_runner_handle_rejects_secret_authored_source_diagnostic_type() -> None:
    secret = "RunnerDiagnosticSecretCanaryABC123"
    source_error = RuntimeError("opaque runner failure")
    source_error.diagnostic = {"error_type": secret}
    handle = InvocationRunnerHandle(
        _InjectedFailureRunner(source_error),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    with pytest.raises(RunnerExecutionError) as exc_info:
        asyncio.run(handle.exec(ExecCommand.process("fail")))

    error = exc_info.value
    assert error.diagnostic["error_type"] == "RuntimeError"
    assert secret not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_invocation_runner_handle_rejects_secret_authored_unavailable_types() -> None:
    secret = "RunnerUnavailableSecretCanaryABC123"
    source_error = RunnerUnavailableError(
        "opaque runner failure",
        diagnostic={
            "adapter": "microsandbox",
            "last_command": {
                "exit_code": None,
                "timed_out": False,
                "cancelled": False,
                "error_type": secret,
            },
            "probe": {
                "method": "Sandbox.ping",
                "status": "failed",
                "error_type": secret,
                "status_error_type": secret,
            },
        },
    )
    handle = InvocationRunnerHandle(
        _InjectedFailureRunner(source_error),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    with pytest.raises(RunnerUnavailableError) as exc_info:
        asyncio.run(handle.exec(ExecCommand.process("fail")))

    error = exc_info.value
    assert error.diagnostic["error_type"] == "RunnerUnavailableError"
    assert "error_type" not in error.diagnostic["last_command"]
    assert error.diagnostic["probe"] == {
        "method": "Sandbox.ping",
        "status": "failed",
    }
    assert secret not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_invocation_runner_handle_rejects_secret_authored_cleanup_type() -> None:
    secret = "RunnerCleanupSecretCanaryABC123"
    secret_error_type = type(secret, (RuntimeError,), {})

    class SecretTypedCleanupRunner(_BlockingRunner):
        async def exec(
            self,
            command: ExecCommand,
            **kwargs,
        ) -> ExecResult:
            try:
                return await super().exec(command, **kwargs)
            except asyncio.CancelledError as cancellation:
                raise BaseExceptionGroup(
                    "runner cleanup failed",
                    [cancellation, secret_error_type("opaque cleanup failure")],
                ) from None

    runner = SecretTypedCleanupRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    async def scenario() -> tuple[asyncio.CancelledError, bool]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(handle.exec(ExecCommand.process("blocked")))
        await runner.started.wait()
        task.cancel("caller cancellation")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, task.cancelled()

    cancellation, cancelled = asyncio.run(scenario())

    assert cancelled is True
    assert isinstance(cancellation.__cause__, ExceptionGroup)
    assert len(cancellation.__cause__.exceptions) == 1
    cleanup_failure = cancellation.__cause__.exceptions[0]
    assert isinstance(cleanup_failure, RunnerExecutionError)
    assert cleanup_failure.diagnostic["error_type"] == "Exception"
    assert secret not in repr(cancellation)
    assert secret not in repr(cancellation.__cause__)


def test_invocation_runner_handle_rejects_hostile_instance_adapter_safely() -> None:
    handle = InvocationRunnerHandle(
        _HostileInstanceAdapterRunner(),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(
                (
                    "adapter-secret-canary-ABCDEFGHIJKLMNOP",
                    "runner-secret-canary-ABCDEFGHIJKLMNOP",
                )
            ),
        ),
    )

    with pytest.raises(RunnerExecutionError) as exc_info:
        asyncio.run(handle.exec(ExecCommand.process("fail")))

    error = exc_info.value
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.diagnostic == {
        "type": "cayu.runner_execution_error.v1",
        "adapter": "unknown",
        "status": "failed",
        "error_type": "RuntimeError",
        "timed_out": False,
        "cancelled": False,
    }
    assert "secret-canary" not in repr(error)


def test_invocation_runner_handle_sanitizes_synchronous_dispatch_failure() -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    runner = _SynchronousFailureRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        ),
    )

    with pytest.raises(RunnerExecutionError) as exc_info:
        asyncio.run(
            handle.exec(
                ExecCommand.process("emit", secret),
                env={"WORKLOAD_TOKEN": secret},
                stdin=secret,
            )
        )

    error = exc_info.value
    assert error.args == ("Runner command execution failed.",)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.diagnostic["error_type"] == "RuntimeError"
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if is_cayu_source_filename(frame.f_code.co_filename):
            assert all(
                value is not handle and value is not runner for value in frame.f_locals.values()
            )
            for name, value in frame.f_locals.items():
                assert secret not in repr(value), (
                    frame.f_code.co_filename,
                    frame.f_code.co_name,
                    name,
                )
        traceback = traceback.tb_next


def test_invocation_runner_handle_forwards_environment_removals() -> None:
    runner = _RevisionRunner(ExecResult())
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    async def scenario() -> None:
        runner.started = asyncio.Event()
        runner.release = asyncio.Event()
        runner.release.set()
        await handle.exec(
            ExecCommand.process("git", "status"),
            env_remove=("GIT_CONFIG_COUNT", "GIT_DIR"),
        )

    asyncio.run(scenario())

    assert runner.last_kwargs["env_remove"] == ("GIT_CONFIG_COUNT", "GIT_DIR")


@pytest.mark.parametrize("invalid_remove", (None, [], {}))
def test_invocation_runner_handle_preserves_falsey_invalid_environment_removals(
    invalid_remove,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault = StaticVault({"runner-secret": "runner-secret-value-canary"})
    resolve_calls = 0
    original_resolve = vault.resolve

    async def counting_resolve(ref, *, scope=None):
        nonlocal resolve_calls
        resolve_calls += 1
        return await original_resolve(ref, scope=scope)

    monkeypatch.setattr(vault, "resolve", counting_resolve)
    marker = tmp_path / "invalid-environment-dispatched"
    runner = LocalRunner(
        tmp_path,
        secret_env={"API_TOKEN": SecretRef(name="runner-secret")},
        secret_resolver=vault,
    )
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    with pytest.raises(RunnerExecutionError) as exc_info:
        asyncio.run(
            handle.exec(
                ExecCommand.process(
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ),
                env_remove=invalid_remove,  # type: ignore[arg-type]
            )
        )

    assert exc_info.value.diagnostic["error_type"] == "TypeError"
    assert resolve_calls == 0
    assert not marker.exists()


def test_invocation_runner_handle_owns_environment_before_operation_checkpoint(
    tmp_path,
) -> None:
    environment = {"ORIGINAL": "value"}

    def mutate_environment() -> None:
        environment.clear()
        environment["INVALID\x00NAME"] = "changed"

    def snapshot_provider() -> InvocationRedactorSnapshot:
        asyncio.get_running_loop().call_soon(mutate_environment)
        return InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        )

    handle = InvocationRunnerHandle(
        LocalRunner(tmp_path),
        redactor_snapshot_provider=snapshot_provider,
    )

    result = asyncio.run(
        handle.exec(
            ExecCommand.process(
                sys.executable,
                "-c",
                "import os; print(os.environ['ORIGINAL'])",
            ),
            env=environment,
        )
    )

    assert result.exit_code == 0
    assert result.stdout == "value\n"
    assert environment == {"INVALID\x00NAME": "changed"}


def test_invocation_runner_handle_owns_command_before_operation_checkpoint(
    tmp_path,
) -> None:
    marker = tmp_path / "mutated-command-ran"
    command = ExecCommand.process(sys.executable, "-c", "print('owned-command')")

    def mutate_command() -> None:
        assert command.argv is not None
        command.argv[:] = [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ]

    def snapshot_provider() -> InvocationRedactorSnapshot:
        asyncio.get_running_loop().call_soon(mutate_command)
        return InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        )

    handle = InvocationRunnerHandle(
        LocalRunner(tmp_path),
        redactor_snapshot_provider=snapshot_provider,
    )

    result = asyncio.run(handle.exec(command))

    assert result.exit_code == 0
    assert result.stdout == "owned-command\n"
    assert not marker.exists()


def test_invocation_runner_handle_preserves_custom_adapter_cleanup_evidence() -> None:
    runner = _RevisionRunner(
        ExecResult(
            artifacts=[
                {
                    "type": "cayu.runner_cleanup.v1",
                    "adapter": "third-party-secret-bearing-adapter",
                    "action": "kill_command",
                    "status": "failed",
                    "timeout_s": 5.0,
                    "error_type": "RuntimeError",
                    "error": "workload-secret-",
                },
                {
                    "type": "cayu.runner_cleanup.v1",
                    "adapter": ["malformed"],
                    "action": ["kill_command"],
                    "status": {"failed": True},
                    "timeout_s": 5.0,
                },
            ]
        )
    )
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor("workload-secret-canary-ABCDEFGHIJKLMNOP"),
        ),
    )

    async def scenario() -> ExecResult:
        runner.started = asyncio.Event()
        runner.release = asyncio.Event()
        runner.release.set()
        return await handle.exec(ExecCommand.process("cleanup"))

    result = asyncio.run(scenario())

    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "unknown",
            "action": "kill_command",
            "status": "failed",
            "timeout_s": 5.0,
            "error_type": "RuntimeError",
        }
    ]


def test_invocation_runner_handle_sanitizes_cleanup_without_replacing_cancellation() -> None:
    runner = _CleanupFailureRunner()
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor("workload-secret-canary-ABCDEFGHIJKLMNOP"),
        ),
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(handle.exec(ExecCommand.process("blocked")))
        await runner.started.wait()
        task.cancel("caller cancelled")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert type(cancellation) is asyncio.CancelledError
    assert cancellation.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_command",
            "status": "failed",
            "timeout_s": 5.0,
            "error_type": "RuntimeError",
        }
    ]
