from __future__ import annotations

import asyncio
import hashlib
import shutil
import subprocess
import sys
import threading
import warnings
from collections.abc import AsyncIterator

import pytest
from tests.provider_traceback_assertions import is_cayu_source_filename

import cayu.runtime._tool_round_executor as tool_round_executor_module
import cayu.tools._operation_boundary as operation_boundary_module
import cayu.tools._runner as runner_module
from cayu._exception_groups import exception_cause, iter_exception_tree
from cayu._exception_state import set_exception_state
from cayu.artifacts import ArtifactMetadata, LocalArtifactStore
from cayu.core import AgentSpec, EventType, Message, Tool, ToolContext, ToolResult, ToolSpec
from cayu.environments import (
    DeterministicWorkspaceBinding,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    GitRepositoryBinding,
    NativeBinding,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runners import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    LocalRunner,
    Runner,
    attach_cancellation_artifacts,
)
from cayu.runtime import (
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    EventQuery,
    InMemorySessionStore,
    InterruptSessionRequest,
    RunRequest,
    ToolApprovalDecision,
    ToolApprovalRequest,
    UserInputResponse,
)
from cayu.tools import ExecCommandTool, UserInputTool
from cayu.vaults import SecretRedactor, SecretRef, StaticVault
from cayu.workspaces import (
    LocalWorkspace,
    WorkspaceIdentity,
    WorkspacePathRevision,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationLimits,
    WorkspaceRevisionObservationStatus,
)


async def collect_events(app: CayuApp, request: RunRequest):
    return [event async for event in app.run(request)]


class _ScriptedProvider(ModelProvider):
    name = "scripted"

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.requests += 1
        if self.requests == 1:
            yield ModelStreamEvent.tool_call(
                id="call-shell",
                name="exec_command",
                arguments={
                    "argv": [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; Path('shell.txt').write_text('created')",
                    ]
                },
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _BulkProvider(ModelProvider):
    name = "bulk"

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.requests += 1
        if self.requests == 1:
            yield ModelStreamEvent.tool_call(id="call-bulk", name="bulk_write", arguments={})
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _CancelProvider(ModelProvider):
    name = "cancel"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.tool_call(
            id="call-cancel",
            name="cancel_mutation",
            arguments={},
        )
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class _SingleToolProvider(ModelProvider):
    name = "single-workspace-tool"

    def __init__(self, *, tool_name: str, arguments: dict) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.requests = 0
        self.seen_requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.seen_requests.append(request)
        self.requests += 1
        if self.requests == 1:
            yield ModelStreamEvent.tool_call(
                id="call-workspace",
                name=self.tool_name,
                arguments=self.arguments,
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _DetachedRunnerThenFollowingProvider(ModelProvider):
    name = "detached-runner-then-following"

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.requests += 1
        if self.requests == 1:
            yield ModelStreamEvent.tool_call(
                id="call-detached-runner",
                name="detached_runner_mutation",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if self.requests == 2:
            yield ModelStreamEvent.tool_call(
                id="call-following-tool",
                name="following_tool",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _AwaitedRunnerThenFollowingProvider(_DetachedRunnerThenFollowingProvider):
    name = "awaited-runner-then-following"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if self.requests == 0:
            del request
            self.requests += 1
            yield ModelStreamEvent.tool_call(
                id="call-awaited-runner",
                name="awaited_runner_mutation",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        async for event in super().stream(request):
            yield event


class _AwaitedRunnerAndFollowingProvider(ModelProvider):
    name = "awaited-runner-and-following"

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.requests += 1
        if self.requests == 1:
            yield ModelStreamEvent.tool_call(
                id="call-awaited-runner",
                name="awaited_runner_mutation",
                arguments={},
            )
            yield ModelStreamEvent.tool_call(
                id="call-following-tool",
                name="following_tool",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _PreDispatchRunnerReuseProvider(ModelProvider):
    name = "pre-dispatch-runner-reuse"

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.requests += 1
        if self.requests <= 2:
            yield ModelStreamEvent.tool_call(
                id=f"call-pre-dispatch-{self.requests}",
                name="awaited_runner_mutation",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _ConcurrentFactoryMutationProvider(ModelProvider):
    name = "concurrent-factory-mutation"

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.requests += 1
        if self.requests <= 2:
            yield ModelStreamEvent.tool_call(
                id="call-awaited-runner",
                name="awaited_runner_mutation",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _SiblingSecretProvider(ModelProvider):
    name = "sibling-secret"

    def __init__(self, secret_path: str) -> None:
        self.secret_path = secret_path
        self.requests = 0
        self.seen_requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.seen_requests.append(request)
        self.requests += 1
        if self.requests == 1:
            yield ModelStreamEvent.tool_call(
                id="call-write-secret-path",
                name="private_workspace_write",
                arguments={"path": self.secret_path},
            )
            yield ModelStreamEvent.tool_call(
                id="call-resolve-sibling-secret",
                name="resolve_workspace_path_secret",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _UserInputThenMutationProvider(ModelProvider):
    name = "user-input-mutation"

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.requests += 1
        if self.requests == 1:
            yield ModelStreamEvent.tool_call(
                id="call-input",
                name="ask_user",
                arguments={"question": "Continue?"},
            )
            yield ModelStreamEvent.tool_call(
                id="call-resumed-shell",
                name="exec_command",
                arguments={
                    "argv": [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; Path('resumed.txt').write_text('created')",
                    ]
                },
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _BulkWriteTool(Tool):
    spec = ToolSpec(
        name="bulk_write",
        parallel_safe=False,
        workspace_mutation=True,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        assert ctx.workspace is not None
        for index in range(65):
            content = b"workspace-secret-value" if index == 0 else b"bounded"
            await ctx.workspace.create_bytes(f"generated/{index:03}.txt", content)
        return ToolResult(content="created bounded files")


class _NoopWorkspaceMutationTool(Tool):
    spec = ToolSpec(
        name="noop_workspace_mutation",
        parallel_safe=False,
        workspace_mutation=True,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(content="no mutation requested")


class _MalformedListWorkspace(LocalWorkspace):
    async def list(self, pattern: str = "**/*", *, limit: int | None = None):
        del pattern, limit
        return object()


class _FailingObserverBinding(NativeBinding):
    async def observe_revision(self, bound):
        del bound
        return object()


class _IdentityDriftBinding(DeterministicWorkspaceBinding):
    def __init__(self) -> None:
        super().__init__()
        self.observations = 0

    async def observe_revision(self, bound):
        observation = await super().observe_revision(bound)
        self.observations += 1
        if self.observations == 1:
            return observation
        return observation.model_copy(
            update={
                "identity": WorkspaceIdentity(
                    workspace_id="foreign-workspace",
                    observer=type(self).__name__,
                )
            }
        )


class _OversizedObserverBinding(NativeBinding):
    async def observe_revision(self, bound):
        identity = WorkspaceIdentity(
            workspace_id=bound.workspace.id,
            observer=type(self).__name__,
        )
        path_count = WorkspaceRevisionObservationLimits().max_paths + 1
        return WorkspaceRevisionObservation(
            identity=identity,
            status=WorkspaceRevisionObservationStatus.SUPPORTED,
            revision="sha256:" + "a" * 64,
            paths=tuple(
                WorkspacePathRevision(path=f"generated/{index:05}.txt", present=True)
                for index in range(path_count)
            ),
            total_paths=path_count,
        )


class _ObserverCanary:
    def __repr__(self) -> str:
        return "PRIVATE_OBSERVER_CANARY"


class _MalformedObserverBinding(DeterministicWorkspaceBinding):
    async def observe_revision(self, bound):
        observation = await super().observe_revision(bound)
        return observation.model_copy(update={"status": _ObserverCanary()})


class _DuplicatePathObserverBinding(DeterministicWorkspaceBinding):
    async def observe_revision(self, bound):
        observation = await super().observe_revision(bound)
        if not observation.paths:
            return observation
        duplicate = (*observation.paths, observation.paths[0])
        return observation.model_copy(update={"paths": duplicate, "total_paths": len(duplicate)})


class _ChildCancelledObserverBinding(NativeBinding):
    async def observe_revision(self, bound):
        del bound
        raise asyncio.CancelledError("observer-owned cancellation")


class _StalledObserverBinding(DeterministicWorkspaceBinding):
    def __init__(self) -> None:
        super().__init__()
        self.observations = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def observe_revision(self, bound):
        self.observations += 1
        if self.observations == 1:
            return await super().observe_revision(bound)
        self.started.set()
        await self.release.wait()
        return await super().observe_revision(bound)


class _FailAfterObservationStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def append_event(self, session_id, event):
        if (
            not self.failed
            and event.type == EventType.WORKSPACE_REVISION_OBSERVED
            and event.payload.get("phase") == "after"
        ):
            self.failed = True
            raise ConnectionError("workspace receipt append failed")
        await super().append_event(session_id, event)


class _ChildCancelledAfterObservationStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def append_event(self, session_id, event):
        if (
            not self.failed
            and event.type == EventType.WORKSPACE_REVISION_OBSERVED
            and event.payload.get("phase") == "after"
        ):
            self.failed = True
            raise asyncio.CancelledError("event-store-owned cancellation")
        await super().append_event(session_id, event)


class _BlockingAfterObservationStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.blocked = False

    async def append_event(self, session_id, event):
        if (
            not self.blocked
            and event.type == EventType.WORKSPACE_REVISION_OBSERVED
            and event.payload.get("phase") == "after"
        ):
            self.blocked = True
            self.started.set()
            await self.release.wait()
        await super().append_event(session_id, event)


class _FailingTerminalAfterBlockingCaptureStore(_BlockingAfterObservationStore):
    async def append_event(self, session_id, event):
        if event.type == EventType.TOOL_CALL_COMPLETED:
            raise ConnectionError("terminal publication failed")
        await super().append_event(session_id, event)


class _BlockingTerminalAfterBlockingCaptureStore(_BlockingAfterObservationStore):
    def __init__(self) -> None:
        super().__init__()
        self.terminal_started = asyncio.Event()
        self.terminal_release = asyncio.Event()
        self.terminal_blocked = False

    async def append_event(self, session_id, event):
        if not self.terminal_blocked and event.type == EventType.TOOL_CALL_COMPLETED:
            self.terminal_blocked = True
            self.terminal_started.set()
            await self.terminal_release.wait()
        await super().append_event(session_id, event)


class _ChildCancelledArtifactStore(LocalArtifactStore):
    async def put_bytes(self, *args, **kwargs):
        del args, kwargs
        raise asyncio.CancelledError("artifact-store-owned cancellation")


class _MalformedArtifactStore(LocalArtifactStore):
    async def put_bytes(self, *args, **kwargs):
        metadata = await super().put_bytes(*args, **kwargs)
        return ArtifactMetadata.model_construct(
            id=_ObserverCanary(),
            filename=metadata.filename,
            content_type=metadata.content_type,
            size_bytes=metadata.size_bytes,
            scope=metadata.scope,
            session_id=metadata.session_id,
            agent_name=metadata.agent_name,
            environment_name=metadata.environment_name,
            created_at=metadata.created_at,
            metadata=metadata.metadata,
        )


class _StalledArtifactStore(LocalArtifactStore):
    def __init__(self, root) -> None:
        super().__init__(root)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def put_bytes(self, *args, **kwargs):
        self.started.set()
        await self.release.wait()
        return await super().put_bytes(*args, **kwargs)


class _CancellingMutationTool(Tool):
    spec = ToolSpec(
        name="cancel_mutation",
        parallel_safe=False,
        workspace_mutation=True,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        assert ctx.workspace is not None
        await ctx.workspace.create_bytes("cancelled-write.txt", b"written before cancellation")
        raise asyncio.CancelledError("tool cancellation canary")


class _ResolveWorkspacePathSecretTool(Tool):
    spec = ToolSpec(
        name="resolve_workspace_path_secret",
        parallel_safe=False,
        workspace_mutation=True,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        assert ctx.vault is not None
        await ctx.vault.resolve(SecretRef(name="workspace_path"))
        return ToolResult(content="resolved")


class _PrivateWorkspaceWriteTool(Tool):
    spec = ToolSpec(
        name="private_workspace_write",
        parallel_safe=False,
        workspace_mutation=True,
    )

    @property
    def _publish_arguments(self) -> bool:
        return False

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        await ctx.workspace.create_bytes(args["path"], b"private")
        return ToolResult(content="written")


class _BlockingThreadWorkspace(LocalWorkspace):
    def __init__(self, root, *, workspace_id: str) -> None:
        super().__init__(root, workspace_id=workspace_id)
        self.dispatched = threading.Event()
        self.release = threading.Event()

    async def create_bytes(self, path: str, content: bytes):
        await asyncio.to_thread(self._blocking_create, path, content)
        return await super().create_bytes(path, content)

    def _blocking_create(self, path: str, content: bytes) -> None:
        self.dispatched.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test mutation release timed out")


class _BlockingWorkspaceMutationTool(Tool):
    spec = ToolSpec(
        name="blocking_workspace_mutation",
        parallel_safe=False,
        workspace_mutation=True,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        assert ctx.workspace is not None
        await ctx.workspace.create_bytes("settled.txt", b"settled")
        return ToolResult(content="unexpected")


class _GroupedFailureWorkspace(LocalWorkspace):
    def __init__(self, root, *, workspace_id: str) -> None:
        super().__init__(root, workspace_id=workspace_id)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def create_bytes(self, path: str, content: bytes):
        del path, content
        self.started.set()
        await self.release.wait()
        raise BaseExceptionGroup(
            "PRIVATE_MUTATION_GROUP_CANARY",
            [
                asyncio.CancelledError("PRIVATE_MUTATION_CANCEL_CANARY"),
                RuntimeError("PRIVATE_MUTATION_FAILURE_CANARY"),
            ],
        )


class _DetachedWorkspaceMutationTool(Tool):
    spec = ToolSpec(
        name="detached_workspace_mutation",
        parallel_safe=False,
        workspace_mutation=True,
    )

    def __init__(self, *, dispatched: asyncio.Event) -> None:
        self.dispatched = dispatched

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        assert ctx.workspace is not None
        mutation = asyncio.create_task(
            ctx.workspace.create_bytes("grouped-failure.txt", b"not-written"),
            name="test-detached-workspace-mutation",
        )
        await self.dispatched.wait()
        del mutation
        return ToolResult(content="tool completed before mutation settlement")


class _SupervisoryDetachedWorkspaceMutationTool(Tool):
    spec = ToolSpec(
        name="supervisory_detached_workspace_mutation",
        parallel_safe=False,
        workspace_mutation=True,
    )

    def __init__(self, failure: BaseException) -> None:
        self.failure = failure

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        assert ctx.workspace is not None
        mutation = asyncio.create_task(
            ctx.workspace.create_bytes("settled-after-supervisory-exit.txt", b"settled"),
            name="test-supervisory-detached-workspace-mutation",
        )
        await asyncio.sleep(0)
        del mutation
        raise self.failure


class _AsyncBlockingWorkspace(LocalWorkspace):
    def __init__(self, root, *, workspace_id: str) -> None:
        super().__init__(root, workspace_id=workspace_id)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def create_bytes(self, path: str, content: bytes):
        self.started.set()
        await self.release.wait()
        return await super().create_bytes(path, content)


class _DetachedThenBlockingWorkspaceMutationTool(Tool):
    spec = ToolSpec(
        name="detached_then_blocking_workspace_mutation",
        parallel_safe=False,
        workspace_mutation=True,
    )

    def __init__(self, *, dispatched: asyncio.Event) -> None:
        self.dispatched = dispatched

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        assert ctx.workspace is not None
        mutation = asyncio.create_task(
            ctx.workspace.create_bytes("settled-after-cancellation.txt", b"settled"),
            name="test-detached-blocking-workspace-mutation",
        )
        await self.dispatched.wait()
        del mutation
        await asyncio.Event().wait()
        raise AssertionError("Cancelled tool execution unexpectedly resumed.")


class _BarrierWorkspaceMutationRunner(Runner):
    def __init__(self, root) -> None:
        self.root = root
        self.default_cwd = str(root)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

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
        self.started.set()
        await self.release.wait()
        (self.root / "detached-runner.txt").write_bytes(b"settled")
        return ExecResult(stdout="mutated", stdout_bytes=7)


class _DeferredBackgroundMutationRunner(Runner):
    pending_command_settlement_cancellation_safe = True

    def __init__(self, root) -> None:
        self.root = root
        self.default_cwd = str(root)
        self.started = asyncio.Event()
        self.settlement_started = asyncio.Event()
        self.release_mutation = asyncio.Event()
        self._mutation: asyncio.Task[None] | None = None

    async def exec(
        self,
        command: ExecCommand,
        **kwargs,
    ) -> ExecResult:
        del command, kwargs
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancellation:
            self._mutation = asyncio.create_task(
                self._finish_mutation(),
                name="test-deferred-runner-workspace-mutation",
            )
            attach_cancellation_artifacts(
                cancellation,
                [
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "e2b",
                        "action": "kill_command",
                        "status": "deferred",
                        "timeout_s": 5.0,
                    }
                ],
            )
            raise
        raise AssertionError("Deferred mutation runner unexpectedly resumed.")

    async def _finish_mutation(self) -> None:
        await self.release_mutation.wait()
        (self.root / "deferred-runner.txt").write_bytes(b"settled")

    async def await_pending_command_settlement(self) -> bool:
        self.settlement_started.set()
        assert self._mutation is not None
        await asyncio.shield(self._mutation)
        return True


class _PreDispatchCancellingMutationRunner(Runner):
    def __init__(self, root) -> None:
        self.default_cwd = str(root)
        self.cancel_preflight = True
        self.exec_calls = 0
        self.cancelled_task: asyncio.Task | None = None

    def preflight_exec(self, command: ExecCommand, **kwargs) -> None:
        del command, kwargs
        if self.cancel_preflight:
            self.cancel_preflight = False
            current = asyncio.current_task()
            assert current is not None
            self.cancelled_task = current
            current.cancel("cancel before public runner dispatch")

    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        del command, kwargs
        self.exec_calls += 1
        return ExecResult()


class _BlockedDeferredResultMutationRunner(Runner):
    pending_command_settlement_cancellation_safe = True

    def __init__(self, root) -> None:
        self.root = root
        self.default_cwd = str(root)
        self.settlement_started = asyncio.Event()
        self.release_mutation = asyncio.Event()
        self.mutation_finished = asyncio.Event()
        self.settlement_calls = 0
        self._mutation: asyncio.Task[None] | None = None

    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        del command, kwargs
        self._mutation = asyncio.create_task(
            self._finish_mutation(),
            name="test-supervisory-deferred-runner-mutation",
        )
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

    async def _finish_mutation(self) -> None:
        await self.release_mutation.wait()
        (self.root / "supervisory-settled.txt").write_bytes(b"settled")
        self.mutation_finished.set()

    async def await_pending_command_settlement(self) -> bool:
        self.settlement_calls += 1
        self.settlement_started.set()
        assert self._mutation is not None
        await asyncio.shield(self._mutation)
        return True


class _BlockedSupervisoryDispatchMutationRunner(Runner):
    pending_command_settlement_cancellation_safe = True

    def __init__(self, root) -> None:
        self.root = root
        self.default_cwd = str(root)
        self.started = asyncio.Event()
        self.release_command = asyncio.Event()
        self.settlement_started = asyncio.Event()
        self.release_mutation = asyncio.Event()
        self.mutation_finished = asyncio.Event()
        self.settlement_calls = 0
        self._mutation: asyncio.Task[None] | None = None

    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        del command, kwargs
        self._mutation = asyncio.create_task(
            self._finish_mutation(),
            name="test-supervisory-runner-dispatch-mutation",
        )
        self.started.set()
        await self.release_command.wait()
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

    async def _finish_mutation(self) -> None:
        await self.release_mutation.wait()
        (self.root / "supervisory-dispatch-settled.txt").write_bytes(b"settled")
        self.mutation_finished.set()

    async def await_pending_command_settlement(self) -> bool:
        self.settlement_calls += 1
        self.settlement_started.set()
        assert self._mutation is not None
        await asyncio.shield(self._mutation)
        return True


class _FailingDeferredBackgroundMutationRunner(Runner):
    pending_command_settlement_cancellation_safe = True

    def __init__(self, root) -> None:
        self.root = root
        self.default_cwd = str(root)
        self.started = asyncio.Event()
        self.release_mutation = asyncio.Event()
        self.mutation_finished = asyncio.Event()
        self.reconciliation_started = asyncio.Event()
        self._mutation: asyncio.Task[None] | None = None
        self.settlement_calls = 0

    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        del command, kwargs
        self._mutation = asyncio.create_task(
            self._finish_mutation(),
            name="test-uncertain-runner-workspace-mutation",
        )
        self.started.set()
        return ExecResult(
            timed_out=True,
            artifacts=[
                {
                    "type": "cayu.runner_cleanup.v1",
                    "adapter": "e2b",
                    "action": "kill_command",
                    "status": "deferred",
                    "timeout_s": 5.0,
                }
            ],
        )

    async def _finish_mutation(self) -> None:
        await self.release_mutation.wait()
        (self.root / "uncertain-runner.txt").write_bytes(b"late")
        self.mutation_finished.set()

    async def await_pending_command_settlement(self) -> bool:
        self.settlement_calls += 1
        if self.settlement_calls == 1:
            raise RuntimeError("PRIVATE_RUNNER_SETTLEMENT_CANARY")
        self.reconciliation_started.set()
        await self.mutation_finished.wait()
        return True


class _HostileArtifactDiscriminator:
    def __init__(self) -> None:
        self.compared = False

    def __eq__(self, other):
        del other
        self.compared = True
        raise RuntimeError("PRIVATE_ARTIFACT_EQUALITY_CANARY")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return "PRIVATE_ARTIFACT_REPR_CANARY"


class _HostileArtifactBackgroundMutationRunner(Runner):
    pending_command_settlement_cancellation_safe = True

    def __init__(self, root) -> None:
        self.root = root
        self.default_cwd = str(root)
        self.discriminator = _HostileArtifactDiscriminator()
        self.started = asyncio.Event()
        self.release_mutation = asyncio.Event()
        self.mutation_finished = asyncio.Event()
        self._mutation: asyncio.Task[None] | None = None

    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        del command, kwargs
        self._mutation = asyncio.create_task(
            self._finish_mutation(),
            name="test-hostile-artifact-runner-workspace-mutation",
        )
        self.started.set()
        failure = RuntimeError("Runner command failed.")
        assert set_exception_state(
            failure,
            "artifacts",
            [
                {
                    "type": self.discriminator,
                    "action": "kill_command",
                    "status": "deferred",
                }
            ],
        )
        raise failure

    async def _finish_mutation(self) -> None:
        await self.release_mutation.wait()
        (self.root / "hostile-artifact-runner.txt").write_bytes(b"late")
        self.mutation_finished.set()

    async def await_pending_command_settlement(self) -> bool:
        await self.mutation_finished.wait()
        return True


class _TrackingFinalizeBinding(DeterministicWorkspaceBinding):
    def __init__(self) -> None:
        super().__init__()
        self.finalize_calls = 0
        self.abandon_calls = 0

    async def finalize(self, bound, *, outcome=None, metadata=None):
        self.finalize_calls += 1
        return await super().finalize(
            bound,
            outcome=outcome,
            metadata=metadata,
        )

    def abandon(self, bound) -> bool:
        self.abandon_calls += 1
        return super().abandon(bound)


class _AwaitedRunnerMutationTool(Tool):
    spec = ToolSpec(
        name="awaited_runner_mutation",
        parallel_safe=False,
        workspace_mutation=True,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        assert ctx.runner is not None
        await ctx.runner.exec(ExecCommand.process("mutate-workspace"))
        return ToolResult(content="runner mutation completed")


class _SupervisoryAwaitedRunnerMutationTool(_AwaitedRunnerMutationTool):
    def __init__(self) -> None:
        self.dispatching = False
        self.task: asyncio.Task | None = None

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.task = asyncio.current_task()
        self.dispatching = True
        try:
            return await super().run(ctx, args)
        finally:
            self.dispatching = False


class _DetachedRunnerMutationTool(Tool):
    spec = ToolSpec(
        name="detached_runner_mutation",
        parallel_safe=False,
        workspace_mutation=True,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        assert ctx.runner is not None
        mutation = asyncio.create_task(
            ctx.runner.exec(ExecCommand.process("mutate-workspace")),
            name="test-detached-runner-mutation",
        )
        await asyncio.sleep(0)
        del mutation
        return ToolResult(content="runner mutation dispatched")


class _FollowingTool(Tool):
    spec = ToolSpec(name="following_tool", parallel_safe=False)

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        self.started.set()
        return ToolResult(content="following tool completed")


def test_workspace_mutation_classification_requires_exclusive_effectful_tool() -> None:
    with pytest.raises(ValueError, match="parallel_safe=False"):
        ToolSpec(name="unsafe_parallel_mutation", workspace_mutation=True)

    with pytest.raises(ValueError, match="cannot declare ToolEffect.NONE"):
        ToolSpec(
            name="missing_mutation_effect",
            effect="none",
            parallel_safe=False,
            workspace_mutation=True,
        )


def test_cayu_app_records_git_workspace_mutation_receipt_for_shell_tool(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    (tmp_path / "README.md").write_text("baseline\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-m", "baseline")

    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="git-workspace"),
                runner=LocalRunner(tmp_path),
                binding=GitRepositoryBinding(
                    repo_url="https://example.invalid/repository.git",
                    fetch=False,
                    require_clean=False,
                    verify_remote_url=False,
                ),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        public_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-receipt",
                    messages=[Message.text("user", "create a file")],
                )
            )
        ]
        durable = await store.query_events(EventQuery(session_id="session-receipt"))
        return public_events, [record.event for record in durable]

    public_events, durable_events = asyncio.run(run())
    receipt_events = [
        event
        for event in durable_events
        if event.type
        in {
            EventType.WORKSPACE_REVISION_OBSERVED,
            EventType.WORKSPACE_MUTATION_RECORDED,
        }
    ]

    assert [event.type for event in receipt_events] == [
        EventType.WORKSPACE_REVISION_OBSERVED,
        EventType.WORKSPACE_REVISION_OBSERVED,
        EventType.WORKSPACE_MUTATION_RECORDED,
    ]
    before, after, receipt = receipt_events
    assert before.payload["phase"] == "before"
    assert after.payload["phase"] == "after"
    assert before.payload["window_id"] == after.payload["window_id"]
    assert receipt.payload["window_id"] == before.payload["window_id"]
    assert receipt.payload["before_observation_id"] == before.id
    assert receipt.payload["after_observation_id"] == after.id
    assert receipt.payload["session_run_epoch"] == 1
    assert receipt.payload["model_step"] == 1
    assert receipt.payload["tool_call_id"] == "call-shell"
    assert receipt.payload["workspace_id"] == "git-workspace"
    assert receipt.payload["status"] == "changed"
    assert receipt.payload["paths"] == [
        {"path": "shell.txt", "change": "added", "renamed_from": None}
    ]
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in public_events)
    assert "created" not in before.model_dump_json()
    assert "created" not in after.model_dump_json()
    assert "created" not in receipt.model_dump_json()


def test_detached_runner_mutation_settles_before_receipt_and_following_tool(
    tmp_path,
) -> None:
    async def run():
        workspace = LocalWorkspace(tmp_path, workspace_id="detached-runner-workspace")
        runner = _BarrierWorkspaceMutationRunner(tmp_path)
        following = _FollowingTool()
        provider = _DetachedRunnerThenFollowingProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=runner,
                workspace=workspace,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_DetachedRunnerMutationTool(), following],
        )

        consumer = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-detached-runner-mutation",
                    messages=[Message.text("user", "write")],
                ),
            )
        )
        await runner.started.wait()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert not consumer.done()
        assert not following.started.is_set()
        assert provider.requests == 1
        before_release = await store.query_events(
            EventQuery(session_id="session-detached-runner-mutation")
        )
        assert not any(
            record.event.type is EventType.WORKSPACE_MUTATION_RECORDED for record in before_release
        )

        runner.release.set()
        public_events = await consumer
        durable = await store.query_events(
            EventQuery(session_id="session-detached-runner-mutation")
        )
        return public_events, [record.event for record in durable], following, provider

    public_events, durable_events, following, provider = asyncio.run(run())

    receipt = next(
        event for event in durable_events if event.type is EventType.WORKSPACE_MUTATION_RECORDED
    )
    assert receipt.payload["status"] == "changed"
    assert receipt.payload["paths"] == [
        {"path": "detached-runner.txt", "change": "added", "renamed_from": None}
    ]
    assert following.started.is_set()
    assert provider.requests == 3
    assert any(event.type is EventType.TOOL_CALL_COMPLETED for event in public_events)


@pytest.mark.parametrize(
    "failure_factory",
    [
        lambda: SystemExit(17),
        lambda: GeneratorExit("supervisor abandoned tool execution"),
        lambda: BaseExceptionGroup(
            "supervisory tool failures",
            [SystemExit(17), RuntimeError("secondary tool cleanup failure")],
        ),
    ],
    ids=("system-exit", "generator-exit", "grouped-system-exit"),
)
def test_supervisory_tool_exit_fences_environment_reuse_until_detached_mutation_settles(
    tmp_path,
    failure_factory,
) -> None:
    async def run():
        workspace = _AsyncBlockingWorkspace(
            tmp_path,
            workspace_id="supervisory-detached-workspace",
        )
        provider = _SingleToolProvider(
            tool_name="supervisory_detached_workspace_mutation",
            arguments={},
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=workspace,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_SupervisoryDetachedWorkspaceMutationTool(failure_factory())],
        )

        with pytest.raises(BaseException) as raised:
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-supervisory-detached-first",
                    messages=[Message.text("user", "write")],
                ),
            )
        if isinstance(raised.value, GeneratorExit):
            assert raised.value.args == ("supervisor abandoned tool execution",)
        else:
            signals = [
                candidate
                for candidate in iter_exception_tree(raised.value)
                if isinstance(candidate, SystemExit)
            ]
            assert len(signals) == 1
            assert signals[0].code == 17
        assert workspace.started.is_set()
        assert not (tmp_path / "settled-after-supervisory-exit.txt").exists()

        contender = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-supervisory-detached-second",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert contender.done() is False
        assert provider.requests == 1

        workspace.release.set()
        contender_events = await contender
        durable = await store.query_events(
            EventQuery(session_id="session-supervisory-detached-first")
        )
        return contender_events, [record.event for record in durable], provider

    contender_events, durable_events, provider = asyncio.run(run())

    assert provider.requests == 2
    assert any(event.type is EventType.SESSION_COMPLETED for event in contender_events)
    assert not any(event.type is EventType.WORKSPACE_MUTATION_RECORDED for event in durable_events)
    assert (tmp_path / "settled-after-supervisory-exit.txt").read_bytes() == b"settled"


def test_cancelled_runner_mutation_returns_promptly_and_fences_reuse(tmp_path) -> None:
    async def run():
        workspace = LocalWorkspace(tmp_path, workspace_id="cancelled-runner-workspace")
        runner = _DeferredBackgroundMutationRunner(tmp_path)
        following = _FollowingTool()
        provider = _AwaitedRunnerAndFollowingProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=runner,
                workspace=workspace,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_AwaitedRunnerMutationTool(), following],
        )
        session_id = "session-cancelled-runner-mutation"
        consumer = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "write")],
                ),
            )
        )
        await runner.started.wait()
        consumer.cancel("cancel runner mutation")
        with pytest.raises(asyncio.CancelledError, match="cancel runner mutation"):
            await asyncio.wait_for(consumer, timeout=1)
        assert consumer.cancelling() == 1
        assert consumer.cancelled() is True
        assert following.started.is_set() is False
        before_release = await store.query_events(EventQuery(session_id=session_id))
        assert not any(
            record.event.type is EventType.WORKSPACE_MUTATION_RECORDED for record in before_release
        )

        contender = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-after-cancelled-runner-mutation",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )
        await runner.settlement_started.wait()
        assert contender.done() is False
        assert provider.requests == 1
        runner.release_mutation.set()
        contender_events = await contender
        durable = await store.query_events(EventQuery(session_id=session_id))
        return [record.event for record in durable], contender_events, following, provider

    durable_events, contender_events, following, provider = asyncio.run(run())

    assert not any(event.type is EventType.WORKSPACE_MUTATION_RECORDED for event in durable_events)
    assert (tmp_path / "deferred-runner.txt").read_bytes() == b"settled"
    assert any(event.type is EventType.SESSION_COMPLETED for event in contender_events)
    assert following.started.is_set() is False
    assert provider.requests == 2


def test_public_runner_cancellation_before_dispatch_does_not_quarantine_reuse(
    tmp_path,
) -> None:
    async def run():
        workspace = LocalWorkspace(tmp_path, workspace_id="pre-dispatch-runner-workspace")
        runner = _PreDispatchCancellingMutationRunner(tmp_path)
        provider = _PreDispatchRunnerReuseProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=runner,
                workspace=workspace,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_AwaitedRunnerMutationTool()],
        )

        first_events: list = []
        first_failure: BaseException | None = None
        first_operation = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-runner-cancelled-before-dispatch",
                    messages=[Message.text("user", "cancel before dispatch")],
                ),
            )
        )
        try:
            first_events = await first_operation
        except BaseException as exc:
            first_failure = exc
        assert runner.exec_calls == 0

        second_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session-runner-reused-after-pre-dispatch-cancel",
                messages=[Message.text("user", "reuse environment")],
            ),
        )
        return first_events, first_failure, first_operation, second_events, runner, provider

    first_events, first_failure, first_operation, second_events, runner, provider = asyncio.run(
        run()
    )

    assert first_failure is not None or any(
        event.type in {EventType.SESSION_INTERRUPTED, EventType.SESSION_FAILED}
        for event in first_events
    )
    assert runner.cancelled_task is not None
    assert runner.cancelled_task is first_operation
    assert runner.cancelled_task.cancelling() == 1
    assert runner.cancelled_task.cancelled() is True
    assert runner.exec_calls == 1
    assert provider.requests == 3
    assert any(event.type is EventType.SESSION_COMPLETED for event in second_events)


def test_timed_out_runner_mutation_returns_promptly_and_fences_reuse(
    tmp_path,
) -> None:
    async def run():
        workspace = LocalWorkspace(tmp_path, workspace_id="timed-out-runner-workspace")
        runner = _DeferredBackgroundMutationRunner(tmp_path)
        following = _FollowingTool()
        provider = _AwaitedRunnerAndFollowingProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            tool_timeout_seconds=0.01,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=runner,
                workspace=workspace,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_AwaitedRunnerMutationTool(), following],
        )
        session_id = "session-timed-out-runner-mutation"
        consumer = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "write")],
                ),
            )
        )
        public_events = await asyncio.wait_for(consumer, timeout=1)
        assert following.started.is_set() is False
        assert provider.requests == 1
        before_release = await store.query_events(EventQuery(session_id=session_id))
        assert not any(
            record.event.type is EventType.WORKSPACE_MUTATION_RECORDED for record in before_release
        )

        contender = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-after-timed-out-runner-mutation",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )
        await runner.settlement_started.wait()
        assert contender.done() is False
        assert provider.requests == 1
        runner.release_mutation.set()
        contender_events = await contender
        durable = await store.query_events(EventQuery(session_id=session_id))
        return (
            public_events,
            [record.event for record in durable],
            contender_events,
            following,
            provider,
        )

    public_events, durable_events, contender_events, following, provider = asyncio.run(run())

    assert not any(event.type is EventType.WORKSPACE_MUTATION_RECORDED for event in durable_events)
    assert (tmp_path / "deferred-runner.txt").read_bytes() == b"settled"
    terminal = next(event for event in public_events if event.type is EventType.TOOL_CALL_FAILED)
    assert terminal.payload["terminal_outcome"] == "tool_execution_timeout"
    assert (
        terminal.payload["workspace_mutation_capture_detail_code"] == "mutation_settlement_unproven"
    )
    assert any(event.type is EventType.SESSION_COMPLETED for event in contender_events)
    assert following.started.is_set() is False
    assert provider.requests == 2


def test_supervisory_runner_signal_retains_environment_fence_until_settlement(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run():
        workspace = LocalWorkspace(tmp_path, workspace_id="supervisory-runner-workspace")
        runner = _BlockedDeferredResultMutationRunner(tmp_path)
        following = _FollowingTool()
        provider = _AwaitedRunnerAndFollowingProvider()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=runner,
                workspace=workspace,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_AwaitedRunnerMutationTool(), following],
        )

        original_wait = runner_module.await_shielded_task_outcome
        interrupted = False

        async def supervisory_wait(task, **kwargs):
            nonlocal interrupted
            if not interrupted and task.get_name() == "cayu-runner-mutation-settlement":
                interrupted = True
                await runner.settlement_started.wait()
                raise GeneratorExit("supervising tool invocation was abandoned")
            return await original_wait(task, **kwargs)

        monkeypatch.setattr(runner_module, "await_shielded_task_outcome", supervisory_wait)

        first_failure: BaseException | None = None
        try:
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-supervisory-runner-signal",
                    messages=[Message.text("user", "mutate")],
                ),
            )
        except BaseException as exc:
            first_failure = exc
        assert interrupted is True
        assert runner.mutation_finished.is_set() is False
        assert following.started.is_set() is False

        contender = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-after-supervisory-runner-signal",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert contender.done() is False
        assert provider.requests == 1
        assert runner.settlement_calls == 1

        runner.release_mutation.set()
        contender_events = await contender
        return first_failure, contender_events, runner, following, provider

    first_failure, contender_events, runner, following, provider = asyncio.run(run())

    assert first_failure is not None
    assert any(
        isinstance(candidate, GeneratorExit) for candidate in iter_exception_tree(first_failure)
    )
    assert runner.mutation_finished.is_set()
    assert runner.settlement_calls == 1
    assert following.started.is_set() is False
    assert provider.requests == 2
    assert any(event.type is EventType.SESSION_COMPLETED for event in contender_events)


def test_supervisory_exit_during_runner_dispatch_fences_reuse_through_deferred_settlement(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run():
        workspace = LocalWorkspace(tmp_path, workspace_id="supervisory-dispatch-workspace")
        runner = _BlockedSupervisoryDispatchMutationRunner(tmp_path)
        tool = _SupervisoryAwaitedRunnerMutationTool()
        following = _FollowingTool()
        provider = _AwaitedRunnerAndFollowingProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=runner,
                workspace=workspace,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[tool, following],
        )

        original_shield = operation_boundary_module.asyncio.shield
        interrupted = False

        async def supervisory_shield(awaitable):
            nonlocal interrupted
            if not interrupted and tool.dispatching and asyncio.current_task() is tool.task:
                await runner.started.wait()
                interrupted = True
                raise GeneratorExit("supervising runner dispatch was abandoned")
            return await original_shield(awaitable)

        monkeypatch.setattr(operation_boundary_module.asyncio, "shield", supervisory_shield)

        first_failure: BaseException | None = None
        try:
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-supervisory-runner-dispatch",
                    messages=[Message.text("user", "mutate")],
                ),
            )
        except BaseException as exc:
            first_failure = exc
        assert interrupted is True
        assert runner.started.is_set()
        assert runner.mutation_finished.is_set() is False
        assert following.started.is_set() is False
        before_release = await store.query_events(
            EventQuery(session_id="session-supervisory-runner-dispatch")
        )
        assert not any(
            record.event.type is EventType.WORKSPACE_MUTATION_RECORDED for record in before_release
        )

        contender = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-after-supervisory-runner-dispatch",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert contender.done() is False
        assert provider.requests == 1
        assert runner.settlement_calls == 0

        runner.release_command.set()
        await runner.settlement_started.wait()
        await asyncio.sleep(0)
        assert contender.done() is False
        assert provider.requests == 1
        assert runner.mutation_finished.is_set() is False

        runner.release_mutation.set()
        contender_events = await contender
        return first_failure, contender_events, runner, following, provider

    first_failure, contender_events, runner, following, provider = asyncio.run(run())

    assert isinstance(first_failure, GeneratorExit)
    assert first_failure.args == ("supervising runner dispatch was abandoned",)
    assert runner.mutation_finished.is_set()
    assert runner.settlement_calls == 1
    assert following.started.is_set() is False
    assert provider.requests == 2
    assert any(event.type is EventType.SESSION_COMPLETED for event in contender_events)
    assert (tmp_path / "supervisory-dispatch-settled.txt").read_bytes() == b"settled"


def test_unproven_runner_mutation_stops_before_receipt_and_following_tool(
    tmp_path,
    caplog,
    capsys,
) -> None:
    async def run():
        workspace = LocalWorkspace(tmp_path, workspace_id="uncertain-runner-workspace")
        runner = _FailingDeferredBackgroundMutationRunner(tmp_path)
        binding = _TrackingFinalizeBinding()
        following = _FollowingTool()
        provider = _AwaitedRunnerAndFollowingProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=runner,
                workspace=workspace,
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_AwaitedRunnerMutationTool(), following],
        )
        session_id = "session-uncertain-runner-mutation"
        public_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "write")],
            ),
        )
        assert runner.started.is_set()
        assert runner.mutation_finished.is_set() is False
        durable = await store.query_events(EventQuery(session_id=session_id))
        transcript = await store.load_transcript(session_id)
        runner.release_mutation.set()
        await runner.mutation_finished.wait()
        return (
            public_events,
            [record.event for record in durable],
            transcript,
            following,
            provider,
            binding,
        )

    with warnings.catch_warnings(record=True) as captured_warnings:
        (
            public_events,
            durable_events,
            transcript,
            following,
            provider,
            binding,
        ) = asyncio.run(run())
    captured = capsys.readouterr()

    assert not any(event.type is EventType.WORKSPACE_MUTATION_RECORDED for event in durable_events)
    terminals = [
        event
        for event in durable_events
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
    ]
    assert len(terminals) == 1, [event.type for event in durable_events]
    terminal = terminals[0]
    assert terminal.payload["workspace_mutation_capture_status"] == "failed"
    assert (
        terminal.payload["workspace_mutation_capture_detail_code"] == "mutation_settlement_unproven"
    )
    assert any(event.type is EventType.SESSION_FAILED for event in public_events)
    assert following.started.is_set() is False
    assert provider.requests == 1
    assert binding.finalize_calls == 0
    assert binding.abandon_calls == 0
    combined = repr(
        (
            [event.model_dump(mode="json") for event in public_events],
            [event.model_dump(mode="json") for event in durable_events],
            [message.model_dump(mode="json") for message in transcript],
            captured_warnings,
            [record.getMessage() for record in caplog.records],
            captured.out,
            captured.err,
        )
    )
    assert "PRIVATE_RUNNER_SETTLEMENT_CANARY" not in combined


def test_unproven_runner_mutation_fences_environment_reuse_until_settled(
    tmp_path,
) -> None:
    async def run():
        workspace = LocalWorkspace(tmp_path, workspace_id="shared-runner-workspace")
        runner = _FailingDeferredBackgroundMutationRunner(tmp_path)
        binding = _TrackingFinalizeBinding()
        following = _FollowingTool()
        provider = _AwaitedRunnerAndFollowingProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=runner,
                workspace=workspace,
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_AwaitedRunnerMutationTool(), following],
        )

        first = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session-uncertain-runner-first",
                messages=[Message.text("user", "write")],
            ),
        )
        assert any(event.type is EventType.SESSION_FAILED for event in first)
        assert runner.mutation_finished.is_set() is False
        assert binding.finalize_calls == 0

        second = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-uncertain-runner-second",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )
        await runner.reconciliation_started.wait()
        await asyncio.sleep(0)
        assert second.done() is False
        assert provider.requests == 1
        assert binding.finalize_calls == 0
        assert binding.abandon_calls == 0

        runner.release_mutation.set()
        second_events = await second
        first_durable = await store.query_events(
            EventQuery(session_id="session-uncertain-runner-first")
        )
        return first_durable, second_events, provider, following, binding, runner

    first_durable, second_events, provider, following, binding, runner = asyncio.run(run())

    assert not any(
        record.event.type is EventType.WORKSPACE_MUTATION_RECORDED for record in first_durable
    )
    assert any(event.type is EventType.SESSION_COMPLETED for event in second_events)
    assert runner.mutation_finished.is_set()
    assert provider.requests == 2
    assert following.started.is_set() is False
    assert binding.finalize_calls == 1
    # The retained first-session binding and the normally completed second
    # session own distinct bound generations, so each is abandoned exactly
    # once after the shared mutation fence proves quiescence.
    assert binding.abandon_calls == 2


def test_concurrent_factory_mutations_retain_every_child_settlement_owner(
    tmp_path,
) -> None:
    class ConcurrentFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.initial_pair_created = asyncio.Event()
            self.runners: dict[str, _FailingDeferredBackgroundMutationRunner] = {}
            self.bindings: dict[str, _TrackingFinalizeBinding] = {}

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            root = tmp_path / request.session_id
            root.mkdir()
            runner = _FailingDeferredBackgroundMutationRunner(root)
            binding = _TrackingFinalizeBinding()
            self.runners[request.session_id] = runner
            self.bindings[request.session_id] = binding
            if len(self.runners) >= 2:
                self.initial_pair_created.set()
            await self.initial_pair_created.wait()
            return EnvironmentFactoryResult(
                Environment(
                    EnvironmentSpec(name=request.environment_name),
                    workspace=LocalWorkspace(
                        root,
                        workspace_id=f"factory-{request.session_id}",
                    ),
                    runner=runner,
                    binding=binding,
                )
            )

    async def run():
        factory = ConcurrentFactory()
        provider = _ConcurrentFactoryMutationProvider()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_AwaitedRunnerMutationTool()],
        )

        first_session = "session-factory-uncertain-a"
        second_session = "session-factory-uncertain-b"
        first_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=first_session,
                    messages=[Message.text("user", "write a")],
                ),
            )
        )
        second_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=second_session,
                    messages=[Message.text("user", "write b")],
                ),
            )
        )
        first_events, second_events = await asyncio.gather(first_task, second_task)
        assert any(event.type is EventType.SESSION_FAILED for event in first_events)
        assert any(event.type is EventType.SESSION_FAILED for event in second_events)
        assert provider.requests == 2

        first_runner = factory.runners[first_session]
        second_runner = factory.runners[second_session]
        first_binding = factory.bindings[first_session]
        second_binding = factory.bindings[second_session]
        contender = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-factory-after-uncertainty",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )
        await asyncio.gather(
            first_runner.reconciliation_started.wait(),
            second_runner.reconciliation_started.wait(),
        )

        first_runner.release_mutation.set()
        await first_runner.mutation_finished.wait()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert contender.done() is False
        assert provider.requests == 2
        assert len(factory.runners) == 2
        assert second_binding.finalize_calls == 0
        assert second_binding.abandon_calls == 0

        second_runner.release_mutation.set()
        contender_events = await contender
        return (
            contender_events,
            provider,
            factory,
            first_runner,
            second_runner,
            first_binding,
            second_binding,
        )

    (
        contender_events,
        provider,
        factory,
        first_runner,
        second_runner,
        first_binding,
        second_binding,
    ) = asyncio.run(run())

    assert any(event.type is EventType.SESSION_COMPLETED for event in contender_events)
    assert provider.requests == 3
    assert len(factory.runners) == 3
    assert first_runner.settlement_calls == 2
    assert second_runner.settlement_calls == 2
    assert first_binding.finalize_calls == 0
    assert second_binding.finalize_calls == 0
    assert first_binding.abandon_calls == 1
    assert second_binding.abandon_calls == 1


def test_operator_cleanup_drain_settles_retained_runner_mutation_fence(
    tmp_path,
) -> None:
    async def run():
        workspace = LocalWorkspace(tmp_path, workspace_id="drained-runner-workspace")
        runner = _FailingDeferredBackgroundMutationRunner(tmp_path)
        binding = _TrackingFinalizeBinding()
        provider = _AwaitedRunnerAndFollowingProvider()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=runner,
                workspace=workspace,
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_AwaitedRunnerMutationTool(), _FollowingTool()],
        )

        first = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session-uncertain-runner-drain",
                messages=[Message.text("user", "write")],
            ),
        )
        assert any(event.type is EventType.SESSION_FAILED for event in first)

        drain = asyncio.create_task(app.drain_environment_cleanups(timeout_s=10))
        await runner.reconciliation_started.wait()
        await asyncio.sleep(0)
        assert drain.done() is False
        assert binding.finalize_calls == 0
        assert binding.abandon_calls == 0

        runner.release_mutation.set()
        return await drain, runner, provider, binding

    drained, runner, provider, binding = asyncio.run(run())

    assert drained is True
    assert runner.mutation_finished.is_set()
    assert provider.requests == 1
    # The durable finalize-failure evidence already made this owner safe to
    # abandon; the drain must not retroactively synchronize the failed run.
    assert binding.finalize_calls == 0
    assert binding.abandon_calls == 1


def test_cancelled_environment_reuse_wait_keeps_settlement_probe_owned(
    tmp_path,
) -> None:
    async def run():
        workspace = LocalWorkspace(tmp_path, workspace_id="cancelled-reuse-workspace")
        runner = _FailingDeferredBackgroundMutationRunner(tmp_path)
        binding = _TrackingFinalizeBinding()
        provider = _AwaitedRunnerAndFollowingProvider()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=runner,
                workspace=workspace,
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_AwaitedRunnerMutationTool(), _FollowingTool()],
        )

        first = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session-uncertain-runner-before-cancelled-reuse",
                messages=[Message.text("user", "write")],
            ),
        )
        assert any(event.type is EventType.SESSION_FAILED for event in first)

        cancelled_waiter = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-cancelled-runner-reuse",
                    messages=[Message.text("user", "wait")],
                ),
            )
        )
        await runner.reconciliation_started.wait()
        cancelled_waiter.cancel("stop waiting for runner settlement")
        with pytest.raises(asyncio.CancelledError, match="stop waiting for runner settlement"):
            await cancelled_waiter
        assert cancelled_waiter.cancelling() == 1
        assert cancelled_waiter.cancelled()
        assert runner.mutation_finished.is_set() is False
        assert provider.requests == 1
        assert binding.finalize_calls == 0
        assert binding.abandon_calls == 0

        runner.release_mutation.set()
        await runner.mutation_finished.wait()
        third = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session-runner-reuse-after-cancelled-wait",
                messages=[Message.text("user", "continue")],
            ),
        )
        return third, runner, provider, binding

    third, runner, provider, binding = asyncio.run(run())

    assert any(event.type is EventType.SESSION_COMPLETED for event in third)
    assert runner.settlement_calls == 2
    assert provider.requests == 2
    assert binding.finalize_calls == 1
    assert binding.abandon_calls == 2


def test_hostile_runner_artifact_fails_closed_without_diagnostic_leakage(
    tmp_path,
    caplog,
    capsys,
) -> None:
    async def run():
        workspace = LocalWorkspace(tmp_path, workspace_id="hostile-artifact-workspace")
        runner = _HostileArtifactBackgroundMutationRunner(tmp_path)
        binding = _TrackingFinalizeBinding()
        following = _FollowingTool()
        provider = _AwaitedRunnerAndFollowingProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=runner,
                workspace=workspace,
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_AwaitedRunnerMutationTool(), following],
        )
        session_id = "session-hostile-runner-artifact"
        public_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "write")],
            ),
        )
        durable = await store.query_events(EventQuery(session_id=session_id))
        transcript = await store.load_transcript(session_id)
        assert runner.mutation_finished.is_set() is False
        runner.release_mutation.set()
        await runner.mutation_finished.wait()
        return (
            public_events,
            [record.event for record in durable],
            transcript,
            runner,
            following,
            binding,
        )

    with warnings.catch_warnings(record=True) as captured_warnings:
        (
            public_events,
            durable_events,
            transcript,
            runner,
            following,
            binding,
        ) = asyncio.run(run())
    captured = capsys.readouterr()

    assert runner.discriminator.compared is False
    assert not any(event.type is EventType.WORKSPACE_MUTATION_RECORDED for event in durable_events)
    terminal = next(
        event
        for event in durable_events
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
    )
    assert terminal.payload["workspace_mutation_capture_status"] == "failed"
    assert (
        terminal.payload["workspace_mutation_capture_detail_code"] == "mutation_settlement_unproven"
    )
    assert any(event.type is EventType.SESSION_FAILED for event in public_events)
    assert following.started.is_set() is False
    assert binding.finalize_calls == 0
    assert binding.abandon_calls == 0
    combined = repr(
        (
            [event.model_dump(mode="json") for event in public_events],
            [event.model_dump(mode="json") for event in durable_events],
            [message.model_dump(mode="json") for message in transcript],
            captured_warnings,
            [record.getMessage() for record in caplog.records],
            captured.out,
            captured.err,
        )
    )
    assert "PRIVATE_ARTIFACT_EQUALITY_CANARY" not in combined
    assert "PRIVATE_ARTIFACT_REPR_CANARY" not in combined


def test_runner_settlement_signal_is_sanitized_at_public_runtime_boundary(
    tmp_path,
    caplog,
    capsys,
) -> None:
    class SignalSettlementRunner(Runner):
        pending_command_settlement_cancellation_safe = True

        def __init__(self) -> None:
            self.default_cwd = str(tmp_path)

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
            raise SystemExit("PRIVATE_RUNTIME_SETTLEMENT_SIGNAL_CANARY")

    async def run():
        workspace = LocalWorkspace(tmp_path, workspace_id="signal-runner-workspace")
        runner = SignalSettlementRunner()
        following = _FollowingTool()
        provider = _AwaitedRunnerAndFollowingProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=runner,
                workspace=workspace,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_AwaitedRunnerMutationTool(), following],
        )
        public_events = []
        try:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-runner-settlement-signal",
                    messages=[Message.text("user", "write")],
                )
            ):
                public_events.append(event)
        except BaseException as failure:
            durable = await store.query_events(
                EventQuery(session_id="session-runner-settlement-signal")
            )
            transcript = await store.load_transcript("session-runner-settlement-signal")
            return (
                failure,
                public_events,
                [record.event for record in durable],
                transcript,
                following,
                provider,
            )
        raise AssertionError("Settlement process signal did not propagate.")

    with warnings.catch_warnings(record=True) as captured_warnings:
        (
            failure,
            public_events,
            durable_events,
            transcript,
            following,
            provider,
        ) = asyncio.run(run())
    captured = capsys.readouterr()

    signals = [
        candidate for candidate in iter_exception_tree(failure) if isinstance(candidate, SystemExit)
    ]
    assert len(signals) == 1
    assert signals[0].code == 1
    assert not any(event.type is EventType.WORKSPACE_MUTATION_RECORDED for event in durable_events)
    assert following.started.is_set() is False
    assert provider.requests == 1
    for candidate in iter_exception_tree(failure):
        traceback = candidate.__traceback__
        while traceback is not None:
            if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
                assert "PRIVATE_RUNTIME_SETTLEMENT_SIGNAL_CANARY" not in repr(
                    tuple(traceback.tb_frame.f_locals.values())
                )
            traceback = traceback.tb_next
    combined = repr(
        (
            failure,
            [event.model_dump(mode="json") for event in public_events],
            [event.model_dump(mode="json") for event in durable_events],
            [message.model_dump(mode="json") for message in transcript],
            captured_warnings,
            [record.getMessage() for record in caplog.records],
            captured.out,
            captured.err,
        )
    )
    assert "PRIVATE_RUNTIME_SETTLEMENT_SIGNAL_CANARY" not in combined


def test_cayu_app_records_git_workspace_mutation_receipt_before_initial_commit(
    tmp_path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="unborn-git-workspace"),
                runner=LocalRunner(tmp_path),
                binding=GitRepositoryBinding(
                    repo_url="https://example.invalid/repository.git",
                    fetch=False,
                    require_clean=False,
                    verify_remote_url=False,
                ),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        public_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-unborn-receipt",
                    messages=[Message.text("user", "create a file")],
                )
            )
        ]
        durable = await store.query_events(EventQuery(session_id="session-unborn-receipt"))
        return public_events, [record.event for record in durable]

    public_events, durable_events = asyncio.run(run())
    receipt = next(
        event for event in durable_events if event.type is EventType.WORKSPACE_MUTATION_RECORDED
    )
    observations = [
        event for event in durable_events if event.type is EventType.WORKSPACE_REVISION_OBSERVED
    ]

    assert [event.payload["head_revision"] for event in observations] == [None, None]
    assert receipt.payload["status"] == "changed"
    assert receipt.payload["paths"] == [
        {"path": "shell.txt", "change": "added", "renamed_from": None}
    ]
    assert any(event.type is EventType.SESSION_COMPLETED for event in public_events)


def test_cayu_app_records_durable_no_change_workspace_mutation_receipt(tmp_path) -> None:
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            _SingleToolProvider(tool_name="noop_workspace_mutation", arguments={}),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="no-change-workspace"),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_NoopWorkspaceMutationTool()],
        )
        public = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session-no-change-workspace-receipt",
                messages=[Message.text("user", "observe without changing anything")],
            ),
        )
        durable = await store.query_events(
            EventQuery(session_id="session-no-change-workspace-receipt")
        )
        return public, [record.event for record in durable]

    public_events, durable_events = asyncio.run(run())
    receipt_events = [
        event
        for event in durable_events
        if event.type
        in {
            EventType.WORKSPACE_REVISION_OBSERVED,
            EventType.WORKSPACE_MUTATION_RECORDED,
        }
    ]

    assert [event.type for event in receipt_events] == [
        EventType.WORKSPACE_REVISION_OBSERVED,
        EventType.WORKSPACE_REVISION_OBSERVED,
        EventType.WORKSPACE_MUTATION_RECORDED,
    ]
    before, after, receipt = receipt_events
    assert [before.payload["phase"], after.payload["phase"]] == ["before", "after"]
    assert [before.payload["status"], after.payload["status"]] == [
        "supported",
        "supported",
    ]
    assert receipt.payload["status"] == "no_change"
    assert receipt.payload["paths"] == []
    assert receipt.payload["before_observation_id"] == before.id
    assert receipt.payload["after_observation_id"] == after.id
    assert any(event.type is EventType.SESSION_COMPLETED for event in public_events)


def test_malformed_deterministic_workspace_result_is_typed_capture_failure(tmp_path) -> None:
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            _SingleToolProvider(tool_name="noop_workspace_mutation", arguments={}),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=_MalformedListWorkspace(
                    tmp_path,
                    workspace_id="malformed-result-workspace",
                ),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_NoopWorkspaceMutationTool()],
        )
        public = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session-malformed-workspace-result",
                messages=[Message.text("user", "observe")],
            ),
        )
        durable = await store.query_events(
            EventQuery(session_id="session-malformed-workspace-result")
        )
        return public, [record.event for record in durable]

    public_events, durable_events = asyncio.run(run())

    observations = [
        event for event in durable_events if event.type == EventType.WORKSPACE_REVISION_OBSERVED
    ]
    assert [event.payload["status"] for event in observations] == ["failed", "failed"]
    assert all(event.payload["detail_code"] == "workspace_list_failed" for event in observations)
    receipt = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_MUTATION_RECORDED
    )
    assert receipt.payload["status"] == "failed"
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in public_events)


def test_workspace_receipt_waits_for_dynamic_secret_scope_before_publication(
    tmp_path,
    caplog,
    capsys,
) -> None:
    secret_path = "PRIVATE_WORKSPACE_PATH_CANARY"
    workspace_root = tmp_path / "workspace"
    artifact_root = tmp_path / "artifacts"
    workspace_root.mkdir()
    (workspace_root / secret_path).write_text("existing\n", encoding="utf-8")
    for index in range(39):
        (workspace_root / f"visible-{index:02}.txt").write_text(
            "existing\n",
            encoding="utf-8",
        )
    store = InMemorySessionStore()
    artifacts = LocalArtifactStore(artifact_root)
    app = CayuApp(session_store=store, enable_logging=False)
    provider = _SingleToolProvider(tool_name="resolve_workspace_path_secret", arguments={})
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=LocalWorkspace(
                workspace_root,
                workspace_id="workspace-secret-scope",
            ),
            artifact_store=artifacts,
            binding=DeterministicWorkspaceBinding(),
            vault=StaticVault({"workspace_path": secret_path}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[_ResolveWorkspacePathSecretTool()],
    )

    async def run():
        with warnings.catch_warnings(record=True) as captured_warnings:
            public = await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-dynamic-receipt-secret",
                    messages=[Message.text("user", "resolve")],
                ),
            )
        durable = await store.query_events(EventQuery(session_id="session-dynamic-receipt-secret"))
        artifact_contents = []
        for record in durable:
            artifact_id = record.event.payload.get("manifest_artifact_id")
            if type(artifact_id) is str:
                artifact_contents.append((await artifacts.read_bytes(artifact_id)).content)
        return public, durable, artifact_contents, captured_warnings

    public_events, durable, artifact_contents, captured_warnings = asyncio.run(run())
    captured = capsys.readouterr()
    combined = repr(
        (
            [event.model_dump(mode="json") for event in public_events],
            [record.event.model_dump(mode="json") for record in durable],
            provider.seen_requests,
            artifact_contents,
            captured_warnings,
            caplog.records,
            captured.out,
            captured.err,
        )
    )

    assert secret_path not in combined
    assert len(artifact_contents) == 2
    receipt_events = [
        record.event
        for record in durable
        if record.event.type
        in {EventType.WORKSPACE_REVISION_OBSERVED, EventType.WORKSPACE_MUTATION_RECORDED}
    ]
    assert len(receipt_events) == 3


def test_private_workspace_arguments_quarantine_receipt_paths(
    tmp_path,
    caplog,
    capsys,
) -> None:
    private_path = "PRIVATE_ARGUMENT_PATH_CANARY.txt"
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    provider = _SingleToolProvider(
        tool_name="private_workspace_write",
        arguments={"path": private_path},
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=LocalWorkspace(tmp_path, workspace_id="workspace-private-argument"),
            binding=DeterministicWorkspaceBinding(),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[_PrivateWorkspaceWriteTool()],
    )

    with warnings.catch_warnings(record=True) as captured_warnings:
        public_events = asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-private-receipt",
                    messages=[Message.text("user", "write")],
                ),
            )
        )
    durable = asyncio.run(store.query_events(EventQuery(session_id="session-private-receipt")))
    captured = capsys.readouterr()
    combined = repr(
        (
            [event.model_dump(mode="json") for event in public_events],
            [record.event.model_dump(mode="json") for record in durable],
            provider.seen_requests,
            captured_warnings,
            caplog.records,
            captured.out,
            captured.err,
        )
    )

    assert private_path not in combined
    receipt = next(
        record.event
        for record in durable
        if record.event.type == EventType.WORKSPACE_MUTATION_RECORDED
    )
    assert receipt.payload["status"] == "truncated"
    assert receipt.payload["detail_code"] == "workspace_evidence_quarantined"
    assert receipt.payload["paths"] == []


def test_multi_call_workspace_receipt_cannot_precede_sibling_secret_scope(
    tmp_path,
    caplog,
    capsys,
) -> None:
    secret_path = "PRIVATE_SIBLING_WORKSPACE_CANARY.txt"
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    provider = _SiblingSecretProvider(secret_path)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=LocalWorkspace(tmp_path, workspace_id="workspace-sibling-secret"),
            binding=DeterministicWorkspaceBinding(),
            vault=StaticVault({"workspace_path": secret_path}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[_PrivateWorkspaceWriteTool(), _ResolveWorkspacePathSecretTool()],
    )

    with warnings.catch_warnings(record=True) as captured_warnings:
        public_events = asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-sibling-receipt-secret",
                    messages=[Message.text("user", "write and resolve")],
                ),
            )
        )
    durable = asyncio.run(
        store.query_events(EventQuery(session_id="session-sibling-receipt-secret"))
    )
    captured = capsys.readouterr()
    combined = repr(
        (
            [event.model_dump(mode="json") for event in public_events],
            [record.event.model_dump(mode="json") for record in durable],
            provider.seen_requests,
            captured_warnings,
            caplog.records,
            captured.out,
            captured.err,
        )
    )

    assert secret_path not in combined
    receipts = [
        record.event
        for record in durable
        if record.event.type == EventType.WORKSPACE_MUTATION_RECORDED
    ]
    assert len(receipts) == 2
    assert all(
        receipt.payload["detail_code"] == "workspace_evidence_quarantined" for receipt in receipts
    )


def test_stream_abandonment_remains_authoritative_during_staged_settlement_failure(
    tmp_path,
) -> None:
    async def run() -> tuple[EventType, bool]:
        workspace = LocalWorkspace(tmp_path, workspace_id="staged-settlement-workspace")
        runner = _FailingDeferredBackgroundMutationRunner(tmp_path)
        following = _FollowingTool()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(_AwaitedRunnerAndFollowingProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=runner,
                workspace=workspace,
                binding=DeterministicWorkspaceBinding(),
                vault=StaticVault({"unused_dynamic_secret": "private"}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_AwaitedRunnerMutationTool(), following],
        )

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="session-staged-settlement-abandonment",
                messages=[Message.text("user", "mutate and continue")],
            )
        )
        terminal_type: EventType | None = None
        async for event in stream:
            if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}:
                terminal_type = event.type
                assert (
                    event.payload["workspace_mutation_capture_detail_code"]
                    == "mutation_settlement_unproven"
                )
                break
        assert terminal_type is not None

        # Async-generator closure injects a real GeneratorExit at the staged
        # terminal yield. It must not be replaced by the settlement failure.
        try:
            await stream.aclose()
        finally:
            runner.release_mutation.set()
            await runner.mutation_finished.wait()
        return terminal_type, following.started.is_set()

    terminal_type, following_started = asyncio.run(run())

    assert terminal_type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
    assert following_started is False


def test_approval_resume_preserves_workspace_receipt_model_step(tmp_path) -> None:
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )
        paused = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-approval-receipt",
                    messages=[Message.text("user", "create a file")],
                )
            )
        ]
        requested = next(
            event for event in paused if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval = requested.payload["approval"]
        _ = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="session-approval-receipt",
                    approval_id=approval["approval_id"],
                    tool_round_id=approval["tool_round_id"],
                    tool_call_id=approval["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        durable = await store.query_events(EventQuery(session_id="session-approval-receipt"))
        return [record.event for record in durable]

    durable_events = asyncio.run(run())

    receipt = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_MUTATION_RECORDED
    )
    assert receipt.payload["model_step"] == 1
    assert receipt.payload["tool_call_id"] == "call-shell"


def test_user_input_resume_preserves_workspace_receipt_model_step(tmp_path) -> None:
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_UserInputThenMutationProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[UserInputTool(), ExecCommandTool()],
        )
        paused = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-input-receipt",
                    messages=[Message.text("user", "create a file")],
                )
            )
        ]
        awaiting = next(
            event for event in paused if event.type == EventType.SESSION_AWAITING_USER_INPUT
        )
        _ = [
            event
            async for event in app.resolve_user_input(
                UserInputResponse(
                    session_id="session-input-receipt",
                    input_id=awaiting.payload["input_id"],
                    answer="yes",
                )
            )
        ]
        durable = await store.query_events(EventQuery(session_id="session-input-receipt"))
        return [record.event for record in durable]

    durable_events = asyncio.run(run())

    receipt = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_MUTATION_RECORDED
    )
    assert receipt.payload["model_step"] == 1
    assert receipt.payload["tool_call_id"] == "call-resumed-shell"


def test_large_workspace_receipt_uses_integrity_checked_artifact_reference(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")
    workspace_root = tmp_path / "workspace"
    artifact_root = tmp_path / "artifacts"
    workspace_root.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=workspace_root,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    (workspace_root / "README.md").write_text("baseline\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-m", "baseline")

    async def run():
        store = InMemorySessionStore()
        artifacts = LocalArtifactStore(artifact_root)
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            secret_redactor=SecretRedactor(["workspace-secret-value"]),
        )
        app.register_provider(_BulkProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="git-workspace"),
                runner=LocalRunner(workspace_root),
                artifact_store=artifacts,
                binding=GitRepositoryBinding(
                    repo_url="https://example.invalid/repository.git",
                    fetch=False,
                    require_clean=False,
                    verify_remote_url=False,
                ),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="bulk-model"),
            tools=[_BulkWriteTool()],
        )
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-bulk",
                    messages=[Message.text("user", "write files")],
                )
            )
        ]
        records = await store.query_events(EventQuery(session_id="session-bulk"))
        events = [record.event for record in records]
        after = next(
            event
            for event in events
            if event.type == EventType.WORKSPACE_REVISION_OBSERVED
            and event.payload["phase"] == "after"
        )
        receipt = next(
            event for event in events if event.type == EventType.WORKSPACE_MUTATION_RECORDED
        )
        after_artifact = await artifacts.read_bytes(after.payload["manifest_artifact_id"])
        receipt_artifact = await artifacts.read_bytes(receipt.payload["manifest_artifact_id"])
        return after, receipt, after_artifact, receipt_artifact

    after, receipt, after_artifact, receipt_artifact = asyncio.run(run())

    for event, artifact, expected_paths in (
        (after, after_artifact, 66),
        (receipt, receipt_artifact, 65),
    ):
        assert event.payload["paths"] == []
        assert event.payload["total_paths"] == expected_paths
        assert event.payload["manifest_artifact_size_bytes"] == artifact.total_bytes
        assert (
            event.payload["manifest_artifact_sha256"]
            == hashlib.sha256(artifact.content).hexdigest()
        )
        assert artifact.metadata.metadata["sha256"] == event.payload["manifest_artifact_sha256"]
        assert b"workspace-secret-value" not in artifact.content
        assert len(event.model_dump_json().encode("utf-8")) < 16_000


@pytest.mark.parametrize(
    ("artifact_store_type", "expected_status", "expected_detail_code"),
    [
        (
            _ChildCancelledArtifactStore,
            "truncated",
            "manifest_artifact_write_failed",
        ),
        (
            _MalformedArtifactStore,
            "failed",
            "manifest_artifact_reference_invalid",
        ),
    ],
)
def test_artifact_store_failures_are_bounded_without_replacing_tool_outcome(
    tmp_path,
    artifact_store_type,
    expected_status,
    expected_detail_code,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = tmp_path / "artifacts"
    workspace_root.mkdir()

    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_BulkProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=artifact_store_type(artifact_root),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="bulk-model"),
            tools=[_BulkWriteTool()],
        )
        public = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-artifact-child-cancellation",
                    messages=[Message.text("user", "create files")],
                )
            )
        ]
        durable = await store.query_events(
            EventQuery(session_id="session-artifact-child-cancellation")
        )
        return public, [record.event for record in durable]

    public_events, durable_events = asyncio.run(run())

    receipt = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_MUTATION_RECORDED
    )
    assert receipt.payload["status"] == expected_status
    assert receipt.payload["detail_code"] == expected_detail_code
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in public_events)


def test_stalled_receipt_artifact_write_is_bounded_without_replacing_tool_outcome(
    tmp_path,
    monkeypatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = tmp_path / "artifacts"
    workspace_root.mkdir()
    monkeypatch.setattr(
        tool_round_executor_module,
        "_WORKSPACE_ARTIFACT_WRITE_TIMEOUT_SECONDS",
        0.01,
    )

    async def run():
        store = InMemorySessionStore()
        artifacts = _StalledArtifactStore(artifact_root)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_BulkProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=artifacts,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="bulk-model"),
            tools=[_BulkWriteTool()],
        )
        public = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-stalled-receipt-artifact",
                    messages=[Message.text("user", "create files")],
                )
            )
        ]
        assert artifacts.started.is_set()
        artifacts.release.set()
        await asyncio.sleep(0)
        durable = await store.query_events(
            EventQuery(session_id="session-stalled-receipt-artifact")
        )
        return public, [record.event for record in durable]

    public_events, durable_events = asyncio.run(run())

    after = next(
        event
        for event in durable_events
        if event.type == EventType.WORKSPACE_REVISION_OBSERVED and event.payload["phase"] == "after"
    )
    assert after.payload["status"] == "truncated"
    assert after.payload["detail_code"] == "manifest_artifact_write_failed"
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in public_events)


def test_receipt_artifacts_inside_workspace_do_not_contaminate_tool_delta(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.email", "tester@example.com")
    git("config", "user.name", "Test User")
    (tmp_path / ".gitignore").write_text(".artifacts/\n", encoding="utf-8")
    for index in range(40):
        (tmp_path / f"tracked-{index:02}.txt").write_text("baseline\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "baseline")

    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="git-workspace"),
                runner=LocalRunner(tmp_path),
                artifact_store=LocalArtifactStore(tmp_path / ".artifacts"),
                binding=GitRepositoryBinding(
                    repo_url="https://example.invalid/repository.git",
                    fetch=False,
                    require_clean=False,
                    verify_remote_url=False,
                ),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-artifacts-inside-workspace",
                    messages=[Message.text("user", "create a file")],
                )
            )
        ]
        records = await store.query_events(
            EventQuery(session_id="session-artifacts-inside-workspace")
        )
        return [record.event for record in records]

    durable_events = asyncio.run(run())

    observations = [
        event for event in durable_events if event.type == EventType.WORKSPACE_REVISION_OBSERVED
    ]
    receipt = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_MUTATION_RECORDED
    )
    assert [event.payload["status"] for event in observations] == [
        "truncated",
        "truncated",
    ]
    assert all(
        event.payload["detail_code"] == "manifest_artifact_store_inside_workspace"
        for event in observations
    )
    assert receipt.payload["status"] == "changed"
    assert receipt.payload["paths"] == [
        {"path": "shell.txt", "change": "added", "renamed_from": None}
    ]
    assert list((tmp_path / ".artifacts").iterdir()) == []


@pytest.mark.parametrize(
    "binding",
    [
        _FailingObserverBinding(),
        _IdentityDriftBinding(),
        _DuplicatePathObserverBinding(),
    ],
)
def test_revision_observer_failure_is_visible_without_replacing_tool_outcome(
    tmp_path,
    binding,
) -> None:
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        public = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-observer-failure",
                    messages=[Message.text("user", "create a file")],
                )
            )
        ]
        durable = await store.query_events(EventQuery(session_id="session-observer-failure"))
        return public, [record.event for record in durable]

    public_events, durable_events = asyncio.run(run())

    observations = [
        event for event in durable_events if event.type == EventType.WORKSPACE_REVISION_OBSERVED
    ]
    receipt = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_MUTATION_RECORDED
    )
    assert observations[-1].payload["status"] == "failed"
    assert observations[-1].payload["detail_code"] == "revision_observer_failed"
    assert receipt.payload["status"] == "failed"
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in public_events)
    assert (tmp_path / "shell.txt").read_text(encoding="utf-8") == "created"
    assert "foreign-workspace" not in repr(
        [event.model_dump(mode="json") for event in durable_events]
    )


def test_revision_observer_runtime_limit_is_typed_without_replacing_tool_outcome(
    tmp_path,
) -> None:
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=_OversizedObserverBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        public = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-observer-limit",
                    messages=[Message.text("user", "create a file")],
                )
            )
        ]
        durable = await store.query_events(EventQuery(session_id="session-observer-limit"))
        return public, [record.event for record in durable]

    public_events, durable_events = asyncio.run(run())

    observations = [
        event for event in durable_events if event.type == EventType.WORKSPACE_REVISION_OBSERVED
    ]
    receipt = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_MUTATION_RECORDED
    )
    assert [event.payload["status"] for event in observations] == [
        "truncated",
        "truncated",
    ]
    assert all(
        event.payload["detail_code"] == "revision_observer_limit_exceeded" for event in observations
    )
    assert receipt.payload["status"] == "truncated"
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in public_events)


def test_malformed_revision_observer_is_sanitized_before_serialization(
    tmp_path,
    caplog,
    capsys,
) -> None:
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=_MalformedObserverBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        with warnings.catch_warnings(record=True) as captured_warnings:
            public = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="session-malformed-observer",
                        messages=[Message.text("user", "create a file")],
                    )
                )
            ]
        durable = await store.query_events(EventQuery(session_id="session-malformed-observer"))
        return public, [record.event for record in durable], captured_warnings

    public_events, durable_events, captured_warnings = asyncio.run(run())
    captured = capsys.readouterr()
    combined = repr(
        (
            public_events,
            durable_events,
            captured_warnings,
            caplog.records,
            captured.out,
            captured.err,
        )
    )

    assert "PRIVATE_OBSERVER_CANARY" not in combined
    observations = [
        event for event in durable_events if event.type == EventType.WORKSPACE_REVISION_OBSERVED
    ]
    assert [event.payload["status"] for event in observations] == ["failed", "failed"]
    assert all(event.payload["detail_code"] == "revision_observer_failed" for event in observations)
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in public_events)


def test_observer_owned_cancellation_is_a_bounded_capture_failure(tmp_path) -> None:
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=_ChildCancelledObserverBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        public = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-child-cancelled-observer",
                    messages=[Message.text("user", "create a file")],
                )
            )
        ]
        durable = await store.query_events(
            EventQuery(session_id="session-child-cancelled-observer")
        )
        return public, [record.event for record in durable]

    public_events, durable_events = asyncio.run(run())

    observations = [
        event for event in durable_events if event.type == EventType.WORKSPACE_REVISION_OBSERVED
    ]
    assert [event.payload["status"] for event in observations] == ["failed", "failed"]
    assert all(event.payload["detail_code"] == "revision_observer_failed" for event in observations)
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in public_events)


def test_stalled_observer_is_bounded_without_replacing_tool_outcome(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        tool_round_executor_module,
        "_WORKSPACE_OBSERVATION_TIMEOUT_SECONDS",
        0.01,
    )

    async def run():
        binding = _StalledObserverBinding()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        public = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-stalled-observer",
                    messages=[Message.text("user", "create a file")],
                )
            )
        ]
        assert binding.started.is_set()
        binding.release.set()
        await asyncio.sleep(0)
        durable = await store.query_events(EventQuery(session_id="session-stalled-observer"))
        return public, [record.event for record in durable]

    public_events, durable_events = asyncio.run(run())

    observations = [
        event for event in durable_events if event.type == EventType.WORKSPACE_REVISION_OBSERVED
    ]
    assert [event.payload["status"] for event in observations] == ["supported", "failed"]
    assert observations[-1].payload["detail_code"] == "revision_observer_timeout"
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in public_events)


def test_caller_cancellation_during_after_observation_preserves_tool_terminal(
    tmp_path,
) -> None:
    async def run():
        binding = _StalledObserverBinding()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )

        async def consume() -> None:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="session-cancelled-after-observer",
                        messages=[Message.text("user", "create a file")],
                    )
                )
            ]

        consumer = asyncio.create_task(consume())
        await binding.started.wait()
        consumer.cancel("cancel while observing workspace")
        with pytest.raises(asyncio.CancelledError, match="cancel while observing workspace"):
            await consumer
        assert consumer.cancelling() == 0
        assert consumer.cancelled() is True
        binding.release.set()
        await asyncio.sleep(0)
        durable = await store.query_events(
            EventQuery(session_id="session-cancelled-after-observer")
        )
        return [record.event for record in durable]

    durable_events = asyncio.run(run())

    terminal = next(
        event for event in durable_events if event.type == EventType.TOOL_CALL_COMPLETED
    )
    assert terminal.payload["workspace_mutation_capture_status"] == "interrupted"
    assert (
        terminal.payload["workspace_mutation_capture_detail_code"]
        == "receipt_publication_interrupted"
    )


def test_operator_interruption_during_receipt_append_preserves_tool_terminal(
    tmp_path,
) -> None:
    async def run():
        async def collect(stream):
            return [event async for event in stream]

        store = _BlockingAfterObservationStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )

        session_id = "session-interrupted-receipt-append"
        consumer = asyncio.create_task(
            collect(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "create a file")],
                    )
                )
            )
        )
        await store.started.wait()
        interruption = asyncio.create_task(
            collect(
                app.interrupt_session(
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="operator interrupted receipt capture",
                    )
                )
            )
        )
        for _ in range(100):
            if consumer.cancelling():
                break
            await asyncio.sleep(0)
        assert consumer.cancelling() == 1
        store.release.set()
        public, interrupted = await asyncio.gather(consumer, interruption)
        durable = await store.query_events(EventQuery(session_id=session_id))
        return public, interrupted, [record.event for record in durable]

    public_events, interruption_events, durable_events = asyncio.run(run())

    terminal = next(
        event for event in durable_events if event.type == EventType.TOOL_CALL_COMPLETED
    )
    assert terminal.payload["workspace_mutation_capture_status"] == "interrupted"
    assert (
        terminal.payload["workspace_mutation_capture_detail_code"]
        == "receipt_publication_interrupted"
    )
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in public_events)
    assert [event.type for event in interruption_events] == [EventType.SESSION_INTERRUPTED]


def test_terminal_failure_after_capture_cancellation_preserves_caller_cancellation(
    tmp_path,
) -> None:
    async def run():
        store = _FailingTerminalAfterBlockingCaptureStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )

        async def consume() -> None:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="session-cancelled-terminal-failure",
                        messages=[Message.text("user", "create a file")],
                    )
                )
            ]

        consumer = asyncio.create_task(consume())
        await store.started.wait()
        consumer.cancel("cancel during workspace capture")
        store.release.set()
        with pytest.raises(
            asyncio.CancelledError, match="cancel during workspace capture"
        ) as raised:
            await consumer
        assert consumer.cancelling() == 0
        assert consumer.cancelled() is True
        cause = exception_cause(raised.value)
        assert isinstance(cause, RuntimeError)
        assert "terminal publication failed" not in str(cause)
        durable = await store.query_events(
            EventQuery(session_id="session-cancelled-terminal-failure")
        )
        return [record.event for record in durable]

    durable_events = asyncio.run(run())

    assert not any(event.type == EventType.SESSION_FAILED for event in durable_events)
    assert (tmp_path / "shell.txt").read_text(encoding="utf-8") == "created"


def test_later_terminal_cancellation_does_not_replace_capture_cancellation(tmp_path) -> None:
    async def run():
        store = _BlockingTerminalAfterBlockingCaptureStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )

        async def consume() -> None:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="session-repeated-capture-cancellation",
                        messages=[Message.text("user", "create a file")],
                    )
                )
            ]

        consumer = asyncio.create_task(consume())
        await store.started.wait()
        consumer.cancel("first cancellation during workspace capture")
        store.release.set()
        await store.terminal_started.wait()
        consumer.cancel("later cancellation during terminal publication")
        with pytest.raises(asyncio.CancelledError) as raised:
            await consumer
        assert raised.value.args == ("first cancellation during workspace capture",)
        assert exception_cause(raised.value) is not None
        assert consumer.cancelling() == 0
        assert consumer.cancelled() is True
        store.terminal_release.set()

    asyncio.run(run())

    assert (tmp_path / "shell.txt").read_text(encoding="utf-8") == "created"


@pytest.mark.parametrize(
    "store_type",
    [_FailAfterObservationStore, _ChildCancelledAfterObservationStore],
)
def test_workspace_capture_publication_failure_preserves_tool_terminal(
    tmp_path,
    store_type,
) -> None:
    async def run():
        store = store_type()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            secret_redactor=SecretRedactor(["failed", "receipt_publication_failed"]),
        )
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        public = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-capture-publication-failure",
                    messages=[Message.text("user", "create a file")],
                )
            )
        ]
        durable = await store.query_events(
            EventQuery(session_id="session-capture-publication-failure")
        )
        return public, [record.event for record in durable]

    public_events, durable_events = asyncio.run(run())

    terminal = next(
        event for event in durable_events if event.type == EventType.TOOL_CALL_COMPLETED
    )
    assert terminal.payload["workspace_mutation_capture_status"] == "failed"
    assert (
        terminal.payload["workspace_mutation_capture_detail_code"] == "receipt_publication_failed"
    )
    public_terminal = next(
        event for event in public_events if event.type == EventType.TOOL_CALL_COMPLETED
    )
    assert public_terminal.payload["workspace_mutation_capture_status"] == "failed"
    assert (
        public_terminal.payload["workspace_mutation_capture_detail_code"]
        == "receipt_publication_failed"
    )
    assert (tmp_path / "shell.txt").read_text(encoding="utf-8") == "created"


def test_abandoning_after_workspace_event_does_not_erase_tool_terminal(tmp_path) -> None:
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="session-abandoned-receipt-stream",
                messages=[Message.text("user", "create a file")],
            )
        )
        async for event in stream:
            if (
                event.type == EventType.WORKSPACE_REVISION_OBSERVED
                and event.payload.get("phase") == "after"
            ):
                break
        await stream.aclose()
        durable = await store.query_events(
            EventQuery(session_id="session-abandoned-receipt-stream")
        )
        return [record.event for record in durable]

    durable_events = asyncio.run(run())

    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in durable_events)
    assert any(event.type == EventType.WORKSPACE_MUTATION_RECORDED for event in durable_events)


def test_workspace_receipt_closes_before_tool_cancellation_propagates(tmp_path) -> None:
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_CancelProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="cancel-model"),
            tools=[_CancellingMutationTool()],
        )
        with pytest.raises(asyncio.CancelledError, match="tool cancellation canary"):
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="session-cancel-receipt",
                        messages=[Message.text("user", "write then cancel")],
                    )
                )
            ]
        durable = await store.query_events(EventQuery(session_id="session-cancel-receipt"))
        return [record.event for record in durable]

    durable_events = asyncio.run(run())

    receipt = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_MUTATION_RECORDED
    )
    assert receipt.payload["status"] == "changed"
    assert receipt.payload["paths"] == [
        {
            "path": "cancelled-write.txt",
            "change": "added",
            "renamed_from": None,
        }
    ]
    assert (tmp_path / "cancelled-write.txt").read_bytes() == b"written before cancellation"
    assert "tool cancellation canary" not in repr(
        [event.model_dump(mode="json") for event in durable_events]
    )


def test_workspace_receipt_waits_for_cancellation_opaque_mutation_after_timeout(tmp_path) -> None:
    workspace = _BlockingThreadWorkspace(tmp_path, workspace_id="blocking-workspace")
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        tool_timeout_seconds=0.05,
    )
    app.register_provider(
        _SingleToolProvider(tool_name="blocking_workspace_mutation", arguments={}),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=workspace,
            binding=DeterministicWorkspaceBinding(),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[_BlockingWorkspaceMutationTool()],
    )

    async def run():
        consumer = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-opaque-mutation-timeout",
                    messages=[Message.text("user", "write")],
                ),
            )
        )
        await asyncio.to_thread(workspace.dispatched.wait, 1)
        await asyncio.sleep(0.1)
        assert not consumer.done()
        durable_before_release = await store.query_events(
            EventQuery(session_id="session-opaque-mutation-timeout")
        )
        assert not any(
            record.event.type == EventType.WORKSPACE_MUTATION_RECORDED
            for record in durable_before_release
        )
        workspace.release.set()
        events = await consumer
        durable = await store.query_events(EventQuery(session_id="session-opaque-mutation-timeout"))
        return events, [record.event for record in durable]

    events, durable_events = asyncio.run(run())

    receipt = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_MUTATION_RECORDED
    )
    assert receipt.payload["status"] == "changed"
    assert receipt.payload["paths"] == [
        {"path": "settled.txt", "change": "added", "renamed_from": None}
    ]
    terminal = next(event for event in events if event.type == EventType.TOOL_CALL_FAILED)
    assert terminal.payload["terminal_outcome"] == "tool_execution_timeout"


def test_workspace_receipt_waits_for_cancellation_opaque_mutation_after_task_cancel(
    tmp_path,
) -> None:
    workspace = _BlockingThreadWorkspace(tmp_path, workspace_id="cancelled-workspace")
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        _SingleToolProvider(tool_name="blocking_workspace_mutation", arguments={}),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=workspace,
            binding=DeterministicWorkspaceBinding(),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[_BlockingWorkspaceMutationTool()],
    )

    async def run():
        consumer = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-opaque-mutation-cancel",
                    messages=[Message.text("user", "write")],
                ),
            )
        )
        await asyncio.to_thread(workspace.dispatched.wait, 1)
        consumer.cancel("cancel after mutation dispatch")
        await asyncio.sleep(0.05)
        assert not consumer.done()
        durable_before_release = await store.query_events(
            EventQuery(session_id="session-opaque-mutation-cancel")
        )
        assert not any(
            record.event.type == EventType.WORKSPACE_MUTATION_RECORDED
            for record in durable_before_release
        )
        workspace.release.set()
        with pytest.raises(asyncio.CancelledError, match="cancel after mutation dispatch"):
            await consumer
        assert consumer.cancelling() == 1
        assert consumer.cancelled() is True
        durable = await store.query_events(EventQuery(session_id="session-opaque-mutation-cancel"))
        return [record.event for record in durable]

    durable_events = asyncio.run(run())

    receipt = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_MUTATION_RECORDED
    )
    assert receipt.payload["status"] == "changed"
    assert receipt.payload["paths"] == [
        {"path": "settled.txt", "change": "added", "renamed_from": None}
    ]


def test_store_cancellation_group_during_mutation_settlement_is_operational_failure(
    tmp_path,
    caplog,
    capsys,
) -> None:
    async def run():
        workspace = _GroupedFailureWorkspace(
            tmp_path,
            workspace_id="grouped-failure-workspace",
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            _SingleToolProvider(tool_name="detached_workspace_mutation", arguments={}),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=workspace,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_DetachedWorkspaceMutationTool(dispatched=workspace.started)],
        )
        consumer = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-grouped-mutation-failure",
                    messages=[Message.text("user", "write")],
                ),
            )
        )
        await workspace.started.wait()
        await asyncio.sleep(0)
        assert not consumer.done()
        workspace.release.set()
        public = await consumer
        assert consumer.cancelling() == 0
        assert consumer.cancelled() is False
        durable = await store.query_events(
            EventQuery(session_id="session-grouped-mutation-failure")
        )
        return public, durable

    with warnings.catch_warnings(record=True) as captured_warnings:
        public_events, durable = asyncio.run(run())
    captured = capsys.readouterr()
    terminal = next(event for event in public_events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert terminal.payload["workspace_mutation_capture_status"] == "failed"
    assert (
        terminal.payload["workspace_mutation_capture_detail_code"] == "receipt_publication_failed"
    )
    combined = repr(
        (
            [event.model_dump(mode="json") for event in public_events],
            [record.event.model_dump(mode="json") for record in durable],
            captured_warnings,
            caplog.records,
            captured.out,
            captured.err,
        )
    )
    assert "PRIVATE_MUTATION" not in combined


def test_caller_cancellation_remains_authoritative_over_mutation_failure_group(
    tmp_path,
) -> None:
    async def run():
        workspace = _GroupedFailureWorkspace(
            tmp_path,
            workspace_id="cancelled-grouped-failure-workspace",
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            _SingleToolProvider(tool_name="detached_workspace_mutation", arguments={}),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=workspace,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_DetachedWorkspaceMutationTool(dispatched=workspace.started)],
        )
        consumer = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-cancelled-grouped-mutation-failure",
                    messages=[Message.text("user", "write")],
                ),
            )
        )
        await workspace.started.wait()
        consumer.cancel("caller cancelled grouped mutation")
        await asyncio.sleep(0)
        assert not consumer.done()
        workspace.release.set()
        with pytest.raises(
            asyncio.CancelledError,
            match="caller cancelled grouped mutation",
        ) as raised:
            await consumer
        assert consumer.cancelling() == 1
        assert consumer.cancelled() is True
        cause = exception_cause(raised.value)
        assert isinstance(cause, RuntimeError)
        assert str(cause) == "Runner command execution failed."
        assert "PRIVATE_MUTATION" not in repr(raised.value)
        assert "PRIVATE_MUTATION" not in repr(cause)

    asyncio.run(run())


def test_repeated_cancellation_during_mutation_settlement_preserves_original(
    tmp_path,
) -> None:
    async def run():
        workspace = _AsyncBlockingWorkspace(
            tmp_path,
            workspace_id="repeated-cancellation-workspace",
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            _SingleToolProvider(
                tool_name="detached_then_blocking_workspace_mutation",
                arguments={},
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=workspace,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_DetachedThenBlockingWorkspaceMutationTool(dispatched=workspace.started)],
        )
        consumer = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-repeated-mutation-cancellation",
                    messages=[Message.text("user", "write")],
                ),
            )
        )
        await workspace.started.wait()
        consumer.cancel("first mutation cancellation")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not consumer.done()
        consumer.cancel("second mutation cancellation")
        await asyncio.sleep(0)
        assert not consumer.done()
        workspace.release.set()
        with pytest.raises(
            asyncio.CancelledError,
            match="first mutation cancellation",
        ):
            await consumer
        assert consumer.cancelling() == 2
        assert consumer.cancelled() is True
        durable = await store.query_events(
            EventQuery(session_id="session-repeated-mutation-cancellation")
        )
        return [record.event for record in durable]

    durable_events = asyncio.run(run())

    receipt_events = [
        event for event in durable_events if event.type == EventType.WORKSPACE_MUTATION_RECORDED
    ]
    assert receipt_events, [event.type for event in durable_events]
    receipt = receipt_events[0]
    assert receipt.payload["status"] == "changed"
    assert receipt.payload["paths"] == [
        {
            "path": "settled-after-cancellation.txt",
            "change": "added",
            "renamed_from": None,
        }
    ]


def test_supervisory_exit_during_cancelled_mutation_close_fences_environment_reuse(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run():
        workspace = _AsyncBlockingWorkspace(
            tmp_path,
            workspace_id="supervisory-cancelled-close-workspace",
        )
        provider = _SingleToolProvider(
            tool_name="detached_then_blocking_workspace_mutation",
            arguments={},
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=workspace,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_DetachedThenBlockingWorkspaceMutationTool(dispatched=workspace.started)],
        )

        original_boundary = tool_round_executor_module.await_invocation_operation
        original_shield = operation_boundary_module.asyncio.shield
        inside_cancelled_close = False
        supervisory_delivered = False

        async def tracked_boundary(operation_factory, **kwargs):
            nonlocal inside_cancelled_close
            is_cancelled_close = (
                getattr(operation_factory, "__name__", None) == "close_workspace_mutation_window"
                and kwargs.get("cancellation") is not None
            )
            if not is_cancelled_close:
                return await original_boundary(operation_factory, **kwargs)
            inside_cancelled_close = True
            try:
                return await original_boundary(operation_factory, **kwargs)
            finally:
                inside_cancelled_close = False

        async def supervisory_shield(awaitable):
            nonlocal supervisory_delivered
            if inside_cancelled_close and not supervisory_delivered:
                supervisory_delivered = True
                raise GeneratorExit("supervisor abandoned cancellation cleanup")
            return await original_shield(awaitable)

        monkeypatch.setattr(
            tool_round_executor_module,
            "await_invocation_operation",
            tracked_boundary,
        )
        monkeypatch.setattr(operation_boundary_module.asyncio, "shield", supervisory_shield)

        consumer = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-supervisory-cancelled-close",
                    messages=[Message.text("user", "write")],
                ),
            )
        )
        await workspace.started.wait()
        consumer.cancel("cancel before supervisory cleanup exit")
        with pytest.raises(GeneratorExit, match="supervisor abandoned cancellation cleanup"):
            await consumer
        assert supervisory_delivered is True
        assert consumer.cancelling() == 1
        assert consumer.cancelled() is False
        assert not (tmp_path / "settled-after-cancellation.txt").exists()

        contender = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-after-supervisory-cancelled-close",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert contender.done() is False
        assert provider.requests == 1

        workspace.release.set()
        contender_events = await contender
        durable = await store.query_events(
            EventQuery(session_id="session-supervisory-cancelled-close")
        )
        return contender_events, [record.event for record in durable], provider

    contender_events, durable_events, provider = asyncio.run(run())

    assert provider.requests == 2
    assert any(event.type is EventType.SESSION_COMPLETED for event in contender_events)
    assert not any(event.type is EventType.WORKSPACE_MUTATION_RECORDED for event in durable_events)
    assert (tmp_path / "settled-after-cancellation.txt").read_bytes() == b"settled"
