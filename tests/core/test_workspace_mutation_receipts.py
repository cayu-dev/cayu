from __future__ import annotations

import asyncio
import copy
import hashlib
import shutil
import subprocess
import sys
import threading
import traceback
import warnings
from collections.abc import AsyncIterator
from typing import Any

import pytest
from tests.provider_traceback_assertions import is_cayu_source_filename

import cayu.runtime._environment_lifecycle as environment_lifecycle_module
import cayu.runtime._session_engine as session_engine_module
import cayu.runtime._tool_round_executor as tool_round_executor_module
import cayu.tools._operation_boundary as operation_boundary_module
import cayu.tools._runner as runner_module
from cayu._exception_groups import exception_cause, iter_exception_tree
from cayu._exception_state import set_exception_state
from cayu._validation import canonical_durable_json_bytes
from cayu._workspace_mutation import WorkspaceMutationSettlementError
from cayu.artifacts import ArtifactMetadata, ArtifactScope, LocalArtifactStore
from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    ExecutionProfileBehaviorIdentity,
    Message,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
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
    RunnerExecutionError,
    attach_cancellation_artifacts,
)
from cayu.runtime import (
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    EventQuery,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    InMemoryBudgetStore,
    InMemorySessionStore,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
    RuntimePublicationRequest,
    RuntimePublicationResult,
    SessionIdentity,
    SessionStatus,
    ToolApprovalDecision,
    ToolApprovalRequest,
    UserInputResponse,
)
from cayu.runtime._environment_operation_boundary import await_environment_operation
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._model_errors import (
    _BillingIdentityResolutionCancelled,
    detach_billing_identity_cancellation_group,
)
from cayu.runtime.app import _close_delegated_event_stream
from cayu.runtime.sessions import runtime_publication_request_digest
from cayu.runtime.workspace_observation_recovery import (
    WorkspaceObservationLifecycle,
    _admit_workspace_observation_intent,
    _project_workspace_observation_authority,
    publish_workspace_observation_transition,
    retain_workspace_observation_pending_cancellation_requests,
    workspace_observation_event_digest,
    workspace_observation_observer_authority_matches,
    workspace_observation_pending_cancellation_requests,
    workspace_observations_from_checkpoint,
)
from cayu.tools import ExecCommandTool, UserInputTool
from cayu.vaults import SecretRedactor, SecretRef, StaticVault
from cayu.workspaces import (
    LocalWorkspace,
    WorkspaceIdentity,
    WorkspacePathRevision,
    WorkspaceRevisionDelta,
    WorkspaceRevisionDeltaStatus,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationLimits,
    WorkspaceRevisionObservationStatus,
)


def _admit_test_workspace_observation_intent(
    lifecycle: WorkspaceObservationLifecycle,
) -> Any:
    return _admit_workspace_observation_intent(
        lifecycle,
        redactor=SecretRedactor(),
        configured_workspace_id=(
            None if lifecycle.workspace_id == "workspace-unavailable" else lifecycle.workspace_id
        ),
        configured_artifact_store_id=lifecycle.artifact_store_id,
    )


async def collect_events(app: CayuApp, request: RunRequest):
    return [event async for event in app.run(request)]


def _portable_environment_spec(name: str) -> EnvironmentSpec:
    """Declare equivalent test environments portable across recovery app instances."""

    return EnvironmentSpec(
        name=name,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name=f"tests:workspace-mutation-receipts:{name}",
            behavior_version="stable",
            implementation_version="test-build",
        ),
    )


@pytest.mark.parametrize(
    (
        "durable_authority",
        "configured_observer_is_runtime_owned",
        "expected",
    ),
    [
        ("runtime_builtin", True, True),
        ("runtime_builtin", False, False),
        ("configured", True, False),
        ("configured", False, True),
    ],
)
def test_workspace_observer_authority_matching_requires_equal_provenance(
    durable_authority,
    configured_observer_is_runtime_owned: bool,
    expected: bool,
) -> None:
    assert (
        workspace_observation_observer_authority_matches(
            "DeterministicWorkspaceBinding",
            durable_authority,
            "DeterministicWorkspaceBinding",
            configured_observer_is_runtime_owned=configured_observer_is_runtime_owned,
            session_id="session-observer-authority-provenance",
            public_authority_alias_codec=None,
        )
        is expected
    )


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


class _CompletionThenSingleToolProvider(_SingleToolProvider):
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.seen_requests.append(request)
        self.requests += 1
        if self.requests == 1:
            yield ModelStreamEvent.text_delta("ready")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        if self.requests == 2:
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
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:bulk-write-tool",
            behavior_version="stable",
            implementation_version="test-build",
        ),
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
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:noop-workspace-mutation-tool",
            behavior_version="1",
            implementation_version="1",
        ),
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


class _OneShotGeneratorExitObserverBinding(DeterministicWorkspaceBinding):
    def __init__(self) -> None:
        super().__init__()
        self.observations = 0

    async def observe_revision(self, bound):
        self.observations += 1
        if self.observations == 1:
            raise GeneratorExit("observer supervisory exit")
        return await super().observe_revision(bound)


class _ConcurrentGeneratorExitObserverBinding(DeterministicWorkspaceBinding):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancel_target: asyncio.Task | None = None
        self.cancellation_sent = False

    async def observe_revision(self, bound):
        del bound
        self.started.set()
        await self.release.wait()
        assert self.cancel_target is not None
        if not self.cancellation_sent:
            self.cancellation_sent = True
            self.cancel_target.cancel("observer caller cancellation")
        raise GeneratorExit("observer concurrent supervisory exit")


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


class _ThreadBackedBeforeObserverBinding(DeterministicWorkspaceBinding):
    def __init__(self) -> None:
        super().__init__()
        self.observations = 0
        self.started = threading.Event()
        self.release = threading.Event()

    async def observe_revision(self, bound):
        self.observations += 1
        if self.observations == 1:
            self.started.set()
            await asyncio.to_thread(self.release.wait)
        return await super().observe_revision(bound)


class _FailAfterObservationStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        if not self.failed and any(
            event.type == EventType.WORKSPACE_REVISION_OBSERVED
            and event.payload.get("phase") == "after"
            for event in request.events
        ):
            self.failed = True
            raise ConnectionError("workspace receipt append failed")
        return await super().publish_runtime_publication(
            session_id,
            request=request,
            **kwargs,
        )


class _ChildCancelledAfterObservationStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        if not self.failed and any(
            event.type == EventType.WORKSPACE_REVISION_OBSERVED
            and event.payload.get("phase") == "after"
            for event in request.events
        ):
            self.failed = True
            raise asyncio.CancelledError("event-store-owned cancellation")
        return await super().publish_runtime_publication(
            session_id,
            request=request,
            **kwargs,
        )


class _BlockingAfterObservationStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.blocked = False

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        if not self.blocked and any(
            event.type == EventType.WORKSPACE_REVISION_OBSERVED
            and event.payload.get("phase") == "after"
            for event in request.events
        ):
            self.blocked = True
            self.started.set()
            await self.release.wait()
        return await super().publish_runtime_publication(
            session_id,
            request=request,
            **kwargs,
        )


class _CancelledFailedWorkspacePublicationStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.publication_started = asyncio.Event()
        self.release_publication = asyncio.Event()
        self.publication_id: str | None = None
        self.initial_failure = ConnectionError("initial workspace publication failed")
        self.reconciliation_failure = TimeoutError("workspace receipt read failed")

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        if (
            self.publication_id is None
            and request.kind == "workspace-observation"
            and request.intent.get("phase") == "before-evidence"
        ):
            self.publication_id = request.publication_id
            self.publication_started.set()
            await self.release_publication.wait()
            raise self.initial_failure
        return await super().publish_runtime_publication(
            session_id,
            request=request,
            **kwargs,
        )

    async def load_runtime_publication_receipt(self, session_id, publication_id):
        if publication_id == self.publication_id:
            raise self.reconciliation_failure
        return await super().load_runtime_publication_receipt(session_id, publication_id)


_MALFORMED_WORKSPACE_ACK_CANARY = "f" * 64


def _malformed_workspace_publication_result(
    result: RuntimePublicationResult,
    *,
    request: RuntimePublicationRequest,
    wrong_type_fields: bool,
) -> RuntimePublicationResult:
    receipt = result.receipt.model_copy(
        update={
            "publication_id": request.publication_id,
            "kind": request.kind,
            "interaction_id": request.interaction_id,
            "intent": request.intent,
            "request_digest": runtime_publication_request_digest(
                result.session.id,
                request,
            ),
            "publication_digest": _MALFORMED_WORKSPACE_ACK_CANARY,
            "source_status": result.session.status,
            "source_run_epoch": (True if wrong_type_fields else result.receipt.source_run_epoch),
            "appended_event_ids": tuple(event.id for event in request.events),
            "referenced_events": request.referenced_events,
        }
    )
    return result.model_copy(
        update={
            "session": (
                result.session.model_copy(update={"run_epoch": True})
                if wrong_type_fields
                else result.session
            ),
            "receipt": receipt,
            "replayed": "not-a-boolean" if wrong_type_fields else False,
        }
    )


class _CommittedMalformedWorkspaceAcknowledgementStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.corrupted = False
        self.before_evidence_attempts = 0

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        result = await super().publish_runtime_publication(
            session_id,
            request=request,
            **kwargs,
        )
        if request.kind == "workspace-observation" and request.intent.get("phase") == (
            "before-evidence"
        ):
            self.before_evidence_attempts += 1
            if not self.corrupted:
                self.corrupted = True
                return _malformed_workspace_publication_result(
                    result,
                    request=request,
                    wrong_type_fields=True,
                )
        return result


class _UncommittedMalformedWorkspaceAcknowledgementStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.prior_result: RuntimePublicationResult | None = None
        self.before_evidence_attempts = 0

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        if request.kind == "workspace-observation" and request.intent.get("phase") == (
            "before-evidence"
        ):
            self.before_evidence_attempts += 1
            assert self.prior_result is not None
            return _malformed_workspace_publication_result(
                self.prior_result,
                request=request,
                wrong_type_fields=False,
            )
        result = await super().publish_runtime_publication(
            session_id,
            request=request,
            **kwargs,
        )
        self.prior_result = result
        return result


class _UncommittedConflictingWorkspaceAcknowledgementStore(
    _UncommittedMalformedWorkspaceAcknowledgementStore
):
    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        if request.kind == "workspace-observation" and request.intent.get("phase") == (
            "before-evidence"
        ):
            self.before_evidence_attempts += 1
            assert self.prior_result is not None
            result = _malformed_workspace_publication_result(
                self.prior_result,
                request=request,
                wrong_type_fields=False,
            )
            receipt = result.receipt.model_copy(update={"publication_digest": "0" * 64})
            publication_digest = hashlib.sha256(
                canonical_durable_json_bytes(
                    receipt.model_dump(mode="json", exclude={"publication_digest"}),
                    "workspace_observation_publication_receipt",
                )
            ).hexdigest()
            return result.model_copy(
                update={
                    "receipt": receipt.model_copy(update={"publication_digest": publication_digest})
                }
            )
        result = await InMemorySessionStore.publish_runtime_publication(
            self,
            session_id,
            request=request,
            **kwargs,
        )
        self.prior_result = result
        return result


class _TracebackMalformedWorkspaceAcknowledgementStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.returned_session = None

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        del session_id, request, kwargs
        assert self.returned_session is not None
        return RuntimePublicationResult.model_construct(
            session=self.returned_session,
            receipt=_MALFORMED_WORKSPACE_ACK_CANARY,
            replayed=False,
        )


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


class _BlockingInterruptionCleanupStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_started = asyncio.Event()
        self.cleanup_release = asyncio.Event()
        self.cleanup_blocked = False
        self.fail_next_run_fence_release = False

    async def load(self, session_id):
        task = asyncio.current_task()
        if (
            not self.cleanup_blocked
            and task is not None
            and task.get_name() == "cayu-session-interruption-cleanup"
        ):
            self.cleanup_blocked = True
            self.cleanup_started.set()
            await self.cleanup_release.wait()
        else:
            # A restored cancellation must be consumed before even a
            # suspending persistence preflight. The old check-then-own path
            # lost the authoritative failure at this checkpoint.
            await asyncio.sleep(0)
        return await super().load(session_id)

    async def release_run_fence(self, session_id):
        await super().release_run_fence(session_id)
        if self.fail_next_run_fence_release:
            self.fail_next_run_fence_release = False
            raise RuntimeError("interruption run-fence release failed")


class _WorkspaceObservationProcessLoss(BaseException):
    pass


class _WorkspaceObservationRecoveryFactory(EnvironmentFactory):
    def __init__(self, root, *, workspace_id: str) -> None:
        self.root = root
        self.workspace_id = workspace_id
        self.create_calls = 0

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:workspace-observation-recovery-factory",
            behavior_version="1",
            implementation_version="1",
        )

    async def create(
        self,
        request: EnvironmentFactoryRequest,
    ) -> EnvironmentFactoryResult:
        self.create_calls += 1
        self.root.mkdir(parents=True, exist_ok=True)
        return EnvironmentFactoryResult(
            Environment(
                _portable_environment_spec(request.environment_name),
                workspace=LocalWorkspace(self.root, workspace_id=self.workspace_id),
                runner=LocalRunner(self.root),
                binding=DeterministicWorkspaceBinding(),
            )
        )


class _WorkspaceObservationProcessLossStore(InMemorySessionStore):
    def __init__(self, *, phase: str) -> None:
        super().__init__()
        self.phase = phase
        self.failed = False
        self.cancel_workspace_observation_read = False
        self.workspace_observation_read_cancelled = False
        self.cancel_workspace_observation_mutation = False
        self.workspace_observation_mutation_cancelled = False
        self.hide_workspace_delta = False
        self.workspace_delta_event_id = None

    async def load_checkpoint(self, session_id):
        task = asyncio.current_task()
        if (
            self.cancel_workspace_observation_read
            and not self.workspace_observation_read_cancelled
            and task is not None
            and task.get_name() == "cayu-workspace-observation-store-read"
        ):
            self.workspace_observation_read_cancelled = True
            raise asyncio.CancelledError("store-owned workspace observation cancellation")
        return await super().load_checkpoint(session_id)

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        result = await super().publish_runtime_publication(
            session_id,
            request=request,
            **kwargs,
        )
        if (
            request.kind == "workspace-observation"
            and request.intent.get("phase") == "delta-publication"
            and result.receipt.appended_event_ids
        ):
            self.workspace_delta_event_id = result.receipt.appended_event_ids[0]
        if (
            not self.failed
            and request.kind == "workspace-observation"
            and request.intent.get("phase") == self.phase
        ):
            self.failed = True
            raise _WorkspaceObservationProcessLoss(self.phase)
        return result

    async def transform_checkpoint(self, session_id, transform):
        task = asyncio.current_task()
        if (
            self.cancel_workspace_observation_mutation
            and not self.workspace_observation_mutation_cancelled
            and task is not None
            and task.get_name() == "cayu-workspace-observation-store-mutation"
        ):
            self.workspace_observation_mutation_cancelled = True
            raise asyncio.CancelledError("store-owned workspace observation mutation cancellation")
        result = await super().transform_checkpoint(session_id, transform)
        if not self.failed and self.phase == "terminal-stage":
            checkpoint = await self.load_checkpoint(session_id)
            observations = None if checkpoint is None else checkpoint.get("workspace_observations")
            pending_round = None if checkpoint is None else checkpoint.get("pending_tool_round")
            if (
                type(observations) is dict
                and len(observations) == 1
                and next(iter(observations.values())).get("phase") == "before_captured"
                and type(pending_round) is dict
                and pending_round.get("staged_terminals")
            ):
                self.failed = True
                raise _WorkspaceObservationProcessLoss(self.phase)
        return result

    async def query_events(self, query):
        records = await super().query_events(query)
        if self.hide_workspace_delta and query.event_id == self.workspace_delta_event_id:
            return [
                record
                for record in records
                if record.event.type is not EventType.WORKSPACE_MUTATION_RECORDED
            ]
        return records


class _ChildCancelledArtifactStore(LocalArtifactStore):
    async def put_bytes(self, *args, **kwargs):
        del args, kwargs
        raise asyncio.CancelledError("artifact-store-owned cancellation")


class _GeneratorExitArtifactStore(LocalArtifactStore):
    async def put_bytes(self, *args, **kwargs):
        del args, kwargs
        raise GeneratorExit("artifact-store supervisory exit")


class _ConcurrentGeneratorExitArtifactStore(LocalArtifactStore):
    def __init__(self, root) -> None:
        super().__init__(root)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancel_target: asyncio.Task | None = None

    async def put_bytes(self, *args, **kwargs):
        del args, kwargs
        self.started.set()
        await self.release.wait()
        assert self.cancel_target is not None
        self.cancel_target.cancel("artifact caller cancellation")
        raise GeneratorExit("artifact-store concurrent supervisory exit")


class _CommitThenRaiseArtifactStore(LocalArtifactStore):
    async def put_bytes(self, *args, **kwargs):
        await super().put_bytes(*args, **kwargs)
        raise ConnectionError("artifact acknowledgement lost")


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
        self.finished = asyncio.Event()

    async def put_bytes(self, *args, **kwargs):
        self.started.set()
        await self.release.wait()
        result = await super().put_bytes(*args, **kwargs)
        self.finished.set()
        return result


class _ReadTrackingArtifactStore(LocalArtifactStore):
    def __init__(self, root, *, store_id: str) -> None:
        super().__init__(root, store_id=store_id)
        self.reads = 0

    async def read_bytes(self, *args, **kwargs):
        self.reads += 1
        return await super().read_bytes(*args, **kwargs)


class _ConflictingReadArtifactStore(LocalArtifactStore):
    def __init__(self, root, *, store_id: str, metadata_update: dict) -> None:
        super().__init__(root, store_id=store_id)
        self._metadata_update = metadata_update

    async def read_bytes(self, *args, **kwargs):
        result = await super().read_bytes(*args, **kwargs)
        return result.model_copy(
            update={
                "metadata": result.metadata.model_copy(
                    update=self._metadata_update,
                )
            }
        )


class _NoneReadArtifactStore(LocalArtifactStore):
    async def read_bytes(self, *args, **kwargs):
        del args, kwargs
        return None


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


class _BlockingAfterWriteMutationTool(Tool):
    spec = ToolSpec(
        name="blocking_after_write_mutation",
        parallel_safe=False,
        workspace_mutation=True,
    )

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        assert ctx.workspace is not None
        await ctx.workspace.create_bytes("cancelled-write.txt", b"written before cancellation")
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("Cancelled mutation tool unexpectedly resumed.")


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


class _ResolveWorkspaceBranchSecretTool(Tool):
    spec = ToolSpec(
        name="resolve_workspace_branch_secret",
        parallel_safe=False,
        workspace_mutation=True,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        assert ctx.vault is not None
        assert ctx.runner is not None
        resolved = await ctx.vault.resolve(SecretRef(name="workspace_branch"))
        await ctx.runner.exec(
            ExecCommand.process(
                "git",
                "switch",
                "-c",
                resolved.value.get_secret_value(),
            )
        )
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


class _GeneratorExitWorkspace(LocalWorkspace):
    async def create_bytes(self, path: str, content: bytes):
        del path, content
        raise GeneratorExit("workspace mutation supervisory exit")


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


class _MutationQuiescenceTrackingFinalizeBinding(_TrackingFinalizeBinding):
    def __init__(self, runner: _FailingDeferredBackgroundMutationRunner) -> None:
        super().__init__()
        self._runner = runner
        self.observations_while_mutating = 0

    async def observe_revision(self, bound):
        if self._runner.started.is_set() and not self._runner.mutation_finished.is_set():
            self.observations_while_mutating += 1
        return await super().observe_revision(bound)


class _CancellationResistantFinalObserverBinding(_TrackingFinalizeBinding):
    def __init__(self) -> None:
        super().__init__()
        self.observations = 0
        self.final_observation_started = asyncio.Event()
        self.release_final_observation = asyncio.Event()
        self.final_observation_finished = asyncio.Event()

    async def observe_revision(self, bound):
        self.observations += 1
        if self.observations != 3:
            return await super().observe_revision(bound)
        self.final_observation_started.set()
        try:
            await self.release_final_observation.wait()
        except asyncio.CancelledError:
            # Model a custom observer whose underlying operation cannot be
            # stopped by task cancellation after dispatch.
            await self.release_final_observation.wait()
        observation = await super().observe_revision(bound)
        self.final_observation_finished.set()
        return observation


class _CallerCancellingFinalObserverBinding(_TrackingFinalizeBinding):
    def __init__(self, cancellation_requests: int) -> None:
        super().__init__()
        self.cancellation_requests = cancellation_requests
        self.observations = 0
        self.cancel_target: asyncio.Task | None = None

    async def observe_revision(self, bound):
        self.observations += 1
        if self.observations == 3:
            assert self.cancel_target is not None
            for request in range(self.cancellation_requests):
                self.cancel_target.cancel(f"final observer cancellation {request + 1}")
        return await super().observe_revision(bound)


class _CancellationResistantAfterObserverBinding(_TrackingFinalizeBinding):
    def __init__(self) -> None:
        super().__init__()
        self.observations = 0
        self.after_observation_started = asyncio.Event()
        self.release_after_observation = asyncio.Event()
        self.after_observation_finished = asyncio.Event()

    async def observe_revision(self, bound):
        self.observations += 1
        if self.observations != 2:
            return await super().observe_revision(bound)
        self.after_observation_started.set()
        try:
            await self.release_after_observation.wait()
        except asyncio.CancelledError:
            # Model an observer whose workspace access cannot be stopped once
            # it has crossed the custom binding boundary.
            await self.release_after_observation.wait()
        observation = await super().observe_revision(bound)
        self.after_observation_finished.set()
        return observation


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


def test_workspace_mutation_without_binding_records_unsupported_evidence(tmp_path) -> None:
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            _SingleToolProvider(
                tool_name="noop_workspace_mutation",
                arguments={},
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="unconfigured-binding-workspace"),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_NoopWorkspaceMutationTool()],
        )

        public_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session-unconfigured-workspace-binding",
                messages=[Message.text("user", "observe")],
            ),
        )
        durable = await store.query_events(
            EventQuery(session_id="session-unconfigured-workspace-binding")
        )
        return public_events, [record.event for record in durable]

    public_events, durable_events = asyncio.run(run())

    observations = [
        event for event in durable_events if event.type is EventType.WORKSPACE_REVISION_OBSERVED
    ]
    assert len(observations) == 2
    assert {event.payload["phase"] for event in observations} == {"before", "after"}
    assert {event.payload["status"] for event in observations} == {"unsupported"}
    assert {event.payload["observer"] for event in observations} == {"UnconfiguredWorkspaceBinding"}

    receipt = next(
        event for event in durable_events if event.type is EventType.WORKSPACE_MUTATION_RECORDED
    )
    assert receipt.payload["status"] == "unsupported"
    assert receipt.payload["observer"] == "UnconfiguredWorkspaceBinding"

    finalization = next(
        event for event in durable_events if event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
    )
    assert finalization.payload["status"] == "incomplete"
    assert finalization.payload["detail_code"] == "workspace_revision_evidence_incomplete"
    assert not any(
        event.payload.get("workspace_mutation_capture_detail_code") == "receipt_publication_failed"
        for event in public_events
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
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            # Exact built-in observer identities carry structural runtime
            # provenance; a custom binding with the same name remains subject
            # to configured-authority secret admission below.
            secret_redactor=SecretRedactor(["before", "after", "changed", "GitRepositoryBinding"]),
        )
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
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
    assert {event.payload["observer"] for event in receipt_events} == {"GitRepositoryBinding"}
    assert receipt.payload["status"] == "changed"
    assert receipt.payload["paths"] == [
        {"path": "shell.txt", "change": "added", "renamed_from": None}
    ]
    finalization = next(
        event
        for event in durable_events
        if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
    )
    final_revision = dict(finalization.payload["final_revision"])
    assert final_revision.pop("observer") != "GitRepositoryBinding"
    assert "GitRepositoryBinding" not in finalization.model_dump_json()
    assert final_revision == {
        "workspace_id": "git-workspace",
        "status": "supported",
        "revision": after.payload["revision"],
        "head_revision": after.payload["head_revision"],
        "branch": after.payload["branch"],
        "path_scope": "complete",
        "total_paths": after.payload["total_paths"],
        "detail_code": None,
    }
    assert "tool_call_id" not in finalization.payload["final_revision"]
    assert "window_id" not in finalization.payload["final_revision"]
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in public_events), [
        str(event.type) for event in public_events[-20:]
    ]
    observation_terminal = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_OBSERVATION_FINALIZED
    )
    assert observation_terminal.payload["status"] == "complete"
    assert observation_terminal.payload["detail_code"] is None
    assert observation_terminal.payload["observer"] == "GitRepositoryBinding"
    public_observation_terminal = next(
        event for event in public_events if event.type == EventType.WORKSPACE_OBSERVATION_FINALIZED
    )
    assert public_observation_terminal.payload["status"] == "complete"
    assert public_observation_terminal.payload["detail_code"] is None
    assert public_observation_terminal.payload["observer"] != "GitRepositoryBinding"
    assert "GitRepositoryBinding" not in public_observation_terminal.model_dump_json()
    assert "created" not in before.model_dump_json()
    assert "created" not in after.model_dump_json()
    assert "created" not in receipt.model_dump_json()


@pytest.mark.parametrize(
    "identity_source",
    ["workspace", "artifact_store", "observer", "observer_builtin_name_spoof"],
)
def test_configured_observation_identity_is_admitted_before_durable_intent(
    tmp_path,
    caplog,
    capsys,
    identity_source,
) -> None:
    secret_identity = (
        "GitRepositoryBinding"
        if identity_source == "observer_builtin_name_spoof"
        else f"PRIVATE_CONFIGURED_{identity_source.upper()}_ID_CANARY"
    )

    async def run():
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            secret_redactor=SecretRedactor([secret_identity]),
        )
        app.register_provider(_ScriptedProvider(), default=True)
        binding_type = type(secret_identity, (DeterministicWorkspaceBinding,), {})
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(
                    tmp_path,
                    workspace_id=(
                        secret_identity if identity_source == "workspace" else "workspace"
                    ),
                ),
                runner=LocalRunner(tmp_path),
                binding=(
                    binding_type()
                    if identity_source in {"observer", "observer_builtin_name_spoof"}
                    else DeterministicWorkspaceBinding()
                ),
                artifact_store=(
                    LocalArtifactStore(tmp_path / "artifacts", store_id=secret_identity)
                    if identity_source == "artifact_store"
                    else None
                ),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        with warnings.catch_warnings(record=True) as captured_warnings:
            public = await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-configured-workspace-secret",
                    messages=[Message.text("user", "create a file")],
                ),
            )
        durable = await store.query_events(
            EventQuery(session_id="session-configured-workspace-secret")
        )
        checkpoint = await store.load_checkpoint("session-configured-workspace-secret")
        transcript = await store.load_transcript("session-configured-workspace-secret")
        return (
            public,
            [record.event for record in durable],
            checkpoint,
            transcript,
            captured_warnings,
        )

    public_events, durable_events, checkpoint, transcript, captured_warnings = asyncio.run(run())
    captured = capsys.readouterr()
    assert not workspace_observations_from_checkpoint(checkpoint)
    assert not any(
        event.type
        in {
            EventType.WORKSPACE_REVISION_OBSERVED,
            EventType.WORKSPACE_MUTATION_RECORDED,
        }
        for event in durable_events
    )
    assert any(event.type is EventType.SESSION_FAILED for event in public_events)
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
    assert secret_identity not in combined


def test_dynamic_observation_identity_is_opaque_before_tool_secret_resolution(
    tmp_path,
    caplog,
    capsys,
) -> None:
    secret_identity = "PRIVATE_DYNAMIC_WORKSPACE_ID_CANARY"

    class AbortAfterIntentBinding(NativeBinding):
        async def observe_revision(self, bound):
            del bound
            raise SystemExit(23)

    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            _SingleToolProvider(tool_name="noop_workspace_mutation", arguments={}),
            default=True,
        )
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id=secret_identity),
                binding=AbortAfterIntentBinding(),
                vault=StaticVault({"workspace_identity": secret_identity}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_NoopWorkspaceMutationTool()],
        )
        with (
            warnings.catch_warnings(record=True) as captured_warnings,
            pytest.raises(SystemExit) as raised,
        ):
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-dynamic-identity-intent",
                    messages=[Message.text("user", "observe")],
                ),
            )
        checkpoint = await store.load_checkpoint("session-dynamic-identity-intent")
        durable_before_recovery = await store.query_events(
            EventQuery(session_id="session-dynamic-identity-intent")
        )
        await store.release_run_fence("session-dynamic-identity-intent")
        await store.update_status(
            "session-dynamic-identity-intent",
            SessionStatus.INTERRUPTED,
        )
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_provider = _SingleToolProvider(
            tool_name="noop_workspace_mutation",
            arguments={},
        )
        recovery_app.register_provider(recovery_provider, default=True)
        recovery_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id=secret_identity),
                binding=AbortAfterIntentBinding(),
                vault=StaticVault({"workspace_identity": secret_identity}),
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_NoopWorkspaceMutationTool()],
        )
        with pytest.raises(
            RuntimeError,
            match="cannot safely continue with opaque provider state",
        ):
            await recovery_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="session-dynamic-identity-intent")
            )
        durable_after_recovery = await store.query_events(
            EventQuery(session_id="session-dynamic-identity-intent")
        )
        return (
            raised.value,
            checkpoint,
            durable_before_recovery,
            recovery_provider.requests,
            durable_after_recovery,
            captured_warnings,
        )

    (
        error,
        checkpoint,
        durable_before_recovery,
        recovery_requests,
        durable_after_recovery,
        captured_warnings,
    ) = asyncio.run(run())
    captured = capsys.readouterr()
    assert error.code == 23
    observations = workspace_observations_from_checkpoint(checkpoint)
    assert len(observations) == 1
    (lifecycle,) = observations.values()
    assert lifecycle.workspace_id != secret_identity
    assert lifecycle.workspace_id.startswith("cayu_authority_")
    assert recovery_requests == 0
    assert any(
        record.event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
        for record in durable_after_recovery
    )
    combined = repr(
        (
            checkpoint,
            [record.event.model_dump(mode="json") for record in durable_before_recovery],
            [record.event.model_dump(mode="json") for record in durable_after_recovery],
            captured_warnings,
            caplog.records,
            captured.out,
            captured.err,
        )
    )
    assert secret_identity not in combined


def test_cancellation_resistant_final_observer_fences_finalization_and_reuse(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        environment_lifecycle_module,
        "_FINAL_WORKSPACE_OBSERVATION_TIMEOUT_SECONDS",
        0.01,
    )

    async def run():
        binding = _CancellationResistantFinalObserverBinding()
        provider = _SingleToolProvider(
            tool_name="noop_workspace_mutation",
            arguments={},
        )
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="final-observer-workspace"),
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_NoopWorkspaceMutationTool()],
        )

        first = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session-final-observer-first",
                messages=[Message.text("user", "observe")],
            ),
        )
        assert binding.final_observation_started.is_set()
        assert binding.final_observation_finished.is_set() is False
        assert binding.finalize_calls == 0
        assert any(event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED for event in first)

        contender = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-final-observer-second",
                    messages=[Message.text("user", "reuse")],
                ),
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert contender.done() is False
        assert provider.requests == 2
        assert binding.finalize_calls == 0
        assert binding.abandon_calls == 0

        binding.release_final_observation.set()
        second = await contender
        return first, second, binding, provider

    first, second, binding, provider = asyncio.run(run())

    assert binding.final_observation_finished.is_set()
    assert provider.requests == 3
    assert any(event.type is EventType.SESSION_COMPLETED for event in first)
    assert any(event.type is EventType.SESSION_COMPLETED for event in second)
    assert binding.abandon_calls >= 1


@pytest.mark.parametrize("cancellation_requests", [1, 2])
def test_final_workspace_observer_restores_caller_cancellation_requests(
    tmp_path,
    cancellation_requests: int,
) -> None:
    async def run() -> tuple[asyncio.Task[list[Event]], _CallerCancellingFinalObserverBinding]:
        binding = _CallerCancellingFinalObserverBinding(cancellation_requests)
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(
            _SingleToolProvider(
                tool_name="noop_workspace_mutation",
                arguments={},
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="final-cancel-workspace"),
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_NoopWorkspaceMutationTool()],
        )
        consumer = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=f"session-final-observer-cancel-{cancellation_requests}",
                    messages=[Message.text("user", "observe")],
                ),
            )
        )
        binding.cancel_target = consumer
        with pytest.raises(asyncio.CancelledError):
            await consumer
        return consumer, binding

    consumer, binding = asyncio.run(run())

    assert consumer.cancelling() == cancellation_requests
    assert consumer.cancelled() is True
    assert binding.observations == 3
    assert binding.finalize_calls == 0


def test_workspace_cancellation_authority_survives_billing_detachment() -> None:
    canary = "workspace-billing-cancellation-canary"
    original = BaseExceptionGroup(
        f"billing group {canary}",
        [
            _BillingIdentityResolutionCancelled(f"billing cancellation {canary}"),
            GeneratorExit(f"observer exit {canary}"),
        ],
    )
    retain_workspace_observation_pending_cancellation_requests(original, 2)

    detached = detach_billing_identity_cancellation_group(original)

    assert detached is not None
    assert workspace_observation_pending_cancellation_requests(detached) == 2
    assert canary not in repr(
        [(type(error).__name__, error.args) for error in iter_exception_tree(detached)]
    )


def test_workspace_cancellation_authority_survives_environment_detachment() -> None:
    canary = "workspace-environment-cancellation-canary"

    async def run() -> BaseExceptionGroup:
        original = BaseExceptionGroup(
            f"environment group {canary}",
            [
                asyncio.CancelledError(f"environment cancellation {canary}"),
                GeneratorExit(f"environment exit {canary}"),
            ],
        )
        retain_workspace_observation_pending_cancellation_requests(original, 2)

        async def fail() -> None:
            raise original

        with pytest.raises(BaseExceptionGroup) as exc_info:
            await await_environment_operation(
                fail,
                operation_name="Workspace observation environment operation",
                redactor=SecretRedactor([canary]),
            )
        return exc_info.value

    detached = asyncio.run(run())

    assert workspace_observation_pending_cancellation_requests(detached) == 2
    assert canary not in repr(
        [(type(error).__name__, error.args) for error in iter_exception_tree(detached)]
    )


def test_delegated_stream_close_adds_late_cancellation_to_retained_requests() -> None:
    async def run() -> tuple[asyncio.Task[None], BaseExceptionGroup]:
        close_started = asyncio.Event()
        close_release = asyncio.Event()

        async def stream() -> AsyncIterator[Event]:
            try:
                yield Event(
                    type=EventType.SESSION_STARTED,
                    session_id="session-delegated-close-cancellation",
                )
                await asyncio.Future()
            finally:
                close_started.set()
                await close_release.wait()

        authoritative_failure = BaseExceptionGroup(
            "Workspace observation control.",
            [
                asyncio.CancelledError("original workspace cancellation"),
                GeneratorExit("original workspace process control"),
            ],
        )
        retain_workspace_observation_pending_cancellation_requests(
            authoritative_failure,
            2,
        )

        async def delegated_stream() -> AsyncIterator[Event]:
            async with _close_delegated_event_stream(stream()) as owned_stream:
                await owned_stream.__anext__()
                raise authoritative_failure
                yield  # pragma: no cover - establishes the async-generator shape

        async def consume() -> None:
            async with _close_delegated_event_stream(delegated_stream()) as owned_stream:
                await owned_stream.__anext__()

        consumer = asyncio.create_task(consume())
        await close_started.wait()
        consumer.cancel("late delegated cleanup cancellation")
        await asyncio.sleep(0)
        assert consumer.done() is False
        close_release.set()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await consumer
        return consumer, exc_info.value

    consumer, failure = asyncio.run(run())

    assert consumer.cancelling() == 3
    assert consumer.cancelled() is False
    cancellations = [
        error for error in iter_exception_tree(failure) if isinstance(error, asyncio.CancelledError)
    ]
    assert len(cancellations) == 2
    assert (
        sum(
            cancellation.args == ("late delegated cleanup cancellation",)
            for cancellation in cancellations
        )
        == 1
    )
    assert workspace_observation_pending_cancellation_requests(failure) == 3


def test_delegated_stream_close_preserves_late_cancellation_for_untagged_group() -> None:
    async def run() -> tuple[asyncio.Task[None], BaseExceptionGroup]:
        close_started = asyncio.Event()
        close_release = asyncio.Event()

        async def stream() -> AsyncIterator[Event]:
            try:
                yield Event(
                    type=EventType.SESSION_STARTED,
                    session_id="session-delegated-untagged-cancellation",
                )
                await asyncio.Future()
            finally:
                close_started.set()
                await close_release.wait()

        authoritative_failure = BaseExceptionGroup(
            "Untagged delegated stream control.",
            [
                asyncio.CancelledError("older untagged cancellation"),
                GeneratorExit("older untagged supervisory exit"),
            ],
        )

        async def delegated_stream() -> AsyncIterator[Event]:
            async with _close_delegated_event_stream(stream()) as owned_stream:
                await owned_stream.__anext__()
                raise authoritative_failure
                yield  # pragma: no cover - establishes the async-generator shape

        async def consume() -> None:
            async with _close_delegated_event_stream(delegated_stream()) as owned_stream:
                await owned_stream.__anext__()

        consumer = asyncio.create_task(consume())
        await close_started.wait()
        consumer.cancel("late untagged cleanup cancellation")
        await asyncio.sleep(0)
        assert consumer.done() is False
        close_release.set()
        with pytest.raises(BaseExceptionGroup) as raised:
            await consumer
        return consumer, raised.value

    consumer, failure = asyncio.run(run())

    cancellations = [
        candidate
        for candidate in iter_exception_tree(failure)
        if isinstance(candidate, asyncio.CancelledError)
    ]
    assert (
        sum(candidate.args == ("older untagged cancellation",) for candidate in cancellations) == 1
    )
    assert (
        sum(
            candidate.args == ("late untagged cleanup cancellation",) for candidate in cancellations
        )
        == 1
    )
    assert (
        sum(
            type(candidate) is GeneratorExit
            and candidate.args == ("older untagged supervisory exit",)
            for candidate in iter_exception_tree(failure)
        )
        == 1
    )
    assert workspace_observation_pending_cancellation_requests(failure) == 1
    assert consumer.cancelling() == 1
    assert consumer.cancelled() is False


def test_delegated_stream_close_preserves_pending_entry_cancellation_for_untagged_group() -> None:
    async def run() -> tuple[asyncio.Task[None], BaseExceptionGroup]:
        authoritative_failure = BaseExceptionGroup(
            "Untagged historical delegated stream control.",
            [
                asyncio.CancelledError("older untagged cancellation"),
                GeneratorExit("older untagged supervisory exit"),
            ],
        )

        async def stream() -> AsyncIterator[Event]:
            yield Event(
                type=EventType.SESSION_STARTED,
                session_id="session-delegated-entry-cancellation",
            )
            await asyncio.Future()

        async def consume() -> None:
            async with _close_delegated_event_stream(stream()) as owned_stream:
                await owned_stream.__anext__()
                current_task = asyncio.current_task()
                assert current_task is not None
                current_task.cancel("pending cleanup-entry cancellation")
                raise authoritative_failure

        consumer = asyncio.create_task(consume())
        with pytest.raises(BaseExceptionGroup) as raised:
            await consumer
        return consumer, raised.value

    consumer, failure = asyncio.run(run())

    failures = list(iter_exception_tree(failure))
    assert (
        sum(
            isinstance(candidate, asyncio.CancelledError)
            and candidate.args == ("older untagged cancellation",)
            for candidate in failures
        )
        == 1
    )
    assert (
        sum(
            isinstance(candidate, asyncio.CancelledError)
            and candidate.args == ("pending cleanup-entry cancellation",)
            for candidate in failures
        )
        == 1
    )
    assert (
        sum(
            type(candidate) is GeneratorExit
            and candidate.args == ("older untagged supervisory exit",)
            for candidate in failures
        )
        == 1
    )
    assert workspace_observation_pending_cancellation_requests(failure) == 1
    assert consumer.cancelling() == 1
    assert consumer.cancelled() is False


def test_delegated_stream_close_counts_checkpoint_cancellation_once() -> None:
    async def run() -> tuple[asyncio.Task[None], BaseExceptionGroup]:
        authoritative_failure = BaseExceptionGroup(
            "Delegated stream process control.",
            [GeneratorExit("delegated checkpoint supervisory exit")],
        )

        async def stream() -> AsyncIterator[Event]:
            yield Event(
                type=EventType.SESSION_STARTED,
                session_id="session-delegated-checkpoint-cancellation",
            )
            await asyncio.Future()

        async def delegated_stream() -> AsyncIterator[Event]:
            async with _close_delegated_event_stream(stream()) as owned_stream:
                await owned_stream.__anext__()
                current_task = asyncio.current_task()
                assert current_task is not None
                asyncio.get_running_loop().call_soon(
                    current_task.cancel,
                    "checkpoint cleanup cancellation",
                )
                raise authoritative_failure
                yield  # pragma: no cover - establishes the async-generator shape

        async def consume() -> None:
            async with _close_delegated_event_stream(delegated_stream()) as owned_stream:
                await owned_stream.__anext__()

        consumer = asyncio.create_task(consume())
        with pytest.raises(BaseExceptionGroup) as raised:
            await consumer
        return consumer, raised.value

    consumer, failure = asyncio.run(run())

    cancellations = [
        candidate
        for candidate in iter_exception_tree(failure)
        if isinstance(candidate, asyncio.CancelledError)
    ]
    assert len(cancellations) == 1
    assert cancellations[0].args == ("checkpoint cleanup cancellation",)
    assert workspace_observation_pending_cancellation_requests(failure) == 1
    assert consumer.cancelling() == 1
    assert consumer.cancelled() is False


def test_delegated_stream_close_distinguishes_restored_and_late_cancellation() -> None:
    async def run() -> tuple[asyncio.Task[None], BaseExceptionGroup]:
        close_started = asyncio.Event()
        close_release = asyncio.Event()
        authoritative_failure = BaseExceptionGroup(
            "Delegated stream control with restored cancellation.",
            [
                asyncio.CancelledError("authoritative workspace cancellation"),
                GeneratorExit("authoritative workspace supervisory exit"),
            ],
        )
        retain_workspace_observation_pending_cancellation_requests(
            authoritative_failure,
            1,
        )

        async def stream() -> AsyncIterator[Event]:
            try:
                yield Event(
                    type=EventType.SESSION_STARTED,
                    session_id="session-delegated-restored-and-late-cancellation",
                )
                await asyncio.Future()
            finally:
                close_started.set()
                await close_release.wait()

        async def consume() -> None:
            async with _close_delegated_event_stream(stream()) as owned_stream:
                await owned_stream.__anext__()
                current_task = asyncio.current_task()
                assert current_task is not None
                current_task.cancel("restored workspace cancellation")
                raise authoritative_failure

        consumer = asyncio.create_task(consume())
        await close_started.wait()
        consumer.cancel("late delegated cleanup cancellation")
        await asyncio.sleep(0)
        assert consumer.done() is False
        close_release.set()
        with pytest.raises(BaseExceptionGroup) as raised:
            await consumer
        return consumer, raised.value

    consumer, failure = asyncio.run(run())

    cancellations = [
        candidate
        for candidate in iter_exception_tree(failure)
        if isinstance(candidate, asyncio.CancelledError)
    ]
    assert (
        sum(
            candidate.args == ("authoritative workspace cancellation",)
            for candidate in cancellations
        )
        == 1
    )
    assert (
        sum(candidate.args == ("restored workspace cancellation",) for candidate in cancellations)
        == 0
    )
    assert (
        sum(
            candidate.args == ("late delegated cleanup cancellation",)
            for candidate in cancellations
        )
        == 1
    )
    assert workspace_observation_pending_cancellation_requests(failure) == 2
    assert consumer.cancelling() == 2
    assert consumer.cancelled() is False


def test_delegated_stream_close_aggregates_process_control_only_failure_with_cancellation() -> None:
    async def run() -> tuple[asyncio.Task[None], BaseExceptionGroup]:
        close_started = asyncio.Event()
        close_release = asyncio.Event()

        async def stream() -> AsyncIterator[Event]:
            try:
                yield Event(
                    type=EventType.SESSION_STARTED,
                    session_id="session-delegated-process-control-cancellation",
                )
                await asyncio.Future()
            finally:
                close_started.set()
                await close_release.wait()

        authoritative_failure = BaseExceptionGroup(
            "Delegated stream process control.",
            [
                GeneratorExit("delegated supervisory exit"),
                RuntimeError("delegated ordinary failure"),
            ],
        )

        async def delegated_stream() -> AsyncIterator[Event]:
            async with _close_delegated_event_stream(stream()) as owned_stream:
                await owned_stream.__anext__()
                raise authoritative_failure
                yield  # pragma: no cover - establishes the async-generator shape

        async def consume() -> None:
            async with _close_delegated_event_stream(delegated_stream()) as owned_stream:
                await owned_stream.__anext__()

        consumer = asyncio.create_task(consume())
        await close_started.wait()
        consumer.cancel("concurrent delegated cancellation")
        await asyncio.sleep(0)
        assert consumer.done() is False
        close_release.set()
        with pytest.raises(BaseExceptionGroup) as raised:
            await consumer
        return consumer, raised.value

    consumer, failure = asyncio.run(run())

    failures = list(iter_exception_tree(failure))
    assert (
        sum(
            type(candidate) is GeneratorExit and candidate.args == ("delegated supervisory exit",)
            for candidate in failures
        )
        == 1
    )
    assert (
        sum(
            isinstance(candidate, asyncio.CancelledError)
            and candidate.args == ("concurrent delegated cancellation",)
            for candidate in failures
        )
        == 1
    )
    assert workspace_observation_pending_cancellation_requests(failure) == 1
    assert consumer.cancelling() == 1
    assert consumer.cancelled() is False


def test_delegated_stream_close_aggregates_child_process_control_with_cancellation() -> None:
    async def run() -> tuple[asyncio.Task[None], BaseExceptionGroup]:
        close_started = asyncio.Event()
        close_release = asyncio.Event()

        async def stream() -> AsyncIterator[Event]:
            try:
                yield Event(
                    type=EventType.SESSION_STARTED,
                    session_id="session-delegated-child-process-control",
                )
                await asyncio.Future()
            finally:
                close_started.set()
                await close_release.wait()
                raise SystemExit(17)

        async def consume() -> None:
            async with _close_delegated_event_stream(stream()) as owned_stream:
                await owned_stream.__anext__()

        consumer = asyncio.create_task(consume())
        await close_started.wait()
        consumer.cancel("concurrent child-cleanup cancellation")
        await asyncio.sleep(0)
        assert consumer.done() is False
        close_release.set()
        with pytest.raises(BaseExceptionGroup) as raised:
            await consumer
        return consumer, raised.value

    consumer, failure = asyncio.run(run())

    failures = list(iter_exception_tree(failure))
    assert (
        sum(type(candidate) is SystemExit and candidate.args == (17,) for candidate in failures)
        == 1
    )
    assert (
        sum(
            isinstance(candidate, asyncio.CancelledError)
            and candidate.args == ("concurrent child-cleanup cancellation",)
            for candidate in failures
        )
        == 1
    )
    assert workspace_observation_pending_cancellation_requests(failure) == 1
    assert consumer.cancelling() == 1
    assert consumer.cancelled() is False


def test_delegated_stream_child_cancellation_is_an_operational_cleanup_failure() -> None:
    async def run() -> tuple[asyncio.Task[None], RuntimeError]:
        async def stream() -> AsyncIterator[Event]:
            try:
                yield Event(
                    type=EventType.SESSION_STARTED,
                    session_id="session-delegated-child-cancellation",
                )
                await asyncio.Future()
            finally:
                raise asyncio.CancelledError("child stream cleanup cancellation")

        async def consume() -> None:
            async with _close_delegated_event_stream(stream()) as owned_stream:
                await owned_stream.__anext__()

        consumer = asyncio.create_task(consume())
        with pytest.raises(RuntimeError) as raised:
            await consumer
        return consumer, raised.value

    consumer, failure = asyncio.run(run())

    assert failure.args == (
        "Delegated runtime stream cleanup was cancelled without caller cancellation.",
    )
    child_cancellation = exception_cause(failure)
    assert isinstance(child_cancellation, asyncio.CancelledError)
    assert child_cancellation.args == ("child stream cleanup cancellation",)
    assert consumer.cancelling() == 0
    assert consumer.cancelled() is False


def test_thread_backed_before_observer_timeout_fences_tool_dispatch_and_reuse(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        tool_round_executor_module,
        "_WORKSPACE_OBSERVATION_TIMEOUT_SECONDS",
        0.01,
    )

    async def run():
        binding = _ThreadBackedBeforeObserverBinding()
        provider = _SingleToolProvider(
            tool_name="noop_workspace_mutation",
            arguments={},
        )
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="thread-observer-workspace"),
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_NoopWorkspaceMutationTool()],
        )

        first = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session-thread-observer-first",
                messages=[Message.text("user", "observe")],
            ),
        )
        assert binding.started.is_set()
        assert binding.release.is_set() is False
        assert any(event.type is EventType.SESSION_FAILED for event in first)

        contender = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-thread-observer-second",
                    messages=[Message.text("user", "reuse")],
                ),
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert contender.done() is False
        assert provider.requests == 1

        binding.release.set()
        second = await contender
        return second, binding, provider

    second, binding, provider = asyncio.run(run())

    assert binding.observations >= 2
    assert provider.requests == 2
    assert any(event.type is EventType.SESSION_COMPLETED for event in second)


def test_cancellation_resistant_after_observer_fences_finalization_and_reuse(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        tool_round_executor_module,
        "_WORKSPACE_OBSERVATION_TIMEOUT_SECONDS",
        0.01,
    )

    async def run():
        binding = _CancellationResistantAfterObserverBinding()
        provider = _SingleToolProvider(
            tool_name="noop_workspace_mutation",
            arguments={},
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="after-observer-workspace"),
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_NoopWorkspaceMutationTool()],
        )

        first = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-after-observer-first",
                    messages=[Message.text("user", "observe")],
                ),
            )
        )
        await binding.after_observation_started.wait()
        for _ in range(100):
            durable = await store.query_events(
                EventQuery(session_id="session-after-observer-first")
            )
            if any(record.event.type is EventType.SESSION_COMPLETED for record in durable):
                break
            await asyncio.sleep(0.001)
        else:
            raise AssertionError("First run did not publish its terminal event.")

        assert first.done() is False
        assert binding.after_observation_finished.is_set() is False
        assert binding.finalize_calls == 0
        assert not any(
            record.event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
            for record in durable
        )

        contender = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-after-observer-second",
                    messages=[Message.text("user", "reuse")],
                ),
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert contender.done() is False
        assert provider.requests == 2
        assert binding.finalize_calls == 0

        binding.release_after_observation.set()
        first_events, second_events = await asyncio.gather(first, contender)
        first_durable = await store.query_events(
            EventQuery(session_id="session-after-observer-first")
        )
        return first_events, second_events, first_durable, binding, provider

    first, second, durable, binding, provider = asyncio.run(run())

    assert binding.after_observation_finished.is_set()
    assert provider.requests == 3
    assert any(event.type is EventType.SESSION_COMPLETED for event in first)
    assert any(event.type is EventType.SESSION_COMPLETED for event in second)
    observations = [
        record.event
        for record in durable
        if record.event.type is EventType.WORKSPACE_REVISION_OBSERVED
    ]
    assert [event.payload["status"] for event in observations] == ["supported", "failed"]
    assert observations[-1].payload["detail_code"] == "revision_observer_timeout"
    assert binding.finalize_calls == 1


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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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


def test_workspace_mutation_generator_exit_propagates_without_false_terminal_evidence(
    tmp_path,
) -> None:
    async def run():
        provider = _SingleToolProvider(
            tool_name="blocking_workspace_mutation",
            arguments={},
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=_GeneratorExitWorkspace(
                    tmp_path,
                    workspace_id="generator-exit-mutation-workspace",
                ),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_BlockingWorkspaceMutationTool()],
        )

        with pytest.raises(GeneratorExit, match="workspace mutation supervisory exit"):
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-workspace-mutation-generator-exit",
                    messages=[Message.text("user", "mutate")],
                ),
            )
        durable = await store.query_events(
            EventQuery(session_id="session-workspace-mutation-generator-exit")
        )
        return [record.event for record in durable], provider

    durable_events, provider = asyncio.run(run())

    assert provider.requests == 1
    assert not any(
        event.type
        in {
            EventType.WORKSPACE_MUTATION_RECORDED,
            EventType.WORKSPACE_OBSERVATION_FINALIZED,
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
            EventType.SESSION_FAILED,
        }
        for event in durable_events
    )


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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
        binding = _MutationQuiescenceTrackingFinalizeBinding(runner)
        following = _FollowingTool()
        provider = _AwaitedRunnerAndFollowingProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
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
        assert binding.observations_while_mutating == 0
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
    assert not any("final_revision" in record.event.payload for record in first_durable)
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
                    _portable_environment_spec(request.environment_name),
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
            _portable_environment_spec("dynamic"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
            _portable_environment_spec("local"),
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


def test_terminal_workspace_revision_quarantines_dynamic_secret_scope(
    tmp_path,
    caplog,
    capsys,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is required")
    secret_branch = "PRIVATE_WORKSPACE_BRANCH_CANARY"

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

    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    provider = _SingleToolProvider(
        tool_name="resolve_workspace_branch_secret",
        arguments={},
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            _portable_environment_spec("local"),
            workspace=LocalWorkspace(tmp_path, workspace_id="git-secret-workspace"),
            runner=LocalRunner(tmp_path),
            binding=GitRepositoryBinding(
                repo_url="https://example.invalid/repository.git",
                fetch=False,
                require_clean=False,
                verify_remote_url=False,
            ),
            vault=StaticVault({"workspace_branch": secret_branch}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[_ResolveWorkspaceBranchSecretTool()],
    )

    async def run():
        with warnings.catch_warnings(record=True) as captured_warnings:
            public = await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-dynamic-final-revision-secret",
                    messages=[Message.text("user", "resolve")],
                ),
            )
        durable = await store.query_events(
            EventQuery(session_id="session-dynamic-final-revision-secret")
        )
        return public, durable, captured_warnings

    public_events, durable, captured_warnings = asyncio.run(run())
    captured = capsys.readouterr()
    finalization = next(
        record.event
        for record in durable
        if record.event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
    )
    assert finalization.payload["final_snapshot"] is None
    projected_authority = _project_workspace_observation_authority(
        session_id="session-dynamic-final-revision-secret",
        configured_workspace_id="git-secret-workspace",
        configured_observer="GitRepositoryBinding",
        configured_artifact_store_id=None,
        observer_is_runtime_owned=True,
        secret_resolution_scope="dynamic",
        redactor=SecretRedactor(),
        public_authority_alias_codec=store.public_authority_alias_codec,
    )
    assert finalization.payload["final_revision"] == {
        "workspace_id": projected_authority.workspace_id,
        "observer": "GitRepositoryBinding",
        "status": "truncated",
        "revision": None,
        "head_revision": None,
        "branch": None,
        "path_scope": "complete",
        "total_paths": 1,
        "detail_code": "final_revision_secret_scope_unavailable",
    }
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
    assert secret_branch not in combined
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
            _portable_environment_spec("local"),
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
            _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
        artifacts = LocalArtifactStore(artifact_root, store_id="artifact-store")
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            # Decimal digits collide with runtime-generated event, interaction,
            # and commitment identities. Their structural provenance must keep
            # those identities admissible without trusting configured input.
            secret_redactor=SecretRedactor(["workspace-secret-value", *"0123456789"]),
        )
        app.register_provider(_BulkProvider(), default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
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
        finalized = next(
            event for event in events if event.type == EventType.WORKSPACE_OBSERVATION_FINALIZED
        )
        after_artifact = await artifacts.read_bytes(after.payload["manifest_artifact_id"])
        receipt_artifact = await artifacts.read_bytes(receipt.payload["manifest_artifact_id"])
        return after, receipt, finalized, after_artifact, receipt_artifact

    after, receipt, finalized, after_artifact, receipt_artifact = asyncio.run(run())

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

    assert (
        finalized.payload["tool_outcome_event_digest"]
        == receipt.payload["tool_outcome_event_digest"]
    )
    assert finalized.payload["mutation_event_digest"] == workspace_observation_event_digest(receipt)
    assert (
        finalized.payload["revision_after_artifact_sha256"]
        == after.payload["manifest_artifact_sha256"]
    )
    assert (
        finalized.payload["revision_delta_artifact_sha256"]
        == receipt.payload["manifest_artifact_sha256"]
    )


@pytest.mark.parametrize(
    ("artifact_store_type", "expected_status", "expected_detail_code"),
    [
        (
            _ChildCancelledArtifactStore,
            "truncated",
            "manifest_artifact_write_unsettled",
        ),
        (
            _CommitThenRaiseArtifactStore,
            "truncated",
            "manifest_artifact_write_unsettled",
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
                _portable_environment_spec("local"),
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
    finalized = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_OBSERVATION_FINALIZED
    )
    assert finalized.payload["status"] == "incomplete"
    assert finalized.payload["detail_code"] == "workspace_revision_evidence_incomplete"
    artifact_states = {
        value for key, value in finalized.payload.items() if key.endswith("_artifact_state")
    }
    assert artifact_states == {"intent"}
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in public_events)


def test_artifact_store_generator_exit_propagates_without_false_terminal_evidence(
    tmp_path,
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
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=_GeneratorExitArtifactStore(artifact_root),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="bulk-model"),
            tools=[_BulkWriteTool()],
        )

        with pytest.raises(GeneratorExit, match="artifact-store supervisory exit"):
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-artifact-generator-exit",
                    messages=[Message.text("user", "create files")],
                ),
            )
        durable = await store.query_events(EventQuery(session_id="session-artifact-generator-exit"))
        return [record.event for record in durable]

    durable_events = asyncio.run(run())

    assert not any(
        event.type
        in {
            EventType.WORKSPACE_MUTATION_RECORDED,
            EventType.WORKSPACE_OBSERVATION_FINALIZED,
            EventType.SESSION_FAILED,
        }
        for event in durable_events
    )


def test_artifact_store_process_control_is_not_lost_to_concurrent_cancellation(
    tmp_path,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = tmp_path / "artifacts"
    workspace_root.mkdir()

    async def run():
        artifact_store = _ConcurrentGeneratorExitArtifactStore(artifact_root)
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(_BulkProvider(), default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=artifact_store,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="bulk-model"),
            tools=[_BulkWriteTool()],
        )
        consumer = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-artifact-concurrent-control",
                    messages=[Message.text("user", "create files")],
                ),
            )
        )
        artifact_store.cancel_target = consumer
        await artifact_store.started.wait()
        artifact_store.release.set()
        with pytest.raises(BaseExceptionGroup) as raised:
            await consumer
        return raised.value, consumer

    failure, consumer = asyncio.run(run())

    failures = list(iter_exception_tree(failure))
    assert any(
        type(candidate) is GeneratorExit
        and candidate.args == ("artifact-store concurrent supervisory exit",)
        for candidate in failures
    )
    assert any(
        type(candidate) is asyncio.CancelledError
        and candidate.args == ("artifact caller cancellation",)
        for candidate in failures
    )
    assert consumer.cancelling() == 1
    assert consumer.cancelled() is False


@pytest.mark.parametrize("run_fence_release_failure", [False, True])
@pytest.mark.parametrize("entrance", ["run", "resume"])
def test_interrupted_tool_preserves_artifact_store_supervisory_exit(
    tmp_path,
    monkeypatch,
    entrance: str,
    run_fence_release_failure: bool,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = tmp_path / "artifacts"
    workspace_root.mkdir()
    monkeypatch.setattr(
        tool_round_executor_module,
        "_WORKSPACE_RECEIPT_INLINE_PATH_LIMIT",
        0,
    )

    async def run():
        tool = _BlockingAfterWriteMutationTool()
        store = _BlockingInterruptionCleanupStore()
        app = CayuApp(session_store=store, enable_logging=False)
        provider_type = (
            _SingleToolProvider if entrance == "run" else _CompletionThenSingleToolProvider
        )
        app.register_provider(
            provider_type(
                tool_name=tool.spec.name,
                arguments={},
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=_GeneratorExitArtifactStore(artifact_root),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="cancel-model"),
            tools=[tool],
        )
        session_id = f"session-interrupted-artifact-generator-exit-{entrance}"
        if entrance == "resume":
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "complete before resume")],
                ),
            )
        store.fail_next_run_fence_release = run_fence_release_failure

        async def consume() -> list[Event]:
            if entrance == "run":
                return await collect_events(
                    app,
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "write then cancel")],
                    ),
                )
            return [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "write then cancel")],
                    )
                )
            ]

        consumer = asyncio.create_task(consume())
        await tool.started.wait()
        consumer.cancel("caller cancellation during workspace mutation")
        await store.cleanup_started.wait()
        consumer.cancel("late cancellation during interruption cleanup")
        await asyncio.sleep(0)
        assert consumer.done() is False
        store.cleanup_release.set()
        with pytest.raises(BaseExceptionGroup) as raised:
            await consumer
        durable = await store.query_events(EventQuery(session_id=session_id))
        persisted = await store.load(session_id)
        assert persisted is not None
        return raised.value, consumer, persisted.status, [record.event for record in durable]

    failure, consumer, status, durable_events = asyncio.run(run())

    failures = list(iter_exception_tree(failure))
    assert any(
        type(candidate) is asyncio.CancelledError
        and candidate.args == ("caller cancellation during workspace mutation",)
        for candidate in failures
    )
    assert any(
        type(candidate) is asyncio.CancelledError
        and candidate.args == ("late cancellation during interruption cleanup",)
        for candidate in failures
    )
    assert any(
        type(candidate) is GeneratorExit and candidate.args == ("artifact-store supervisory exit",)
        for candidate in failures
    )
    assert (
        any(
            type(candidate) is RuntimeError
            and candidate.args == ("interruption run-fence release failed",)
            for candidate in failures
        )
        is run_fence_release_failure
    )
    assert workspace_observation_pending_cancellation_requests(failure) == 2
    assert consumer.cancelling() == 2
    assert consumer.cancelled() is False
    assert status is SessionStatus.INTERRUPTED
    assert (
        sum(
            type(candidate) is ValueError
            and candidate.args
            == (
                "Tool-round publication requires exactly one terminal event for every "
                "pending call; missing: call-workspace.",
            )
            for candidate in failures
        )
        == 1
    )
    paused_events = [
        event for event in durable_events if event.type is EventType.INTERACTION_PAUSED
    ]
    assert len(paused_events) == 1
    assert paused_events[0].payload["pending_action_kind"] == "tool_recovery"
    assert any(event.type is EventType.SESSION_INTERRUPTED for event in durable_events)
    assert not any(
        event.type
        in {
            EventType.WORKSPACE_MUTATION_RECORDED,
            EventType.WORKSPACE_OBSERVATION_FINALIZED,
        }
        for event in durable_events
    )


def test_grouped_interruption_does_not_transfer_cancellation_to_stream_closer(
    tmp_path,
    monkeypatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    artifact_root = tmp_path / "artifacts"
    workspace_root.mkdir()
    monkeypatch.setattr(
        tool_round_executor_module,
        "_WORKSPACE_RECEIPT_INLINE_PATH_LIMIT",
        0,
    )

    async def run():
        tool = _BlockingAfterWriteMutationTool()
        store = _BlockingInterruptionCleanupStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            _SingleToolProvider(
                tool_name=tool.spec.name,
                arguments={},
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=_GeneratorExitArtifactStore(artifact_root),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="cancel-model"),
            tools=[tool],
        )
        session_id = "session-close-buffered-interruption-event"
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "write then close")],
            )
        )

        async def consume_until_failure():
            delivered_events: list[Event] = []
            try:
                async for event in stream:
                    delivered_events.append(event)
            except BaseException as failure:
                task = asyncio.current_task()
                assert task is not None
                return delivered_events, failure, task.cancelling()
            raise AssertionError("The authoritative interruption failure was not delivered.")

        consumer = asyncio.create_task(consume_until_failure())
        await tool.started.wait()
        consumer.cancel("caller cancellation before buffered delivery")
        await store.cleanup_started.wait()
        store.cleanup_release.set()
        delivered_events, failure, cancellation_requests = await consumer

        async def close_stream() -> int:
            await stream.aclose()
            await asyncio.sleep(0)
            task = asyncio.current_task()
            assert task is not None
            return task.cancelling()

        closer = asyncio.create_task(close_stream())
        closer_cancellation_requests = await closer
        durable = await store.query_events(EventQuery(session_id=session_id))
        return (
            delivered_events,
            failure,
            cancellation_requests,
            consumer,
            closer_cancellation_requests,
            closer,
            [record.event for record in durable],
        )

    (
        delivered_events,
        failure,
        cancellation_requests,
        consumer,
        closer_cancellation_requests,
        closer,
        durable_events,
    ) = asyncio.run(run())

    failures = list(iter_exception_tree(failure))
    assert any(
        type(candidate) is asyncio.CancelledError
        and candidate.args == ("caller cancellation before buffered delivery",)
        for candidate in failures
    )
    assert any(
        type(candidate) is GeneratorExit and candidate.args == ("artifact-store supervisory exit",)
        for candidate in failures
    )
    assert workspace_observation_pending_cancellation_requests(failure) == 1
    assert not any(
        event.type in {EventType.INTERACTION_INTERRUPTED, EventType.SESSION_INTERRUPTED}
        for event in delivered_events
    )
    assert cancellation_requests == 1
    assert consumer.cancelled() is False
    assert closer_cancellation_requests == 0
    assert closer.cancelled() is False
    paused_events = [
        event for event in durable_events if event.type is EventType.INTERACTION_PAUSED
    ]
    assert len(paused_events) == 1
    assert paused_events[0].payload["pending_action_kind"] == "tool_recovery"
    assert any(event.type is EventType.SESSION_INTERRUPTED for event in durable_events)


def test_interruption_failure_deduplication_handles_deep_groups_iteratively() -> None:
    failure: BaseException = RuntimeError("deep cleanup failure")
    for _ in range(2_000):
        failure = BaseExceptionGroup("nested cleanup", [failure])

    assert (
        session_engine_module._failure_without_existing_exception_identities(
            failure,
            set(),
        )
        is failure
    )


def test_interruption_cleanup_does_not_invoke_failure_truthiness() -> None:
    class HostileTruthinessFailure(RuntimeError):
        def __bool__(self) -> bool:
            raise AssertionError("failure truthiness must not run")

    preserved = asyncio.CancelledError("authoritative cancellation")
    hostile = HostileTruthinessFailure("private cleanup diagnostic")
    failure = BaseExceptionGroup("cleanup failures", [preserved, hostile])

    classified = session_engine_module._session_interruption_cleanup_child_error(
        failure,
        operation="Session interruption cleanup",
        preserved_failure=preserved,
    )

    assert isinstance(classified, BaseExceptionGroup)
    classified_failures = list(iter_exception_tree(classified))
    assert any(candidate is preserved for candidate in classified_failures)
    assert any(candidate is hostile for candidate in classified_failures)


def test_interruption_cleanup_failure_preserves_historical_task_cancellation() -> None:
    async def run() -> BaseException:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("handled historical cancellation")
        with pytest.raises(asyncio.CancelledError, match="handled historical cancellation"):
            await asyncio.sleep(0)
        assert current.cancelling() == 1

        authoritative = BaseExceptionGroup(
            "Workspace observation control.",
            [
                asyncio.CancelledError("owned workspace cancellation"),
                GeneratorExit("owned workspace supervisory exit"),
            ],
        )
        retain_workspace_observation_pending_cancellation_requests(authoritative, 1)
        propagated = session_engine_module._session_interruption_failure_with_additional_control(
            authoritative,
            RuntimeError("interruption cleanup failed"),
            group_message="Session interruption finalization failed.",
            historical_cancellation_requests=1,
        )

        assert current.cancelling() == 1
        await asyncio.sleep(0)
        assert current.cancelling() == 1
        current.uncancel()
        return propagated

    failure = asyncio.run(run())

    failures = list(iter_exception_tree(failure))
    assert (
        sum(
            type(candidate) is asyncio.CancelledError
            and candidate.args == ("owned workspace cancellation",)
            for candidate in failures
        )
        == 1
    )
    assert (
        sum(
            type(candidate) is RuntimeError and candidate.args == ("interruption cleanup failed",)
            for candidate in failures
        )
        == 1
    )
    assert workspace_observation_pending_cancellation_requests(failure) == 1


def test_interruption_cleanup_handles_deep_preserved_failure_group_iteratively() -> None:
    preserved = asyncio.CancelledError("authoritative cancellation")
    failure: BaseException = preserved
    for _ in range(2_000):
        failure = BaseExceptionGroup("nested cleanup", [failure])

    classified = session_engine_module._session_interruption_cleanup_child_error(
        failure,
        operation="Session interruption cleanup",
        preserved_failure=preserved,
    )

    assert isinstance(classified, BaseExceptionGroup)
    assert any(candidate is preserved for candidate in iter_exception_tree(classified))


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
                _portable_environment_spec("local"),
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
        await artifacts.finished.wait()
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
    assert after.payload["detail_code"] == "manifest_artifact_write_unsettled"
    finalized = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_OBSERVATION_FINALIZED
    )
    assert {
        value for key, value in finalized.payload.items() if key.endswith("_artifact_state")
    } == {"intent"}
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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


def test_observer_owned_cancellation_fails_closed_before_tool_dispatch(tmp_path) -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _ScriptedProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
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
        with pytest.raises(
            WorkspaceMutationSettlementError,
            match="settlement could not be proven",
        ) as reuse_error:
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-child-cancelled-observer-reuse",
                    messages=[Message.text("user", "reuse")],
                ),
            )
        return public, reuse_error.value, [record.event for record in durable], provider

    public_events, reuse_error, durable_events, provider = asyncio.run(run())

    assert provider.requests == 1
    assert (tmp_path / "shell.txt").exists() is False
    assert not any(
        event.type
        in {
            EventType.WORKSPACE_REVISION_OBSERVED,
            EventType.TOOL_CALL_COMPLETED,
        }
        for event in durable_events
    )
    assert any(event.type is EventType.SESSION_FAILED for event in public_events)
    assert str(reuse_error) == "Workspace mutation settlement could not be proven."
    assert any(
        event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED for event in durable_events
    )


def test_observer_owned_generator_exit_propagates_and_does_not_quarantine_reuse(
    tmp_path,
) -> None:
    async def run():
        binding = _OneShotGeneratorExitObserverBinding()
        provider = _SingleToolProvider(
            tool_name="noop_workspace_mutation",
            arguments={},
        )
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="generator-exit-workspace"),
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_NoopWorkspaceMutationTool()],
        )

        with pytest.raises(GeneratorExit, match="observer supervisory exit"):
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-observer-generator-exit",
                    messages=[Message.text("user", "observe")],
                ),
            )
        second = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session-observer-generator-exit-reuse",
                messages=[Message.text("user", "reuse")],
            ),
        )
        return second, binding, provider

    second, binding, provider = asyncio.run(run())

    assert any(event.type is EventType.SESSION_COMPLETED for event in second)
    assert binding.observations >= 2
    assert provider.requests == 2


@pytest.mark.parametrize("historical_cancellation", [False, True])
def test_observer_process_control_is_not_lost_to_concurrent_cancellation(
    tmp_path,
    historical_cancellation: bool,
) -> None:
    async def run():
        binding = _ConcurrentGeneratorExitObserverBinding()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(
            _SingleToolProvider(
                tool_name="noop_workspace_mutation",
                arguments={},
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="concurrent-control-workspace"),
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_NoopWorkspaceMutationTool()],
        )

        async def consume() -> list[Event]:
            if historical_cancellation:
                current = asyncio.current_task()
                assert current is not None
                current.cancel("handled historical cancellation")
                with pytest.raises(asyncio.CancelledError, match="handled historical cancellation"):
                    await asyncio.sleep(0)
            return await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-observer-concurrent-control",
                    messages=[Message.text("user", "observe")],
                ),
            )

        consumer = asyncio.create_task(consume())
        binding.cancel_target = consumer
        await binding.started.wait()
        binding.release.set()
        with pytest.raises(BaseExceptionGroup) as raised:
            await consumer
        return raised.value, consumer

    failure, consumer = asyncio.run(run())

    failures = list(iter_exception_tree(failure))
    assert any(
        type(candidate) is GeneratorExit
        and candidate.args == ("observer concurrent supervisory exit",)
        for candidate in failures
    )
    assert (
        sum(
            type(candidate) is asyncio.CancelledError
            and candidate.args == ("observer caller cancellation",)
            for candidate in failures
        )
        == 1
    )
    expected_cancellation_requests = 1 + int(historical_cancellation)
    assert (
        workspace_observation_pending_cancellation_requests(failure)
        == expected_cancellation_requests
    )
    assert consumer.cancelling() == expected_cancellation_requests
    assert consumer.cancelled() is False


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
                _portable_environment_spec("local"),
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


@pytest.mark.parametrize("cancellation_requests", [1, 2])
def test_caller_cancellation_during_after_observation_preserves_tool_terminal(
    tmp_path,
    cancellation_requests: int,
) -> None:
    async def run():
        binding = _StalledObserverBinding()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
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
                        session_id=(f"session-cancelled-after-observer-{cancellation_requests}"),
                        messages=[Message.text("user", "create a file")],
                    )
                )
            ]

        consumer = asyncio.create_task(consume())
        await binding.started.wait()
        for _request in range(cancellation_requests):
            consumer.cancel("cancel while observing workspace")
        with pytest.raises(asyncio.CancelledError, match="cancel while observing workspace"):
            await consumer
        assert consumer.cancelling() == cancellation_requests
        assert consumer.cancelled() is True
        binding.release.set()
        await asyncio.sleep(0)
        durable = await store.query_events(
            EventQuery(session_id=f"session-cancelled-after-observer-{cancellation_requests}")
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
        assert consumer.cancelling() == 1
        assert consumer.cancelled() is True
        cleanup_group = exception_cause(raised.value)
        assert isinstance(cleanup_group, BaseExceptionGroup)
        cleanup_failures = tuple(iter_exception_tree(cleanup_group))
        assert any(isinstance(failure, ValueError) for failure in cleanup_failures)
        closure_failure = exception_cause(cleanup_group)
        assert isinstance(closure_failure, RuntimeError)
        assert str(closure_failure) == "Interrupted tool-round closure failed."
        assert all(
            "terminal publication failed" not in str(failure)
            for failure in (*cleanup_failures, closure_failure)
        )
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
                _portable_environment_spec("local"),
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
        assert consumer.cancelling() == 2
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
            secret_redactor=SecretRedactor(["before", "failed", "receipt_publication_failed"]),
        )
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
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
    observation_terminal = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_OBSERVATION_FINALIZED
    )
    assert observation_terminal.payload["status"] == "failed"
    assert observation_terminal.payload["detail_code"] == "receipt_publication_failed"
    public_observation_terminal = next(
        event for event in public_events if event.type == EventType.WORKSPACE_OBSERVATION_FINALIZED
    )
    assert public_observation_terminal.payload["status"] == "failed"
    assert public_observation_terminal.payload["detail_code"] == "receipt_publication_failed"
    assert (tmp_path / "shell.txt").read_text(encoding="utf-8") == "created"


def test_committed_malformed_workspace_acknowledgement_reconciles_exact_receipt(
    tmp_path,
) -> None:
    async def run():
        store = _CommittedMalformedWorkspaceAcknowledgementStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
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
        public = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session-malformed-committed-workspace-ack",
                messages=[Message.text("user", "create a file")],
            ),
        )
        durable = await store.query_events(
            EventQuery(session_id="session-malformed-committed-workspace-ack")
        )
        return store, public, [record.event for record in durable]

    store, public_events, durable_events = asyncio.run(run())

    assert store.before_evidence_attempts == 2
    assert (
        sum(
            event.type == EventType.WORKSPACE_REVISION_OBSERVED
            and event.payload.get("phase") == "before"
            for event in public_events
        )
        == 1
    )
    assert (
        sum(
            event.type == EventType.WORKSPACE_REVISION_OBSERVED
            and event.payload.get("phase") == "before"
            for event in durable_events
        )
        == 1
    )
    assert sum(event.type == EventType.TOOL_CALL_STARTED for event in durable_events) == 1
    assert any(event.type == EventType.SESSION_COMPLETED for event in durable_events)
    assert (tmp_path / "shell.txt").read_text(encoding="utf-8") == "created"


def test_uncommitted_malformed_workspace_acknowledgement_is_not_fanned_out(
    tmp_path,
    caplog,
    capsys,
) -> None:
    async def run():
        store = _UncommittedMalformedWorkspaceAcknowledgementStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
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
        public = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session-malformed-uncommitted-workspace-ack",
                messages=[Message.text("user", "create a file")],
            ),
        )
        durable = await store.query_events(
            EventQuery(session_id="session-malformed-uncommitted-workspace-ack")
        )
        return store, public, [record.event for record in durable]

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        store, public_events, durable_events = asyncio.run(run())
    captured_output = capsys.readouterr()

    assert store.before_evidence_attempts == 1
    assert not any(
        event.type == EventType.WORKSPACE_REVISION_OBSERVED
        and event.payload.get("phase") == "before"
        for event in public_events
    )
    assert not any(
        event.type == EventType.WORKSPACE_REVISION_OBSERVED
        and event.payload.get("phase") == "before"
        for event in durable_events
    )
    finalized = next(
        event for event in durable_events if event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
    )
    assert finalized.payload["status"] == "failed"
    assert finalized.payload["detail_code"] == "receipt_publication_failed"
    assert any(event.type is EventType.SESSION_COMPLETED for event in durable_events)
    assert _MALFORMED_WORKSPACE_ACK_CANARY not in repr(public_events)
    assert _MALFORMED_WORKSPACE_ACK_CANARY not in repr(durable_events)
    assert _MALFORMED_WORKSPACE_ACK_CANARY not in caplog.text
    assert _MALFORMED_WORKSPACE_ACK_CANARY not in captured_output.out
    assert _MALFORMED_WORKSPACE_ACK_CANARY not in captured_output.err
    assert not any(
        _MALFORMED_WORKSPACE_ACK_CANARY in str(warning.message) for warning in captured_warnings
    )


def test_cancelled_workspace_publication_preserves_initial_and_reconciliation_failures(
    tmp_path,
) -> None:
    async def run():
        store = _CancelledFailedWorkspacePublicationStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
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
        consumer = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-workspace-publication-failure-cancelled",
                    messages=[Message.text("user", "create a file")],
                ),
            )
        )
        await store.publication_started.wait()
        consumer.cancel("workspace publication caller cancellation")
        store.release_publication.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await consumer
        return store, raised.value, consumer.cancelling(), consumer.cancelled()

    store, cancellation, cancelling, cancelled = asyncio.run(run())
    cause = exception_cause(cancellation)
    assert cause is not None
    failures = list(iter_exception_tree(cause))
    bounded_failures = [
        candidate for candidate in failures if type(candidate) is RunnerExecutionError
    ]
    assert [failure.diagnostic["error_type"] for failure in bounded_failures] == [
        "ConnectionError",
        "TimeoutError",
    ]
    assert all(candidate is not store.initial_failure for candidate in failures)
    assert all(candidate is not store.reconciliation_failure for candidate in failures)
    assert cancellation.args == ("workspace publication caller cancellation",)
    assert cancelling == 1
    assert cancelled is True


def test_self_consistent_conflicting_workspace_acknowledgement_is_not_fanned_out(
    tmp_path,
) -> None:
    async def run():
        store = _UncommittedConflictingWorkspaceAcknowledgementStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
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
        public = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="session-conflicting-uncommitted-workspace-ack",
                messages=[Message.text("user", "create a file")],
            ),
        )
        durable = await store.query_events(
            EventQuery(session_id="session-conflicting-uncommitted-workspace-ack")
        )
        return store, public, [record.event for record in durable]

    store, public_events, durable_events = asyncio.run(run())

    assert store.before_evidence_attempts == 1
    assert not any(
        event.type is EventType.WORKSPACE_REVISION_OBSERVED
        and event.payload.get("phase") == "before"
        for event in (*public_events, *durable_events)
    )
    finalized = next(
        event for event in durable_events if event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
    )
    assert finalized.payload["status"] == "failed"
    assert finalized.payload["detail_code"] == "receipt_publication_failed"
    assert any(event.type is EventType.SESSION_COMPLETED for event in durable_events)


def test_rejected_workspace_acknowledgement_is_absent_from_traceback_locals() -> None:
    async def run() -> RuntimeError:
        store = _TracebackMalformedWorkspaceAcknowledgementStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session-malformed-workspace-ack-traceback",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="scripted", model="scripted-model"),
        )
        store.returned_session = session
        event_writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=(),
        )
        lifecycle = WorkspaceObservationLifecycle(
            session_id=session.id,
            window_id="wmut-malformed-ack-traceback",
            source_run_epoch=session.run_epoch,
            binding_generation_id="wbind-malformed-ack-traceback",
            workspace_id="workspace-malformed-ack-traceback",
            observer="TracebackWorkspaceBinding",
            observer_authority="configured",
            artifact_store_id=None,
            agent_name="assistant",
            environment_name="local",
            tool_name="mutate",
            tool_call_id="call-malformed-ack-traceback",
            model_step_id="mstep-malformed-ack-traceback",
            model_attempt_id="matt-malformed-ack-traceback",
            tool_round_id="tround-malformed-ack-traceback",
            model_step=1,
        )
        with pytest.raises(
            RuntimeError,
            match="returned invalid acknowledgement",
        ) as raised:
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=None,
                current=lifecycle,
                phase="intent",
                intent_admission=_admit_test_workspace_observation_intent(lifecycle),
            )
        return raised.value

    error = asyncio.run(run())
    captured = traceback.TracebackException.from_exception(error, capture_locals=True)
    cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
    retained = repr([(frame.name, frame.locals) for frame in cayu_frames])

    assert _MALFORMED_WORKSPACE_ACK_CANARY not in retained
    assert error.__cause__ is None
    assert error.__context__ is None


def test_abandoning_after_workspace_event_does_not_erase_tool_terminal(tmp_path) -> None:
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_ScriptedProvider(), default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
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


@pytest.mark.parametrize(
    (
        "crash_phase",
        "checkpoint_phase",
        "terminal_status",
        "tool_ran",
        "hide_workspace_delta",
    ),
    [
        ("intent", "intent", "ambiguous", False, False),
        ("before-capture", "before_captured", "ambiguous", False, False),
        ("terminal-stage", "before_captured", "incomplete", True, False),
        ("tool-outcome", "tool_outcome_staged", "incomplete", True, False),
        ("before-evidence", "tool_outcome_staged", "incomplete", True, False),
        ("after-capture", "after_captured", "incomplete", True, False),
        ("delta-publication", "delta_published", "complete", True, False),
        ("delta-publication", "delta_published", "incomplete", True, True),
        ("terminal", None, "complete", True, False),
    ],
)
def test_fresh_process_recovers_workspace_observation_crash_boundaries_without_redispatch(
    tmp_path,
    crash_phase: str,
    checkpoint_phase: str | None,
    terminal_status: str,
    tool_ran: bool,
    hide_workspace_delta: bool,
) -> None:
    async def run():
        store = _WorkspaceObservationProcessLossStore(phase=crash_phase)
        first_provider = _ScriptedProvider()
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(first_provider, default=True)
        first_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        session_id = (
            f"session-workspace-observation-process-loss-{crash_phase}-{hide_workspace_delta}"
        )
        with pytest.raises(_WorkspaceObservationProcessLoss, match=crash_phase):
            _ = [
                event
                async for event in first_app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "create a file")],
                    )
                )
            ]

        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        observations = checkpoint.get("workspace_observations")
        if checkpoint_phase is None:
            assert observations is None
        else:
            assert type(observations) is dict and len(observations) == 1
            assert next(iter(observations.values()))["phase"] == checkpoint_phase
        assert first_provider.requests == 1
        assert (tmp_path / "shell.txt").exists() is tool_ran

        store.failed = True
        store.hide_workspace_delta = hide_workspace_delta
        await store.release_run_fence(session_id)
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        recovery_provider = _ScriptedProvider()
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_app.register_provider(recovery_provider, default=True)
        recovery_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        competing_provider = _ScriptedProvider()
        competing_app = CayuApp(session_store=store, enable_logging=False)
        competing_app.register_provider(competing_provider, default=True)
        competing_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        competing_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        recoveries = await asyncio.gather(
            recovery_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            ),
            competing_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            ),
        )
        durable = await store.query_events(EventQuery(session_id=session_id))
        return (
            first_provider,
            recovery_provider,
            competing_provider,
            recoveries,
            [record.event for record in durable],
            await store.load_checkpoint(session_id),
        )

    (
        first_provider,
        recovery_provider,
        competing_provider,
        recoveries,
        durable_events,
        checkpoint,
    ) = asyncio.run(run())

    assert first_provider.requests == 1
    assert recovery_provider.requests == 0
    assert competing_provider.requests == 0
    assert sum(
        IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION in recovery.actions
        for recovery in recoveries
    ) == (1 if checkpoint_phase is not None else 0)
    assert (
        sum(
            IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in recovery.actions
            for recovery in recoveries
        )
        == 1
    )

    finalized = [
        event for event in durable_events if event.type == EventType.WORKSPACE_OBSERVATION_FINALIZED
    ]
    assert len(finalized) == 1
    assert finalized[0].payload["status"] == terminal_status
    terminal_type = EventType.TOOL_CALL_COMPLETED if tool_ran else EventType.TOOL_CALL_FAILED
    terminal = next(event for event in durable_events if event.type == terminal_type)
    if (crash_phase == "delta-publication" and not hide_workspace_delta) or (
        crash_phase == "terminal"
    ):
        assert terminal.payload["workspace_mutation_capture_status"] == "recorded"
    elif hide_workspace_delta:
        assert terminal.payload["workspace_mutation_capture_status"] == "failed"
        assert (
            terminal.payload["workspace_mutation_capture_detail_code"]
            == "workspace_delta_evidence_missing"
        )
    elif tool_ran:
        assert terminal.payload["workspace_mutation_capture_status"] == "failed"
    assert checkpoint is not None
    assert "workspace_observations" not in checkpoint


@pytest.mark.parametrize(
    ("crash_phase", "terminal_status", "detail_code"),
    [
        (
            "intent",
            "ambiguous",
            "worker_lost_before_tool_outcome_was_durable",
        ),
        (
            "after-capture",
            "incomplete",
            "worker_lost_before_workspace_observation_completed",
        ),
        (
            "delta-publication",
            "incomplete",
            "workspace_revision_evidence_incomplete",
        ),
    ],
)
def test_fresh_process_factory_observation_recovery_closes_without_reconnect(
    tmp_path,
    crash_phase: str,
    terminal_status: str,
    detail_code: str,
) -> None:
    async def run():
        store = _WorkspaceObservationProcessLossStore(phase=crash_phase)
        first_factory = _WorkspaceObservationRecoveryFactory(
            tmp_path / "first-factory-workspace",
            workspace_id="factory-workspace",
        )
        first_provider = _ScriptedProvider()
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(first_provider, default=True)
        first_app.register_environment_factory(
            _portable_environment_spec("dynamic"),
            first_factory,
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        session_id = f"session-factory-observation-recovery-{crash_phase}"
        with pytest.raises(_WorkspaceObservationProcessLoss, match=crash_phase):
            await collect_events(
                first_app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create a file")],
                ),
            )

        checkpoint_before = await store.load_checkpoint(session_id)
        assert checkpoint_before is not None
        assert workspace_observations_from_checkpoint(checkpoint_before)
        store.failed = True
        await store.release_run_fence(session_id)
        await store.update_status(session_id, SessionStatus.INTERRUPTED)

        # The replacement registration intentionally has no access to the
        # historical concrete workspace. Recovery must not invoke it merely to
        # collect evidence or compare identities.
        recovery_factory = _WorkspaceObservationRecoveryFactory(
            tmp_path / "replacement-factory-workspace",
            workspace_id="replacement-workspace",
        )
        recovery_provider = _ScriptedProvider()
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_app.register_provider(recovery_provider, default=True)
        recovery_app.register_environment_factory(
            _portable_environment_spec("dynamic"),
            recovery_factory,
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        if crash_phase == "intent":
            with pytest.raises(
                RuntimeError,
                match="cannot safely continue with opaque provider state",
            ):
                await recovery_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                )
            recovery = None
        else:
            recovery = await recovery_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
        durable = await store.query_events(EventQuery(session_id=session_id))
        checkpoint_after = await store.load_checkpoint(session_id)
        return (
            first_factory,
            recovery_factory,
            first_provider,
            recovery_provider,
            recovery,
            [record.event for record in durable],
            checkpoint_after,
        )

    (
        first_factory,
        recovery_factory,
        first_provider,
        recovery_provider,
        recovery,
        durable_events,
        checkpoint,
    ) = asyncio.run(run())

    assert first_factory.create_calls == 1
    assert recovery_factory.create_calls == 0
    assert first_provider.requests == 1
    assert recovery_provider.requests == 0
    if recovery is not None:
        assert IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION in recovery.actions
    finalized = [
        event for event in durable_events if event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
    ]
    assert len(finalized) == 1
    assert finalized[0].payload["status"] == terminal_status
    assert finalized[0].payload["detail_code"] == detail_code
    assert checkpoint is not None
    assert "workspace_observations" not in checkpoint


@pytest.mark.parametrize(
    ("crash_phase", "cancel_kind", "expected_operation"),
    [
        (
            "after-capture",
            "read",
            "Workspace observation recovery checkpoint read",
        ),
        (
            "terminal-stage",
            "mutation",
            "Workspace observation terminal-stage repair",
        ),
    ],
)
def test_workspace_observation_recovery_does_not_fabricate_caller_cancellation(
    tmp_path,
    crash_phase,
    cancel_kind,
    expected_operation,
) -> None:
    async def run():
        store = _WorkspaceObservationProcessLossStore(phase=crash_phase)
        first_provider = _ScriptedProvider()
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(first_provider, default=True)
        first_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        session_id = f"session-workspace-observation-recovery-child-cancellation-{cancel_kind}"
        with pytest.raises(_WorkspaceObservationProcessLoss, match=crash_phase):
            await collect_events(
                first_app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create a file")],
                ),
            )

        store.failed = True
        if cancel_kind == "read":
            store.cancel_workspace_observation_read = True
        else:
            store.cancel_workspace_observation_mutation = True
        await store.release_run_fence(session_id)
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        recovery_provider = _ScriptedProvider()
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_app.register_provider(recovery_provider, default=True)
        recovery_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        recovery_task = asyncio.create_task(
            recovery_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
        )
        with pytest.raises(
            RuntimeError,
            match=f"{expected_operation} was cancelled without caller cancellation",
        ):
            await recovery_task
        return recovery_task, first_provider, recovery_provider, store

    recovery_task, first_provider, recovery_provider, store = asyncio.run(run())

    if cancel_kind == "read":
        assert store.workspace_observation_read_cancelled is True
        assert store.workspace_observation_mutation_cancelled is False
    else:
        assert store.workspace_observation_read_cancelled is False
        assert store.workspace_observation_mutation_cancelled is True
    assert recovery_task.cancelling() == 0
    assert recovery_task.cancelled() is False
    assert first_provider.requests == 1
    assert recovery_provider.requests == 0


@pytest.mark.parametrize(
    (
        "crash_phase",
        "delete_before_recovery",
        "recovery_store_id",
        "metadata_update",
        "artifact_state",
        "commit_then_raise",
        "malformed_read_result",
    ),
    [
        (
            "artifact-revision-after-intent",
            False,
            "artifact-store",
            None,
            "intent",
            False,
            False,
        ),
        (
            "artifact-revision-after-published",
            False,
            "artifact-store",
            None,
            "orphaned",
            False,
            False,
        ),
        ("after-capture", False, "artifact-store", None, "referenced", False, False),
        ("after-capture", True, "artifact-store", None, "missing", False, False),
        ("delta-publication", True, "artifact-store", None, "missing", False, False),
        ("delta-publication", False, "artifact-store", None, "failed", False, True),
        ("after-capture", False, "artifact-store", None, "orphaned", True, False),
        (
            "after-capture",
            False,
            "artifact-store",
            {"filename": "foreign.json"},
            "failed",
            False,
            False,
        ),
        (
            "after-capture",
            False,
            "artifact-store",
            {"session_id": "foreign-session"},
            "failed",
            False,
            False,
        ),
        (
            "after-capture",
            False,
            "artifact-store",
            {"scope": ArtifactScope.ENVIRONMENT},
            "failed",
            False,
            False,
        ),
        (
            "after-capture",
            False,
            "artifact-store",
            {
                "metadata": {
                    "schema_version": 1,
                    "kind": "revision-before",
                    "sha256": "a" * 64,
                    "window_id": "foreign-window",
                }
            },
            "failed",
            False,
            False,
        ),
    ],
)
def test_workspace_observation_recovery_reports_partial_artifact_state(
    tmp_path,
    crash_phase: str,
    delete_before_recovery: bool,
    recovery_store_id: str,
    metadata_update: dict | None,
    artifact_state: str,
    commit_then_raise: bool,
    malformed_read_result: bool,
) -> None:
    async def run():
        store = _WorkspaceObservationProcessLossStore(phase=crash_phase)
        artifact_root = tmp_path / "artifacts"
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        artifact_store = (
            _CommitThenRaiseArtifactStore(artifact_root, store_id="artifact-store")
            if commit_then_raise
            else LocalArtifactStore(artifact_root, store_id="artifact-store")
        )
        first_provider = _BulkProvider()
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(first_provider, default=True)
        first_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=artifact_store,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_BulkWriteTool()],
        )
        session_id = f"session-workspace-artifact-recovery-{crash_phase}-{delete_before_recovery}"
        with pytest.raises(_WorkspaceObservationProcessLoss, match=crash_phase):
            _ = [
                event
                async for event in first_app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "write many files")],
                    )
                )
            ]
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        observations = checkpoint["workspace_observations"]
        lifecycle = next(iter(observations.values()))
        artifact = next(
            item for item in lifecycle["artifacts"] if item["evidence_kind"] == "revision-after"
        )
        if delete_before_recovery:
            await artifact_store.delete(artifact["artifact_id"])

        store.failed = True
        store.hide_workspace_delta = crash_phase == "delta-publication"
        await store.release_run_fence(session_id)
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        recovery_provider = _BulkProvider()
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_app.register_provider(recovery_provider, default=True)
        recovery_artifact_store = (
            _NoneReadArtifactStore(
                artifact_root,
                store_id=recovery_store_id,
            )
            if malformed_read_result
            else LocalArtifactStore(
                artifact_root,
                store_id=recovery_store_id,
            )
            if metadata_update is None
            else _ConflictingReadArtifactStore(
                artifact_root,
                store_id=recovery_store_id,
                metadata_update=metadata_update,
            )
        )
        recovery_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=recovery_artifact_store,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_BulkWriteTool()],
        )
        recovery = await recovery_app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        durable = await store.query_events(EventQuery(session_id=session_id))
        return first_provider, recovery_provider, recovery, [record.event for record in durable]

    first_provider, recovery_provider, recovery, durable_events = asyncio.run(run())

    assert first_provider.requests == 1
    assert recovery_provider.requests == 0
    assert IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION in recovery.actions
    finalized = next(
        event for event in durable_events if event.type == EventType.WORKSPACE_OBSERVATION_FINALIZED
    )
    assert finalized.payload["status"] == "incomplete"
    assert finalized.payload["revision_after_artifact_state"] == artifact_state
    assert finalized.payload["revision_after_artifact_id"].startswith("art_")
    assert len(finalized.payload["revision_after_artifact_sha256"]) == 64
    assert finalized.payload["revision_after_artifact_size_bytes"] > 0
    if malformed_read_result:
        assert finalized.payload["detail_code"] == "workspace_artifact_verification_failed"
    if crash_phase == "delta-publication":
        terminal = next(
            event for event in durable_events if event.type is EventType.TOOL_CALL_COMPLETED
        )
        assert terminal.payload["workspace_mutation_capture_status"] == "failed"
        assert terminal.payload["workspace_mutation_capture_detail_code"] == (
            "workspace_artifact_verification_failed"
            if malformed_read_result
            else "referenced_workspace_artifact_missing"
        )


def test_recovery_does_not_downgrade_failed_delta_when_artifact_is_missing(
    tmp_path,
    monkeypatch,
) -> None:
    def failed_comparison(before, after):
        return WorkspaceRevisionDelta(
            identity=before.identity,
            status=WorkspaceRevisionDeltaStatus.FAILED,
            before_revision=before.revision,
            after_revision=after.revision,
            detail_code="revision_comparison_failed",
        )

    monkeypatch.setattr(
        tool_round_executor_module,
        "compare_workspace_revisions",
        failed_comparison,
    )

    async def run():
        workspace_root = tmp_path / "workspace"
        artifact_root = tmp_path / "artifacts"
        workspace_root.mkdir()
        artifact_store = LocalArtifactStore(
            artifact_root,
            store_id="artifact-store",
        )
        store = _WorkspaceObservationProcessLossStore(phase="delta-publication")
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(_BulkProvider(), default=True)
        first_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=artifact_store,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="bulk-model"),
            tools=[_BulkWriteTool()],
        )
        session_id = "session-failed-workspace-delta-missing-artifact"
        with pytest.raises(_WorkspaceObservationProcessLoss):
            await collect_events(
                first_app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "write many files")],
                ),
            )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        lifecycle = next(iter(checkpoint["workspace_observations"].values()))
        after_artifact = next(
            item for item in lifecycle["artifacts"] if item["evidence_kind"] == "revision-after"
        )
        await artifact_store.delete(after_artifact["artifact_id"])

        store.failed = True
        await store.release_run_fence(session_id)
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_app.register_provider(_BulkProvider(), default=True)
        recovery_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=LocalArtifactStore(
                    artifact_root,
                    store_id="artifact-store",
                ),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="bulk-model"),
            tools=[_BulkWriteTool()],
        )
        await recovery_app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        durable = await store.query_events(EventQuery(session_id=session_id))
        return [record.event for record in durable]

    durable_events = asyncio.run(run())

    finalized = next(
        event for event in durable_events if event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
    )
    assert finalized.payload["status"] == "failed"
    assert finalized.payload["detail_code"] == "workspace_revision_comparison_failed"
    assert finalized.payload["revision_after_artifact_state"] == "missing"


def test_recovery_retains_late_artifact_intent_until_store_cleanup(
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
        store = _WorkspaceObservationProcessLossStore(phase="after-capture")
        artifacts = _StalledArtifactStore(artifact_root)
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(_BulkProvider(), default=True)
        first_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=artifacts,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="bulk-model"),
            tools=[_BulkWriteTool()],
        )
        session_id = "session-late-workspace-artifact-recovery"
        with pytest.raises(_WorkspaceObservationProcessLoss, match="after-capture"):
            await collect_events(
                first_app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "write many files")],
                ),
            )
        assert artifacts.started.is_set()
        assert artifacts.finished.is_set() is False
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        lifecycle = next(iter(checkpoint["workspace_observations"].values()))
        artifact = next(
            item for item in lifecycle["artifacts"] if item["evidence_kind"] == "revision-after"
        )
        assert artifact["state"] == "intent"

        store.failed = True
        await store.release_run_fence(session_id)
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_app.register_provider(_BulkProvider(), default=True)
        recovery_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=artifacts,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="bulk-model"),
            tools=[_BulkWriteTool()],
        )
        await recovery_app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        durable = await store.query_events(EventQuery(session_id=session_id))
        finalized = next(
            record.event
            for record in durable
            if record.event.type == EventType.WORKSPACE_OBSERVATION_FINALIZED
        )
        assert finalized.payload["revision_after_artifact_state"] == "intent"
        assert finalized.payload["revision_after_artifact_id"] == artifact["artifact_id"]
        assert finalized.payload["revision_after_artifact_sha256"] == artifact["sha256"]
        assert finalized.payload["revision_after_artifact_size_bytes"] == artifact["size_bytes"]

        artifacts.release.set()
        await artifacts.finished.wait()
        result = await artifacts.read_bytes(
            artifact["artifact_id"],
            max_bytes=artifact["size_bytes"],
        )
        return result.content, artifact["sha256"]

    content, expected_digest = asyncio.run(run())

    assert hashlib.sha256(content).hexdigest() == expected_digest


@pytest.mark.parametrize(
    ("authority_update", "expected_error"),
    [
        ({}, "duplicate active lifecycles"),
        ({"session_id": "foreign-session"}, "belongs to a different session"),
        ({"agent_name": "foreign-agent"}, "conflicts with its invocation scope"),
        (
            {"environment_name": "foreign-environment"},
            "conflicts with its invocation scope",
        ),
        ({"source_run_epoch": 999}, "belongs to a future run epoch"),
        (
            {"source_run_epoch": 0},
            "conflicts with its active invocation profile",
        ),
        (
            {"interaction_id": "foreign-interaction"},
            "conflicts with its active invocation profile",
        ),
        ({"model_step_id": "model-step-foreign"}, "conflicts with its pending tool round"),
        (
            {"model_attempt_id": "model-attempt-foreign"},
            "conflicts with its pending tool round",
        ),
        ({"tool_round_id": "tool-round-foreign"}, "conflicts with its pending tool round"),
        ({"model_step": 999}, "conflicts with its pending tool round"),
        ({"tool_call_id": "call-foreign"}, "conflicts with its pending tool call"),
        ({"tool_name": "foreign-tool"}, "conflicts with its pending tool call"),
        (
            {"artifact_store_id": "replacement-artifact-store"},
            "conflicts with its registered environment authority",
        ),
    ],
)
def test_workspace_observation_recovery_rejects_foreign_authority_before_artifact_read(
    tmp_path,
    authority_update: dict[str, object],
    expected_error: str,
) -> None:
    async def run():
        store = _WorkspaceObservationProcessLossStore(phase="artifact-revision-after-published")
        artifact_root = tmp_path / "artifacts"
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(_BulkProvider(), default=True)
        first_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=LocalArtifactStore(
                    artifact_root,
                    store_id="artifact-store",
                ),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_BulkWriteTool()],
        )
        session_id = "session-cross-session-workspace-observation"
        with pytest.raises(_WorkspaceObservationProcessLoss):
            await collect_events(
                first_app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "write many files")],
                ),
            )

        def forge_cross_session_lifecycle(_session, checkpoint):
            assert checkpoint is not None
            copied = copy.deepcopy(checkpoint)
            observations = copied["workspace_observations"]
            original_window_id, lifecycle = next(iter(observations.items()))
            foreign = copy.deepcopy(lifecycle)
            foreign.update(authority_update)
            if authority_update:
                # Exercise each conflicting authority independently.  Adding a
                # second lifecycle here would make the duplicate-owner defect
                # authoritative before environment-backed fields (such as the
                # artifact-store identity) can be checked.
                observations[original_window_id] = foreign
            else:
                foreign["window_id"] = "zz-foreign-workspace-observation"
                observations[foreign["window_id"]] = foreign
            return copied

        await store.transform_checkpoint(session_id, forge_cross_session_lifecycle)
        store.failed = True
        await store.release_run_fence(session_id)
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        forged_checkpoint = await store.load_checkpoint(session_id)
        tracking_store = _ReadTrackingArtifactStore(
            artifact_root,
            store_id="artifact-store",
        )
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_app.register_provider(_BulkProvider(), default=True)
        recovery_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=tracking_store,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_BulkWriteTool()],
        )
        with pytest.raises(RuntimeError, match=expected_error):
            await recovery_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
        durable = await store.query_events(EventQuery(session_id=session_id))
        checkpoint = await store.load_checkpoint(session_id)
        return (
            tracking_store,
            [record.event for record in durable],
            forged_checkpoint,
            checkpoint,
        )

    tracking_store, durable_events, forged_checkpoint, checkpoint = asyncio.run(run())

    assert tracking_store.reads == 0
    assert not any(
        event.type == EventType.WORKSPACE_OBSERVATION_FINALIZED for event in durable_events
    )
    assert checkpoint is not None
    assert len(checkpoint["workspace_observations"]) == (1 if authority_update else 2)
    assert forged_checkpoint is not None
    assert checkpoint == forged_checkpoint


@pytest.mark.parametrize("original_is_runtime_builtin", [True, False])
def test_workspace_observation_recovery_rejects_same_name_observer_authority_change(
    tmp_path,
    original_is_runtime_builtin: bool,
) -> None:
    impostor_binding_type = type(
        "DeterministicWorkspaceBinding",
        (DeterministicWorkspaceBinding,),
        {},
    )

    async def run():
        store = _WorkspaceObservationProcessLossStore(phase="intent")
        first_binding = (
            DeterministicWorkspaceBinding()
            if original_is_runtime_builtin
            else impostor_binding_type()
        )
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(_ScriptedProvider(), default=True)
        first_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=first_binding,
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        session_id = (
            "session-workspace-observer-authority-runtime-to-configured"
            if original_is_runtime_builtin
            else "session-workspace-observer-authority-configured-to-runtime"
        )
        with pytest.raises(_WorkspaceObservationProcessLoss, match="intent"):
            await collect_events(
                first_app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create a file")],
                ),
            )

        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        lifecycle = next(iter(checkpoint["workspace_observations"].values()))
        assert lifecycle["observer"] == "DeterministicWorkspaceBinding"
        assert lifecycle["observer_authority"] == (
            "runtime_builtin" if original_is_runtime_builtin else "configured"
        )

        store.failed = True
        await store.release_run_fence(session_id)
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        recovery_binding = (
            impostor_binding_type()
            if original_is_runtime_builtin
            else DeterministicWorkspaceBinding()
        )
        recovery_provider = _ScriptedProvider()
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_app.register_provider(recovery_provider, default=True)
        recovery_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=recovery_binding,
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        with pytest.raises(
            RuntimeError,
            match="conflicts with its registered environment authority",
        ):
            await recovery_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
        durable = await store.query_events(EventQuery(session_id=session_id))
        return recovery_provider, [record.event for record in durable]

    recovery_provider, durable_events = asyncio.run(run())

    assert recovery_provider.requests == 0
    assert not any(
        event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED for event in durable_events
    )


def test_workspace_observation_recovery_rejects_foreign_durable_tool_outcome(
    tmp_path,
) -> None:
    async def run():
        workspace_root = tmp_path / "workspace"
        artifact_root = tmp_path / "artifacts"
        workspace_root.mkdir()
        store = _WorkspaceObservationProcessLossStore(phase="after-capture")
        first_provider = _BulkProvider()
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(first_provider, default=True)
        first_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=LocalArtifactStore(
                    artifact_root,
                    store_id="artifact-store",
                ),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_BulkWriteTool()],
        )
        session_id = "session-foreign-workspace-tool-outcome"
        with pytest.raises(_WorkspaceObservationProcessLoss):
            await collect_events(
                first_app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create a file")],
                ),
            )

        foreign_event: Event | None = None

        def forge_tool_outcome(_session, checkpoint):
            nonlocal foreign_event
            assert checkpoint is not None
            copied = copy.deepcopy(checkpoint)
            pending_round = copied["pending_tool_round"]
            staged_event = Event.model_validate(pending_round["staged_terminals"][0]["event"])
            foreign_event = staged_event.model_copy(
                update={
                    "id": "foreign-workspace-tool-outcome",
                    "agent_name": "foreign-agent",
                },
                deep=True,
            )
            pending_round["staged_terminals"] = []
            lifecycle = next(iter(copied["workspace_observations"].values()))
            lifecycle["tool_outcome_event_id"] = foreign_event.id
            lifecycle["tool_outcome_event_digest"] = workspace_observation_event_digest(
                foreign_event
            )
            return copied

        await store.transform_checkpoint(session_id, forge_tool_outcome)
        assert foreign_event is not None
        await store.append_event(session_id, foreign_event)
        store.failed = True
        await store.release_run_fence(session_id)
        await store.update_status(session_id, SessionStatus.INTERRUPTED)

        recovery_provider = _BulkProvider()
        tracking_store = _ReadTrackingArtifactStore(
            artifact_root,
            store_id="artifact-store",
        )
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_app.register_provider(recovery_provider, default=True)
        recovery_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=tracking_store,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[_BulkWriteTool()],
        )
        with pytest.raises(
            ValueError,
            match="Tool-round durable event has a conflicting agent identity",
        ):
            await recovery_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
        durable = await store.query_events(EventQuery(session_id=session_id))
        return (
            recovery_provider,
            tracking_store,
            [record.event for record in durable],
        )

    recovery_provider, tracking_store, durable_events = asyncio.run(run())

    assert recovery_provider.requests == 0
    assert tracking_store.reads == 0
    finalized = next(
        event for event in durable_events if event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
    )
    assert finalized.payload["status"] == "ambiguous"
    assert finalized.payload["detail_code"] == "durable_tool_outcome_evidence_missing"


@pytest.mark.parametrize(
    "conflict",
    [
        "interaction_id",
        "agent_name",
        "window_id",
        "tool_outcome_event_id",
        "manifest_artifact",
    ],
)
def test_workspace_observation_recovery_rejects_foreign_delta_event(
    tmp_path,
    conflict: str,
) -> None:
    async def run():
        store = _WorkspaceObservationProcessLossStore(phase="delta-publication")
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(_ScriptedProvider(), default=True)
        first_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        session_id = f"session-foreign-workspace-delta-{conflict}"
        with pytest.raises(_WorkspaceObservationProcessLoss):
            await collect_events(
                first_app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create a file")],
                ),
            )
        assert store.workspace_delta_event_id is not None
        records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_id=store.workspace_delta_event_id,
            )
        )
        assert len(records) == 1
        delta_event = records[0].event
        event_update: dict[str, object] = {
            "id": f"foreign-delta-{conflict}",
        }
        payload = dict(delta_event.payload)
        if conflict == "interaction_id":
            event_update["interaction_id"] = "foreign-interaction"
        elif conflict == "agent_name":
            event_update["agent_name"] = "foreign-agent"
        elif conflict == "window_id":
            payload["window_id"] = "foreign-window"
        elif conflict == "tool_outcome_event_id":
            payload["tool_outcome_event_id"] = "foreign-tool-outcome"
        else:
            payload.update(
                {
                    "manifest_artifact_id": "foreign-artifact",
                    "manifest_artifact_sha256": "a" * 64,
                    "manifest_artifact_size_bytes": 1,
                }
            )
        event_update["payload"] = payload
        foreign_delta = delta_event.model_copy(update=event_update, deep=True)
        await store.append_event(session_id, foreign_delta)

        def point_lifecycle_to_foreign_delta(_session, checkpoint):
            assert checkpoint is not None
            copied = copy.deepcopy(checkpoint)
            lifecycle = next(iter(copied["workspace_observations"].values()))
            lifecycle["mutation_event_id"] = foreign_delta.id
            lifecycle["mutation_event_digest"] = workspace_observation_event_digest(foreign_delta)
            return copied

        await store.transform_checkpoint(session_id, point_lifecycle_to_foreign_delta)
        store.failed = True
        await store.release_run_fence(session_id)
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        recovery_provider = _ScriptedProvider()
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_app.register_provider(recovery_provider, default=True)
        recovery_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                runner=LocalRunner(tmp_path),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[ExecCommandTool()],
        )
        recovery = await recovery_app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        durable = await store.query_events(EventQuery(session_id=session_id))
        return recovery_provider, recovery, [record.event for record in durable]

    recovery_provider, recovery, durable_events = asyncio.run(run())

    assert recovery_provider.requests == 0
    assert IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION in recovery.actions
    finalized = next(
        event for event in durable_events if event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
    )
    assert finalized.payload["status"] == "ambiguous"
    assert finalized.payload["detail_code"] == "workspace_delta_evidence_conflict"


def test_workspace_observation_recovery_validates_delta_before_artifact_read(
    tmp_path,
) -> None:
    async def run():
        workspace_root = tmp_path / "workspace"
        artifact_root = tmp_path / "artifacts"
        workspace_root.mkdir()
        store = _WorkspaceObservationProcessLossStore(phase="delta-publication")
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(_BulkProvider(), default=True)
        first_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=LocalArtifactStore(
                    artifact_root,
                    store_id="artifact-store",
                ),
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="bulk-model"),
            tools=[_BulkWriteTool()],
        )
        session_id = "session-foreign-bulk-workspace-delta"
        with pytest.raises(_WorkspaceObservationProcessLoss):
            await collect_events(
                first_app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "write many files")],
                ),
            )
        assert store.workspace_delta_event_id is not None
        records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_id=store.workspace_delta_event_id,
            )
        )
        assert len(records) == 1
        payload = dict(records[0].event.payload)
        payload["window_id"] = "foreign-window"
        foreign_delta = records[0].event.model_copy(
            update={"id": "foreign-bulk-delta", "payload": payload},
            deep=True,
        )
        await store.append_event(session_id, foreign_delta)

        def point_lifecycle_to_foreign_delta(_session, checkpoint):
            assert checkpoint is not None
            copied = copy.deepcopy(checkpoint)
            lifecycle = next(iter(copied["workspace_observations"].values()))
            lifecycle["mutation_event_id"] = foreign_delta.id
            lifecycle["mutation_event_digest"] = workspace_observation_event_digest(foreign_delta)
            return copied

        await store.transform_checkpoint(session_id, point_lifecycle_to_foreign_delta)
        store.failed = True
        await store.release_run_fence(session_id)
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        tracking_store = _ReadTrackingArtifactStore(
            artifact_root,
            store_id="artifact-store",
        )
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_app.register_provider(_BulkProvider(), default=True)
        recovery_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(workspace_root, workspace_id="workspace"),
                artifact_store=tracking_store,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="bulk-model"),
            tools=[_BulkWriteTool()],
        )
        recovery = await recovery_app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        durable = await store.query_events(EventQuery(session_id=session_id))
        return tracking_store, recovery, [record.event for record in durable]

    tracking_store, recovery, durable_events = asyncio.run(run())

    assert tracking_store.reads == 0
    assert IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION in recovery.actions
    finalized = next(
        event for event in durable_events if event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
    )
    assert finalized.payload["status"] == "ambiguous"
    assert finalized.payload["detail_code"] == "workspace_delta_evidence_conflict"


def test_workspace_receipt_closes_before_tool_cancellation_propagates(tmp_path) -> None:
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_CancelProvider(), default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
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
            _portable_environment_spec("local"),
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
            _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
                _portable_environment_spec("local"),
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
