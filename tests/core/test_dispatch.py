from __future__ import annotations

import asyncio
import base64
import os
import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, NamedTuple
from uuid import uuid4

import pytest
from pydantic import SecretStr
from tests.core._execution_profile_fixtures import (
    checkpoint_with_rebound_test_invocation_profile,
    runtime_interaction_started_event,
)

import cayu.runtime._session_engine as session_engine_module
from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    ExecutionProfileBehaviorIdentity,
    Message,
    ProviderStatePart,
    TextPart,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    DispatchHandle,
    DispatchRequest,
    DispatchStatus,
    EventQuery,
    ExecutionProfileComponentClass,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
    ForkSessionRequest,
    IncompleteSessionRecoveryAction,
    InMemorySessionStore,
    InMemoryTaskStore,
    InvocationOrigin,
    InvocationOriginTrust,
    ModelTarget,
    NativeStructuredOutputUnsupported,
    ResumeRequest,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    SessionExecutionSource,
    SessionIdentity,
    SessionInvocation,
    SessionInvocationBinding,
    SessionStatus,
    SessionStatusConflict,
    StructuredOutputSpec,
    Task,
    TaskClaimLost,
    TaskCreate,
    TaskExecutionSource,
    TaskQuery,
    TaskStatus,
    TaskStore,
    TaskStoreDispatcher,
    TaskTerminalizationRequest,
    TaskTerminalKind,
)
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._diagnostics import ExceptionDiagnostic, exception_diagnostic
from cayu.runtime._recovery_coordinator import ModelCompletionBoundaryReconciliation
from cayu.runtime.approvals import PendingToolCallApproval
from cayu.runtime.dispatch import (
    _STALLED_RECOVERED_ACTIONS,
    _dispatch_status_after_event,
    _new_queued_dispatch_envelope,
    _queued_dispatch_task_id,
    _QueuedDispatchEnvelope,
    _QueuedDispatchSettlement,
    _QueuedDispatchSettlementState,
    copy_dispatch_request,
)
from cayu.runtime.execution_profiles import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
    build_execution_profile_identity,
    checkpoint_with_active_invocation_execution_profile,
    execution_profile_from_session_metadata,
)
from cayu.runtime.public_authority import (
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
)
from cayu.runtime.sessions import (
    QueuedDispatchTerminalReceipt,
    _checkpoint_with_session_run_operation,
)
from cayu.runtime.tasks import task_create_with_runtime_invocation
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.vaults import REDACTED_SECRET, SecretRedactor

_DISPATCH_TASK_TYPE = "cayu.dispatch"
_TEST_DISPATCH_ROOTS: dict[str, str] = {}


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self, events: list[list[ModelStreamEvent]]) -> None:
        self.event_batches = events
        self.requests: list[ModelRequest] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:dispatch:fake-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        batch_index = len(self.requests)
        self.requests.append(request)
        if batch_index >= len(self.event_batches):
            raise AssertionError(f"No fake provider event batch for request {batch_index}")
        for event in self.event_batches[batch_index]:
            yield event


class NativeStructuredOutputFakeProvider(FakeProvider):
    supports_native_structured_output = True


class ModelAwareOptionsFakeProvider(FakeProvider):
    def request_fingerprint_options(self, request: ModelRequest) -> dict[str, Any]:
        return {"test": {"effective_model": request.model}}


class PortableMessageRejectingFakeProvider(FakeProvider):
    def preflight_portable_messages(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> None:
        del model, messages, tools
        raise ValueError("Target provider cannot render the portable transcript.")


class ProfileTool(Tool):
    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(
            name=name,
            description="Profile-bound dispatch test tool.",
            input_schema={"type": "object", "properties": {}},
            effect="none",
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name=f"tests:dispatch:profile-tool:{name}",
                behavior_version="1",
                implementation_version="1",
            ),
        )
        super().__init__()
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        self.calls += 1
        return ToolResult(content="ok")


class Harness(NamedTuple):
    app: CayuApp
    store: InMemorySessionStore | SQLiteSessionStore
    tasks: TaskStore
    provider: FakeProvider
    dispatcher: TaskStoreDispatcher


class _SecretFreeDispatchRuntime:
    """Test runtime implementing the mandatory durable-dispatch boundary."""

    @staticmethod
    def redact_dispatch_request(request: DispatchRequest) -> DispatchRequest:
        return copy_dispatch_request(request)

    @staticmethod
    def redact_json(value: Any) -> Any:
        return SecretRedactor().redact_json(value)

    @staticmethod
    def redact_exception_diagnostic(
        error: BaseException,
        *,
        empty_message: str,
        nonportable_message: str,
    ) -> ExceptionDiagnostic:
        return exception_diagnostic(
            error,
            empty_message=empty_message,
            nonportable_message=nonportable_message,
        )

    @staticmethod
    async def session_invocation_for_dispatch(session_id: str) -> SessionInvocationBinding:
        return SessionInvocationBinding(
            id=session_id,
            invocation=SessionInvocation(
                origin=InvocationOrigin(trust=InvocationOriginTrust.UNATTRIBUTED),
                root_invocation_id=_TEST_DISPATCH_ROOTS.setdefault(
                    session_id,
                    str(uuid4()),
                ),
                root_session_id=session_id,
                source=SessionExecutionSource.SDK_RUN,
            ),
        )

    async def _prepare_queued_dispatch(
        self,
        request: DispatchRequest,
        *,
        queue_task_id: str,
    ) -> _QueuedDispatchEnvelope:
        return _test_dispatch_envelope(request, queue_task_id=queue_task_id)

    async def _dispatch_queued(
        self,
        envelope: _QueuedDispatchEnvelope,
    ) -> AsyncIterator[Event]:
        status = DispatchStatus.SUBMITTED
        async for event in self.dispatch_inline(envelope.request):
            status = _dispatch_status_after_event(event, fallback=status)
            yield event
        if status is not DispatchStatus.SUBMITTED:
            self._test_terminal_status = status

    async def _queued_dispatch_requests_match(
        self,
        existing: DispatchRequest,
        candidate: DispatchRequest,
    ) -> bool:
        return existing == candidate

    async def _queued_dispatch_settlement_state(
        self,
        envelope: _QueuedDispatchEnvelope,
    ) -> _QueuedDispatchSettlement:
        del envelope
        terminal_status = getattr(self, "_test_terminal_status", None)
        if terminal_status is None:
            return _QueuedDispatchSettlement(_QueuedDispatchSettlementState.NOT_ADMITTED)
        return _QueuedDispatchSettlement(
            _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE,
            terminal_status=terminal_status,
        )

    async def _list_queued_dispatch_terminal_receipts(self, query):
        del query
        return []

    async def _acknowledge_queued_dispatch(
        self,
        envelope: _QueuedDispatchEnvelope,
        *,
        dispatch_status: DispatchStatus,
        receipt=None,
    ) -> None:
        del envelope, dispatch_status, receipt


def _batch(text: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def _build(
    batches: list[list[ModelStreamEvent]],
    *,
    task_store: TaskStore | None = None,
    task_type: str = _DISPATCH_TASK_TYPE,
    recover_stalled_sessions_after_seconds: int | None = None,
    secret_redactor: SecretRedactor | None = None,
    session_store: InMemorySessionStore | SQLiteSessionStore | None = None,
    runtime_hooks: list[RuntimeHook] | None = None,
) -> Harness:
    store = InMemorySessionStore() if session_store is None else session_store
    tasks = task_store if task_store is not None else InMemoryTaskStore()
    provider = FakeProvider(batches)
    dispatcher = TaskStoreDispatcher(
        tasks,
        task_type=task_type,
        recover_stalled_sessions_after_seconds=recover_stalled_sessions_after_seconds,
    )
    app = CayuApp(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        enable_logging=False,
        secret_redactor=secret_redactor,
        runtime_hooks=[] if runtime_hooks is None else runtime_hooks,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return Harness(app, store, tasks, provider, dispatcher)


def _configured_app(
    *,
    session_store,
    task_store,
    dispatcher: TaskStoreDispatcher,
    provider: ModelProvider,
    tools: list[Tool] | None = None,
) -> CayuApp:
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        dispatcher=dispatcher,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[] if tools is None else tools,
    )
    return app


def _create_resumable_session(app: CayuApp, session_id: str) -> None:
    async def run() -> None:
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "first request")],
            )
        ):
            pass

    asyncio.run(run())


def _dispatch_request(session_id: str, dispatch_id: str) -> DispatchRequest:
    return DispatchRequest(
        session_id=session_id,
        dispatch_id=dispatch_id,
        messages=[Message.text("user", "queued work")],
    )


def _test_dispatch_envelope(
    request: DispatchRequest,
    *,
    queue_task_id: str,
) -> _QueuedDispatchEnvelope:
    profile = build_execution_profile_identity(
        runtime_name="test",
        runtime_version="1",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt=None,
        direct_tools=(),
        tool_catalogue_revision=f"sha256:{'c' * 64}",
    )
    return _new_queued_dispatch_envelope(
        queue_task_id=queue_task_id,
        request=request,
        session_instance_fingerprint="1" * 64,
        source_profile=profile,
        required_profile=profile,
    )


def test_submit_enqueues_pending_task_without_running() -> None:
    # Only the initial run consumes a batch; the dispatch must NOT run on submit.
    h = _build([_batch("first answer")])
    _create_resumable_session(h.app, "sess_submit")

    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_submit", "d_submit")))

    assert handle.status == DispatchStatus.SUBMITTED
    assert handle.backend == "task_store"
    assert handle.dispatch_id == "d_submit"
    # Not run yet: the provider only saw the initial request.
    assert len(h.provider.requests) == 1

    # The work was persisted as a claimable (session-unbound) PENDING dispatch task that
    # carries the serialized request; the target session_id rides in the payload.
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))
    assert task is not None
    assert task.type == _DISPATCH_TASK_TYPE
    assert task.status == TaskStatus.PENDING
    assert task.session_id is None
    assert task.input["dispatch"]["request"]["session_id"] == "sess_submit"
    assert task.input["dispatch"]["request"]["dispatch_id"] == "d_submit"
    assert "operation_kind" not in task.input["dispatch"]
    assert "prepared_subagent" not in task.input["dispatch"]
    assert (
        task.input["dispatch"]["required_profile"]["fingerprint"]
        == (handle.metadata["required_execution_profile_fingerprint"])
    )
    session = asyncio.run(h.store.load("sess_submit"))
    assert session is not None
    assert task.invocation.origin == session.invocation.origin
    assert task.invocation.root_invocation_id == session.invocation.root_invocation_id
    assert task.invocation.root_session_id == session.invocation.root_session_id
    assert task.invocation.source is TaskExecutionSource.TASK_DISPATCH


def test_worker_accepts_revision_40_resume_envelope_without_new_default_fields() -> None:
    h = _build([_batch("first answer"), _batch("queued answer")])
    session_id = "sess_prior_revision_40_envelope"
    _create_resumable_session(h.app, session_id)

    async def scenario() -> tuple[DispatchHandle | None, Task]:
        request = _dispatch_request(session_id, "d_prior_revision_40_envelope")
        queue_task_id = _queued_dispatch_task_id(request, task_type=_DISPATCH_TASK_TYPE)
        envelope = await h.app._prepare_queued_dispatch(
            request,
            queue_task_id=queue_task_id,
        )
        persisted = envelope.model_dump(mode="json")
        persisted.pop("operation_kind")
        persisted.pop("prepared_subagent")
        binding = await h.app.session_invocation_for_dispatch(session_id)
        await h.tasks.create_task(
            task_create_with_runtime_invocation(
                TaskCreate(
                    task_id=queue_task_id,
                    type=_DISPATCH_TASK_TYPE,
                    parent_task_id=request.task_id,
                    input={"dispatch": persisted},
                ),
                source=TaskExecutionSource.TASK_DISPATCH,
                session_invocation=binding,
            )
        )
        handle = await h.dispatcher.process_next(
            h.app,
            worker_id="worker_prior_revision_40_envelope",
        )
        task = await h.tasks.load_task(queue_task_id)
        assert task is not None
        return handle, task

    handle, task = asyncio.run(scenario())

    assert handle is not None
    assert handle.status is DispatchStatus.COMPLETED
    assert task.status is TaskStatus.COMPLETED
    assert len(h.provider.requests) == 2


def test_submit_redacts_workload_secrets_before_durable_dispatch_write() -> None:
    secret = "dispatch-boundary-canary"
    replacement_secret = "dispatch-boundary-replacement-canary"
    h = _build(
        [_batch("first answer")],
        secret_redactor=SecretRedactor([secret, replacement_secret]),
    )
    _create_resumable_session(h.app, "sess_dispatch_redaction")
    request = DispatchRequest(
        session_id="sess_dispatch_redaction",
        dispatch_id="d_redacted",
        messages=[Message.text("user", f"queued {secret}")],
        metadata={"note": f"metadata {secret}"},
    )

    handle = asyncio.run(h.app.dispatch(request))
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))

    assert task is not None
    serialized = str(task.input)
    assert secret not in serialized
    assert REDACTED_SECRET in serialized

    replacement = request.model_copy(
        update={
            "messages": [Message.text("user", f"queued {replacement_secret}")],
            "metadata": {"note": f"metadata {replacement_secret}"},
        }
    )
    replayed = asyncio.run(h.app.dispatch(replacement))
    assert replayed.metadata["queue_task_id"] == handle.metadata["queue_task_id"]
    assert replayed.metadata["idempotent_submission"] is True


def test_exact_submit_retry_reuses_one_profile_bound_queue_task() -> None:
    h = _build([_batch("initial"), _batch("queued")])
    _create_resumable_session(h.app, "sess_submit_retry")
    request = _dispatch_request("sess_submit_retry", "d_submit_retry")

    first = asyncio.run(h.app.dispatch(request))
    second = asyncio.run(h.app.dispatch(request))

    assert second.metadata["queue_task_id"] == first.metadata["queue_task_id"]
    assert second.metadata["dispatch_operation_id"] == first.metadata["dispatch_operation_id"]
    assert second.metadata["idempotent_submission"] is True
    tasks = asyncio.run(h.tasks.list_tasks(TaskQuery(type=_DISPATCH_TASK_TYPE)))
    assert len(tasks) == 1

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_submit_retry"))
    assert result is not None
    assert result.status is DispatchStatus.COMPLETED
    assert len(h.provider.requests) == 2


def test_queued_dispatch_operation_identity_binds_source_and_governed_profiles() -> None:
    request = _dispatch_request("sess_profile_tuple", "d_profile_tuple")
    source_profile = build_execution_profile_identity(
        runtime_name="test",
        runtime_version="1",
        provider_name="fake",
        model="source-model",
        durable_system_prompt=None,
        direct_tools=(),
        tool_catalogue_revision=f"sha256:{'c' * 64}",
    )
    required_profile = build_execution_profile_identity(
        runtime_name="test",
        runtime_version="1",
        provider_name="fake",
        model="governed-model",
        durable_system_prompt=None,
        direct_tools=(),
        tool_catalogue_revision=f"sha256:{'c' * 64}",
    )
    envelope = _new_queued_dispatch_envelope(
        queue_task_id="queue-profile-tuple",
        request=request,
        session_instance_fingerprint="2" * 64,
        source_profile=source_profile,
        required_profile=required_profile,
    )
    tampered = envelope.model_dump(mode="json")
    tampered["source_profile"] = required_profile.model_dump(mode="json")

    with pytest.raises(ValueError, match="conflicts with its authority tuple"):
        _QueuedDispatchEnvelope.model_validate(tampered)


def test_submit_retry_survives_public_session_alias_rotation_without_private_digest() -> None:
    def key(byte: int) -> SecretStr:
        encoded = base64.urlsafe_b64encode(bytes([byte]) * 32).decode("ascii").rstrip("=")
        return SecretStr(encoded)

    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="new",
            keys={"old": key(1), "new": key(2)},
        )
    )
    store = InMemorySessionStore(public_authority_alias_codec=codec)
    h = _build(
        [_batch("initial"), _batch("queued")],
        session_store=store,
    )
    private_session_id = "short-private-session-id"
    _create_resumable_session(h.app, private_session_id)
    new_alias, old_alias = codec.aliases(private_session_id, field_name="session_id")
    dispatch_id = "d_alias_rotation_retry"

    first = asyncio.run(h.app.dispatch(_dispatch_request(old_alias, dispatch_id)))
    second = asyncio.run(h.app.dispatch(_dispatch_request(new_alias, dispatch_id)))

    assert second.metadata["queue_task_id"] == first.metadata["queue_task_id"]
    assert second.metadata["dispatch_operation_id"] == first.metadata["dispatch_operation_id"]
    assert second.metadata["idempotent_submission"] is True
    task = asyncio.run(h.tasks.load_task(first.metadata["queue_task_id"]))
    assert task is not None
    serialized = str(task.input)
    assert private_session_id not in serialized
    assert "resolved_session_id_sha256" not in serialized
    assert sha256(private_session_id.encode("utf-8")).hexdigest() not in serialized
    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_alias_rotation"))
    assert result is not None
    assert result.status is DispatchStatus.COMPLETED
    assert len(h.provider.requests) == 2


def test_reused_dispatch_id_with_different_session_fails_closed() -> None:
    h = _build([_batch("first"), _batch("second")])
    _create_resumable_session(h.app, "sess_dispatch_identity_a")
    _create_resumable_session(h.app, "sess_dispatch_identity_b")
    dispatch_id = "d_cross_session_conflict"
    first = asyncio.run(h.app.dispatch(_dispatch_request("sess_dispatch_identity_a", dispatch_id)))

    with pytest.raises(RuntimeError, match="conflicts with the queued dispatch authority"):
        asyncio.run(h.app.dispatch(_dispatch_request("sess_dispatch_identity_b", dispatch_id)))

    tasks = asyncio.run(h.tasks.list_tasks(TaskQuery(type=_DISPATCH_TASK_TYPE)))
    assert [task.id for task in tasks] == [first.metadata["queue_task_id"]]


def test_queued_dispatch_rejects_recreated_session_with_same_id_and_profile() -> None:
    h = _build([_batch("original"), _batch("replacement")])
    session_id = "sess_recreated_queued_target"
    _create_resumable_session(h.app, session_id)
    original = asyncio.run(h.store.load(session_id))
    assert original is not None
    submitted = asyncio.run(
        h.app.dispatch(_dispatch_request(session_id, "d_recreated_queued_target"))
    )

    asyncio.run(h.store.delete_session(session_id))
    _create_resumable_session(h.app, session_id)
    replacement = asyncio.run(h.store.load(session_id))
    assert replacement is not None
    assert replacement.invocation.root_invocation_id != (original.invocation.root_invocation_id)
    assert execution_profile_from_session_metadata(replacement.metadata) == (
        execution_profile_from_session_metadata(original.metadata)
    )
    with pytest.raises(RuntimeError, match="target session instance changed"):
        asyncio.run(h.app.dispatch(_dispatch_request(session_id, "d_recreated_queued_target")))

    result = asyncio.run(
        h.dispatcher.process_next(h.app, worker_id="worker_recreated_queued_target")
    )
    task = asyncio.run(h.tasks.load_task(submitted.metadata["queue_task_id"]))

    assert result is not None
    assert result.status is DispatchStatus.FAILED
    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert task.error is not None
    assert task.error["error"] == "Queued dispatch target session instance changed."
    assert task.error["dispatch_operation_id"] == submitted.metadata["dispatch_operation_id"]
    assert (
        task.error["session_instance_fingerprint"]
        == submitted.metadata["session_instance_fingerprint"]
    )
    assert (
        task.error["source_execution_profile_fingerprint"]
        == submitted.metadata["source_execution_profile_fingerprint"]
    )
    assert (
        task.error["required_execution_profile_fingerprint"]
        == submitted.metadata["required_execution_profile_fingerprint"]
    )
    assert len(h.provider.requests) == 2


def test_terminal_dispatch_retry_rejects_recreated_session_instance() -> None:
    h = _build([_batch("original"), _batch("queued"), _batch("replacement")])
    session_id = "sess_recreated_terminal_dispatch_target"
    request = _dispatch_request(session_id, "d_recreated_terminal_dispatch_target")
    _create_resumable_session(h.app, session_id)
    submitted = asyncio.run(h.app.dispatch(request))
    completed = asyncio.run(
        h.dispatcher.process_next(
            h.app,
            worker_id="worker_recreated_terminal_dispatch_target",
        )
    )
    assert completed is not None
    assert completed.status is DispatchStatus.COMPLETED
    task = asyncio.run(h.tasks.load_task(submitted.metadata["queue_task_id"]))
    assert task is not None
    assert task.status is TaskStatus.COMPLETED

    asyncio.run(h.store.delete_session(session_id))
    _create_resumable_session(h.app, session_id)

    with pytest.raises(RuntimeError, match="target session instance changed"):
        asyncio.run(h.app.dispatch(request))

    assert len(h.provider.requests) == 3


def test_exact_submit_retry_reuses_original_envelope_after_model_target_adoption() -> None:
    h = _build([_batch("initial"), _batch("queued")])
    _create_resumable_session(h.app, "sess_submit_target_retry")
    session = asyncio.run(h.store.load("sess_submit_target_retry"))
    assert session is not None
    baseline_profile = execution_profile_from_session_metadata(session.metadata)
    request = _dispatch_request("sess_submit_target_retry", "d_submit_target_retry").model_copy(
        update={"target": ModelTarget(provider_name="fake", model="upgraded-model")},
        deep=True,
    )

    first = asyncio.run(h.app.dispatch(request))
    queued_task = asyncio.run(h.tasks.load_task(first.metadata["queue_task_id"]))
    assert queued_task is not None
    envelope = _QueuedDispatchEnvelope.model_validate(queued_task.input["dispatch"])
    assert envelope.source_profile == baseline_profile
    assert envelope.required_profile != envelope.source_profile
    assert first.metadata["source_execution_profile_fingerprint"] == baseline_profile.fingerprint
    assert (
        first.metadata["required_execution_profile_fingerprint"]
        == envelope.required_profile.fingerprint
    )
    completed = asyncio.run(
        h.dispatcher.process_next(h.app, worker_id="worker_submit_target_retry")
    )
    second = asyncio.run(h.app.dispatch(request))

    assert completed is not None
    assert completed.status is DispatchStatus.COMPLETED
    assert second.metadata["queue_task_id"] == first.metadata["queue_task_id"]
    assert second.metadata["dispatch_operation_id"] == first.metadata["dispatch_operation_id"]
    assert second.metadata["idempotent_submission"] is True
    tasks = asyncio.run(h.tasks.list_tasks(TaskQuery(type=_DISPATCH_TASK_TYPE)))
    assert len(tasks) == 1


def test_sqlite_target_adoption_redelivery_accepts_governed_profile(tmp_path) -> None:
    async def scenario() -> tuple[DispatchStatus, TaskStatus, list[dict], str, str]:
        session_path = tmp_path / "target-recovery-sessions.sqlite"
        task_path = tmp_path / "target-recovery-tasks.sqlite"
        sessions = SQLiteSessionStore(session_path)
        tasks = SQLiteTaskStore(task_path)
        dispatcher = TaskStoreDispatcher(tasks, task_type=_DISPATCH_TASK_TYPE)
        producer_tool = ProfileTool("profile_tool")
        producer = _configured_app(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            provider=FakeProvider([_batch("initial")]),
            tools=[producer_tool],
        )
        session_id = "sess_target_recovery"
        interaction_id = "interaction_target_recovery"
        try:
            async for _ in producer.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "initial")],
                )
            ):
                pass
            request = _dispatch_request(session_id, "d_target_recovery").model_copy(
                update={"target": ModelTarget(provider_name="fake", model="upgraded-model")},
                deep=True,
            )
            submitted = await producer.dispatch(request)
            queued_task = await tasks.load_task(submitted.metadata["queue_task_id"])
            session = await sessions.load(session_id)
            checkpoint = await sessions.load_checkpoint(session_id)
            assert queued_task is not None
            assert session is not None
            active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            assert active_profile is not None
            envelope = _QueuedDispatchEnvelope.model_validate(queued_task.input["dispatch"])
            assert envelope.source_profile != envelope.required_profile

            recovery_checkpoint = checkpoint_with_active_invocation_execution_profile(
                checkpoint,
                session_id=session_id,
                interaction_id=interaction_id,
                run_epoch=session.run_epoch - 1,
                profile=envelope.required_profile,
                expected=active_profile,
            )
            recovery_checkpoint.pop("last_model_step_publication", None)
            pending_round = tool_round_recovery.PendingToolRound(
                model_step_id=f"mstep_{'1' * 32}",
                model_attempt_id=f"matt_{'2' * 32}",
                tool_round_id=f"tround_{'3' * 32}",
                agent_name="assistant",
                execution_profile_fingerprint=envelope.required_profile.fingerprint,
                tool_calls=[
                    PendingToolCallApproval(
                        tool_call_id="call-target-recovery",
                        tool_name="profile_tool",
                        arguments={},
                    )
                ],
            )
            recovery_checkpoint[tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY] = (
                pending_round.model_dump(mode="json")
            )
            await sessions.checkpoint(session_id, recovery_checkpoint)
            await sessions.append_event(
                session_id,
                runtime_interaction_started_event(
                    producer,
                    session_id=session_id,
                    interaction_id=interaction_id,
                    agent_name="assistant",
                ),
            )
        finally:
            await sessions.close()
            await tasks.close()

        # Model/profile admission committed before the worker disappeared. Reopen the
        # persistent stores from the resulting recovery checkpoint, with no live owner.
        connection = sqlite3.connect(session_path)
        try:
            connection.execute(
                "UPDATE cayu_sessions SET status = ?, model = ? WHERE id = ?",
                (SessionStatus.INTERRUPTED.value, "upgraded-model", session_id),
            )
            connection.commit()
        finally:
            connection.close()

        restarted_sessions = SQLiteSessionStore(session_path)
        restarted_tasks = SQLiteTaskStore(task_path)
        restarted_dispatcher = TaskStoreDispatcher(
            restarted_tasks,
            task_type=_DISPATCH_TASK_TYPE,
        )
        worker_tool = ProfileTool("profile_tool")
        worker_provider = FakeProvider([_batch("recovered")])
        worker = _configured_app(
            session_store=restarted_sessions,
            task_store=restarted_tasks,
            dispatcher=restarted_dispatcher,
            provider=worker_provider,
            tools=[worker_tool],
        )
        try:
            result = await restarted_dispatcher.process_next(
                worker,
                worker_id="worker_target_recovery",
            )
            assert result is not None
            terminal_task = await restarted_tasks.load_task(submitted.metadata["queue_task_id"])
            assert terminal_task is not None
            terminal_events = await restarted_sessions.load_events(session_id)
            failed_payloads = [
                event.payload for event in terminal_events if event.type == EventType.SESSION_FAILED
            ]
            return (
                result.status,
                terminal_task.status,
                failed_payloads,
                envelope.source_profile.fingerprint,
                envelope.required_profile.fingerprint,
            )
        finally:
            await restarted_sessions.close()
            await restarted_tasks.close()

    status, task_status, failed_payloads, source_fingerprint, required_fingerprint = asyncio.run(
        scenario()
    )

    assert status is DispatchStatus.INTERRUPTED
    assert task_status is TaskStatus.COMPLETED
    assert failed_payloads == []
    assert source_fingerprint != required_fingerprint


def test_submit_reconciles_queue_publication_acknowledgement_loss() -> None:
    class CommitThenRaiseCreateStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def create_task(self, request: TaskCreate):
            await super().create_task(request)
            raise ConnectionError("queue publication acknowledgement lost")

    tasks = CommitThenRaiseCreateStore()
    h = _build([_batch("initial")], task_store=tasks)
    _create_resumable_session(h.app, "sess_submit_ack_loss")

    submitted = asyncio.run(
        h.app.dispatch(_dispatch_request("sess_submit_ack_loss", "d_submit_ack_loss"))
    )

    assert submitted.metadata["idempotent_submission"] is True
    task = asyncio.run(tasks.load_task(submitted.metadata["queue_task_id"]))
    assert task is not None
    assert task.status is TaskStatus.PENDING


def test_submit_reconciles_terminal_peer_after_duplicate_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PeerTerminalThenRaiseStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        on_created = None

        async def create_task(self, request: TaskCreate):
            await super().create_task(request)
            assert self.on_created is not None
            await self.on_created()
            raise ConnectionError("duplicate queue publication")

    tasks = PeerTerminalThenRaiseStore()
    h = _build(
        [_batch("initial"), _batch("queued")],
        task_store=tasks,
    )
    session_id = "sess_duplicate_terminal_publication"
    _create_resumable_session(h.app, session_id)
    original_acknowledge = h.app._acknowledge_queued_dispatch

    async def scenario() -> tuple[DispatchHandle, Task, dict[str, Any] | None, int]:
        acknowledgement_calls = 0

        async def lose_first_acknowledgement(
            envelope: _QueuedDispatchEnvelope,
            *,
            dispatch_status: DispatchStatus,
            receipt: QueuedDispatchTerminalReceipt | None = None,
        ) -> None:
            nonlocal acknowledgement_calls
            acknowledgement_calls += 1
            if acknowledgement_calls == 1:
                raise ConnectionError("peer terminal acknowledgement lost")
            await original_acknowledge(
                envelope,
                dispatch_status=dispatch_status,
                receipt=receipt,
            )

        monkeypatch.setattr(
            h.app,
            "_acknowledge_queued_dispatch",
            lose_first_acknowledgement,
        )

        async def terminalize_peer() -> None:
            with pytest.raises(ConnectionError, match="peer terminal acknowledgement lost"):
                await h.dispatcher.process_next(h.app, worker_id="publication_peer")

        tasks.on_created = terminalize_peer
        submitted = await h.app.dispatch(
            _dispatch_request(session_id, "d_duplicate_terminal_publication")
        )
        terminal_task = await tasks.load_task(submitted.metadata["queue_task_id"])
        assert terminal_task is not None
        return (
            submitted,
            terminal_task,
            await h.store.load_checkpoint(session_id),
            acknowledgement_calls,
        )

    submitted, terminal_task, checkpoint, acknowledgement_calls = asyncio.run(scenario())

    assert submitted.metadata["idempotent_submission"] is True
    assert terminal_task.status is TaskStatus.COMPLETED
    assert checkpoint is None or "queued_dispatch_terminal_receipts" not in checkpoint
    assert acknowledgement_calls == 2
    assert len(h.provider.requests) == 2


def test_submit_rejects_protocol_minimal_runtime_before_persisting_secret_input() -> None:
    secret = "protocol-minimal-dispatch-input-secret"
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)

    class ProtocolMinimalRuntime:
        async def dispatch_inline(self, request: DispatchRequest) -> AsyncIterator[Event]:
            del request
            if False:
                yield

    request = DispatchRequest(
        session_id="sess_protocol_minimal",
        dispatch_id="d_protocol_minimal",
        messages=[Message.text("user", secret)],
    )

    with pytest.raises(TypeError, match="redact_dispatch_request"):
        asyncio.run(dispatcher.submit(ProtocolMinimalRuntime(), request))

    assert asyncio.run(tasks.list_tasks(TaskQuery(type=_DISPATCH_TASK_TYPE))) == []


def test_worker_rejects_protocol_minimal_runtime_before_persisting_secret_failure() -> None:
    secret = "protocol-minimal-dispatch-failure-secret"
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)

    class ProtocolMinimalRuntime:
        @staticmethod
        def redact_dispatch_request(request: DispatchRequest) -> DispatchRequest:
            return copy_dispatch_request(request)

        @staticmethod
        def redact_json(value: Any) -> Any:
            return value

        async def dispatch_inline(self, request: DispatchRequest) -> AsyncIterator[Event]:
            del request
            if False:
                yield
            raise RuntimeError(secret)

    task = asyncio.run(
        tasks.create_task(
            TaskCreate(
                type=_DISPATCH_TASK_TYPE,
                input={
                    "dispatch": _dispatch_request(
                        "sess_protocol_minimal_failure",
                        "d_protocol_minimal_failure",
                    ).model_dump(mode="json")
                },
            )
        )
    )

    with pytest.raises(TypeError, match="redact_exception_diagnostic"):
        asyncio.run(dispatcher.process_next(ProtocolMinimalRuntime(), worker_id="worker_a"))

    persisted = asyncio.run(tasks.load_task(task.id))
    assert persisted is not None
    assert persisted.status is TaskStatus.PENDING
    assert secret not in str(persisted)


def test_submit_rejects_secret_bearing_structured_output_without_changing_schema() -> None:
    secret = "dispatch-schema-canary"
    h = _build(
        [_batch("first answer")],
        secret_redactor=SecretRedactor(secret),
    )
    _create_resumable_session(h.app, "sess_dispatch_schema")
    request = DispatchRequest(
        session_id="sess_dispatch_schema",
        dispatch_id="d_schema",
        messages=[Message.text("user", "queued work")],
        structured_output=StructuredOutputSpec(
            json_schema={
                "type": "object",
                "properties": {"answer": {"const": secret}},
                "required": ["answer"],
            }
        ),
    )

    with pytest.raises(ValueError, match="changing execution semantics"):
        asyncio.run(h.app.dispatch(request))

    assert len(h.provider.requests) == 1


def test_submit_rejects_unsupported_native_output_before_queue_publication() -> None:
    h = _build([_batch("first answer"), _batch("valid queued answer")])
    session_id = "sess_dispatch_native_preflight"
    _create_resumable_session(h.app, session_id)
    native_request = DispatchRequest(
        session_id=session_id,
        dispatch_id="d_native_preflight",
        messages=[Message.text("user", "queued native work")],
        structured_output=StructuredOutputSpec(
            json_schema={"type": "object"},
            strategy="native",
        ),
    )

    with pytest.raises(NativeStructuredOutputUnsupported):
        asyncio.run(h.app.dispatch(native_request))

    assert asyncio.run(h.tasks.list_tasks(TaskQuery(type=_DISPATCH_TASK_TYPE))) == []
    submitted = asyncio.run(
        h.app.dispatch(_dispatch_request(session_id, "d_after_native_preflight"))
    )
    completed = asyncio.run(
        h.dispatcher.process_next(h.app, worker_id="worker_after_native_preflight")
    )
    queued_task = asyncio.run(h.tasks.load_task(submitted.metadata["queue_task_id"]))

    assert completed is not None
    assert completed.status is DispatchStatus.COMPLETED
    assert queued_task is not None
    assert queued_task.status is TaskStatus.COMPLETED
    assert len(h.provider.requests) == 2


def test_worker_permanently_rejects_unsupported_native_output_without_fifo_starvation() -> None:
    store = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)
    producer_provider = NativeStructuredOutputFakeProvider([_batch("initial answer")])
    producer = _configured_app(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=producer_provider,
    )
    session_id = "sess_dispatch_native_worker_rejection"
    _create_resumable_session(producer, session_id)
    native_submission = asyncio.run(
        producer.dispatch(
            DispatchRequest(
                session_id=session_id,
                dispatch_id="d_native_worker_rejection",
                messages=[Message.text("user", "queued native work")],
                structured_output=StructuredOutputSpec(
                    json_schema={"type": "object"},
                    strategy="native",
                ),
            )
        )
    )
    valid_submission = asyncio.run(
        producer.dispatch(_dispatch_request(session_id, "d_after_native_worker_rejection"))
    )
    worker_provider = FakeProvider([_batch("valid queued answer")])
    worker = _configured_app(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=worker_provider,
    )

    rejected = asyncio.run(dispatcher.process_next(worker, worker_id="worker_native_rejection"))
    completed = asyncio.run(dispatcher.process_next(worker, worker_id="worker_after_rejection"))
    rejected_task = asyncio.run(tasks.load_task(native_submission.metadata["queue_task_id"]))
    completed_task = asyncio.run(tasks.load_task(valid_submission.metadata["queue_task_id"]))

    assert rejected is not None
    assert rejected.status is DispatchStatus.FAILED
    assert rejected_task is not None
    assert rejected_task.status is TaskStatus.FAILED
    assert rejected_task.error is not None
    assert (
        rejected_task.error["required_execution_profile_fingerprint"]
        == (native_submission.metadata["required_execution_profile_fingerprint"])
    )
    assert completed is not None
    assert completed.dispatch_id == "d_after_native_worker_rejection"
    assert completed.status is DispatchStatus.COMPLETED
    assert completed_task is not None
    assert completed_task.status is TaskStatus.COMPLETED
    assert len(producer_provider.requests) == 1
    assert len(worker_provider.requests) == 1


def test_worker_permanently_rejects_target_portability_without_fifo_starvation() -> None:
    store = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)
    producer_provider = FakeProvider([_batch("initial answer")])
    producer = _configured_app(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=producer_provider,
    )
    session_id = "sess_dispatch_target_portability_rejection"
    _create_resumable_session(producer, session_id)
    invalid_submission = asyncio.run(
        producer.dispatch(
            _dispatch_request(session_id, "d_target_portability_rejection").model_copy(
                update={"target": ModelTarget(provider_name="fake", model="upgraded-model")},
                deep=True,
            )
        )
    )
    valid_submission = asyncio.run(
        producer.dispatch(_dispatch_request(session_id, "d_after_target_portability_rejection"))
    )
    worker_provider = PortableMessageRejectingFakeProvider([_batch("valid queued answer")])
    worker = _configured_app(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=worker_provider,
    )

    rejected = asyncio.run(
        dispatcher.process_next(worker, worker_id="worker_target_portability_rejection")
    )
    completed = asyncio.run(
        dispatcher.process_next(worker, worker_id="worker_after_portability_rejection")
    )
    rejected_task = asyncio.run(tasks.load_task(invalid_submission.metadata["queue_task_id"]))
    completed_task = asyncio.run(tasks.load_task(valid_submission.metadata["queue_task_id"]))

    assert rejected is not None
    assert rejected.status is DispatchStatus.FAILED
    assert rejected_task is not None
    assert rejected_task.status is TaskStatus.FAILED
    assert rejected_task.error is not None
    assert (
        rejected_task.error["required_execution_profile_fingerprint"]
        == (invalid_submission.metadata["required_execution_profile_fingerprint"])
    )
    assert completed is not None
    assert completed.dispatch_id == "d_after_target_portability_rejection"
    assert completed.status is DispatchStatus.COMPLETED
    assert completed_task is not None
    assert completed_task.status is TaskStatus.COMPLETED
    assert len(producer_provider.requests) == 1
    assert len(worker_provider.requests) == 1


def test_worker_rejects_nonportable_target_request_without_fifo_starvation() -> None:
    store = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)
    producer_provider = FakeProvider([_batch("initial answer")])
    producer = _configured_app(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=producer_provider,
    )
    session_id = "sess_dispatch_nonportable_target_request"
    _create_resumable_session(producer, session_id)
    invalid_submission = asyncio.run(
        producer.dispatch(
            DispatchRequest(
                session_id=session_id,
                dispatch_id="d_nonportable_target_request",
                messages=[
                    Message(
                        role="assistant",
                        content=(
                            TextPart(text="caller-supplied assistant state"),
                            ProviderStatePart(
                                provider="fake",
                                state={"type": "response_ref", "id": "untrusted"},
                            ),
                        ),
                    )
                ],
                target=ModelTarget(provider_name="fake", model="upgraded-model"),
            )
        )
    )
    valid_submission = asyncio.run(
        producer.dispatch(_dispatch_request(session_id, "d_after_nonportable_target_request"))
    )
    worker_provider = FakeProvider([_batch("valid queued answer")])
    worker = _configured_app(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=worker_provider,
    )

    rejected = asyncio.run(
        dispatcher.process_next(worker, worker_id="worker_nonportable_target_request")
    )
    completed = asyncio.run(
        dispatcher.process_next(worker, worker_id="worker_after_nonportable_target_request")
    )
    rejected_task = asyncio.run(tasks.load_task(invalid_submission.metadata["queue_task_id"]))
    completed_task = asyncio.run(tasks.load_task(valid_submission.metadata["queue_task_id"]))

    assert rejected is not None
    assert rejected.status is DispatchStatus.FAILED
    assert rejected_task is not None
    assert rejected_task.status is TaskStatus.FAILED
    assert rejected_task.error is not None
    assert (
        rejected_task.error["required_execution_profile_fingerprint"]
        == invalid_submission.metadata["required_execution_profile_fingerprint"]
    )
    assert completed is not None
    assert completed.dispatch_id == "d_after_nonportable_target_request"
    assert completed.status is DispatchStatus.COMPLETED
    assert completed_task is not None
    assert completed_task.status is TaskStatus.COMPLETED
    assert len(producer_provider.requests) == 1
    assert len(worker_provider.requests) == 1


def test_inline_dispatch_rejects_secret_bearing_structured_output_before_session_claim() -> None:
    secret = "inline-dispatch-schema-canary"
    h = _build(
        [_batch("first answer")],
        secret_redactor=SecretRedactor(secret),
    )
    session_id = "sess_inline_dispatch_schema"
    _create_resumable_session(h.app, session_id)

    async def scenario() -> None:
        with pytest.raises(ValueError, match="workload secret"):
            async for _ in h.app.dispatch_inline(
                DispatchRequest(
                    session_id=session_id,
                    dispatch_id="d_inline_schema",
                    messages=[Message.text("user", "queued work")],
                    structured_output=StructuredOutputSpec(
                        json_schema={"type": "string", "const": secret},
                    ),
                )
            ):
                pass

    asyncio.run(scenario())
    session = asyncio.run(h.store.load(session_id))

    assert session is not None
    assert session.status is SessionStatus.COMPLETED
    assert len(h.provider.requests) == 1


def test_process_next_does_not_persist_unclassified_runtime_failure(monkeypatch) -> None:
    secret = "dispatch-failure-canary"
    h = _build(
        [_batch("first answer"), _batch("dispatch answer")],
        secret_redactor=SecretRedactor(secret),
    )
    _create_resumable_session(h.app, "sess_dispatch_failure")
    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_dispatch_failure", "d_failure")))
    original_dispatch = h.app._dispatch_queued

    async def fail_dispatch(envelope):
        del envelope
        if False:
            yield
        raise RuntimeError(f"dispatch failed with {secret}")

    monkeypatch.setattr(h.app, "_dispatch_queued", fail_dispatch)
    with pytest.raises(RuntimeError, match="dispatch failed"):
        asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))

    assert task is not None
    assert task.status == TaskStatus.PENDING
    assert task.error is None
    assert secret not in str(task.model_dump(mode="json"))

    monkeypatch.setattr(h.app, "_dispatch_queued", original_dispatch)
    completed = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_b"))
    assert completed is not None
    assert completed.status is DispatchStatus.COMPLETED
    assert len(h.provider.requests) == 2


def test_failure_redaction_preserves_exact_dispatch_terminal_authority() -> None:
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)
    redactor = SecretRedactor(DispatchStatus.FAILED.value)

    class RedactingFailureRuntime(_SecretFreeDispatchRuntime):
        @staticmethod
        def redact_json(value: Any) -> Any:
            return redactor.redact_json(value)

        async def dispatch_inline(self, request: DispatchRequest) -> AsyncIterator[Event]:
            del request
            self._test_failure_durable = True
            if False:
                yield
            raise RuntimeError("worker failed before session admission")

        async def _queued_dispatch_settlement_state(
            self,
            envelope: _QueuedDispatchEnvelope,
        ) -> _QueuedDispatchSettlement:
            del envelope
            if not getattr(self, "_test_failure_durable", False):
                return _QueuedDispatchSettlement(_QueuedDispatchSettlementState.NOT_ADMITTED)
            return _QueuedDispatchSettlement(
                _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE,
                terminal_status=DispatchStatus.FAILED,
            )

    async def scenario() -> tuple[DispatchHandle, Task, DispatchHandle]:
        runtime = RedactingFailureRuntime()
        request = _dispatch_request("sess_redacted_authority", "d_redacted_authority")
        submitted = await dispatcher.submit(runtime, request)
        result = await dispatcher.process_next(runtime, worker_id="worker_redacted_authority")
        assert result is not None
        task = await tasks.load_task(submitted.metadata["queue_task_id"])
        assert task is not None
        replayed = await dispatcher.submit(runtime, request)
        return result, task, replayed

    result, task, replayed = asyncio.run(scenario())

    assert result.status is DispatchStatus.FAILED
    assert task.status is TaskStatus.FAILED
    assert task.error is not None
    assert task.error["status"] == DispatchStatus.FAILED.value
    assert task.error["dispatch_operation_id"] == result.metadata["dispatch_operation_id"]
    assert task.error["error"] == f"worker {REDACTED_SECRET} before session admission"
    assert replayed.metadata["idempotent_submission"] is True


def test_stalled_recovery_log_redacts_workload_secret(
    monkeypatch,
    caplog,
) -> None:
    secret = "dispatch-recovery-log-canary"
    h = _build(
        [_batch("first answer")],
        recover_stalled_sessions_after_seconds=0,
        secret_redactor=SecretRedactor(secret),
    )
    _create_resumable_session(h.app, "sess_dispatch_recovery_log")
    asyncio.run(h.app.dispatch(_dispatch_request("sess_dispatch_recovery_log", "d_recovery_log")))
    asyncio.run(
        h.store.transition_status(
            "sess_dispatch_recovery_log",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
    )

    async def fail_recovery(request):
        del request
        raise RuntimeError(f"recovery failed with {secret}")

    monkeypatch.setattr(h.app, "recover_incomplete_session", fail_recovery)
    with caplog.at_level("WARNING", logger="cayu.runtime.dispatch"):
        result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_recovery"))

    assert result is not None
    assert result.metadata.get("requeued") is True
    assert secret not in caplog.text
    assert REDACTED_SECRET in caplog.text


def test_process_next_claims_runs_and_completes() -> None:
    h = _build([_batch("first answer"), _batch("dispatch answer")])
    _create_resumable_session(h.app, "sess_run")
    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_run", "d_run")))

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is not None
    assert result.status == DispatchStatus.COMPLETED
    assert result.dispatch_id == "d_run"
    # The dispatched run actually executed (second provider request).
    assert len(h.provider.requests) == 2
    # The queue task is completed and the session ran to completion.
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))
    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    session = asyncio.run(h.store.load("sess_run"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_transient_session_provenance_read_releases_valid_dispatch_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _build([_batch("first answer"), _batch("dispatch answer")])
    session_id = "sess_transient_dispatch_provenance"
    _create_resumable_session(h.app, session_id)
    submitted = asyncio.run(
        h.app.dispatch(_dispatch_request(session_id, "d_transient_dispatch_provenance"))
    )
    original_loader = h.app.session_invocation_for_dispatch
    attempts = 0

    async def fail_once(candidate_session_id: str) -> SessionInvocationBinding:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary invocation provenance read failure")
        return await original_loader(candidate_session_id)

    monkeypatch.setattr(h.app, "session_invocation_for_dispatch", fail_once)

    with pytest.raises(ConnectionError, match="temporary invocation provenance"):
        asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_transient_a"))

    pending = asyncio.run(h.tasks.load_task(submitted.metadata["queue_task_id"]))
    assert pending is not None
    assert pending.status is TaskStatus.PENDING
    assert pending.worker_id is None
    assert len(h.provider.requests) == 1

    completed = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_transient_b"))
    assert completed is not None
    assert completed.status is DispatchStatus.COMPLETED
    assert len(h.provider.requests) == 2


def test_transient_settlement_read_releases_valid_dispatch_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _build([_batch("first answer"), _batch("dispatch answer")])
    session_id = "sess_transient_dispatch_settlement"
    _create_resumable_session(h.app, session_id)
    submitted = asyncio.run(
        h.app.dispatch(_dispatch_request(session_id, "d_transient_dispatch_settlement"))
    )
    original_loader = h.app._load_queued_dispatch_terminal_event
    attempts = 0

    async def fail_once(*, private_session_id: str, envelope: _QueuedDispatchEnvelope):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary terminal evidence read failure")
        return await original_loader(
            private_session_id=private_session_id,
            envelope=envelope,
        )

    monkeypatch.setattr(h.app, "_load_queued_dispatch_terminal_event", fail_once)

    with pytest.raises(ConnectionError, match="temporary terminal evidence"):
        asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_settlement_a"))

    pending = asyncio.run(h.tasks.load_task(submitted.metadata["queue_task_id"]))
    assert pending is not None
    assert pending.status is TaskStatus.PENDING
    assert pending.worker_id is None
    assert len(h.provider.requests) == 1

    completed = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_settlement_b"))
    assert completed is not None
    assert completed.status is DispatchStatus.COMPLETED
    assert len(h.provider.requests) == 2


def test_worker_missing_required_agent_fails_dispatch_terminally() -> None:
    store = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)
    producer = _configured_app(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=FakeProvider([_batch("initial")]),
    )
    worker_provider = FakeProvider([])
    worker = CayuApp(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        enable_logging=False,
    )
    worker.register_provider(worker_provider, default=True)

    session_id = "sess_dispatch_missing_worker_agent"
    _create_resumable_session(producer, session_id)
    submitted = asyncio.run(
        producer.dispatch(_dispatch_request(session_id, "d_dispatch_missing_worker_agent"))
    )

    rejected = asyncio.run(dispatcher.process_next(worker, worker_id="worker_missing_agent"))
    task = asyncio.run(tasks.load_task(submitted.metadata["queue_task_id"]))

    assert rejected is not None
    assert rejected.status is DispatchStatus.FAILED
    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert task.error is not None
    assert task.error["error"] == "Queued dispatch required runtime component is unavailable."
    assert (
        task.error["required_execution_profile_fingerprint"]
        == (submitted.metadata["required_execution_profile_fingerprint"])
    )
    assert worker_provider.requests == []
    assert (
        asyncio.run(dispatcher.process_next(worker, worker_id="worker_missing_agent_retry")) is None
    )


def test_operator_cancellation_wins_permanent_authority_rejection() -> None:
    class CancelBeforeAuthorityRejectionStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            if request.kind is TaskTerminalKind.FAILED:
                await super().cancel_task(
                    request.task_id,
                    {"reason": "operator cancelled queued work"},
                )
            return await super().terminalize_task(request)

    store = InMemorySessionStore()
    tasks = CancelBeforeAuthorityRejectionStore()
    dispatcher = TaskStoreDispatcher(tasks)
    producer = _configured_app(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=FakeProvider([_batch("initial")]),
    )
    worker_provider = FakeProvider([])
    worker = CayuApp(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        enable_logging=False,
    )
    worker.register_provider(worker_provider, default=True)

    session_id = "sess_operator_cancel_authority_rejection"
    _create_resumable_session(producer, session_id)
    submitted = asyncio.run(
        producer.dispatch(
            _dispatch_request(
                session_id,
                "d_operator_cancel_authority_rejection",
            )
        )
    )

    result = asyncio.run(
        dispatcher.process_next(worker, worker_id="worker_missing_agent_cancelled")
    )
    task = asyncio.run(tasks.load_task(submitted.metadata["queue_task_id"]))

    assert result is not None
    assert result.status is DispatchStatus.CANCELLED
    assert result.metadata["reclaimed"] is True
    assert task is not None
    assert task.status is TaskStatus.CANCELLED
    assert task.error == {"reason": "operator cancelled queued work"}
    assert worker_provider.requests == []


def test_conflicting_terminal_receipt_fails_dispatch_terminally() -> None:
    h = _build([_batch("initial")])
    session_id = "sess_dispatch_conflicting_receipt"
    _create_resumable_session(h.app, session_id)
    submitted = asyncio.run(
        h.app.dispatch(_dispatch_request(session_id, "d_dispatch_conflicting_receipt"))
    )
    task = asyncio.run(h.tasks.load_task(submitted.metadata["queue_task_id"]))
    assert task is not None
    envelope = _QueuedDispatchEnvelope.model_validate(task.input["dispatch"])

    async def install_conflicting_receipt() -> None:
        def update(_session, checkpoint):
            updated = {} if checkpoint is None else dict(checkpoint)
            updated["queued_dispatch_terminal_receipts"] = {
                "version": 1,
                "receipts": {
                    envelope.dispatch_operation_id: {
                        "queue_task_id": "conflicting-queue-task",
                        "terminal_event_id": envelope.terminal_event_id,
                        "run_epoch": 1,
                    }
                },
            }
            return updated

        await h.store.transform_checkpoint(session_id, update)

    asyncio.run(install_conflicting_receipt())
    rejected = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_conflicting_receipt"))
    failed = asyncio.run(h.tasks.load_task(envelope.queue_task_id))

    assert rejected is not None
    assert rejected.status is DispatchStatus.FAILED
    assert failed is not None
    assert failed.status is TaskStatus.FAILED
    assert failed.error is not None
    assert failed.error["error"] == "Queued dispatch terminal receipt identity conflicts."


def test_worker_process_cannot_substitute_changed_queued_profile() -> None:
    store = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)
    producer_provider = FakeProvider([_batch("initial")])
    producer = _configured_app(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=producer_provider,
        tools=[ProfileTool("original_tool")],
    )
    worker_provider = FakeProvider([])
    worker = _configured_app(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=worker_provider,
        tools=[ProfileTool("replacement_tool")],
    )

    _create_resumable_session(producer, "sess_profile_drift")
    submitted = asyncio.run(
        producer.dispatch(_dispatch_request("sess_profile_drift", "d_profile_drift"))
    )
    result = asyncio.run(dispatcher.process_next(worker, worker_id="worker_changed"))

    assert result is not None
    assert result.status is DispatchStatus.FAILED
    assert worker_provider.requests == []
    task = asyncio.run(tasks.load_task(submitted.metadata["queue_task_id"]))
    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert task.error is not None
    assert task.error["error_type"] == "ExecutionProfileMismatchError"
    assert (
        task.error["required_execution_profile_fingerprint"]
        == (submitted.metadata["required_execution_profile_fingerprint"])
    )
    session = asyncio.run(store.load("sess_profile_drift"))
    assert session is not None
    assert session.status is SessionStatus.COMPLETED


def test_profile_rejection_diagnostic_does_not_persist_private_session_authority() -> None:
    def key(byte: int) -> SecretStr:
        encoded = base64.urlsafe_b64encode(bytes([byte]) * 32).decode("ascii").rstrip("=")
        return SecretStr(encoded)

    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="active",
            keys={"active": key(7)},
        )
    )
    store = InMemorySessionStore(public_authority_alias_codec=codec)
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)
    producer = _configured_app(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=FakeProvider([_batch("initial")]),
        tools=[ProfileTool("original_tool")],
    )
    worker_provider = FakeProvider([])
    worker = _configured_app(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=worker_provider,
        tools=[ProfileTool("replacement_tool")],
    )
    private_session_id = "private-profile-rejection-canary"
    _create_resumable_session(producer, private_session_id)
    public_session_id = codec.aliases(private_session_id, field_name="session_id")[0]
    submitted = asyncio.run(
        producer.dispatch(_dispatch_request(public_session_id, "d_private_profile_rejection"))
    )

    result = asyncio.run(dispatcher.process_next(worker, worker_id="worker_changed_alias"))

    assert result is not None
    assert result.status is DispatchStatus.FAILED
    assert worker_provider.requests == []
    task = asyncio.run(tasks.load_task(submitted.metadata["queue_task_id"]))
    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert task.error is not None
    assert task.error["error_type"] == "ExecutionProfileMismatchError"
    assert task.error["error"] == (
        "Queued dispatch execution profile did not match its durable requirement."
    )
    assert private_session_id not in str(task.error)


def test_separate_worker_process_executes_exact_queued_profile() -> None:
    store = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)
    producer = _configured_app(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=FakeProvider([_batch("initial")]),
        tools=[ProfileTool("stable_tool")],
    )
    worker_provider = FakeProvider([_batch("queued")])
    worker = _configured_app(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=worker_provider,
        tools=[ProfileTool("stable_tool")],
    )

    _create_resumable_session(producer, "sess_profile_exact")
    submitted = asyncio.run(
        producer.dispatch(_dispatch_request("sess_profile_exact", "d_profile_exact"))
    )
    result = asyncio.run(dispatcher.process_next(worker, worker_id="worker_exact"))

    assert result is not None
    assert result.status is DispatchStatus.COMPLETED
    assert len(worker_provider.requests) == 1
    assert (
        result.metadata["required_execution_profile_fingerprint"]
        == (submitted.metadata["required_execution_profile_fingerprint"])
    )


def test_queued_model_target_binds_target_effective_request_policy() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        provider = ModelAwareOptionsFakeProvider([_batch("initial")])
        app = _configured_app(
            session_store=store,
            task_store=tasks,
            dispatcher=dispatcher,
            provider=provider,
        )
        session_id = "sess_queued_target_effective_options"
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "initial")],
            )
        ):
            pass

        submitted = await app.dispatch(
            DispatchRequest(
                session_id=session_id,
                dispatch_id="d_queued_target_effective_options",
                messages=[Message.text("user", "switch model")],
                target=ModelTarget(provider_name="fake", model="upgraded-model"),
            )
        )
        task = await tasks.load_task(submitted.metadata["queue_task_id"])
        assert task is not None
        envelope = _QueuedDispatchEnvelope.model_validate(task.input["dispatch"])
        component_class = ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY
        assert envelope.required_profile.component(component_class) != (
            envelope.source_profile.component(component_class)
        )

        result = await dispatcher.process_next(app, worker_id="worker_target_options")

        assert result is not None
        assert result.status is DispatchStatus.FAILED
        assert len(provider.requests) == 1
        decisions = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_EXECUTION_PROFILE_REJECTED,
            )
        )
        assert len(decisions) == 1
        assert decisions[0].event.payload["changed_component_classes"] == [
            ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY.value,
            ExecutionProfileComponentClass.PROVIDER_TARGET.value,
        ]

    asyncio.run(scenario())


def test_queued_dispatch_preserves_compatible_active_profile_after_release(
    monkeypatch,
) -> None:
    class CompatibleProfilePolicy(ExecutionProfilePolicy):
        def __init__(self) -> None:
            self.requests: list[ExecutionProfilePolicyRequest] = []

        @property
        def identity(self) -> str:
            return "test:dispatch-compatible-profile:v1"

        async def decide(
            self,
            request: ExecutionProfilePolicyRequest,
        ) -> ExecutionProfilePolicyResult:
            self.requests.append(request)
            return ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.COMPATIBLE_REUSE,
                reason="The worker supports this runtime transition.",
            )

    class BlockingFirstProvider(FakeProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            request_index = len(self.requests)
            self.requests.append(request)
            if request_index == 0:
                self.started.set()
                await self.release.wait()
            for event in _batch(f"answer-{request_index}"):
                yield event

    async def scenario() -> None:
        store = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        monkeypatch.setattr(session_engine_module, "_runtime_version", lambda: "old-runtime")
        producer = _configured_app(
            session_store=store,
            task_store=tasks,
            dispatcher=dispatcher,
            provider=FakeProvider([_batch("initial")]),
        )
        session_id = "sess_queued_compatible_active_profile"
        async for _ in producer.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "initial")],
            )
        ):
            pass
        session = await store.load(session_id)
        assert session is not None
        baseline_profile = execution_profile_from_session_metadata(session.metadata)

        monkeypatch.setattr(session_engine_module, "_runtime_version", lambda: "new-runtime")
        policy = CompatibleProfilePolicy()
        provider = BlockingFirstProvider()
        worker = CayuApp(
            session_store=store,
            task_store=tasks,
            dispatcher=dispatcher,
            execution_profile_policy=policy,
            enable_logging=False,
        )
        worker.register_provider(provider, default=True)
        worker.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def run_active_invocation() -> list[Event]:
            return [
                event
                async for event in worker.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "active compatible invocation")],
                    )
                )
            ]

        active_run = asyncio.create_task(run_active_invocation())
        await asyncio.wait_for(provider.started.wait(), timeout=2)
        active_checkpoint = await store.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(active_checkpoint)
        assert active_profile is not None
        submitted = await worker.dispatch(
            _dispatch_request(session_id, "d_compatible_active_profile")
        )
        assert (
            submitted.metadata["required_execution_profile_fingerprint"]
            == active_profile.profile.fingerprint
        )
        assert (
            submitted.metadata["required_execution_profile_fingerprint"]
            != baseline_profile.fingerprint
        )

        busy = await dispatcher.process_next(worker, worker_id="worker_busy")
        assert busy is not None
        assert busy.status is DispatchStatus.SUBMITTED
        assert busy.metadata["requeued"] is True
        assert len(provider.requests) == 1

        provider.release.set()
        await active_run

        completed = await dispatcher.process_next(worker, worker_id="worker_after_release")
        assert completed is not None
        assert completed.status is DispatchStatus.COMPLETED
        assert len(provider.requests) == 2
        assert len(policy.requests) == 2

    asyncio.run(scenario())


def test_queue_preparation_preserves_released_profile_for_pending_tool_recovery() -> None:
    h = _build([_batch("initial")])
    session_id = "sess_released_pending_profile"
    _create_resumable_session(h.app, session_id)
    recovery_profile = build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="queued-recovery-profile",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt=None,
        direct_tools=(),
        tool_catalogue_revision=f"sha256:{'c' * 64}",
    )
    pending_round = tool_round_recovery.PendingToolRound(
        model_step_id=f"mstep_{'1' * 32}",
        model_attempt_id=f"matt_{'2' * 32}",
        tool_round_id=f"tround_{'3' * 32}",
        agent_name="assistant",
        execution_profile_fingerprint=recovery_profile.fingerprint,
        tool_calls=[
            PendingToolCallApproval(
                tool_call_id="call-pending-profile",
                tool_name="profile_tool",
                arguments={},
            )
        ],
    )

    async def scenario() -> tuple[_QueuedDispatchEnvelope, _QueuedDispatchEnvelope, str]:
        session = await h.store.load(session_id)
        checkpoint = await h.store.load_checkpoint(session_id)
        assert session is not None
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert active_profile is not None
        assert active_profile.run_epoch == session.run_epoch - 1
        baseline_profile = execution_profile_from_session_metadata(session.metadata)

        def add_pending_recovery(_session, current):
            updated = checkpoint_with_active_invocation_execution_profile(
                current,
                session_id=session.id,
                interaction_id=active_profile.interaction_id,
                run_epoch=active_profile.run_epoch,
                profile=recovery_profile,
                expected=active_profile,
            )
            updated[tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY] = (
                pending_round.model_dump(mode="json")
            )
            return updated

        await h.store.transform_checkpoint(session_id, add_pending_recovery)
        await h.store.update_status(session_id, SessionStatus.INTERRUPTED)
        recovery_envelope = await h.app._prepare_queued_dispatch(
            _dispatch_request(session_id, "d_released_pending_profile"),
            queue_task_id="queue-released-pending-profile",
        )

        def finish_recovery(_session, current):
            assert current is not None
            updated = dict(current)
            updated.pop(tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY)
            return updated

        await h.store.transform_checkpoint(session_id, finish_recovery)
        ordinary_envelope = await h.app._prepare_queued_dispatch(
            _dispatch_request(session_id, "d_released_clean_profile"),
            queue_task_id="queue-released-clean-profile",
        )
        return recovery_envelope, ordinary_envelope, baseline_profile.fingerprint

    recovery_envelope, ordinary_envelope, baseline_fingerprint = asyncio.run(scenario())

    assert recovery_envelope.required_profile == recovery_profile
    assert ordinary_envelope.required_profile.fingerprint == baseline_fingerprint
    assert ordinary_envelope.required_profile != recovery_profile


def test_redelivery_replays_old_terminal_dispatch_during_newer_profile_ownership() -> None:
    class LoseFirstTerminalClaimStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.lose_first_terminal = True

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            if self.lose_first_terminal:
                self.lose_first_terminal = False
                raise TaskClaimLost("simulated lease loss before task terminalization")
            return await super().terminalize_task(request)

    tasks = LoseFirstTerminalClaimStore()
    h = _build(
        [_batch("initial"), _batch("queued")],
        task_store=tasks,
    )
    _create_resumable_session(h.app, "sess_dispatch_redelivery")
    submitted = asyncio.run(
        h.app.dispatch(_dispatch_request("sess_dispatch_redelivery", "d_dispatch_redelivery"))
    )

    first = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))
    assert first is not None
    assert first.metadata["reclaimed"] is True
    assert len(h.provider.requests) == 2

    async def bind_newer_profile_ownership() -> None:
        session = await h.store.load("sess_dispatch_redelivery")
        checkpoint = await h.store.load_checkpoint("sess_dispatch_redelivery")
        assert session is not None
        released_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert released_profile is not None
        receipts = checkpoint["queued_dispatch_terminal_receipts"]["receipts"]
        terminal_run_epoch = next(iter(receipts.values()))["run_epoch"]
        assert session.run_epoch > terminal_run_epoch
        await h.store.transform_checkpoint(
            session.id,
            lambda _session, current: checkpoint_with_active_invocation_execution_profile(
                current,
                session_id=session.id,
                interaction_id=released_profile.interaction_id,
                run_epoch=session.run_epoch,
                profile=released_profile.profile,
                expected=released_profile,
            ),
        )

    asyncio.run(bind_newer_profile_ownership())
    asyncio.run(
        tasks.release_task(
            submitted.metadata["queue_task_id"],
            "worker_a",
        )
    )

    replayed = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_b"))

    assert replayed is not None
    assert replayed.status is DispatchStatus.COMPLETED
    assert len(h.provider.requests) == 2
    task = asyncio.run(tasks.load_task(submitted.metadata["queue_task_id"]))
    assert task is not None
    assert task.status is TaskStatus.COMPLETED


def test_sqlite_pruning_retains_terminal_evidence_until_queue_acknowledgement(
    tmp_path,
) -> None:
    class LoseFirstTerminalClaimStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.lose_first_terminal = True

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            if self.lose_first_terminal:
                self.lose_first_terminal = False
                raise TaskClaimLost("simulated lease loss before task terminalization")
            return await super().terminalize_task(request)

    sessions = SQLiteSessionStore(tmp_path / "queued-terminal-retention.sqlite")
    tasks = LoseFirstTerminalClaimStore()
    h = _build(
        [_batch("initial"), _batch("queued")],
        task_store=tasks,
        session_store=sessions,
    )
    session_id = "sess_queued_terminal_retention"
    try:
        _create_resumable_session(h.app, session_id)
        submitted = asyncio.run(
            h.app.dispatch(_dispatch_request(session_id, "d_queued_terminal_retention"))
        )
        task_id = submitted.metadata["queue_task_id"]
        envelope_task = asyncio.run(tasks.load_task(task_id))
        assert envelope_task is not None
        envelope = _QueuedDispatchEnvelope.model_validate(envelope_task.input["dispatch"])

        first = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_retention_a"))
        assert first is not None
        assert first.metadata["reclaimed"] is True
        checkpoint = asyncio.run(sessions.load_checkpoint(session_id))
        assert checkpoint is not None
        assert checkpoint["queued_dispatch_terminal_receipts"]["receipts"] == {
            envelope.dispatch_operation_id: {
                "queue_task_id": envelope.queue_task_id,
                "terminal_event_id": envelope.terminal_event_id,
                "run_epoch": 3,
            }
        }
        with pytest.raises(ValueError, match="queued dispatch terminal acknowledgement"):
            asyncio.run(sessions.delete_session(session_id))

        async def fork_while_source_receipt_is_retained() -> None:
            async for _ in h.app.fork_session(
                ForkSessionRequest(
                    source_session_id=session_id,
                    session_id="sess_queued_terminal_retention_child",
                    copy_checkpoint=True,
                )
            ):
                pass

        asyncio.run(fork_while_source_receipt_is_retained())
        child_checkpoint = asyncio.run(
            sessions.load_checkpoint("sess_queued_terminal_retention_child")
        )
        assert (
            child_checkpoint is None or "queued_dispatch_terminal_receipts" not in child_checkpoint
        )

        cutoff = datetime.now(UTC) + timedelta(seconds=1)
        asyncio.run(sessions.prune_events(before=cutoff, session_id=session_id))
        retained = asyncio.run(
            sessions.query_events(
                EventQuery(
                    session_id=session_id,
                    event_id=envelope.terminal_event_id,
                    limit=1,
                )
            )
        )
        assert [record.event.id for record in retained] == [envelope.terminal_event_id]

        asyncio.run(tasks.release_task(task_id, "worker_retention_a"))
        replayed = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_retention_b"))
        assert replayed is not None
        assert replayed.status is DispatchStatus.COMPLETED
        assert len(h.provider.requests) == 2
        checkpoint = asyncio.run(sessions.load_checkpoint(session_id))
        assert checkpoint is None or "queued_dispatch_terminal_receipts" not in checkpoint
    finally:
        asyncio.run(sessions.close())


@pytest.mark.parametrize("retention_phase", ("run_operation", "queue_receipt"))
def test_sqlite_pruning_uses_queued_terminal_retention_markers(
    tmp_path,
    retention_phase: str,
) -> None:
    sessions = SQLiteSessionStore(tmp_path / f"queued-prune-{retention_phase}.sqlite")
    session_id = f"sess_queued_prune_{retention_phase}"
    operation_id = f"operation-{retention_phase}"
    terminal_event_id = f"terminal-{retention_phase}"

    async def scenario() -> None:
        await sessions.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(
                provider_name="fake",
                model="fake-model",
                runtime_name="cayu",
            ),
        )
        terminal_event = Event(
            id=terminal_event_id,
            type=EventType.SESSION_COMPLETED,
            session_id=session_id,
            timestamp=datetime.now(UTC) - timedelta(hours=1),
        )
        await sessions.append_events(session_id, [terminal_event])
        delivery_claim = await sessions.claim_persisted_event_side_effect(
            session_id=session_id,
            event_id=terminal_event_id,
        )
        assert delivery_claim is not None
        await sessions.mark_persisted_event_side_effect_delivered(delivery_claim)
        marker = (
            {
                "session_run_operation": {
                    "version": 1,
                    "operation_id": operation_id,
                    "run_epoch": 1,
                    "terminal_event_id": terminal_event_id,
                    "queue_task_id": f"queue-{retention_phase}",
                }
            }
            if retention_phase == "run_operation"
            else {
                "queued_dispatch_terminal_receipts": {
                    "version": 1,
                    "receipts": {
                        operation_id: {
                            "queue_task_id": f"queue-{retention_phase}",
                            "terminal_event_id": terminal_event_id,
                            "run_epoch": 1,
                        }
                    },
                }
            }
        )
        await sessions.checkpoint(session_id, marker)
        handoffs = await sessions.list_queued_dispatch_terminal_receipts()
        assert [handoff.model_dump(mode="json") for handoff in handoffs] == [
            {
                "session_id": session_id,
                "queue_task_id": f"queue-{retention_phase}",
                "operation_id": operation_id,
                "terminal_event_id": terminal_event_id,
            }
        ]
        cutoff = datetime.now(UTC)

        assert await sessions.prune_events(before=cutoff, session_id=session_id) == 0
        retained = await sessions.query_events(
            EventQuery(
                session_id=session_id,
                event_id=terminal_event_id,
                limit=1,
            )
        )
        assert [record.event.id for record in retained] == [terminal_event_id]

        await sessions.checkpoint(session_id, {})
        assert await sessions.prune_events(before=cutoff, session_id=session_id) == 1
        assert (
            await sessions.query_events(
                EventQuery(
                    session_id=session_id,
                    event_id=terminal_event_id,
                    limit=1,
                )
            )
            == []
        )

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(sessions.close())


def test_terminal_redelivery_settles_independently_of_a_newer_active_invocation() -> None:
    class LoseFirstTerminalClaimStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.lose_first_terminal = True

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            if self.lose_first_terminal:
                self.lose_first_terminal = False
                raise TaskClaimLost("simulated lease loss before task terminalization")
            return await super().terminalize_task(request)

    tasks = LoseFirstTerminalClaimStore()
    h = _build(
        [_batch("initial"), _batch("queued")],
        task_store=tasks,
    )
    _create_resumable_session(h.app, "sess_dispatch_newer_epoch")
    submitted = asyncio.run(
        h.app.dispatch(_dispatch_request("sess_dispatch_newer_epoch", "d_newer_epoch"))
    )
    first = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_old"))
    assert first is not None
    assert first.metadata["reclaimed"] is True
    asyncio.run(tasks.release_task(submitted.metadata["queue_task_id"], "worker_old"))

    interaction_id = "interaction_after_queued_terminal"
    asyncio.run(
        h.store.transition_status_and_checkpoint(
            "sess_dispatch_newer_epoch",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=lambda session, checkpoint: (
                checkpoint_with_rebound_test_invocation_profile(
                    session,
                    checkpoint,
                    interaction_id=interaction_id,
                )
            ),
            interaction_started_event=runtime_interaction_started_event(
                h.app,
                session_id="sess_dispatch_newer_epoch",
                interaction_id=interaction_id,
                agent_name="assistant",
            ),
            interaction_source_messages=[],
        )
    )

    replay = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_new"))

    assert replay is not None
    assert replay.status is DispatchStatus.COMPLETED
    assert replay.metadata.get("requeued") is None
    assert len(h.provider.requests) == 2
    task = asyncio.run(tasks.load_task(submitted.metadata["queue_task_id"]))
    assert task is not None
    assert task.status is TaskStatus.COMPLETED


def test_worker_cancellation_redelivery_replays_without_provider_redispatch() -> None:
    class BlockingProvider(FakeProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            self.started.set()
            await asyncio.Event().wait()
            if False:
                yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def scenario() -> tuple[asyncio.Task, DispatchHandle, int, TaskStatus]:
        store = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        producer = _configured_app(
            session_store=store,
            task_store=tasks,
            dispatcher=dispatcher,
            provider=FakeProvider([_batch("initial")]),
        )
        async for _ in producer.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_cancelled_dispatch",
                messages=[Message.text("user", "initial")],
            )
        ):
            pass
        submitted = await producer.dispatch(
            _dispatch_request("sess_cancelled_dispatch", "d_cancelled_dispatch")
        )
        blocking_provider = BlockingProvider()
        worker = _configured_app(
            session_store=store,
            task_store=tasks,
            dispatcher=dispatcher,
            provider=blocking_provider,
        )
        processing = asyncio.create_task(
            dispatcher.process_next(worker, worker_id="worker_cancelled")
        )
        await asyncio.wait_for(blocking_provider.started.wait(), timeout=2)
        processing.cancel("worker shutdown")
        with pytest.raises(asyncio.CancelledError, match="worker shutdown"):
            await processing
        await tasks.release_task(
            submitted.metadata["queue_task_id"],
            "worker_cancelled",
        )
        replayed = await dispatcher.process_next(worker, worker_id="worker_restarted")
        assert replayed is not None
        task = await tasks.load_task(submitted.metadata["queue_task_id"])
        assert task is not None
        return processing, replayed, len(blocking_provider.requests), task.status

    processing, replayed, provider_calls, task_status = asyncio.run(scenario())

    assert processing.cancelling() == 1
    assert processing.cancelled() is True
    assert replayed.status is DispatchStatus.INTERRUPTED
    assert provider_calls == 1
    assert task_status is TaskStatus.COMPLETED


def test_sqlite_restart_preserves_queued_profile_and_executes_once(tmp_path) -> None:
    async def scenario() -> tuple[DispatchStatus, int, TaskStatus]:
        session_path = tmp_path / "dispatch-sessions.sqlite"
        task_path = tmp_path / "dispatch-tasks.sqlite"
        producer_sessions = SQLiteSessionStore(session_path)
        producer_tasks = SQLiteTaskStore(task_path)
        producer_dispatcher = TaskStoreDispatcher(producer_tasks)
        producer = _configured_app(
            session_store=producer_sessions,
            task_store=producer_tasks,
            dispatcher=producer_dispatcher,
            provider=FakeProvider([_batch("initial")]),
            tools=[ProfileTool("stable_tool")],
        )
        try:
            async for _ in producer.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_sqlite_profile_dispatch",
                    messages=[Message.text("user", "initial")],
                )
            ):
                pass
            submitted = await producer.dispatch(
                _dispatch_request(
                    "sess_sqlite_profile_dispatch",
                    "d_sqlite_profile_dispatch",
                )
            )
        finally:
            await producer_sessions.close()
            await producer_tasks.close()

        worker_sessions = SQLiteSessionStore(session_path)
        worker_tasks = SQLiteTaskStore(task_path)
        worker_dispatcher = TaskStoreDispatcher(worker_tasks)
        worker_provider = FakeProvider([_batch("queued")])
        worker = _configured_app(
            session_store=worker_sessions,
            task_store=worker_tasks,
            dispatcher=worker_dispatcher,
            provider=worker_provider,
            tools=[ProfileTool("stable_tool")],
        )
        try:
            result = await worker_dispatcher.process_next(
                worker,
                worker_id="sqlite-worker",
            )
            assert result is not None
            task = await worker_tasks.load_task(submitted.metadata["queue_task_id"])
            assert task is not None
            return result.status, len(worker_provider.requests), task.status
        finally:
            await worker_sessions.close()
            await worker_tasks.close()

    status, provider_calls, task_status = asyncio.run(scenario())

    assert status is DispatchStatus.COMPLETED
    assert provider_calls == 1
    assert task_status is TaskStatus.COMPLETED


def test_process_next_reconciles_terminalization_acknowledgement_loss() -> None:
    class CommitThenRaiseStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            await super().terminalize_task(request)
            raise ConnectionError("acknowledgement lost")

    tasks = CommitThenRaiseStore()
    h = _build(
        [_batch("first answer"), _batch("dispatch answer")],
        task_store=tasks,
    )
    _create_resumable_session(h.app, "sess_ack_loss")
    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_ack_loss", "d_ack_loss")))

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is not None
    assert result.status is DispatchStatus.COMPLETED
    assert tasks.terminalize_calls == 1
    task = asyncio.run(tasks.load_task(handle.metadata["queue_task_id"]))
    assert task is not None
    assert task.status is TaskStatus.COMPLETED


def test_same_dispatcher_reconciles_cancellation_after_task_terminal_commit() -> None:
    class CommitThenBlockStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.terminal_committed = asyncio.Event()
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            terminal_task = await super().terminalize_task(request)
            self.terminal_committed.set()
            await asyncio.Event().wait()
            return terminal_task

    tasks = CommitThenBlockStore()
    h = _build(
        [_batch("first answer"), _batch("dispatch answer")],
        task_store=tasks,
    )
    session_id = "sess_cancel_after_task_terminal_commit"
    _create_resumable_session(h.app, session_id)
    submitted = asyncio.run(
        h.app.dispatch(
            _dispatch_request(
                session_id,
                "d_cancel_after_task_terminal_commit",
            )
        )
    )
    task_id = submitted.metadata["queue_task_id"]

    async def scenario() -> None:
        worker = asyncio.create_task(
            h.dispatcher.process_next(
                h.app,
                worker_id="worker_cancel_after_task_terminal_commit",
            )
        )
        await tasks.terminal_committed.wait()
        terminal_task = await tasks.load_task(task_id)
        assert terminal_task is not None
        assert terminal_task.status is TaskStatus.COMPLETED

        worker.cancel("cancel after task terminal commit")
        assert worker.cancelling() == 1
        with pytest.raises(asyncio.CancelledError) as raised:
            await worker
        assert raised.value.args == ("cancel after task terminal commit",)
        assert worker.cancelled()

        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "queued_dispatch_terminal_receipts" in checkpoint

        retried_result = await h.dispatcher.process_next(
            h.app,
            worker_id="worker_reconcile_cancelled_terminal_commit",
        )
        assert retried_result is None
        assert tasks.terminalize_calls == 1
        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is None or "queued_dispatch_terminal_receipts" not in checkpoint

    asyncio.run(scenario())
    assert len(h.provider.requests) == 2


def test_same_dispatcher_reconciles_load_failure_after_task_terminal_commit() -> None:
    class FailPostCommitLoadOnceStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.fail_post_commit_load = False
            self.terminalize_calls = 0

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            self.terminalize_calls += 1
            terminal_task = await super().terminalize_task(request)
            self.fail_post_commit_load = True
            return terminal_task

        async def load_task(self, task_id: str):
            if self.fail_post_commit_load:
                self.fail_post_commit_load = False
                raise ConnectionError("post-commit task load unavailable")
            return await super().load_task(task_id)

    tasks = FailPostCommitLoadOnceStore()
    h = _build(
        [_batch("first answer"), _batch("dispatch answer")],
        task_store=tasks,
    )
    session_id = "sess_load_failure_after_task_terminal_commit"
    _create_resumable_session(h.app, session_id)
    submitted = asyncio.run(
        h.app.dispatch(
            _dispatch_request(
                session_id,
                "d_load_failure_after_task_terminal_commit",
            )
        )
    )
    task_id = submitted.metadata["queue_task_id"]

    async def scenario() -> None:
        with pytest.raises(ConnectionError, match="post-commit task load unavailable"):
            await h.dispatcher.process_next(
                h.app,
                worker_id="worker_fail_post_commit_load",
            )

        terminal_task = await tasks.load_task(task_id)
        assert terminal_task is not None
        assert terminal_task.status is TaskStatus.COMPLETED
        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "queued_dispatch_terminal_receipts" in checkpoint

        retried_result = await h.dispatcher.process_next(
            h.app,
            worker_id="worker_reconcile_post_commit_load",
        )
        assert retried_result is None
        assert tasks.terminalize_calls == 1
        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is None or "queued_dispatch_terminal_receipts" not in checkpoint

    asyncio.run(scenario())
    assert len(h.provider.requests) == 2


def test_process_next_rejects_peer_terminalization_without_exact_dispatch_evidence() -> None:
    class PeerWinningStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            await super().complete_task(
                request.task_id,
                {"winner": "peer"},
                worker_id=request.worker_id,
            )
            return await super().terminalize_task(request)

    tasks = PeerWinningStore()
    h = _build(
        [_batch("first answer"), _batch("dispatch answer")],
        task_store=tasks,
    )
    _create_resumable_session(h.app, "sess_peer_winner")
    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_peer_winner", "d_peer_winner")))

    with pytest.raises(RuntimeError, match="conflicting dispatch authority"):
        asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    task = asyncio.run(tasks.load_task(handle.metadata["queue_task_id"]))
    assert task is not None
    assert task.status is TaskStatus.COMPLETED
    assert task.result == {"winner": "peer"}
    checkpoint = asyncio.run(h.store.load_checkpoint("sess_peer_winner"))
    assert checkpoint is not None
    assert "queued_dispatch_terminal_receipts" in checkpoint


def test_terminal_session_status_without_exact_event_keeps_queue_task_reclaimable() -> None:
    class RejectFirstQueuedTerminalEventStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.blocked_event_id: str | None = None

        async def append_event(self, session_id: str, event: Event) -> None:
            if event.id == self.blocked_event_id:
                raise ConnectionError("terminal event was not committed")
            await super().append_event(session_id, event)

    sessions = RejectFirstQueuedTerminalEventStore()
    h = _build(
        [_batch("first answer"), _batch("dispatch answer")],
        session_store=sessions,
        recover_stalled_sessions_after_seconds=0,
    )
    session_id = "sess_missing_queued_terminal_event"
    _create_resumable_session(h.app, session_id)
    submitted = asyncio.run(
        h.app.dispatch(_dispatch_request(session_id, "d_missing_queued_terminal_event"))
    )
    task_id = submitted.metadata["queue_task_id"]
    task = asyncio.run(h.tasks.load_task(task_id))
    assert task is not None
    envelope = _QueuedDispatchEnvelope.model_validate(task.input["dispatch"])
    sessions.blocked_event_id = envelope.terminal_event_id

    first = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert first is not None
    assert first.status is DispatchStatus.SUBMITTED
    assert first.metadata["requeued"] is True
    task = asyncio.run(h.tasks.load_task(task_id))
    assert task is not None
    assert task.status is TaskStatus.PENDING
    records = asyncio.run(
        h.store.query_events(
            EventQuery(
                session_id=session_id,
                event_id=envelope.terminal_event_id,
                limit=1,
            )
        )
    )
    if not records:
        checkpoint = asyncio.run(h.store.load_checkpoint(session_id))
        assert checkpoint is not None
        assert checkpoint["session_run_operation"]["operation_id"] == (
            envelope.dispatch_operation_id
        )

    sessions.blocked_event_id = None
    second = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_b"))

    assert second is not None
    if second.status is DispatchStatus.SUBMITTED:
        second = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_c"))
        assert second is not None
    assert second.status in {
        DispatchStatus.COMPLETED,
        DispatchStatus.FAILED,
        DispatchStatus.INTERRUPTED,
    }
    task = asyncio.run(h.tasks.load_task(task_id))
    assert task is not None
    assert task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}
    records = asyncio.run(
        h.store.query_events(
            EventQuery(
                session_id=session_id,
                event_id=envelope.terminal_event_id,
                limit=1,
            )
        )
    )
    assert [record.event.id for record in records] == [envelope.terminal_event_id]
    assert len(h.provider.requests) == 2


def test_same_dispatcher_acknowledges_terminal_task_after_cancelled_session_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _build([_batch("first answer"), _batch("dispatch answer")])
    session_id = "sess_cancelled_queue_ack"
    _create_resumable_session(h.app, session_id)
    submitted = asyncio.run(h.app.dispatch(_dispatch_request(session_id, "d_cancelled_queue_ack")))
    task_id = submitted.metadata["queue_task_id"]
    original_acknowledge = h.app._acknowledge_queued_dispatch

    async def scenario() -> None:
        acknowledgement_started = asyncio.Event()

        async def block_acknowledgement(
            envelope: _QueuedDispatchEnvelope,
            *,
            dispatch_status: DispatchStatus,
        ) -> None:
            del envelope, dispatch_status
            acknowledgement_started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(
            h.app,
            "_acknowledge_queued_dispatch",
            block_acknowledgement,
        )
        worker = asyncio.create_task(
            h.dispatcher.process_next(h.app, worker_id="worker_cancelled_ack")
        )
        await acknowledgement_started.wait()
        terminal_task = await h.tasks.load_task(task_id)
        assert terminal_task is not None
        assert terminal_task.status is TaskStatus.COMPLETED
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker

        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "queued_dispatch_terminal_receipts" in checkpoint

        monkeypatch.setattr(
            h.app,
            "_acknowledge_queued_dispatch",
            original_acknowledge,
        )
        retried_result = await h.dispatcher.process_next(
            h.app,
            worker_id="worker_retried_ack",
        )
        assert retried_result is None
        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is None or "queued_dispatch_terminal_receipts" not in checkpoint

    asyncio.run(scenario())


def test_operator_cancellation_wins_terminal_race_and_restart_acknowledges_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _build([_batch("first answer"), _batch("dispatch answer")])
    session_id = "sess_operator_cancel_terminal_race"
    _create_resumable_session(h.app, session_id)
    submitted = asyncio.run(
        h.app.dispatch(_dispatch_request(session_id, "d_operator_cancel_terminal_race"))
    )
    task_id = submitted.metadata["queue_task_id"]
    original_commit = h.dispatcher._commit_task_terminal
    original_acknowledge = h.app._acknowledge_queued_dispatch

    async def scenario() -> None:
        terminal_commit_started = asyncio.Event()
        allow_terminal_commit = asyncio.Event()
        acknowledgement_calls = 0

        async def block_terminal_commit(
            *,
            task_id: str,
            worker_id: str,
            kind: TaskTerminalKind,
            payload: dict[str, Any],
        ) -> bool:
            terminal_commit_started.set()
            await allow_terminal_commit.wait()
            return await original_commit(
                task_id=task_id,
                worker_id=worker_id,
                kind=kind,
                payload=payload,
            )

        async def lose_first_acknowledgement(
            envelope: _QueuedDispatchEnvelope,
            *,
            dispatch_status: DispatchStatus,
            receipt: QueuedDispatchTerminalReceipt | None = None,
        ) -> None:
            nonlocal acknowledgement_calls
            acknowledgement_calls += 1
            if acknowledgement_calls == 1:
                raise ConnectionError("cancelled task acknowledgement lost")
            await original_acknowledge(
                envelope,
                dispatch_status=dispatch_status,
                receipt=receipt,
            )

        monkeypatch.setattr(h.dispatcher, "_commit_task_terminal", block_terminal_commit)
        monkeypatch.setattr(
            h.app,
            "_acknowledge_queued_dispatch",
            lose_first_acknowledgement,
        )
        processing = asyncio.create_task(
            h.dispatcher.process_next(h.app, worker_id="worker_operator_cancel")
        )
        await terminal_commit_started.wait()
        cancelled = await h.tasks.cancel_task(
            task_id,
            {"reason": "operator cancelled queued work"},
        )
        assert cancelled.status is TaskStatus.CANCELLED
        allow_terminal_commit.set()
        with pytest.raises(ConnectionError, match="cancelled task acknowledgement lost"):
            await processing

        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "queued_dispatch_terminal_receipts" in checkpoint

        reconciled = await h.dispatcher.process_next(
            h.app,
            worker_id="worker_operator_cancel_restarted",
        )
        assert reconciled is None
        authoritative_task = await h.tasks.load_task(task_id)
        assert authoritative_task is not None
        assert authoritative_task.status is TaskStatus.CANCELLED
        assert authoritative_task.error == {"reason": "operator cancelled queued work"}
        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is None or "queued_dispatch_terminal_receipts" not in checkpoint
        assert acknowledgement_calls == 2

    asyncio.run(scenario())
    assert len(h.provider.requests) == 2


def test_cancelled_queue_task_retains_terminal_receipt_until_hooks_release_profile() -> None:
    class BlockingTerminalHook(RuntimeHook):
        def __init__(self) -> None:
            self.block = False
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            del context
            if not self.block:
                return
            self.started.set()
            await self.release.wait()

    hook = BlockingTerminalHook()
    h = _build(
        [_batch("first answer"), _batch("dispatch answer")],
        runtime_hooks=[hook],
    )
    session_id = "sess_cancelled_queue_terminal_hook"

    async def scenario() -> None:
        async for _ in h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "first request")],
            )
        ):
            pass
        submitted = await h.app.dispatch(
            _dispatch_request(session_id, "d_cancelled_queue_terminal_hook")
        )
        task_id = submitted.metadata["queue_task_id"]
        hook.block = True
        processing = asyncio.create_task(
            h.dispatcher.process_next(h.app, worker_id="worker_blocked_terminal_hook")
        )
        await hook.started.wait()

        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "queued_dispatch_terminal_receipts" in checkpoint
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        session = await h.store.load(session_id)
        assert active_profile is not None
        assert session is not None
        assert not active_invocation_execution_profile_is_released(
            active_profile,
            session_id=session.id,
            run_epoch=session.run_epoch,
        )

        cancelled = await h.tasks.cancel_task(
            task_id,
            {"reason": "operator cancelled during terminal hook"},
        )
        assert cancelled.status is TaskStatus.CANCELLED
        restarted_dispatcher = TaskStoreDispatcher(h.tasks, task_type=_DISPATCH_TASK_TYPE)
        assert (
            await restarted_dispatcher.process_next(
                h.app,
                worker_id="worker_restart_during_terminal_hook",
            )
            is None
        )
        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "queued_dispatch_terminal_receipts" in checkpoint

        hook.release.set()
        completed = await processing
        assert completed is not None
        assert completed.status is DispatchStatus.CANCELLED
        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is None or "queued_dispatch_terminal_receipts" not in checkpoint

    asyncio.run(scenario())
    assert len(h.provider.requests) == 2


def test_same_dispatcher_retries_transient_terminal_acknowledgement_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _build([_batch("first answer"), _batch("dispatch answer")])
    session_id = "sess_transient_queue_ack"
    _create_resumable_session(h.app, session_id)
    submitted = asyncio.run(h.app.dispatch(_dispatch_request(session_id, "d_transient_queue_ack")))
    original_acknowledge = h.app._acknowledge_queued_dispatch

    async def scenario() -> None:
        acknowledgement_calls = 0

        async def fail_once(
            envelope: _QueuedDispatchEnvelope,
            *,
            dispatch_status: DispatchStatus,
            receipt: QueuedDispatchTerminalReceipt | None = None,
        ) -> None:
            nonlocal acknowledgement_calls
            acknowledgement_calls += 1
            if acknowledgement_calls == 1:
                raise ConnectionError("terminal acknowledgement unavailable")
            await original_acknowledge(
                envelope,
                dispatch_status=dispatch_status,
                receipt=receipt,
            )

        monkeypatch.setattr(h.app, "_acknowledge_queued_dispatch", fail_once)
        with pytest.raises(ConnectionError, match="terminal acknowledgement unavailable"):
            await h.dispatcher.process_next(h.app, worker_id="worker_transient_ack")

        terminal_task = await h.tasks.load_task(submitted.metadata["queue_task_id"])
        assert terminal_task is not None
        assert terminal_task.status is TaskStatus.COMPLETED
        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "queued_dispatch_terminal_receipts" in checkpoint

        retried_result = await h.dispatcher.process_next(
            h.app,
            worker_id="worker_transient_ack_retry",
        )
        assert retried_result is None
        assert acknowledgement_calls == 2
        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is None or "queued_dispatch_terminal_receipts" not in checkpoint

    asyncio.run(scenario())
    assert len(h.provider.requests) == 2


def test_restart_acknowledges_old_receipt_while_newer_invocation_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingThirdProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            request_index = len(self.requests)
            self.requests.append(request)
            if request_index == 2:
                self.started.set()
                await self.release.wait()
            for event in _batch(f"answer-{request_index}"):
                yield event

    async def scenario() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks, task_type=_DISPATCH_TASK_TYPE)
        provider = BlockingThirdProvider()
        app = _configured_app(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            provider=provider,
        )
        session_id = "sess_old_receipt_new_invocation"
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "initial")],
            )
        ):
            pass
        submitted = await app.dispatch(
            _dispatch_request(session_id, "d_old_receipt_new_invocation")
        )
        original_acknowledge = app._acknowledge_queued_dispatch

        async def lose_acknowledgement(
            envelope: _QueuedDispatchEnvelope,
            *,
            dispatch_status: DispatchStatus,
            receipt: QueuedDispatchTerminalReceipt | None = None,
        ) -> None:
            del envelope, dispatch_status, receipt
            raise ConnectionError("terminal acknowledgement lost")

        monkeypatch.setattr(app, "_acknowledge_queued_dispatch", lose_acknowledgement)
        with pytest.raises(ConnectionError, match="terminal acknowledgement lost"):
            await dispatcher.process_next(app, worker_id="worker_old_receipt")
        monkeypatch.setattr(app, "_acknowledge_queued_dispatch", original_acknowledge)

        terminal_task = await tasks.load_task(submitted.metadata["queue_task_id"])
        assert terminal_task is not None
        assert terminal_task.status is TaskStatus.COMPLETED
        retained_checkpoint = await sessions.load_checkpoint(session_id)
        assert retained_checkpoint is not None
        receipts = retained_checkpoint["queued_dispatch_terminal_receipts"]["receipts"]
        retained_run_epoch = next(iter(receipts.values()))["run_epoch"]

        async def run_active_invocation() -> list[Event]:
            return [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "newer invocation")],
                    )
                )
            ]

        active_run = asyncio.create_task(run_active_invocation())
        await provider.started.wait()
        active_session = await sessions.load(session_id)
        active_checkpoint = await sessions.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(active_checkpoint)
        assert active_session is not None
        assert active_profile is not None
        assert active_profile.run_epoch > retained_run_epoch
        assert not active_invocation_execution_profile_is_released(
            active_profile,
            session_id=session_id,
            run_epoch=active_session.run_epoch,
        )

        restarted_dispatcher = TaskStoreDispatcher(tasks, task_type=_DISPATCH_TASK_TYPE)
        assert (
            await restarted_dispatcher.process_next(
                app,
                worker_id="worker_ack_old_receipt",
            )
            is None
        )
        acknowledged_checkpoint = await sessions.load_checkpoint(session_id)
        assert acknowledged_checkpoint is not None
        assert "queued_dispatch_terminal_receipts" not in acknowledged_checkpoint
        assert not active_run.done()

        provider.release.set()
        await active_run
        assert len(provider.requests) == 3

    asyncio.run(scenario())


def test_exact_terminal_submit_retry_ignores_newer_invocation_ownership() -> None:
    class BlockingThirdProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            request_index = len(self.requests)
            self.requests.append(request)
            if request_index == 2:
                self.started.set()
                await self.release.wait()
            for event in _batch(f"answer-{request_index}"):
                yield event

    async def scenario() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks, task_type=_DISPATCH_TASK_TYPE)
        provider = BlockingThirdProvider()
        app = _configured_app(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            provider=provider,
        )
        session_id = "sess_terminal_retry_newer_invocation"
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "initial")],
            )
        ):
            pass
        request = _dispatch_request(session_id, "d_terminal_retry_newer_invocation")
        submitted = await app.dispatch(request)
        completed = await dispatcher.process_next(app, worker_id="worker_original")
        assert completed is not None
        assert completed.status is DispatchStatus.COMPLETED
        settled_checkpoint = await sessions.load_checkpoint(session_id)
        assert settled_checkpoint is not None
        assert "session_run_operation" not in settled_checkpoint
        assert "queued_dispatch_terminal_receipts" not in settled_checkpoint

        async def run_active_invocation() -> list[Event]:
            return [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "newer invocation")],
                    )
                )
            ]

        active_run = asyncio.create_task(run_active_invocation())
        await provider.started.wait()
        active_session = await sessions.load(session_id)
        active_checkpoint = await sessions.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(active_checkpoint)
        assert active_session is not None
        assert active_profile is not None
        assert not active_invocation_execution_profile_is_released(
            active_profile,
            session_id=session_id,
            run_epoch=active_session.run_epoch,
        )

        retried = await app.dispatch(request)
        assert retried.metadata["queue_task_id"] == submitted.metadata["queue_task_id"]
        assert retried.metadata["idempotent_submission"] is True
        assert not active_run.done()
        assert len(provider.requests) == 3

        provider.release.set()
        await active_run

    asyncio.run(scenario())


def test_restart_retains_terminal_receipt_when_invocation_profile_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _build([_batch("first answer"), _batch("dispatch answer")])
    session_id = "sess_missing_terminal_profile"
    _create_resumable_session(h.app, session_id)
    submitted = asyncio.run(
        h.app.dispatch(_dispatch_request(session_id, "d_missing_terminal_profile"))
    )

    async def scenario() -> None:
        async def lose_acknowledgement(
            envelope: _QueuedDispatchEnvelope,
            *,
            dispatch_status: DispatchStatus,
            receipt: QueuedDispatchTerminalReceipt | None = None,
        ) -> None:
            del envelope, dispatch_status, receipt
            raise ConnectionError("terminal acknowledgement lost")

        monkeypatch.setattr(
            h.app,
            "_acknowledge_queued_dispatch",
            lose_acknowledgement,
        )
        with pytest.raises(ConnectionError, match="terminal acknowledgement lost"):
            await h.dispatcher.process_next(h.app, worker_id="worker_missing_terminal_profile")

        def remove_profile(_session, checkpoint):
            assert checkpoint is not None
            updated = dict(checkpoint)
            updated.pop(ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY)
            return updated

        await h.store.transform_checkpoint(session_id, remove_profile)
        restarted_dispatcher = TaskStoreDispatcher(h.tasks, task_type=_DISPATCH_TASK_TYPE)
        assert (
            await restarted_dispatcher.process_next(
                h.app,
                worker_id="worker_restart_missing_terminal_profile",
            )
            is None
        )
        task = await h.tasks.load_task(submitted.metadata["queue_task_id"])
        assert task is not None
        assert task.status is TaskStatus.COMPLETED
        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "queued_dispatch_terminal_receipts" in checkpoint

    asyncio.run(scenario())
    assert len(h.provider.requests) == 2


def test_paginated_reconciliation_remembers_unresolved_earlier_page() -> None:
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)

    class PaginatedReceiptRuntime(_SecretFreeDispatchRuntime):
        def __init__(self) -> None:
            self.receipts: list[QueuedDispatchTerminalReceipt] = []
            self.fail_operation_id: str | None = None
            self.failed_once = False

        async def _list_queued_dispatch_terminal_receipts(
            self,
            query,
        ) -> list[QueuedDispatchTerminalReceipt]:
            cursor = (
                None
                if query.after_session_id is None
                else (query.after_session_id, query.after_operation_id)
            )
            return [
                receipt
                for receipt in self.receipts
                if cursor is None or (receipt.session_id, receipt.operation_id) > cursor
            ][: query.limit]

        async def _acknowledge_queued_dispatch(
            self,
            envelope: _QueuedDispatchEnvelope,
            *,
            dispatch_status: DispatchStatus,
            receipt: QueuedDispatchTerminalReceipt | None = None,
        ) -> None:
            del envelope, dispatch_status
            assert receipt is not None
            if receipt.operation_id == self.fail_operation_id and not self.failed_once:
                self.failed_once = True
                raise ConnectionError("early-page acknowledgement unavailable")
            self.receipts = [
                candidate
                for candidate in self.receipts
                if (candidate.session_id, candidate.operation_id)
                != (receipt.session_id, receipt.operation_id)
            ]

    async def scenario() -> tuple[int, int, int]:
        runtime = PaginatedReceiptRuntime()
        for index in range(1000):
            request = _dispatch_request(
                f"sess_receipt_{index:04d}",
                f"d_receipt_{index:04d}",
            )
            submitted = await dispatcher.submit(runtime, request)
            task_id = submitted.metadata["queue_task_id"]
            claimed = await tasks.claim_task(
                f"worker_receipt_{index:04d}",
                TaskQuery(type=_DISPATCH_TASK_TYPE),
                lease_seconds=300,
            )
            assert claimed is not None
            assert claimed.id == task_id
            envelope = _QueuedDispatchEnvelope.model_validate(claimed.input["dispatch"])
            await tasks.complete_task(
                task_id,
                {
                    "status": DispatchStatus.COMPLETED.value,
                    "dispatch_operation_id": envelope.dispatch_operation_id,
                    "session_instance_fingerprint": (envelope.session_instance_fingerprint),
                    "source_execution_profile_fingerprint": (envelope.source_profile.fingerprint),
                    "required_execution_profile_fingerprint": (
                        envelope.required_profile.fingerprint
                    ),
                },
                worker_id=claimed.worker_id,
            )
            runtime.receipts.append(
                QueuedDispatchTerminalReceipt(
                    session_id=request.session_id,
                    queue_task_id=task_id,
                    operation_id=envelope.dispatch_operation_id,
                    terminal_event_id=envelope.terminal_event_id,
                )
            )
        runtime.receipts.sort(key=lambda receipt: (receipt.session_id, receipt.operation_id))
        runtime.fail_operation_id = runtime.receipts[0].operation_id

        assert await dispatcher.process_next(runtime, worker_id="sweep_one") is None
        after_first_sweep = len(runtime.receipts)
        assert await dispatcher.process_next(runtime, worker_id="sweep_two") is None
        after_cycle_end = len(runtime.receipts)
        assert await dispatcher.process_next(runtime, worker_id="sweep_three") is None
        return after_first_sweep, after_cycle_end, len(runtime.receipts)

    after_first_sweep, after_cycle_end, after_retry = asyncio.run(scenario())

    assert after_first_sweep == 1
    assert after_cycle_end == 1
    assert after_retry == 0


def test_receipt_reconciliation_does_not_serialize_mutated_store_record(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    canary = "mutated-receipt-diagnostic-canary"

    class Canary:
        def __repr__(self) -> str:
            return canary

        def __str__(self) -> str:
            return canary

    class MutatedReceiptRuntime(_SecretFreeDispatchRuntime):
        async def _list_queued_dispatch_terminal_receipts(self, query):
            del query
            receipt = QueuedDispatchTerminalReceipt(
                session_id="sess_mutated_receipt",
                queue_task_id="queue_mutated_receipt",
                operation_id="operation_mutated_receipt",
                terminal_event_id="terminal_mutated_receipt",
            )
            object.__setattr__(receipt, "queue_task_id", Canary())
            return [receipt]

    dispatcher = TaskStoreDispatcher(InMemoryTaskStore(), task_type=_DISPATCH_TASK_TYPE)
    with caplog.at_level("WARNING"):
        assert (
            asyncio.run(
                dispatcher.process_next(
                    MutatedReceiptRuntime(),
                    worker_id="worker_mutated_receipt",
                )
            )
            is None
        )

    captured = capsys.readouterr()
    assert canary not in caplog.text
    assert canary not in captured.out
    assert canary not in captured.err
    assert all(canary not in str(record.message) for record in recwarn)


def test_queue_terminalization_uses_exact_session_event_status() -> None:
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks, task_type=_DISPATCH_TASK_TYPE)
    runtime = _SecretFreeDispatchRuntime()
    runtime._test_terminal_status = DispatchStatus.COMPLETED
    request = _dispatch_request("sess_terminal_status", "d_terminal_status")

    async def scenario() -> tuple[DispatchHandle, Task]:
        submitted = await dispatcher.submit(runtime, request)
        task_id = submitted.metadata["queue_task_id"]
        claimed = await tasks.claim_task(
            "worker_terminal_status",
            TaskQuery(type=_DISPATCH_TASK_TYPE),
            lease_seconds=300,
        )
        assert claimed is not None
        envelope = _QueuedDispatchEnvelope.model_validate(claimed.input["dispatch"])
        handle = await dispatcher._terminalize(
            runtime,
            task_id,
            "worker_terminal_status",
            request,
            DispatchStatus.FAILED,
            {
                "status": DispatchStatus.FAILED.value,
                "dispatch_operation_id": envelope.dispatch_operation_id,
                "session_instance_fingerprint": envelope.session_instance_fingerprint,
                "source_execution_profile_fingerprint": (envelope.source_profile.fingerprint),
                "required_execution_profile_fingerprint": (envelope.required_profile.fingerprint),
                "error": "failure after the session terminal event",
            },
            envelope=envelope,
        )
        terminal_task = await tasks.load_task(task_id)
        assert terminal_task is not None
        return handle, terminal_task

    handle, terminal_task = asyncio.run(scenario())

    assert handle.status is DispatchStatus.COMPLETED
    assert terminal_task.status is TaskStatus.COMPLETED
    assert terminal_task.result is not None
    assert terminal_task.result["status"] == DispatchStatus.COMPLETED.value
    assert terminal_task.result["error"] == "failure after the session terminal event"


def test_queue_acknowledgement_rejects_task_status_conflicting_with_terminal_event() -> None:
    h = _build([_batch("first answer")])
    session_id = "sess_terminal_status_conflict"
    _create_resumable_session(h.app, session_id)
    submitted = asyncio.run(
        h.app.dispatch(_dispatch_request(session_id, "d_terminal_status_conflict"))
    )

    async def scenario() -> None:
        task = await h.tasks.load_task(submitted.metadata["queue_task_id"])
        assert task is not None
        envelope = _QueuedDispatchEnvelope.model_validate(task.input["dispatch"])

        def stage_queued_run(current_session, checkpoint):
            active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            assert active_profile is not None
            updated = _checkpoint_with_session_run_operation(
                checkpoint=checkpoint,
                current_session=current_session,
                operation_id=envelope.dispatch_operation_id,
                terminal_event_id=envelope.terminal_event_id,
                queue_task_id=envelope.queue_task_id,
            )
            return checkpoint_with_active_invocation_execution_profile(
                updated,
                session_id=current_session.id,
                interaction_id="interaction_terminal_status_conflict",
                run_epoch=current_session.run_epoch + 1,
                profile=envelope.required_profile,
                expected=active_profile,
            )

        await h.store.transition_status_and_checkpoint(
            session_id,
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=stage_queued_run,
        )
        await h.store.update_status(session_id, SessionStatus.COMPLETED)
        await h.store.append_event(
            session_id,
            Event(
                id=envelope.terminal_event_id,
                type=EventType.SESSION_COMPLETED,
                session_id=session_id,
                payload={"session_run_operation_id": envelope.dispatch_operation_id},
            ),
        )
        await h.store.release_run_fence(session_id)

        with pytest.raises(
            RuntimeError,
            match="task status conflicts with its exact terminal event",
        ):
            await h.app._acknowledge_queued_dispatch(
                envelope,
                dispatch_status=DispatchStatus.FAILED,
            )
        checkpoint = await h.store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["session_run_operation"]["operation_id"] == (
            envelope.dispatch_operation_id
        )

    asyncio.run(scenario())


def test_provider_pending_status_ack_loss_retains_queued_dispatch_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LoseInterruptedStatusAcknowledgement(InMemorySessionStore):
        lose_next_interrupted_ack = False

        async def update_status(self, session_id: str, status: SessionStatus):
            updated = await super().update_status(session_id, status)
            if self.lose_next_interrupted_ack and status is SessionStatus.INTERRUPTED:
                self.lose_next_interrupted_ack = False
                raise ConnectionError("interrupted status acknowledgement lost after commit")
            return updated

    store = LoseInterruptedStatusAcknowledgement()
    h = _build(
        [_batch("first answer"), _batch("unused queued answer")],
        session_store=store,
    )
    session_id = "sess_provider_pending_status_ack_loss"
    _create_resumable_session(h.app, session_id)
    submitted = asyncio.run(
        h.app.dispatch(_dispatch_request(session_id, "d_provider_pending_status_ack_loss"))
    )

    async def scenario() -> None:
        task = await h.tasks.load_task(submitted.metadata["queue_task_id"])
        assert task is not None
        envelope = _QueuedDispatchEnvelope.model_validate(task.input["dispatch"])

        async def pending_provider_operation(session, **kwargs):
            del kwargs
            return ModelCompletionBoundaryReconciliation(
                state="provider_operation_pending",
                session=session,
            )

        monkeypatch.setattr(
            h.app._session_engine._recovery_coordinator,
            "reconcile_model_completion_boundary",
            pending_provider_operation,
        )
        store.lose_next_interrupted_ack = True
        handle = await h.dispatcher.process_next(
            h.app,
            worker_id="worker_provider_pending_ack_loss",
        )

        assert handle is not None
        assert handle.status is DispatchStatus.SUBMITTED
        assert handle.metadata["requeued"] is True
        terminal_task = await h.tasks.load_task(envelope.queue_task_id)
        assert terminal_task is not None
        assert terminal_task.status is TaskStatus.PENDING
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        interrupted = await store.load(session_id)
        assert interrupted is not None
        run_operation = checkpoint["session_run_operation"]
        assert run_operation["version"] == 1
        assert run_operation["operation_id"] == envelope.dispatch_operation_id
        assert run_operation["terminal_event_id"] == envelope.terminal_event_id
        assert run_operation["queue_task_id"] == envelope.queue_task_id
        assert run_operation["run_epoch"] <= interrupted.run_epoch
        assert interrupted.status is SessionStatus.INTERRUPTED

    asyncio.run(scenario())


def test_sqlite_restart_repairs_terminal_task_session_acknowledgement(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        session_path = tmp_path / "dispatch-ack-sessions.sqlite"
        task_path = tmp_path / "dispatch-ack-tasks.sqlite"
        sessions = SQLiteSessionStore(session_path)
        tasks = SQLiteTaskStore(task_path)
        dispatcher = TaskStoreDispatcher(tasks, task_type=_DISPATCH_TASK_TYPE)
        app = _configured_app(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            provider=FakeProvider([_batch("initial"), _batch("queued")]),
        )
        session_id = "sess_sqlite_cancelled_queue_ack"
        acknowledgement_started = asyncio.Event()

        async def block_acknowledgement(
            envelope: _QueuedDispatchEnvelope,
            *,
            dispatch_status: DispatchStatus,
        ) -> None:
            del envelope, dispatch_status
            acknowledgement_started.set()
            await asyncio.Event().wait()

        try:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "initial")],
                )
            ):
                pass
            submitted = await app.dispatch(
                _dispatch_request(session_id, "d_sqlite_cancelled_queue_ack")
            )
            task_id = submitted.metadata["queue_task_id"]
            monkeypatch.setattr(
                app,
                "_acknowledge_queued_dispatch",
                block_acknowledgement,
            )
            processing = asyncio.create_task(
                dispatcher.process_next(app, worker_id="sqlite_cancelled_ack")
            )
            await acknowledgement_started.wait()
            terminal_task = await tasks.load_task(task_id)
            assert terminal_task is not None
            assert terminal_task.status is TaskStatus.COMPLETED
            processing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await processing
            checkpoint = await sessions.load_checkpoint(session_id)
            assert checkpoint is not None
            assert "queued_dispatch_terminal_receipts" in checkpoint
        finally:
            await sessions.close()
            await tasks.close()

        restarted_sessions = SQLiteSessionStore(session_path)
        restarted_tasks = SQLiteTaskStore(task_path)
        restarted_dispatcher = TaskStoreDispatcher(
            restarted_tasks,
            task_type=_DISPATCH_TASK_TYPE,
        )
        restarted_provider = FakeProvider([])
        restarted_app = _configured_app(
            session_store=restarted_sessions,
            task_store=restarted_tasks,
            dispatcher=restarted_dispatcher,
            provider=restarted_provider,
        )
        try:
            restarted_result = await restarted_dispatcher.process_next(
                restarted_app,
                worker_id="sqlite_restarted_ack",
            )
            assert restarted_result is None
            checkpoint = await restarted_sessions.load_checkpoint(session_id)
            assert checkpoint is None or "queued_dispatch_terminal_receipts" not in checkpoint
            assert restarted_provider.requests == []
        finally:
            await restarted_sessions.close()
            await restarted_tasks.close()

    asyncio.run(scenario())


def test_process_next_returns_none_when_queue_empty() -> None:
    h = _build([_batch("first answer")])
    _create_resumable_session(h.app, "sess_empty")
    assert asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a")) is None


def test_busy_session_requeues_dispatch_task() -> None:
    # A second dispatch for a session another worker is already running must be requeued,
    # not failed — the per-session serialization is preserved without losing the work.
    h = _build([_batch("first answer")])
    _create_resumable_session(h.app, "sess_busy")
    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_busy", "d_busy")))
    # Simulate another worker already running the session.
    asyncio.run(
        h.store.transition_status(
            "sess_busy",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
    )

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is not None
    assert result.status == DispatchStatus.SUBMITTED
    assert result.metadata.get("requeued") is True
    # The dispatched run never started, and the task is back to PENDING for a later retry.
    assert len(h.provider.requests) == 1
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))
    assert task is not None
    assert task.status == TaskStatus.PENDING


def test_busy_session_conflict_leaves_fresh_session_alone() -> None:
    # A conflicting session with recent store activity looks live (another worker is
    # really running it), so the dispatcher must requeue without recovering it.
    h = _build([_batch("first answer")])
    _create_resumable_session(h.app, "sess_fresh_conflict")
    asyncio.run(h.app.dispatch(_dispatch_request("sess_fresh_conflict", "d_fresh")))
    asyncio.run(
        h.store.transition_status(
            "sess_fresh_conflict",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
    )

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is not None
    assert result.metadata.get("requeued") is True
    assert "recovered_session" not in result.metadata
    session = asyncio.run(h.store.load("sess_fresh_conflict"))
    assert session is not None
    assert session.status == SessionStatus.RUNNING  # untouched


def test_busy_session_with_old_status_timestamp_but_recent_progress_is_not_recovered() -> None:
    h = _build(
        [_batch("first answer")],
        recover_stalled_sessions_after_seconds=60,
    )
    _create_resumable_session(h.app, "sess_recent_progress")
    asyncio.run(h.app.dispatch(_dispatch_request("sess_recent_progress", "d_recent_progress")))
    asyncio.run(
        h.store.transition_status(
            "sess_recent_progress",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
    )

    async def age_status_then_record_progress() -> None:
        old = datetime.now(UTC) - timedelta(hours=1)
        async with h.store._lock:
            session = h.store._sessions["sess_recent_progress"]
            h.store._sessions[session.id] = session.model_copy(
                update={"updated_at": old, "last_activity_at": old}
            )
        await h.store.checkpoint("sess_recent_progress", {"step": 2})

    asyncio.run(age_status_then_record_progress())
    before = asyncio.run(h.store.load("sess_recent_progress"))
    assert before is not None
    assert before.updated_at < datetime.now(UTC) - timedelta(minutes=30)
    assert before.last_activity_at > datetime.now(UTC) - timedelta(seconds=5)

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is not None
    assert result.metadata.get("requeued") is True
    assert "recovered_session" not in result.metadata
    after = asyncio.run(h.store.load("sess_recent_progress"))
    assert after is not None
    assert after.status == SessionStatus.RUNNING


def test_conflict_after_worker_crash_recovers_stalled_session_and_reruns() -> None:
    # A worker crashed mid-run: its queue task was reclaimed, but the session row is
    # stranded RUNNING, so every re-claim conflicts. With the recovery horizon elapsed
    # (0 here), the dispatcher must recover the session and requeue, and the next
    # claim must run the dispatch to completion instead of conflict-spinning forever.
    h = _build(
        [_batch("first answer"), _batch("dispatch answer")],
        recover_stalled_sessions_after_seconds=0,
    )
    _create_resumable_session(h.app, "sess_crash")
    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_crash", "d_crash")))
    # Simulate the crash: the session is stuck RUNNING with no live run anywhere.
    interaction_id = "interaction_dispatch_worker_crash"
    asyncio.run(
        h.store.transition_status_and_checkpoint(
            "sess_crash",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=lambda session, checkpoint: (
                checkpoint_with_rebound_test_invocation_profile(
                    session,
                    checkpoint,
                    interaction_id=interaction_id,
                )
            ),
            interaction_started_event=runtime_interaction_started_event(
                h.app,
                session_id="sess_crash",
                interaction_id=interaction_id,
                agent_name="assistant",
            ),
            interaction_source_messages=[],
        )
    )

    first = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_b"))

    assert first is not None
    assert first.status == DispatchStatus.SUBMITTED
    assert first.metadata.get("requeued") is True
    assert first.metadata.get("recovered_session") is True
    # The stranded session was finalized to a resumable status, not left RUNNING.
    session = asyncio.run(h.store.load("sess_crash"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    recovery_events = asyncio.run(h.store.load_events("sess_crash"))
    fenced = next(event for event in recovery_events if event.type == EventType.SESSION_RUN_FENCED)
    assert fenced.payload["previous_run_epoch"] == 3
    assert fenced.payload["run_epoch"] == 4

    second = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_b"))

    assert second is not None
    assert second.status == DispatchStatus.COMPLETED
    assert len(h.provider.requests) == 2
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))
    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    session = asyncio.run(h.store.load("sess_crash"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_crashed_queued_run_recovery_preserves_terminal_identity_and_does_not_redispatch() -> None:
    h = _build(
        [_batch("initial")],
        recover_stalled_sessions_after_seconds=0,
    )
    _create_resumable_session(h.app, "sess_queued_crash")
    submitted = asyncio.run(
        h.app.dispatch(_dispatch_request("sess_queued_crash", "d_queued_crash"))
    )
    task = asyncio.run(h.tasks.load_task(submitted.metadata["queue_task_id"]))
    assert task is not None
    envelope = _QueuedDispatchEnvelope.model_validate(task.input["dispatch"])
    interaction_id = "interaction_queued_dispatch_crash"

    def claim_crashed_queued_run(session, checkpoint):
        profiled = checkpoint_with_rebound_test_invocation_profile(
            session,
            checkpoint,
            interaction_id=interaction_id,
        )
        return _checkpoint_with_session_run_operation(
            checkpoint=profiled,
            current_session=session,
            operation_id=envelope.dispatch_operation_id,
            terminal_event_id=envelope.terminal_event_id,
            queue_task_id=envelope.queue_task_id,
        )

    asyncio.run(
        h.store.transition_status_and_checkpoint(
            "sess_queued_crash",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=claim_crashed_queued_run,
            interaction_started_event=runtime_interaction_started_event(
                h.app,
                session_id="sess_queued_crash",
                interaction_id=interaction_id,
                agent_name="assistant",
            ),
            interaction_source_messages=[],
        )
    )

    recovered = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_recovery"))
    replayed = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_replay"))

    assert recovered is not None
    assert recovered.status is DispatchStatus.SUBMITTED
    assert recovered.metadata["requeued"] is True
    assert recovered.metadata["recovered_session"] is True
    assert replayed is not None
    assert replayed.status is DispatchStatus.INTERRUPTED
    assert len(h.provider.requests) == 1
    events = asyncio.run(h.store.load_events("sess_queued_crash"))
    recovered_terminal = next(event for event in events if event.id == envelope.terminal_event_id)
    assert recovered_terminal.payload["session_run_operation_id"] == envelope.dispatch_operation_id
    completed_task = asyncio.run(h.tasks.load_task(submitted.metadata["queue_task_id"]))
    assert completed_task is not None
    assert completed_task.status is TaskStatus.COMPLETED


def test_stalled_dispatch_recovery_recognizes_provider_resolution_repair() -> None:
    assert (
        IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION
        in _STALLED_RECOVERED_ACTIONS
    )


def test_recover_stalled_sessions_after_seconds_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="recover_stalled_sessions_after_seconds"):
        TaskStoreDispatcher(InMemoryTaskStore(), recover_stalled_sessions_after_seconds=-1)


def test_prepared_subagent_task_type_suffix_is_reserved() -> None:
    with pytest.raises(ValueError, match="reserved prepared-subagent task-type suffix"):
        TaskStoreDispatcher(
            InMemoryTaskStore(),
            task_type="acme.dispatch.prepared-subagent.v1",
        )


def test_missing_session_is_rejected_before_queue_publication() -> None:
    h = _build([_batch("first answer")])
    with pytest.raises(KeyError, match="Session not found"):
        asyncio.run(h.app.dispatch(_dispatch_request("sess_missing", "d_missing")))

    assert asyncio.run(h.tasks.list_tasks(TaskQuery(type=_DISPATCH_TASK_TYPE))) == []


def test_unprofiled_session_is_rejected_before_queue_publication() -> None:
    h = _build([])

    async def scenario() -> None:
        await h.store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_unprofiled_dispatch",
                messages=[],
            ),
            identity=SessionIdentity(
                provider_name="fake",
                model="fake-model",
                runtime_name="cayu",
            ),
        )
        with pytest.raises(ValueError, match="no durable execution-profile"):
            await h.app.dispatch(
                _dispatch_request("sess_unprofiled_dispatch", "d_unprofiled_dispatch")
            )

    asyncio.run(scenario())
    assert asyncio.run(h.tasks.list_tasks(TaskQuery(type=_DISPATCH_TASK_TYPE))) == []


def test_worker_rejects_mutated_queued_request_before_dispatch() -> None:
    h = _build([_batch("initial")])
    _create_resumable_session(h.app, "sess_mutated_envelope")

    async def scenario():
        queue_task_id = "mutated-envelope-task"
        envelope = await h.app._prepare_queued_dispatch(
            _dispatch_request("sess_mutated_envelope", "d_mutated_envelope"),
            queue_task_id=queue_task_id,
        )
        payload = envelope.model_dump(mode="json")
        payload["request"]["max_steps"] = 17
        await h.tasks.create_task(
            TaskCreate(
                task_id=queue_task_id,
                type=_DISPATCH_TASK_TYPE,
                input={"dispatch": payload},
            )
        )
        result = await h.dispatcher.process_next(h.app, worker_id="worker_mutated")
        return result, await h.tasks.load_task(queue_task_id)

    result, task = asyncio.run(scenario())

    assert result is None
    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert len(h.provider.requests) == 1


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("session_id", "unexpected-task-session"),
        ("parent_task_id", "unexpected-parent-task"),
    ),
)
def test_worker_rejects_claimed_task_row_that_conflicts_with_envelope(
    field_name: str,
    field_value: str,
) -> None:
    class ConflictingClaimTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def claim_task(self, *args, **kwargs):
            task = await super().claim_task(*args, **kwargs)
            if task is None:
                return None
            return task.model_copy(update={field_name: field_value}, deep=True)

    tasks = ConflictingClaimTaskStore()
    h = _build([_batch("initial")], task_store=tasks)
    _create_resumable_session(h.app, "sess_claimed_row_conflict")
    submitted = asyncio.run(
        h.app.dispatch(
            _dispatch_request(
                "sess_claimed_row_conflict",
                f"d_claimed_row_conflict_{field_name}",
            )
        )
    )

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_row_conflict"))

    assert result is None
    task = asyncio.run(tasks.load_task(submitted.metadata["queue_task_id"]))
    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert len(h.provider.requests) == 1


def test_in_band_run_failure_marks_dispatch_task_failed() -> None:
    # The session exists, but the dispatched run fails mid-stream (the provider has no batch
    # for it, so the run emits a SESSION_FAILED event rather than raising). The queue task
    # must be recorded FAILED — not COMPLETED — so failure queries and retries can see it.
    h = _build([_batch("first answer")])  # no batch for the dispatched run
    _create_resumable_session(h.app, "sess_inband")
    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_inband", "d_inband")))

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is not None
    assert result.status == DispatchStatus.FAILED
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))
    assert task is not None
    assert task.status == TaskStatus.FAILED


def test_invalid_request_payload_fails_task_terminally() -> None:
    # A claimed task whose request payload no longer validates (e.g. an older serialization
    # after a schema change) must be failed terminally, not left to be reclaimed forever.
    h = _build([_batch("first answer")])
    task = asyncio.run(
        h.tasks.create_task(
            TaskCreate(type=_DISPATCH_TASK_TYPE, input={"dispatch": {"bad": "data"}})
        )
    )

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is None
    failed = asyncio.run(h.tasks.load_task(task.id))
    assert failed is not None
    assert failed.status == TaskStatus.FAILED


def test_process_next_rejects_a_non_dispatch_task_with_a_valid_payload() -> None:
    h = _build([_batch("first answer"), _batch("must not execute")])
    _create_resumable_session(h.app, "sess_wrong_source")
    request = _dispatch_request("sess_wrong_source", "d_wrong_source")
    task = asyncio.run(
        h.tasks.create_task(
            TaskCreate(
                type=_DISPATCH_TASK_TYPE,
                input={"request": request.model_dump(mode="json")},
            )
        )
    )

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is None
    failed = asyncio.run(h.tasks.load_task(task.id))
    assert failed is not None
    assert failed.status is TaskStatus.FAILED
    assert len(h.provider.requests) == 1


def test_process_next_rejects_authenticated_dispatch_provenance_with_profile_evidence() -> None:
    h = _build(
        [
            _batch("first answer"),
            _batch("second answer"),
            _batch("must not execute"),
        ]
    )
    _create_resumable_session(h.app, "sess_dispatch_source")
    _create_resumable_session(h.app, "sess_dispatch_target")
    source_binding = asyncio.run(h.app.session_invocation_for_dispatch("sess_dispatch_source"))
    request = _dispatch_request("sess_dispatch_target", "d_wrong_tree")
    queue_task_id = _queued_dispatch_task_id(request, task_type=_DISPATCH_TASK_TYPE)
    envelope = asyncio.run(h.app._prepare_queued_dispatch(request, queue_task_id=queue_task_id))
    task = asyncio.run(
        h.tasks.create_task(
            task_create_with_runtime_invocation(
                TaskCreate(
                    task_id=queue_task_id,
                    type=_DISPATCH_TASK_TYPE,
                    input={"dispatch": envelope.model_dump(mode="json")},
                ),
                source=TaskExecutionSource.TASK_DISPATCH,
                session_invocation=source_binding,
            )
        )
    )

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is not None
    assert result.status is DispatchStatus.FAILED
    assert result.metadata["dispatch_operation_id"] == envelope.dispatch_operation_id
    assert result.metadata["session_instance_fingerprint"] == (
        envelope.session_instance_fingerprint
    )
    assert result.metadata["source_execution_profile_fingerprint"] == (
        envelope.source_profile.fingerprint
    )
    assert result.metadata["required_execution_profile_fingerprint"] == (
        envelope.required_profile.fingerprint
    )
    failed = asyncio.run(h.tasks.load_task(task.id))
    assert failed is not None
    assert failed.status is TaskStatus.FAILED
    assert failed.error is not None
    assert failed.error["dispatch_operation_id"] == envelope.dispatch_operation_id
    assert failed.error["session_instance_fingerprint"] == (envelope.session_instance_fingerprint)
    assert failed.error["source_execution_profile_fingerprint"] == (
        envelope.source_profile.fingerprint
    )
    assert failed.error["required_execution_profile_fingerprint"] == (
        envelope.required_profile.fingerprint
    )
    assert len(h.provider.requests) == 2


def test_submit_rejects_loop_policies() -> None:
    # loop_policies are process-local callables that cannot survive serialization; queuing a
    # dispatch that carries them must fail loudly rather than silently drop them.
    from cayu.runtime import LoopPolicy

    class _NoopPolicy(LoopPolicy):
        pass

    h = _build([_batch("first answer")])
    _create_resumable_session(h.app, "sess_lp")
    request = DispatchRequest(
        session_id="sess_lp",
        dispatch_id="d_lp",
        messages=[Message.text("user", "queued work")],
        loop_policies=(_NoopPolicy(),),
    )
    with pytest.raises(ValueError, match="loop_policies"):
        asyncio.run(h.app.dispatch(request))


def test_durable_request_redaction_rejects_loop_policies_for_custom_dispatchers() -> None:
    from cayu.runtime import LoopPolicy

    class _NoopPolicy(LoopPolicy):
        pass

    h = _build([_batch("first answer")])
    request = DispatchRequest(
        session_id="sess_custom_dispatch_policy",
        dispatch_id="d_custom_policy",
        messages=[Message.text("user", "queued work")],
        loop_policies=(_NoopPolicy(),),
    )

    with pytest.raises(ValueError, match="loop_policies"):
        h.app.redact_dispatch_request(request)


def test_reclaimed_dispatch_is_reprocessable() -> None:
    h = _build([_batch("first answer"), _batch("dispatch answer")])
    _create_resumable_session(h.app, "sess_reclaim")

    async def scenario() -> None:
        handle = await h.app.dispatch(_dispatch_request("sess_reclaim", "d_reclaim"))
        queue_task_id = handle.metadata["queue_task_id"]
        # A worker claims it with a short lease, then "dies" without completing.
        await h.tasks.claim_task(
            "dead_worker", TaskQuery(type=_DISPATCH_TASK_TYPE), lease_seconds=1
        )
        assert await h.dispatcher.process_next(h.app, worker_id="live_worker") is None

        await asyncio.sleep(1.05)
        reclaimed = await h.tasks.reclaim_expired(query=TaskQuery(type=_DISPATCH_TASK_TYPE))
        assert [task.id for task in reclaimed] == [queue_task_id]

        result = await h.dispatcher.process_next(h.app, worker_id="live_worker")
        assert result is not None
        assert result.status == DispatchStatus.COMPLETED

    asyncio.run(scenario())


def test_run_worker_drains_queue_until_stopped() -> None:
    h = _build([_batch("a0"), _batch("b0"), _batch("a1"), _batch("b1")])
    _create_resumable_session(h.app, "sess_w_a")
    _create_resumable_session(h.app, "sess_w_b")

    async def scenario() -> None:
        h_a = await h.app.dispatch(_dispatch_request("sess_w_a", "d_w_a"))
        h_b = await h.app.dispatch(_dispatch_request("sess_w_b", "d_w_b"))
        stop = asyncio.Event()
        worker = asyncio.create_task(
            h.dispatcher.run_worker(h.app, worker_id="worker_a", stop=stop, poll_interval_s=0.01)
        )
        try:
            async with asyncio.timeout(5):
                while True:
                    t_a = await h.tasks.load_task(h_a.metadata["queue_task_id"])
                    t_b = await h.tasks.load_task(h_b.metadata["queue_task_id"])
                    if (
                        t_a is not None
                        and t_b is not None
                        and t_a.status == TaskStatus.COMPLETED
                        and t_b.status == TaskStatus.COMPLETED
                    ):
                        break
                    await asyncio.sleep(0.01)
        finally:
            stop.set()
            await worker

    asyncio.run(scenario())


def test_lease_seconds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="lease_seconds must be a positive integer"):
        TaskStoreDispatcher(InMemoryTaskStore(), lease_seconds=0)


def test_terminal_update_requires_owning_worker() -> None:
    # A worker that does not own the task's active lease cannot terminalize it.
    h = _build([_batch("first answer")])

    async def scenario() -> None:
        task = await h.tasks.create_task(TaskCreate(type=_DISPATCH_TASK_TYPE))
        await h.tasks.claim_task("worker_a", TaskQuery(type=_DISPATCH_TASK_TYPE), lease_seconds=300)

        with pytest.raises(ValueError, match="does not own"):
            await h.tasks.complete_task(task.id, {"ok": True}, worker_id="worker_b")
        with pytest.raises(ValueError, match="does not own"):
            await h.tasks.fail_task(task.id, {"err": True}, worker_id="worker_b")

        done = await h.tasks.complete_task(task.id, {"ok": True}, worker_id="worker_a")
        assert done.status == TaskStatus.COMPLETED

    asyncio.run(scenario())


def test_terminalize_does_not_clobber_a_reclaimed_task() -> None:
    # If a worker lost its lease and the task was reclaimed by another worker, the original
    # worker's terminal write must be rejected and leave the reclaimer's record untouched.
    h = _build([_batch("first answer")])

    async def scenario() -> None:
        task = await h.tasks.create_task(
            TaskCreate(type=_DISPATCH_TASK_TYPE, input={"dispatch": {"x": 1}})
        )
        # The task is now owned by worker_b (stands in for a reclaim by another worker).
        await h.tasks.claim_task("worker_b", TaskQuery(type=_DISPATCH_TASK_TYPE), lease_seconds=300)
        request = _dispatch_request("sess_reclaimed", "d_reclaimed")
        envelope = _test_dispatch_envelope(request, queue_task_id=task.id)

        handle = await h.dispatcher._terminalize(
            _SecretFreeDispatchRuntime(),
            task.id,
            "worker_a",
            request,
            DispatchStatus.COMPLETED,
            {"status": "completed"},
            envelope=envelope,
        )

        assert handle.metadata.get("reclaimed") is True
        reloaded = await h.tasks.load_task(task.id)
        assert reloaded is not None
        assert reloaded.status == TaskStatus.CLAIMED  # not clobbered to COMPLETED
        assert reloaded.worker_id == "worker_b"  # still the reclaimer's

    asyncio.run(scenario())


def test_malformed_dispatch_claim_loss_returns_without_clobbering_new_owner() -> None:
    class ReclaimBeforeMalformedFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def terminalize_task(self, request: TaskTerminalizationRequest):
            if request.worker_id is not None:
                await super().release_task(request.task_id, request.worker_id)
                reclaimed = await super().claim_task("worker_b", lease_seconds=300)
                assert reclaimed is not None
                assert reclaimed.id == request.task_id
            return await super().terminalize_task(request)

    tasks = ReclaimBeforeMalformedFailureStore()
    dispatcher = TaskStoreDispatcher(tasks)

    class UnusedRuntime(_SecretFreeDispatchRuntime):
        async def dispatch_inline(self, request: DispatchRequest) -> AsyncIterator[Event]:
            del request
            if False:
                yield

    async def scenario():
        task = await tasks.create_task(
            TaskCreate(
                type=_DISPATCH_TASK_TYPE,
                input={"dispatch": "not-an-object"},
            )
        )
        result = await dispatcher.process_next(UnusedRuntime(), worker_id="worker_a")
        return result, await tasks.load_task(task.id)

    result, task = asyncio.run(scenario())

    assert result is None
    assert task is not None
    assert task.status is TaskStatus.CLAIMED
    assert task.worker_id == "worker_b"


def test_conflict_requeue_returns_reclaimed_handle_when_control_plane_wins() -> None:
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)
    request = _dispatch_request("sess_conflict_claim_lost", "d_conflict_claim_lost")

    async def scenario():
        class SubmitRuntime(_SecretFreeDispatchRuntime):
            async def dispatch_inline(
                self,
                request: DispatchRequest,
            ) -> AsyncIterator[Event]:
                del request
                if False:
                    yield

        submitted = await dispatcher.submit(SubmitRuntime(), request)
        task_id = submitted.metadata["queue_task_id"]

        class TerminalizingConflictRuntime(_SecretFreeDispatchRuntime):
            async def dispatch_inline(
                self,
                request: DispatchRequest,
            ) -> AsyncIterator[Event]:
                del request
                await tasks.complete_task(task_id, {"winner": "control-plane"})
                if False:
                    yield
                raise SessionStatusConflict("session already running")

        result = await dispatcher.process_next(
            TerminalizingConflictRuntime(),
            worker_id="worker_a",
        )
        return result, await tasks.load_task(task_id)

    result, task = asyncio.run(scenario())

    assert result is not None
    assert result.status is DispatchStatus.SUBMITTED
    assert result.metadata["reclaimed"] is True
    assert result.metadata.get("requeued") is None
    assert task is not None
    assert task.status is TaskStatus.COMPLETED
    assert task.result == {"winner": "control-plane"}


@pytest.mark.parametrize(
    ("rejected_text", "error_code"),
    [
        ("provider failure\u0000with invalid text", "nul_character"),
        ("provider failure\ud800with invalid text", "unicode_surrogate"),
    ],
)
def test_dispatch_failure_with_nonportable_text_is_terminal_and_not_reclaimed(
    rejected_text: str,
    error_code: str,
) -> None:
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)

    class FailingRuntime(_SecretFreeDispatchRuntime):
        async def dispatch_inline(self, request: DispatchRequest) -> AsyncIterator[Event]:
            del request
            self._test_failure_durable = True
            if False:
                yield
            raise RuntimeError(rejected_text)

        async def _queued_dispatch_settlement_state(
            self,
            envelope: _QueuedDispatchEnvelope,
        ) -> _QueuedDispatchSettlement:
            del envelope
            if not getattr(self, "_test_failure_durable", False):
                return _QueuedDispatchSettlement(_QueuedDispatchSettlementState.NOT_ADMITTED)
            return _QueuedDispatchSettlement(
                _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE,
                terminal_status=DispatchStatus.FAILED,
            )

    async def scenario():
        submitted = await dispatcher.submit(
            FailingRuntime(),
            _dispatch_request("sess_nonportable_failure", "d_nonportable_failure"),
        )
        result = await dispatcher.process_next(FailingRuntime(), worker_id="worker_a")
        task_id = submitted.metadata["queue_task_id"]
        task = await tasks.load_task(task_id)
        reclaimed = await tasks.reclaim_expired(query=TaskQuery(type=_DISPATCH_TASK_TYPE))
        second_claim = await tasks.claim_task(
            "worker_b",
            TaskQuery(type=_DISPATCH_TASK_TYPE),
            lease_seconds=300,
        )
        return result, task, reclaimed, second_claim

    result, task, reclaimed, second_claim = asyncio.run(scenario())

    assert result is not None
    assert result.status is DispatchStatus.FAILED
    assert result.metadata.get("reclaimed") is None
    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert task.error is not None
    assert task.error["error"] == "Dispatch failed with a non-portable diagnostic."
    assert task.error["error_type"] == "RuntimeError"
    assert task.error["durable_value_error_code"] == error_code
    assert task.error["durable_value_error_path"] == "$"
    assert task.error["dispatch_operation_id"] == result.metadata["dispatch_operation_id"]
    assert (
        task.error["required_execution_profile_fingerprint"]
        == (result.metadata["required_execution_profile_fingerprint"])
    )
    assert reclaimed == []
    assert second_claim is None


def test_concurrent_workers_claim_distinct_dispatch_tasks(postgres_dsn: str) -> None:
    # In-memory sessions + a real PostgresTaskStore queue: two concurrent workers must
    # claim distinct dispatch tasks through the actual FOR UPDATE SKIP LOCKED path. A
    # per-process-unique task type isolates this run from any leftover rows.
    from cayu.storage import PostgresTaskStore
    from cayu.storage.migrations import SchemaMode

    task_type = f"cayu.dispatch.test.{os.getpid()}"

    async def scenario() -> None:
        tasks = PostgresTaskStore(
            postgres_dsn, min_size=1, max_size=4, schema_mode=SchemaMode.CREATE
        )
        try:
            h = _build(
                [_batch("a0"), _batch("b0"), _batch("a1"), _batch("b1")],
                task_store=tasks,
                task_type=task_type,
            )
            for session_id in ("sess_pg_a", "sess_pg_b"):
                async for _ in h.app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "first request")],
                    )
                ):
                    pass
                await h.app.dispatch(_dispatch_request(session_id, f"d_{session_id}"))

            results = await asyncio.gather(
                h.dispatcher.process_next(h.app, worker_id="worker_a"),
                h.dispatcher.process_next(h.app, worker_id="worker_b"),
            )
            claimed = [r for r in results if r is not None]
            assert len(claimed) == 2
            assert {r.session_id for r in claimed} == {"sess_pg_a", "sess_pg_b"}
            assert all(r.status == DispatchStatus.COMPLETED for r in claimed)
        finally:
            await tasks.close()

    asyncio.run(scenario())


def test_postgres_restart_preserves_queued_profile_and_executes_once(
    postgres_dsn: str,
) -> None:
    from cayu import PostgresSessionStore
    from cayu.storage import PostgresTaskStore
    from cayu.storage.migrations import SchemaMode

    suffix = uuid4().hex
    session_id = f"dispatch-profile-{suffix}"
    task_type = f"cayu.dispatch.profile.{suffix}"

    async def scenario() -> tuple[DispatchStatus, int, TaskStatus]:
        producer_sessions = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        producer_tasks = PostgresTaskStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        producer_dispatcher = TaskStoreDispatcher(
            producer_tasks,
            task_type=task_type,
        )
        producer = _configured_app(
            session_store=producer_sessions,
            task_store=producer_tasks,
            dispatcher=producer_dispatcher,
            provider=FakeProvider([_batch("initial")]),
            tools=[ProfileTool("stable_tool")],
        )
        try:
            async for _ in producer.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "initial")],
                )
            ):
                pass
            submitted = await producer.dispatch(_dispatch_request(session_id, f"dispatch-{suffix}"))
        finally:
            await producer_sessions.close()
            await producer_tasks.close()

        worker_sessions = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        worker_tasks = PostgresTaskStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        worker_dispatcher = TaskStoreDispatcher(worker_tasks, task_type=task_type)
        worker_provider = FakeProvider([_batch("queued")])
        worker = _configured_app(
            session_store=worker_sessions,
            task_store=worker_tasks,
            dispatcher=worker_dispatcher,
            provider=worker_provider,
            tools=[ProfileTool("stable_tool")],
        )
        try:
            result = await worker_dispatcher.process_next(
                worker,
                worker_id=f"worker-{suffix}",
            )
            assert result is not None
            task = await worker_tasks.load_task(submitted.metadata["queue_task_id"])
            assert task is not None
            return result.status, len(worker_provider.requests), task.status
        finally:
            await worker_sessions.close()
            await worker_tasks.close()

    status, provider_calls, task_status = asyncio.run(scenario())

    assert status is DispatchStatus.COMPLETED
    assert provider_calls == 1
    assert task_status is TaskStatus.COMPLETED
