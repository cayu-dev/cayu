from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu import SQLiteSessionStore
from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    Message,
    ThinkingConfig,
    ThinkingPart,
)
from cayu.core.billing import BillingIdentity
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ProviderOperationAdapter,
    ProviderOperationConnection,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
)
from cayu.runtime import (
    AllowAllToolPolicy,
    CayuApp,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    InteractionStatus,
    InteractionSummaryEvidence,
    ResumeRequest,
    RetryPolicy,
    RunLimits,
    RunRequest,
    SessionIdentity,
    SessionStatus,
    SessionStore,
)
from cayu.runtime._model_step_executor import ModelCompletionRecoveryContext
from cayu.runtime._recovery_coordinator import ModelCompletionManualRecoveryRequired
from cayu.runtime.execution_units import ModelAttemptIdentity
from cayu.runtime.provider_operations import (
    ProviderOperationInspectionStatus,
    inspect_provider_operation,
)
from cayu.runtime.sessions import ModelCompletionStageRequest
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
)


class _OfflineOperationAdapter(ProviderOperationAdapter):
    def __init__(
        self,
        status: ProviderOperationStatus,
        *,
        events: tuple[ModelStreamEvent, ...] | None = None,
    ) -> None:
        self.status = status
        self.state = ProviderOperationState(
            operation_id="response_offline_123",
            stream_protocol="responses-v1",
            recovery_metadata={"cursor": 0},
        )
        self.start_calls = 0
        self.retrieve_calls: list[ProviderOperationState] = []
        self.events = events
        self.start_events: tuple[ModelStreamEvent, ...] | None = None

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        del request
        self.start_calls += 1
        if self.start_events is None:
            raise AssertionError("offline recovery must not submit a replacement operation")

        async def events() -> AsyncIterator[ModelStreamEvent]:
            if self.start_events is None:  # pragma: no cover - captured above
                raise AssertionError("start events disappeared")
            for event in self.start_events:
                yield event

        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id=f"response_resumed_{self.start_calls}",
                stream_protocol="responses-v1",
                recovery_metadata={"cursor": 0},
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )

    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.retrieve_calls.append(state)
        events = self.events or (
            ModelStreamEvent.text_delta("finished while offline"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 4},
                }
            ),
        )
        return ProviderOperationSnapshot(
            state=self.state,
            status=self.status,
            events=events if self.status is ProviderOperationStatus.COMPLETED else (),
        )

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        raise AssertionError("offline terminal retrieval must not reconnect a stream")

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        raise AssertionError("offline recovery must not cancel the operation")


class _OfflineOperationProvider(ModelProvider):
    name = "offline-operation"

    def __init__(
        self,
        status: ProviderOperationStatus,
        *,
        events: tuple[ModelStreamEvent, ...] | None = None,
    ) -> None:
        self.adapter = _OfflineOperationAdapter(status, events=events)

    @property
    def provider_operation_mode(self) -> ProviderOperationMode:
        return ProviderOperationMode.BACKGROUND

    @property
    def provider_operations(self) -> ProviderOperationAdapter:
        return self.adapter

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        raise AssertionError("offline recovery must not use the synchronous stream")
        yield  # pragma: no cover


class _IdentityAwareOfflineProvider(_OfflineOperationProvider):
    def __init__(self, identity: BillingIdentity) -> None:
        super().__init__(ProviderOperationStatus.COMPLETED)
        self.identity = identity

    def billing_identity_for_completion(
        self,
        identity: BillingIdentity | None,
        payload: dict,
    ) -> BillingIdentity | None:
        del payload
        assert identity == self.identity
        return identity


class _LookupTool(Tool):
    spec = ToolSpec(
        name="lookup",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        effect=ToolEffect.NONE,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(content="lookup complete")


async def _stage_offline_operation(
    store: SessionStore,
    *,
    session_id: str,
    provider: _OfflineOperationProvider,
    recovery_context: dict | None = None,
) -> Message:
    user_message = Message.text("user", "finish this while no worker is attached")
    interaction_id = f"interaction-{session_id}"
    started_event_id = f"{session_id}:interaction-started"
    started_at = datetime.now(UTC)
    started_event = Event(
        id=started_event_id,
        type=EventType.INTERACTION_STARTED,
        session_id=session_id,
        interaction_id=interaction_id,
        timestamp=started_at,
        agent_name="assistant",
        payload=InteractionSummaryEvidence(
            status=InteractionStatus.ACTIVE,
            start_event_id=started_event_id,
            started_at=started_at,
        ).model_dump(mode="json"),
    )
    session = await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[user_message],
        ),
        identity=SessionIdentity(provider_name=provider.name, model="fake-model"),
        interaction_started_event=started_event,
        interaction_source_messages=[user_message],
    )
    await store.replace_initial_transcript_messages(
        session_id,
        [user_message],
        [user_message],
        interaction_id=interaction_id,
    )
    identity = ModelAttemptIdentity(
        model_step_id="mstep_" + "a" * 32,
        model_attempt_id="matt_" + "b" * 32,
    )
    stage_id = f"{identity.model_step_id}:dispatch:0"
    intent = {
        "schema_version": 1,
        "purpose": "assistant-turn",
        **identity.payload(),
        "logical_step_id": identity.model_step_id,
        "provider_name": provider.name,
        "requested_model": "fake-model",
        "source_transcript_cursor": 1,
        "request_fingerprint": "c" * 64,
    }
    if recovery_context is not None:
        intent["recovery_context"] = recovery_context
    await store.prepare_model_completion_stage(
        session_id,
        request=ModelCompletionStageRequest(
            stage_id=stage_id,
            logical_step_id=identity.model_step_id,
            dispatch_ordinal=0,
            intent=intent,
            reservation_ids=(),
        ),
        expected_statuses={session.status},
        expected_run_epoch=session.run_epoch,
        expected_transcript_cursor=1,
    )
    await store.append_events(
        session_id,
        [
            Event(
                type=EventType.MODEL_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
                agent_name="assistant",
                payload={
                    "provider": provider.name,
                    "model": "fake-model",
                    "step": 1,
                    "attempt": 1,
                    "max_attempts": 1,
                    **identity.payload(),
                },
            ),
            Event(
                type=EventType.PROVIDER_OPERATION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
                agent_name="assistant",
                payload={
                    "provider": provider.name,
                    "model": "fake-model",
                    "step": 1,
                    "attempt": 1,
                    "max_attempts": 1,
                    **identity.payload(),
                    "source_run_epoch": session.run_epoch,
                    "start_id": f"provider-operation:{identity.model_attempt_id}",
                    "state_version": provider.adapter.state.version,
                    "operation_id": provider.adapter.state.operation_id,
                    "stream_protocol": provider.adapter.state.stream_protocol,
                    "status": ProviderOperationStatus.IN_PROGRESS.value,
                    "recovery_metadata": {"cursor": 0},
                },
            ),
        ],
    )
    await store.release_run_fence(session_id)
    return user_message


async def assert_offline_provider_operation_recovery(store: SessionStore) -> None:
    provider = _OfflineOperationProvider(ProviderOperationStatus.COMPLETED)
    user_message = await _stage_offline_operation(
        store,
        session_id="offline-completed",
        provider=provider,
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id="offline-completed",
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )
    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(session_id="offline-completed")
    )

    assert provider.adapter.start_calls == 0
    assert provider.adapter.retrieve_calls == [provider.adapter.state]
    transcript = await store.load_transcript("offline-completed")
    assert transcript[0] == user_message
    assert transcript[1].content[0].text == "finished while offline"
    events = await store.load_events("offline-completed")
    completed_events = [event for event in events if event.type is EventType.MODEL_COMPLETED]
    assert len(completed_events) == 1
    assert completed_events[0].payload["usage_metrics"]["input_tokens"] == 3
    assert completed_events[0].payload["usage_metrics"]["output_tokens"] == 4
    assert await store.load_active_model_completion_stage("offline-completed") is None
    inspection = await inspect_provider_operation(store, "offline-completed")
    assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_RECONCILED


async def assert_pending_provider_operation_later_completes(
    store: SessionStore,
    *,
    initial_status: ProviderOperationStatus,
) -> None:
    provider = _OfflineOperationProvider(initial_status)
    user_message = await _stage_offline_operation(
        store,
        session_id=f"offline-{initial_status.value}-then-completed",
        provider=provider,
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = f"offline-{initial_status.value}-then-completed"

    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )
    pending = await store.load(session_id)
    assert pending is not None
    assert pending.status in {SessionStatus.PENDING, SessionStatus.RUNNING}
    assert await store.load_active_model_completion_stage(session_id) is not None

    provider.adapter.status = ProviderOperationStatus.COMPLETED
    await app.recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))
    transcript = await store.load_transcript(session_id)
    assert transcript[0] == user_message
    assert transcript[1].content[0].text == "finished while offline"
    events = await store.load_events(session_id)
    assert sum(event.type is EventType.MODEL_COMPLETED for event in events) == 1
    assert provider.adapter.start_calls == 0
    assert provider.adapter.retrieve_calls == [provider.adapter.state] * 2


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_completed_operation_is_retrieved_and_published_exactly_once(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "offline-recovery.sqlite3")
        )
        await assert_offline_provider_operation_recovery(store)
        if isinstance(store, SQLiteSessionStore):
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "initial_status",
    [ProviderOperationStatus.QUEUED, ProviderOperationStatus.IN_PROGRESS],
)
@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_pending_operation_later_completes_without_losing_publication_eligibility(
    initial_status: ProviderOperationStatus,
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / f"{initial_status.value}.sqlite3")
        )
        await assert_pending_provider_operation_later_completes(
            store,
            initial_status=initial_status,
        )
        if isinstance(store, SQLiteSessionStore):
            await store.close()

    asyncio.run(scenario())


def test_recovery_rejects_an_operation_after_provider_output_was_accepted() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(ProviderOperationStatus.COMPLETED)
        await _stage_offline_operation(
            store,
            session_id="offline-partial-output",
            provider=provider,
        )
        await store.append_events(
            "offline-partial-output",
            [
                Event(
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id="offline-partial-output",
                    agent_name="assistant",
                    payload={
                        "delta": "already accepted",
                        "model_step_id": "mstep_" + "a" * 32,
                        "model_attempt_id": "matt_" + "b" * 32,
                    },
                )
            ],
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(ModelCompletionManualRecoveryRequired):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="offline-partial-output",
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )

        assert provider.adapter.retrieve_calls == []
        assert await store.load_transcript("offline-partial-output") == [
            Message.text("user", "finish this while no worker is attached")
        ]

    asyncio.run(scenario())


def test_offline_recovery_preserves_request_billing_identity() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        identity = BillingIdentity(
            provider_name="offline-operation",
            resource_id="offline-model",
            request_evidence={"region": "us-test-1"},
        )
        provider = _IdentityAwareOfflineProvider(identity)
        context = ModelCompletionRecoveryContext(
            billing_identity=identity,
        )
        await _stage_offline_operation(
            store,
            session_id="offline-billing-identity",
            provider=provider,
            recovery_context=context.model_dump(mode="json"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="offline-billing-identity",
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        completed = next(
            event
            for event in await store.load_events("offline-billing-identity")
            if event.type is EventType.MODEL_COMPLETED
        )
        assert completed.payload["billing_identity"] == identity.model_dump(mode="json")

    asyncio.run(scenario())


def test_offline_recovery_honors_hidden_thinking_transcript_policy() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(
            ProviderOperationStatus.COMPLETED,
            events=(
                ModelStreamEvent.thinking("private reasoning"),
                ModelStreamEvent.text_delta("public answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
        )
        context = ModelCompletionRecoveryContext(
            thinking=ThinkingConfig(include_in_transcript=False),
        )
        await _stage_offline_operation(
            store,
            session_id="offline-hidden-thinking",
            provider=provider,
            recovery_context=context.model_dump(mode="json"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="offline-hidden-thinking",
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        assistant = (await store.load_transcript("offline-hidden-thinking"))[1]
        assert not any(isinstance(part, ThinkingPart) for part in assistant.content)
        assert assistant.content[0].text == "public answer"

    asyncio.run(scenario())


def test_offline_recovery_accepts_hidden_thinking_only_completion() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(
            ProviderOperationStatus.COMPLETED,
            events=(
                ModelStreamEvent.thinking("private reasoning"),
                ModelStreamEvent.completed({"finish_reason": "length"}),
            ),
        )
        context = ModelCompletionRecoveryContext(
            thinking=ThinkingConfig(include_in_transcript=False),
        )
        session_id = "offline-hidden-thinking-only"
        user_message = await _stage_offline_operation(
            store,
            session_id=session_id,
            provider=provider,
            recovery_context=context.model_dump(mode="json"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == [provider.adapter.state]
        assert await store.load_transcript(session_id) == [user_message]
        events = await store.load_events(session_id)
        completed_events = [event for event in events if event.type is EventType.MODEL_COMPLETED]
        assert len(completed_events) == 1
        assert completed_events[0].payload["transcript_cursor"] == 1
        assert await store.load_active_model_completion_stage(session_id) is None
        inspection = await inspect_provider_operation(store, session_id)
        assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_RECONCILED

    asyncio.run(scenario())


def test_offline_recovery_restores_structured_output_tool_contract() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(
            ProviderOperationStatus.COMPLETED,
            events=(
                ModelStreamEvent.tool_call(
                    id="structured-output-call",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "recovered"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ),
        )
        spec = StructuredOutputSpec(
            name="answer",
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        )
        context = ModelCompletionRecoveryContext(
            request_metadata={"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"},
            structured_output=spec,
            max_steps=3,
            limits=RunLimits(max_tool_calls=2),
            retry_policy=RetryPolicy(max_attempts=2),
            structured_output_attempt=1,
        )
        await _stage_offline_operation(
            store,
            session_id="offline-structured-output",
            provider=provider,
            recovery_context=context.model_dump(mode="json"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="offline-structured-output",
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        event_types = {event.type for event in result.events}
        assert EventType.STRUCTURED_OUTPUT_VALIDATING in event_types
        assert EventType.STRUCTURED_OUTPUT_VALIDATED in event_types
        assert EventType.TOOL_CALL_STARTED not in event_types

    asyncio.run(scenario())


def test_offline_recovery_preserves_an_ordinary_tool_call_during_structured_output() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(
            ProviderOperationStatus.COMPLETED,
            events=(
                ModelStreamEvent.tool_call(
                    id="ordinary-tool-call",
                    name="lookup",
                    arguments={"query": "recovered"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ),
        )
        spec = StructuredOutputSpec(
            name="answer",
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            max_retries=1,
        )
        context = ModelCompletionRecoveryContext(
            structured_output=spec,
            structured_output_attempt=2,
        )
        session_id = "offline-structured-output-ordinary-tool"
        await _stage_offline_operation(
            store,
            session_id=session_id,
            provider=provider,
            recovery_context=context.model_dump(mode="json"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_LookupTool()],
            tool_policy=AllowAllToolPolicy(),
        )

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        events = await store.load_events(session_id)
        completed = next(event for event in events if event.type is EventType.MODEL_COMPLETED)
        assert completed.payload["step_classification"]["type"] == "continue"
        assert EventType.STRUCTURED_OUTPUT_VALIDATING not in {event.type for event in events}
        stage = await store.load_model_completion_stage(
            session_id,
            "mstep_" + "a" * 32 + ":dispatch:0",
        )
        assert stage is not None
        assert stage.publication is not None
        pending_operation = next(
            operation
            for operation in stage.publication.mutation.operations
            if operation.key == "pending_tool_round"
        )
        assert pending_operation.value["structured_output_retries"] == 1
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        checkpoint.pop("pending_tool_approval")
        pending_round = checkpoint["pending_tool_round"]
        pending_round["policy_state"] = "planned"
        pending_round["policy_context_version"] = 1
        for call in pending_round["tool_calls"]:
            call["policy_evidence"] = "authoritative"
            call["policy_decision"] = "allow"
        await store.checkpoint(session_id, checkpoint)

        provider.adapter.start_events = (
            ModelStreamEvent.tool_call(
                id="invalid-finalizer-after-ordinary-tool",
                name=STRUCTURED_OUTPUT_TOOL_NAME,
                arguments={"output": {"wrong": "value"}},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        )
        resumed_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                    structured_output=spec,
                )
            )
        ]
        failed = [
            event for event in resumed_events if event.type is EventType.STRUCTURED_OUTPUT_FAILED
        ]
        assert [event.payload["attempt"] for event in failed] == [2]
        assert EventType.STRUCTURED_OUTPUT_RETRY not in {event.type for event in resumed_events}
        assert provider.adapter.start_calls == 1

    asyncio.run(scenario())


def test_queued_operation_remains_recoverable_without_redispatch() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(ProviderOperationStatus.QUEUED)
        user_message = await _stage_offline_operation(
            store,
            session_id="offline-queued",
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="offline-queued",
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        for _ in range(4):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="offline-queued")
            )

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == [provider.adapter.state] * 5
        assert await store.load_transcript("offline-queued") == [user_message]
        assert await store.load_active_model_completion_stage("offline-queued") is not None
        inspection = await inspect_provider_operation(store, "offline-queued")
        assert inspection.status is ProviderOperationInspectionStatus.RECONNECT_SCHEDULED

    asyncio.run(scenario())


def test_resume_retrieves_queued_operation_then_relinquishes_run_fence() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(ProviderOperationStatus.QUEUED)
        await _stage_offline_operation(
            store,
            session_id="resume-offline-queued",
            provider=provider,
        )
        await store.update_status("resume-offline-queued", SessionStatus.INTERRUPTED)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="resume-offline-queued",
                    messages=[Message.text("user", "continue when the operation finishes")],
                )
            )
        ]

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == [provider.adapter.state]
        assert EventType.SESSION_FAILED not in {event.type for event in events}
        assert EventType.PROVIDER_OPERATION_RECONNECT_STARTED in {event.type for event in events}
        persisted = await store.load("resume-offline-queued")
        assert persisted is not None
        assert persisted.status is SessionStatus.INTERRUPTED
        assert await store.load_active_model_completion_stage("resume-offline-queued") is not None

    asyncio.run(scenario())


def test_competing_offline_recovery_workers_converge_on_one_completion() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(ProviderOperationStatus.COMPLETED)
        await _stage_offline_operation(
            store,
            session_id="competing-offline-recovery",
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        request = IncompleteSessionRecoveryRequest(
            session_id="competing-offline-recovery",
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )

        await asyncio.gather(
            app.recover_incomplete_session(request),
            app.recover_incomplete_session(request),
        )

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == [provider.adapter.state]
        events = await store.load_events("competing-offline-recovery")
        assert sum(event.type is EventType.MODEL_COMPLETED for event in events) == 1
        assert len(await store.load_transcript("competing-offline-recovery")) == 2

    asyncio.run(scenario())
