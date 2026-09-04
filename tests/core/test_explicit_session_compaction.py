from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError
from tests.core._execution_profile_fixtures import (
    profiled_session_identity,
    versioned_test_provider_identity,
)

import cayu.runtime._session_engine as session_engine_module
from cayu import (
    CayuConfig,
    ScriptedModelProvider,
    Tool,
    ToolContext,
    ToolEffect,
    ToolExecutionConfig,
    ToolResult,
    ToolSpec,
)
from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.artifacts import FileAttachmentKind, file_attachment
from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    Message,
    ToolCallPart,
    ToolResultPart,
)
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.providers import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    UsageDialect,
    bedrock_billing_identity,
    completed_bedrock_billing_identity,
)
from cayu.runtime import (
    BillingIdentity,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    CompactSessionRequest,
    ContextCompactor,
    ContextRequest,
    EventQuery,
    EventSink,
    ExecutionProfileComponentClass,
    ExecutionProfileMismatchError,
    ForkSessionRequest,
    InMemoryBudgetLedger,
    InMemoryEventSink,
    InMemorySessionStore,
    ModelCompactor,
    ModelPrice,
    PriceBook,
    PromptCacheCompactor,
    RequestFootprint,
    RequestFootprintConfig,
    RequestVariant,
    ResolutionActor,
    ResumeRequest,
    RetryPolicy,
    RunLimits,
    RunRequest,
    SessionIdentity,
    SessionRunFenced,
    SessionStatus,
    ToolCapabilityCeiling,
)
from cayu.runtime._event_projection import (
    PRIVATE_EVENT_AUTHORITY,
    public_event_id,
    public_event_sequence,
)
from cayu.runtime._run_limits import BudgetReservationLeaseLost
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.runtime.context import ContextBuildError
from cayu.storage import SQLiteSessionStore
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class RecordingCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.requests: list[CompactionRequest] = []

    def provider_budget_identity(self, _session) -> None:
        return None

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:recording-context-compactor",
            behavior_version="1",
            implementation_version="1",
        )

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.requests.append(request)
        summary = "durable compact summary"
        if request.existing_summary is not None:
            summary = f"{request.existing_summary}|{summary}"
        return CompactionResult(
            summary=summary,
            covered_message_count=len(request.messages),
            represented_existing_summary_sha256=(
                hashlib.sha256(request.existing_summary.encode("utf-8")).hexdigest()
                if request.existing_summary is not None
                else None
            ),
            metadata={"compactor": type(self).__name__, "mode": "deterministic"},
        )


async def _public_ids_for_private_event_ids(
    store: Any,
    session_id: str,
    event_ids: list[str],
) -> list[str]:
    records = await store.query_events(EventQuery(session_id=session_id, limit=5000))
    sequence_by_id = {record.event.id: record.sequence for record in records}
    return [public_event_id(sequence_by_id[event_id]) for event_id in event_ids]


async def _private_events_for_public_events(
    store: Any,
    session_id: str,
    events: list[Event],
) -> list[Event]:
    sequences = [public_event_sequence(event.id) for event in events]
    assert all(sequence is not None for sequence in sequences)
    records = await store.query_events(EventQuery(session_id=session_id, limit=5000))
    event_by_sequence = {record.sequence: record.event for record in records}
    return [event_by_sequence[sequence] for sequence in sequences]


class NoFullHistoryReplayStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    async def load_events(self, session_id: str):
        raise AssertionError("compaction replay must use indexed event lookup")


async def _create_profiled_session(
    app: CayuApp,
    store: Any,
    request: RunRequest,
    *,
    identity: SessionIdentity,
    **kwargs: Any,
):
    """Create a low-level compaction fixture with complete runtime authority."""

    registered_agent = app._agents[request.agent_name]
    ceiling = ToolCapabilityCeiling(tool_names=tuple(registered_agent.tools))
    prepared_request = request.model_copy(update={"tool_capability_ceiling": ceiling})
    if identity.execution_profile is None:
        identity = profiled_session_identity(
            provider_name=identity.provider_name,
            model=identity.model,
            agent_name=request.agent_name,
            app=app,
            invocation_loop_policies=request.loop_policies,
            structured_output=request.structured_output,
            thinking=request.thinking,
            request_budget_limits=request.budget_limits,
            causal_budget_id=(request.causal_budget_id or request.task_id or request.session_id),
            max_steps=request.max_steps,
            limits=request.limits,
            retry_policy=app._effective_retry_policy(request.retry_policy),
        )
    return await store.create(prepared_request, identity=identity, **kwargs)


def test_compact_session_rejects_incomplete_session_authority_before_mutation() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        compactor = RecordingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(compactor=compactor),
        )
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="incomplete-compaction-authority",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_transcript_messages(
            session.id,
            [Message.text("user", "must remain untouched")],
        )
        session = await store.update_status(session.id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="no durable execution-profile identity"):
            _ = [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=session.id,
                        idempotency_key="reject-incomplete-authority",
                        expected_run_epoch=session.run_epoch,
                        expected_transcript_cursor=1,
                    )
                )
            ]

        assert compactor.requests == []
        assert await store.load_events(session.id) == []
        assert await store.load_checkpoint(session.id) is None

    asyncio.run(run())


def test_policy_model_compaction_does_not_acknowledge_omitted_history() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = RecordingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        policy = CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=provider,
                model="summary-model",
                max_input_chars=1000,
            ),
            max_user_turns=1,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="summary-model"),
            context_policy=policy,
        )
        session = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_coverage_policy",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="recording-compaction", model="summary-model"),
        )
        transcript = [
            Message.text("user", "OLDEST_MUST_SURVIVE " + "x" * 5000),
            Message.text("assistant", "LATEST_COMPACTABLE"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(session.id, transcript)
        completed = await store.update_status(session.id, SessionStatus.COMPLETED)
        await store.checkpoint(
            session.id,
            {"future_additive_field": {"preserved": True}},
        )
        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=session.id,
                    idempotency_key="coverage-1",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]
        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is not None
        assert checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
        assert checkpoint["future_additive_field"] == {"preserved": True}
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 1
        first_request_count = len(provider.requests)
        first_footprints = [
            RequestFootprint.model_validate(event.payload)
            for event in events
            if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert len(first_footprints) == first_request_count
        assert all(
            footprint.request_variant == RequestVariant.CONTEXT_COMPACTION
            for footprint in first_footprints
        )
        first_prompts = [request.messages[-1].content[0].text for request in provider.requests]
        assert any("OLDEST_MUST_SURVIVE" in prompt for prompt in first_prompts)
        assert all("LATEST_COMPACTABLE" not in prompt for prompt in first_prompts)
        assert "OLDEST_MUST_SURVIVE" in checkpoint["context_compaction"]["summary"]
        completed_event = next(
            event for event in events if event.type == EventType.CONTEXT_COMPACTION_COMPLETED
        )
        assert completed_event.payload["requested_source_start"] == 0
        assert completed_event.payload["requested_source_end"] == 2
        assert completed_event.payload["represented_source_start"] == 0
        assert completed_event.payload["represented_source_end"] == 1
        assert completed_event.payload["represented_message_count"] == 1
        assert completed_event.payload["coverage_mode"] == "partial_prefix"
        assert completed_event.payload["chunk_count"] > 1
        assert completed_event.payload["chunk_mode"] == "hierarchical_atomic_unit"
        assert completed_event.payload["bounded_input"] is True
        assert completed_event.payload["compaction_failed"] is False
        assert "OLDEST_MUST_SURVIVE" not in json.dumps(completed_event.model_dump(mode="json"))
        checkpointed_event = next(
            event for event in events if event.type == EventType.SESSION_CHECKPOINTED
        )
        assert checkpointed_event.payload["compacted_transcript_cursor"] == 1
        assert checkpointed_event.payload["newly_compacted_message_count"] == 1

        first_projection = await policy.build_with_checkpoint(
            ContextRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="summary-model"),
                messages=transcript,
                step=1,
            ),
            checkpoint=checkpoint,
        )
        first_projection_text = json.dumps(
            [message.model_dump(mode="json") for message in first_projection.messages]
        )
        assert first_projection_text.count("OLDEST_MUST_SURVIVE") == 1
        assert first_projection_text.count("LATEST_COMPACTABLE") == 1

        retry_events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=session.id,
                    idempotency_key="coverage-2",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]
        retry_checkpoint = await store.load_checkpoint(session.id)
        assert retry_checkpoint is not None
        assert retry_checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2
        retry_prompts = [
            request.messages[-1].content[0].text
            for request in provider.requests[first_request_count:]
        ]
        retry_footprints = [
            RequestFootprint.model_validate(event.payload)
            for event in retry_events
            if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert len(retry_footprints) == len(retry_prompts)
        assert any("OLDEST_MUST_SURVIVE" in prompt for prompt in retry_prompts)
        assert any("LATEST_COMPACTABLE" in prompt for prompt in retry_prompts)
        retry_projection = await policy.build_with_checkpoint(
            ContextRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="summary-model"),
                messages=transcript,
                step=1,
            ),
            checkpoint=retry_checkpoint,
        )
        retry_projection_text = json.dumps(
            [message.model_dump(mode="json") for message in retry_projection.messages]
        )
        assert retry_projection_text.count("OLDEST_MUST_SURVIVE") == 1
        assert retry_projection_text.count("LATEST_COMPACTABLE") == 1
        assert any(event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in retry_events)

    asyncio.run(run())


def test_compact_session_lost_terminal_ack_keeps_completed_state_unambiguous() -> None:
    class LostTerminalAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            result = await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )
            if not self.failed and any(
                event.type == EventType.SESSION_CHECKPOINTED for event in kwargs.get("events", [])
            ):
                self.failed = True
                raise ConnectionError("terminal acknowledgement lost after commit")
            return result

    async def run() -> None:
        store = LostTerminalAcknowledgementStore()
        compactor = RecordingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_terminal_ack_reconciliation",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="terminal-ack-reconciliation",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        with pytest.raises(ConnectionError, match="terminal acknowledgement lost"):
            async for _event in app.compact_session(request):
                pass

        durable = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        assert EventType.CONTEXT_COMPACTION_COMPLETED in {event.type for event in durable}
        assert EventType.SESSION_CHECKPOINTED in {event.type for event in durable}
        assert EventType.CONTEXT_COMPACTION_FAILED not in {event.type for event in durable}
        operation = await store.load_session_operation(created.id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "completed"

        replay = [event async for event in app.compact_session(request)]
        assert [event.id for event in replay] == await _public_ids_for_private_event_ids(
            store,
            created.id,
            operation["event_ids"],
        )
        assert len(compactor.requests) == 1

    asyncio.run(run())


def test_compact_session_cancellation_during_terminal_reconciliation_propagates() -> None:
    class LostTerminalAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            result = await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )
            if any(
                event.type == EventType.SESSION_CHECKPOINTED for event in kwargs.get("events", [])
            ):
                raise ConnectionError("terminal acknowledgement lost after commit")
            return result

    async def run() -> None:
        store = LostTerminalAcknowledgementStore()
        compactor = RecordingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_terminal_reconciliation_cancellation",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="terminal-reconciliation-cancellation",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )
        reconciliation_started = asyncio.Event()
        allow_reconciliation = asyncio.Event()
        original_is_persisted = app._event_writer.is_persisted

        async def block_terminal_reconciliation(event: Event) -> bool:
            if event.type == EventType.SESSION_CHECKPOINTED:
                reconciliation_started.set()
                await allow_reconciliation.wait()
            return await original_is_persisted(event)

        app._event_writer.is_persisted = block_terminal_reconciliation

        async def collect() -> list[Event]:
            return [event async for event in app.compact_session(request)]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(reconciliation_started.wait(), timeout=5)
        task.cancel("cancel during terminal reconciliation")
        assert task.cancelling() == 1
        allow_reconciliation.set()
        with pytest.raises(
            asyncio.CancelledError,
            match="cancel during terminal reconciliation",
        ):
            await task

        assert task.cancelled()
        durable = await store.load_events(created.id)
        assert EventType.SESSION_CHECKPOINTED in {event.type for event in durable}
        assert EventType.CONTEXT_COMPACTION_FAILED not in {event.type for event in durable}
        operation = await store.load_session_operation(created.id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "completed"

    asyncio.run(run())


@pytest.mark.parametrize(
    ("blocked_event_type", "expected_provider_calls"),
    [
        (EventType.BUDGET_CHECKED, 0),
        (EventType.SESSION_CHECKPOINTED, 1),
    ],
)
def test_compact_session_cancellation_does_not_wait_forever_for_stalled_publication(
    blocked_event_type: EventType,
    expected_provider_calls: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StalledPublicationStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.blocked = asyncio.Event()
            self.blocked_once = False
            self.release = asyncio.Event()
            self.child_cancellation_observed = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if not self.blocked_once and any(
                event.type == blocked_event_type for event in kwargs.get("events", [])
            ):
                self.blocked_once = True
                self.blocked.set()
                while not self.release.is_set():
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        # SQLite's physical worker has this shape: cancelling
                        # its await does not stop the write already in progress.
                        self.child_cancellation_observed.set()
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    class CountingProvider(ModelProvider):
        name = "stalled-publication-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_STORE_WAIT_TIMEOUT_SECONDS",
            0.01,
        )
        store = StalledPublicationStore()
        provider = CountingProvider()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name=provider.name,
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        sink = InMemoryEventSink()
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_stalled_{blocked_event_type.value}",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key=f"stalled-{blocked_event_type.value}",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        async def collect() -> list[Event]:
            return [event async for event in app.compact_session(request)]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(store.blocked.wait(), timeout=1)
        task.cancel("cancel stalled explicit compaction publication")
        assert task.cancelling() == 1
        await asyncio.wait_for(store.child_cancellation_observed.wait(), timeout=1)
        if blocked_event_type == EventType.BUDGET_CHECKED:
            done, _pending = await asyncio.wait({task}, timeout=1)
            # Release only after observing the caller outcome: a stalled
            # pre-dispatch write must not own caller cancellation indefinitely.
            store.release.set()
            assert task in done
            with pytest.raises(
                asyncio.CancelledError,
                match="cancel stalled explicit compaction publication",
            ):
                await task
        else:
            # Post-dispatch completion evidence remains synchronously owned
            # until its physical write resolves.
            assert not task.done()
            store.release.set()
            with pytest.raises(
                asyncio.CancelledError,
                match="cancel stalled explicit compaction publication",
            ):
                await asyncio.wait_for(task, timeout=1)

        assert task.cancelled()
        assert store.child_cancellation_observed.is_set()
        assert provider.calls == expected_provider_calls
        expected_operation_status = (
            "completed" if blocked_event_type == EventType.SESSION_CHECKPOINTED else "failed"
        )
        operation = None
        for _ in range(100):
            operation = await store.load_session_operation(created.id, request.idempotency_key)
            if operation is not None and operation["status"] == expected_operation_status:
                break
            await asyncio.sleep(0.01)
        assert operation is not None
        assert operation["status"] == expected_operation_status
        durable = await store.load_events(created.id)
        if blocked_event_type == EventType.SESSION_CHECKPOINTED:
            assert EventType.CONTEXT_COMPACTION_COMPLETED in {event.type for event in durable}
            assert EventType.CONTEXT_COMPACTION_FAILED not in {event.type for event in durable}
            assert operation["status"] == "completed"
        else:
            assert EventType.CONTEXT_COMPACTION_FAILED in {event.type for event in durable}
            assert operation["status"] == "failed"
            durable_budget_checks = [
                record
                for record in await store.query_events(EventQuery(session_id=created.id))
                if record.event.type == EventType.BUDGET_CHECKED
            ]
            sink_budget_checks = [
                event for event in sink.events if event.type == EventType.BUDGET_CHECKED
            ]
            assert [event.id for event in sink_budget_checks] == [
                public_event_id(record.sequence) for record in durable_budget_checks
            ]

    asyncio.run(run())


def test_compact_session_cancellation_does_not_wait_for_stalled_completion_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingCompletionSink(EventSink):
        def __init__(self) -> None:
            self.blocked = asyncio.Event()
            self.release = asyncio.Event()
            self.cancellation_observed = asyncio.Event()
            self.events: list[Event] = []
            self.completion_delivery_calls = 0

        async def emit(self, event: Event) -> None:
            if (
                event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
            ):
                self.completion_delivery_calls += 1
                if self.completion_delivery_calls == 1:
                    self.blocked.set()
                    while not self.release.is_set():
                        try:
                            await self.release.wait()
                        except asyncio.CancelledError:
                            # Model a delivery transport that continues despite
                            # cancellation of the task awaiting it.
                            self.cancellation_observed.set()
            self.events.append(event.model_copy(deep=True))

    class CompletionProvider(ModelProvider):
        name = "explicit-stalled-completion-sink-provider"

        async def stream(self, request: ModelRequest):
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_STORE_WAIT_TIMEOUT_SECONDS",
            0.01,
        )
        store = InMemorySessionStore()
        sink = BlockingCompletionSink()
        provider = CompletionProvider()
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_stalled_completion_sink",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="stalled-completion-sink",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        async def collect() -> list[Event]:
            return [event async for event in app.compact_session(request)]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(sink.blocked.wait(), timeout=1)
        durable = await store.load_events(created.id)
        completions = [
            event
            for event in durable
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1

        task.cancel("cancel explicit compaction with stalled completion sink")
        assert task.cancelling() == 1
        await asyncio.wait_for(sink.cancellation_observed.wait(), timeout=1)
        done, _pending = await asyncio.wait({task}, timeout=1)
        # The durable handoff owns delivery now, so the caller must finish
        # before the sink is released.
        assert task in done
        with pytest.raises(
            asyncio.CancelledError,
            match="cancel explicit compaction with stalled completion sink",
        ):
            await task
        assert task.cancelled()

        sink.release.set()
        for _ in range(100):
            delivered = [
                event
                for event in sink.events
                if event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
            ]
            operation = await store.load_session_operation(created.id, request.idempotency_key)
            if delivered and operation is not None and operation["status"] == "failed":
                break
            await asyncio.sleep(0.01)
        completion_record = next(
            record
            for record in await store.query_events(EventQuery(session_id=created.id))
            if record.event.id == completions[0].id
        )
        assert [event.id for event in delivered] == [public_event_id(completion_record.sequence)]
        assert sink.completion_delivery_calls == 1
        assert operation is not None
        assert operation["status"] == "failed"

    asyncio.run(run())


def test_compact_session_late_completion_commit_is_not_republished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DelayedCompletionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.blocked = asyncio.Event()
            self.release = asyncio.Event()
            self.committed = asyncio.Event()
            self.blocked_once = False
            self.cancellations = 0

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if not self.blocked_once and any(
                event.type == EventType.MODEL_COMPLETED for event in kwargs.get("events", [])
            ):
                self.blocked_once = True
                self.blocked.set()
                while not self.release.is_set():
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        # Match a SQLite worker whose physical commit continues
                        # after cancellation of the awaiting task.
                        self.cancellations += 1
            result = await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )
            if any(event.type == EventType.MODEL_COMPLETED for event in kwargs.get("events", [])):
                self.committed.set()
            return result

    class CompletionProvider(ModelProvider):
        name = "late-completion-provider"

        async def stream(self, request: ModelRequest):
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_STORE_WAIT_TIMEOUT_SECONDS",
            0.01,
        )
        store = DelayedCompletionStore()
        provider = CompletionProvider()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name=provider.name,
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        original_is_persisted = app._event_writer.is_persisted
        original_fan_out = app._event_writer.fan_out_persisted
        stale_completion_reconciled = False
        completion_fan_out_calls = 0

        async def release_after_stale_completion_check(event: Event) -> bool:
            nonlocal stale_completion_reconciled
            persisted = await original_is_persisted(event)
            if (
                not stale_completion_reconciled
                and event.type == EventType.MODEL_COMPLETED
                and not persisted
            ):
                stale_completion_reconciled = True
                store.release.set()
                await store.committed.wait()
                # Model the exact race: the reconciliation read was stale, but
                # the timed-out physical write completed before failure cleanup.
                return False
            return persisted

        app._event_writer.is_persisted = release_after_stale_completion_check

        async def fail_reconciled_completion_fan_out(events: list[Event]) -> list[Event]:
            nonlocal completion_fan_out_calls
            if any(event.type == EventType.MODEL_COMPLETED for event in events):
                completion_fan_out_calls += 1
                if completion_fan_out_calls == 2:
                    raise RuntimeError("reconciled completion fan-out failed")
            return await original_fan_out(events)

        app._event_writer.fan_out_persisted = fail_reconciled_completion_fan_out
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_late_completion_commit",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="late-completion-commit",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        observed: list[Event] = []

        async def collect() -> None:
            async for event in app.compact_session(request):
                observed.append(event)

        task = asyncio.create_task(collect())
        await asyncio.wait_for(store.blocked.wait(), timeout=1)
        with pytest.raises(ContextBuildError, match="bounded store wait"):
            await asyncio.wait_for(task, timeout=1)
        assert stale_completion_reconciled
        assert store.committed.is_set()
        assert completion_fan_out_calls == 2
        operation = None
        for _ in range(100):
            operation = await store.load_session_operation(created.id, request.idempotency_key)
            if operation is not None and operation["status"] != "running":
                break
            await asyncio.sleep(0.01)

        durable = await store.load_events(created.id)
        completions = [event for event in durable if event.type == EventType.MODEL_COMPLETED]
        assert len(completions) == 1
        assert len({event.id for event in observed}) == len(observed)
        assert any(event.type == EventType.BUDGET_CHECKED for event in observed)
        assert [
            event.id for event in observed if event.type == EventType.MODEL_COMPLETED
        ] == await _public_ids_for_private_event_ids(store, created.id, [completions[0].id])
        assert operation is not None
        assert operation["status"] == "failed"
        assert EventType.CONTEXT_COMPACTION_FAILED in {event.type for event in durable}

    asyncio.run(run())


def test_compact_session_lost_budget_event_ack_releases_without_duplicate() -> None:
    class LostBudgetAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            result = await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )
            if not self.failed and any(
                event.type == EventType.BUDGET_RESERVED for event in kwargs.get("events", [])
            ):
                self.failed = True
                raise ConnectionError("budget event acknowledgement lost")
            return result

    class CountingProvider(ModelProvider):
        name = "lost-budget-ack-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        store = LostBudgetAcknowledgementStore()
        provider = CountingProvider()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=60)
        sink = InMemoryEventSink()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name=provider.name,
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        app = CayuApp(
            session_store=store,
            budget_ledger=ledger,
            event_sinks=[sink],
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
            clock=lambda: datetime(2026, 7, 22, tzinfo=UTC),
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_ack_reconciliation",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="budget-ack-reconciliation",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        with pytest.raises(ContextBuildError, match="budget event acknowledgement lost"):
            async for _event in app.compact_session(request):
                pass

        durable = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        types = [event.type for event in durable]
        assert types.count(EventType.BUDGET_RESERVED) == 1
        assert types.count(EventType.BUDGET_RESERVATION_RELEASED) == 1
        assert types.count(EventType.CONTEXT_COMPACTION_FAILED) == 1
        assert sum(event.type == EventType.BUDGET_RESERVED for event in sink.events) == 1
        assert (
            sum(event.type == EventType.BUDGET_RESERVATION_RELEASED for event in sink.events) == 1
        )
        assert provider.calls == 0
        operation = await store.load_session_operation(created.id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "failed"

    asyncio.run(run())


@pytest.mark.parametrize("action", ["interrupt", "notify"])
def test_compact_session_lost_limit_event_ack_is_not_duplicated(action: str) -> None:
    class LostLimitAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            result = await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )
            if not self.failed and any(
                event.type == EventType.BUDGET_LIMIT_REACHED for event in kwargs.get("events", [])
            ):
                self.failed = True
                raise ConnectionError("limit event acknowledgement lost")
            return result

    class CountingProvider(ModelProvider):
        name = "lost-limit-ack-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            yield ModelStreamEvent.completed({"model": request.model})

    async def run() -> None:
        store = LostLimitAcknowledgementStore()
        provider = CountingProvider()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name=provider.name,
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.000001"),
                        pricing=pricing,
                        action=action,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_limit_ack_{action}",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_event(
            created.id,
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id=created.id,
                agent_name="assistant",
                payload={
                    "provider_name": provider.name,
                    "requested_model": "summary-model",
                    "model": "summary-model",
                    "usage_metrics": {
                        "input_tokens": 10,
                        "output_tokens": 0,
                        "total_tokens": 10,
                    },
                },
            ),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key=f"limit-ack-{action}",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        with pytest.raises(ConnectionError, match="limit event acknowledgement lost"):
            async for _event in app.compact_session(request):
                pass

        durable = await store.load_events(created.id)
        assert sum(event.type == EventType.BUDGET_LIMIT_REACHED for event in durable) == 1
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in durable) == 1
        assert provider.calls == 0
        operation = await store.load_session_operation(created.id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "failed"

    asyncio.run(run())


def test_compact_session_attempt_ack_expiry_blocks_first_provider_dispatch() -> None:
    accepted_at = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
    now = {"value": accepted_at}

    class ExpiringAttemptAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            result = await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )
            if any(event.type == EventType.BUDGET_RESERVED for event in kwargs.get("events", [])):
                now["value"] = accepted_at + timedelta(minutes=6)
            return result

    class CountingProvider(ModelProvider):
        name = "expired-attempt-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            yield ModelStreamEvent.completed({"model": request.model})

    async def run() -> None:
        store = ExpiringAttemptAcknowledgementStore(ownership_clock=lambda: now["value"])
        provider = CountingProvider()
        sink = InMemoryEventSink()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name=provider.name,
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_attempt_ack_expiry",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(ContextBuildError, match="claim.*expired"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="attempt-ack-expiry",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert provider.calls == 0
        durable_reservations = [
            record
            for record in await store.query_events(EventQuery(session_id=created.id))
            if record.event.type == EventType.BUDGET_RESERVED
        ]
        sink_reservations = [
            event for event in sink.events if event.type == EventType.BUDGET_RESERVED
        ]
        assert len(durable_reservations) == 1
        assert [event.id for event in sink_reservations] == [
            public_event_id(record.sequence) for record in durable_reservations
        ]

    asyncio.run(run())


def test_compact_session_completion_ack_expiry_blocks_next_hierarchy_dispatch() -> None:
    accepted_at = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
    now = {"value": accepted_at}

    class ExpiringCompletionAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        expired = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            result = await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )
            if not self.expired and any(
                event.type == EventType.MODEL_COMPLETED for event in kwargs.get("events", [])
            ):
                self.expired = True
                now["value"] = accepted_at + timedelta(minutes=6)
            return result

    class HierarchyProvider(ModelProvider):
        name = "expired-hierarchy-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            yield ModelStreamEvent.text_delta(f"summary {self.calls}")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        store = ExpiringCompletionAcknowledgementStore(ownership_clock=lambda: now["value"])
        provider = HierarchyProvider()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_completion_ack_expiry",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(ContextBuildError, match="claim.*expired"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="completion-ack-expiry",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert provider.calls == 1

    asyncio.run(run())


def test_compact_session_invalid_usage_fails_closed_on_next_strict_budget_check() -> None:
    class InvalidUsageProvider(ModelProvider):
        name = "invalid-usage-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage_metrics": {
                        "input_tokens": 2**63,
                        "output_tokens": 1,
                        "total_tokens": 2**63,
                    },
                }
            )

    async def run() -> None:
        store = InMemorySessionStore()
        provider = InvalidUsageProvider()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name=provider.name,
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        allow_unpriced=False,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_invalid_usage_strict_budget",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(ContextBuildError, match="integer_out_of_range"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="invalid-usage-first",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass
        with pytest.raises(RuntimeError, match="budget limit reached"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="invalid-usage-second",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert provider.calls == 1
        completions = [
            event
            for event in await store.load_events(created.id)
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert "usage_metrics" not in completions[0].payload
        assert completions[0].payload["usage_unavailable_reason"] == (
            "invalid compaction usage telemetry"
        )

    asyncio.run(run())


def test_explicit_compaction_retry_rejects_live_provider_drift_before_dispatch(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("must not dispatch"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ],
            name="compaction-provider",
        )
        consume_batch = provider._consume_batch

        def consume_then_mutate(request: ModelRequest) -> tuple[ModelStreamEvent, ...]:
            batch = consume_batch(request)
            if len(provider.requests) == 1:
                monkeypatch.setattr(provider, "_operation_adapter", object())
                raise ModelProviderError(
                    "compactor overloaded",
                    provider=provider.name,
                    status_code=503,
                    retryable=True,
                )
            return batch

        monkeypatch.setattr(provider, "_consume_batch", consume_then_mutate)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
                ),
                max_user_turns=1,
            ),
        )
        session_id = "explicit-compaction-live-provider-drift"
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(session_id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(ContextBuildError) as caught:
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=session_id,
                    idempotency_key="explicit-live-provider-drift",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        profile_mismatch = caught.value.__cause__
        assert isinstance(profile_mismatch, ExecutionProfileMismatchError)
        assert profile_mismatch.changed_component_classes == (
            ExecutionProfileComponentClass.CONTEXT_COMPACTION,
        )
        assert len(provider.requests) == 1
        compaction_completions = [
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(compaction_completions) == 1
        assert compaction_completions[0].payload["compaction_outcome"] == "provider_error"

    asyncio.run(scenario())


def test_explicit_compaction_lost_completion_ack_is_restart_safe() -> None:
    class LostCompletionAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            events = kwargs.get("events", [])
            is_compaction_completion = any(
                event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
                for event in events
            )
            result = await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )
            if is_compaction_completion and not self.failed:
                self.failed = True
                raise ConnectionError("explicit completion acknowledgement lost")
            return result

    class RetryableThenSuccessfulProvider(ModelProvider):
        name = "compaction-provider"

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return versioned_test_provider_identity(self)

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest):
            self.requests.append(request)
            if len(self.requests) == 1:
                raise ModelProviderError(
                    "provider overloaded",
                    provider=self.name,
                    status_code=503,
                    retryable=True,
                )
            yield ModelStreamEvent.text_delta("must not retry after restart")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        store = LostCompletionAcknowledgementStore()
        provider = RetryableThenSuccessfulProvider()
        sink = InMemoryEventSink()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name=provider.name,
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        policy = BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=pricing,
                ),
            )
        )

        def new_app() -> CayuApp:
            app = CayuApp(
                session_store=store,
                budget_policy=policy,
                event_sinks=[sink],
                enable_logging=False,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=ModelCompactor(
                        provider=provider,
                        model="summary-model",
                        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
                    ),
                    max_user_turns=1,
                ),
            )
            return app

        session_id = "sess_explicit_compaction_lost_ack_restart"
        creation_app = new_app()
        created = await _create_profiled_session(
            creation_app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(session_id, transcript)
        completed = await store.update_status(session_id, SessionStatus.COMPLETED)

        first_events: list[Event] = []
        with pytest.raises(ContextBuildError, match="provider overloaded"):
            async for event in creation_app.compact_session(
                CompactSessionRequest(
                    session_id=session_id,
                    idempotency_key="lost-ack-first-process",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                first_events.append(event)

        assert len(provider.requests) == 1
        durable_after_first = [
            record.event
            for record in await store.query_events(EventQuery(session_id=session_id, limit=100))
        ]
        completions = [
            event
            for event in durable_after_first
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["compaction_outcome"] == "provider_error"
        assert (
            sum(
                event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
                for event in sink.events
            )
            == 1
        )
        assert (
            sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in durable_after_first)
            == 1
        )

        # A fresh app instance reconstructs the strict budget from durable event
        # evidence. The unknown first dispatch fails closed before the provider's
        # otherwise-successful second response can run.
        restart_events: list[Event] = []
        with pytest.raises(RuntimeError, match="budget limit reached"):
            async for event in new_app().compact_session(
                CompactSessionRequest(
                    session_id=session_id,
                    idempotency_key="lost-ack-second-process",
                    expected_run_epoch=created.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                restart_events.append(event)

        assert len(provider.requests) == 1
        assert any(event.type == EventType.BUDGET_LIMIT_REACHED for event in restart_events)
        durable_after_restart = [
            record.event
            for record in await store.query_events(EventQuery(session_id=session_id, limit=100))
        ]
        assert (
            sum(
                event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
                for event in durable_after_restart
            )
            == 1
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint

    asyncio.run(run())


def test_unfinished_explicit_compaction_fails_closed_after_restart() -> None:
    class UnfinishedThenSuccessfulProvider(ModelProvider):
        name = "compaction-provider"

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return versioned_test_provider_identity(self)

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest):
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.text_delta("partial summary")
                return
            yield ModelStreamEvent.text_delta("must not run")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        store = InMemorySessionStore()
        provider = UnfinishedThenSuccessfulProvider()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name=provider.name,
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        budget_policy = BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=pricing,
                ),
            )
        )

        def new_app() -> CayuApp:
            app = CayuApp(
                session_store=store,
                budget_policy=budget_policy,
                enable_logging=False,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=ModelCompactor(
                        provider=provider,
                        model="summary-model",
                    ),
                    max_user_turns=1,
                ),
            )
            return app

        creation_app = new_app()
        created = await _create_profiled_session(
            creation_app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_unfinished_restart",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        first_events: list[Event] = []
        with pytest.raises(
            ContextBuildError,
            match="stream ended without a completed event",
        ):
            async for event in creation_app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="explicit-unfinished-first-process",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                first_events.append(event)

        uncertain = next(
            event
            for event in first_events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        )
        assert uncertain.payload["compaction_outcome"] == "unfinished_stream"
        assert uncertain.payload["usage_unavailable_reason"]

        restart_events: list[Event] = []
        with pytest.raises(RuntimeError, match="budget limit reached"):
            async for event in new_app().compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="explicit-unfinished-second-process",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                restart_events.append(event)

        assert len(provider.requests) == 1
        assert any(event.type == EventType.BUDGET_LIMIT_REACHED for event in restart_events)
        durable = await store.query_events(EventQuery(session_id=created.id, limit=100))
        completions = [
            record
            for record in durable
            if record.event.type == EventType.MODEL_COMPLETED
            and record.event.payload.get("purpose") == "context_compaction"
        ]
        assert [record.sequence for record in completions] == [public_event_sequence(uncertain.id)]

    asyncio.run(run())


class FailingCompletionPublishStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def publish_session_operation_guarded_with_store_time(self, session_id: str, **kwargs):
        events = kwargs.get("events", [])
        if not self.failed and any(
            event.type == EventType.SESSION_CHECKPOINTED for event in events
        ):
            self.failed = True
            raise RuntimeError("simulated completion publication failure")
        return await super().publish_session_operation_guarded_with_store_time(session_id, **kwargs)


class BlockingCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def provider_budget_identity(self, _session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.started.set()
        await self.release.wait()
        return CompactionResult(summary="summary", covered_message_count=len(request.messages))


class BlockingCompactionProvider(ModelProvider):
    name = "compaction-provider"

    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def stream(self, request: ModelRequest):
        self.calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        yield ModelStreamEvent.text_delta("provider summary")
        yield ModelStreamEvent.completed(
            {"model": request.model, "usage": {"input_tokens": 8, "output_tokens": 2}}
        )


def _test_compaction_heartbeat_state(
    claim_expires_at: datetime,
) -> session_engine_module._SessionOperationClaimHeartbeatState:
    state = session_engine_module._SessionOperationClaimHeartbeatState()
    state.confirm_claim(
        claim_expires_at,
        claim_deadline_monotonic=(
            time.monotonic() + session_engine_module._SESSION_OPERATION_CLAIM_LEASE.total_seconds()
        ),
    )
    return state


class OverlappingCompactor(ModelCompactor):
    def __init__(self) -> None:
        provider = OverlappingCompactionProvider()
        super().__init__(
            provider=provider,
            model="summary-model",
            max_input_chars=100_000,
        )
        self.started = provider.started
        self.release = provider.release

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:overlapping-model-compactor",
            behavior_version="1",
            implementation_version="1",
        )


class OverlappingCompactionProvider(ModelProvider):
    name = "overlap-compactor"

    def __init__(self) -> None:
        self.started = [asyncio.Event(), asyncio.Event()]
        self.release = [asyncio.Event(), asyncio.Event()]
        self.calls = 0

    async def stream(self, request: ModelRequest):
        del request
        call = self.calls
        self.calls += 1
        self.started[call].set()
        await self.release[call].wait()
        yield ModelStreamEvent.text_delta(f"summary from attempt {call + 1}")
        yield ModelStreamEvent.completed(
            {
                "model": "summary-model",
                "usage": {"input_tokens": call + 1, "output_tokens": 1},
            }
        )


class CompletingProvider(ModelProvider):
    name = "fake"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:completing-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest):
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({})


class UsageCompactionProvider(ModelProvider):
    name = "compaction-provider"

    def __init__(self, *, summary: str = "provider summary") -> None:
        self.calls = 0
        self.summary = summary

    async def stream(self, request: ModelRequest):
        self.calls += 1
        yield ModelStreamEvent.text_delta(self.summary)
        yield ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}})


class ReservationInspectingCompactionProvider(ModelProvider):
    name = "compaction-provider"

    def __init__(self, store: InMemorySessionStore) -> None:
        self.store = store
        self.session_id: str | None = None
        self.calls = 0
        self.durable_types_before_dispatch: list[EventType] | None = None

    async def stream(self, request: ModelRequest):
        self.calls += 1
        assert self.session_id is not None
        self.durable_types_before_dispatch = [
            record.event.type
            for record in await self.store.query_events(
                EventQuery(session_id=self.session_id, limit=100)
            )
        ]
        yield ModelStreamEvent.text_delta("provider summary")
        yield ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}})


class RecordingCompactionProvider(ModelProvider):
    name = "recording-compaction"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        prompt = request.messages[-1].content[0].text
        preserved = [
            marker for marker in ("OLDEST_MUST_SURVIVE", "LATEST_COMPACTABLE") if marker in prompt
        ]
        yield ModelStreamEvent.text_delta(" ".join(preserved) or "provider summary")
        yield ModelStreamEvent.completed({})


class ContextualBedrockCompactionProvider(ModelProvider):
    name = "bedrock"
    billing_provider_name = "bedrock"

    def __init__(self, identity: BillingIdentity, *, report_usage: bool = True) -> None:
        self.identity = identity
        self.report_usage = report_usage
        self.calls = 0

    async def billing_identity_for_request(
        self,
        request: ModelRequest,
    ) -> BillingIdentity:
        assert request.model == self.identity.resource_id
        return self.identity

    def billing_identity_for_completion(
        self,
        identity: BillingIdentity | None,
        payload: dict,
    ) -> BillingIdentity | None:
        assert identity == self.identity
        return completed_bedrock_billing_identity(
            self.identity,
            effective_service_tier="default",
        )

    async def stream(self, request: ModelRequest):
        self.calls += 1
        yield ModelStreamEvent.text_delta("provider summary")
        completed: dict[str, object] = {"model": request.model}
        if self.report_usage:
            completed["usage"] = {"input_tokens": 8, "output_tokens": 2}
        yield ModelStreamEvent.completed(completed)


class FinalRenewalFailureBudgetLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__()
        self.heartbeat_calls = 0

    async def heartbeat(self, *, reservation_id: str) -> bool:
        self.heartbeat_calls += 1
        if self.heartbeat_calls == 2:
            return False
        return await super().heartbeat(reservation_id=reservation_id)


class InitialRenewalFailureBudgetLedger(InMemoryBudgetLedger):
    async def heartbeat(self, *, reservation_id: str) -> bool:
        return False


class HeartbeatCancellationBudgetLedger(FinalRenewalFailureBudgetLedger):
    @property
    def reservation_ttl_seconds(self) -> int:
        return 0


class FailingSecondReservationBudgetLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__()
        self.reserve_calls = 0
        self.release_calls = 0

    async def reserve(self, **kwargs):
        self.reserve_calls += 1
        if self.reserve_calls == 2:
            raise RuntimeError("simulated reservation store failure")
        return await super().reserve(**kwargs)

    async def release(self, **kwargs):
        self.release_calls += 1
        return await super().release(**kwargs)


class FailingSecondReleaseBudgetLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__()
        self.release_calls = 0
        self.reservation_ids: list[str] = []

    async def reserve(self, **kwargs):
        result = await super().reserve(**kwargs)
        if result.record is not None:
            self.reservation_ids.append(result.record.reservation_id)
        return result

    async def release(self, **kwargs):
        self.release_calls += 1
        if self.release_calls == 2:
            raise RuntimeError("simulated second release failure")
        return await super().release(**kwargs)


class UndeclaredProviderCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.calls = 0

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.calls += 1
        return CompactionResult(
            summary="undeclared provider summary",
            covered_message_count=len(request.messages),
            model_completed_payloads=[
                {
                    "provider_name": "compaction-provider",
                    "model": "summary-model",
                    "usage": {"input_tokens": 8, "output_tokens": 2},
                }
            ],
        )


class UnsafeProviderIdentityCompactor(ContextCompactor):
    def __init__(self, identity: tuple[str, str]) -> None:
        self.identity = identity
        self.calls = 0

    def provider_budget_identity(self, _session) -> tuple[str, str]:
        return self.identity

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.calls += 1
        return CompactionResult(
            summary="must not execute", covered_message_count=len(request.messages)
        )


class CancellationCompletingProvider(ModelProvider):
    name = "compaction-provider"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest):
        self.calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            yield ModelStreamEvent.text_delta("completed while cancellation was handled")
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}})
            return
        raise AssertionError("blocking compactor unexpectedly resumed")


class FailOnceCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.calls = 0

    def provider_budget_identity(self, _session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("provider prompt echoed: secret instructions")
        return CompactionResult(
            summary="retry summary", covered_message_count=len(request.messages)
        )


class AdversarialTelemetryCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.billing_identity = BillingIdentity(
            provider_name="fake-compactor",
            resource_id="summary-model-v1",
        )

    def provider_budget_identity(self, _session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        return CompactionResult(
            summary="private summary text",
            covered_message_count=len(request.messages),
            metadata={
                "summary": "metadata summary leak",
                "instructions": "metadata instruction leak",
                "huge": "m" * 100_000,
            },
            model_completed_payloads=[
                {
                    "provider_name": "fake-compactor",
                    "requested_model": "summary-model",
                    "model": "summary-model-v1",
                    "usage_metrics": {
                        "provider_name": "fake-compactor",
                        "requested_model": "summary-model",
                        "model": "summary-model-v1",
                        "billing_identity": self.billing_identity.model_dump(mode="json"),
                        "input_tokens": 8,
                        "output_tokens": 2,
                        "total_tokens": 10,
                    },
                    "summary": "model event summary leak",
                    "instructions": "model event instruction leak",
                    "provider_state": {"secret": "opaque continuation leak"},
                    "arbitrary": "p" * 100_000,
                }
            ],
        )


def test_compact_session_request_requires_idempotency_and_source_fences() -> None:
    with pytest.raises(ValidationError):
        CompactSessionRequest(  # type: ignore[call-arg]
            session_id="sess_compact_contract",
            expected_run_epoch=0,
            expected_transcript_cursor=3,
        )

        with pytest.raises(ValidationError):
            CompactSessionRequest(
                session_id="sess_compact_contract",
                idempotency_key="compact-1",
                expected_run_epoch=0,
                expected_transcript_cursor=3,
                instructions="x" * 4097,
            )

    with pytest.raises(ValidationError):
        CompactSessionRequest(
            session_id="sess_compact_contract",
            idempotency_key="compact-1",
            expected_run_epoch=0,
            expected_transcript_cursor=3,
            instructions="invalid surrogate: \ud800",
        )

    for field_name, value in (
        ("session_id", "invalid-surrogate-\ud800"),
        ("session_id", "invalid-nul-\x00"),
        ("idempotency_key", "invalid-nul-\x00"),
        ("instructions", "invalid-nul-\x00"),
    ):
        values = {
            "session_id": "sess_compact_contract",
            "idempotency_key": "compact-1",
            "expected_run_epoch": 0,
            "expected_transcript_cursor": 3,
            field_name: value,
        }
        with pytest.raises(ValidationError):
            CompactSessionRequest(**values)

    invalid_price = ModelPrice.fixed(
        provider_name="invalid-provider-\ud800",
        model="summary-model",
        input_per_million=Decimal("1"),
        output_per_million=Decimal("1"),
    )
    with pytest.raises(ValidationError):
        CompactSessionRequest(
            session_id="sess_compact_contract",
            idempotency_key="compact-1",
            expected_run_epoch=0,
            expected_transcript_cursor=3,
            budget_limits=(
                BudgetLimit(
                    max_estimated_cost=Decimal("1"),
                    pricing=PriceBook(prices=(invalid_price,)),
                ),
            ),
        )


@pytest.mark.parametrize(
    "values",
    [
        {"summary": "invalid-\x00summary"},
        {"summary": "invalid-\ud800-summary"},
        {"summary": "summary", "metadata": {"nested": "invalid-\x00metadata"}},
        {
            "summary": "summary",
            "model_completed_payloads": [{"model": "invalid-\ud800-model"}],
        },
    ],
)
def test_compaction_result_rejects_text_that_cannot_be_durably_persisted(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CompactionResult.model_validate(values)


def test_size_based_compaction_handles_one_long_user_turn_and_stays_quiet_afterward() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-single-turn",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        compactor = RecordingCompactor()
        policy = CheckpointCompactionContextPolicy(
            compactor=compactor,
            compact_after_estimated_context_tokens=1_600,
            max_recent_context_tokens=1_000,
            reserved_output_tokens=100,
        )
        messages = [
            Message.text("system", "Solve the task precisely."),
            Message.text("user", "Original task " + "q" * 2_000),
            Message.tool_call(
                tool_call_id="old-call",
                tool_name="search",
                arguments={"query": "old evidence"},
            ),
            Message.tool_result(
                tool_call_id="old-call",
                tool_name="search",
                content="old evidence " + "x" * 4_000,
            ),
            Message.tool_call(
                tool_call_id="recent-call",
                tool_name="search",
                arguments={"query": "recent evidence"},
            ),
            Message.tool_result(
                tool_call_id="recent-call",
                tool_name="search",
                content="recent decisive evidence",
            ),
        ]
        request = ContextRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=messages,
            step=3,
        )

        first = await policy.build_with_checkpoint(request, checkpoint=None)

        assert len(compactor.requests) == 1
        compacted = compactor.requests[0].messages
        assert compacted[0].role.value == "assistant"
        assert compacted[-1].role.value == "tool"
        assert "old evidence" in str(compacted[-1].model_dump(mode="json"))
        assert "Original task" not in json.dumps(
            [message.model_dump(mode="json") for message in compacted]
        )
        projected = json.dumps([message.model_dump(mode="json") for message in first.messages])
        assert "durable compact summary" in projected
        assert "Original task " + "q" * 2_000 in projected
        assert "recent decisive evidence" in projected
        assert "old evidence " + "x" * 100 not in projected
        assert first.checkpoint is not None

        second = await policy.build_with_checkpoint(
            request,
            checkpoint=first.checkpoint,
        )

        assert len(compactor.requests) == 1
        assert second.checkpoint is None
        assert "recent decisive evidence" in json.dumps(
            [message.model_dump(mode="json") for message in second.messages]
        )

    asyncio.run(run())


def test_size_based_compaction_requires_a_smaller_recent_context_target() -> None:
    with pytest.raises(ValueError, match="must be configured together"):
        CheckpointCompactionContextPolicy(
            compact_after_estimated_context_tokens=1_000,
        )
    with pytest.raises(ValueError, match="must be less"):
        CheckpointCompactionContextPolicy(
            compact_after_estimated_context_tokens=1_000,
            max_recent_context_tokens=800,
            reserved_output_tokens=200,
        )


def test_size_based_compaction_uses_configured_attachment_retention() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-attachment-retention",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        messages = [
            Message.text("system", "Inspect retained attachments."),
            Message.text("user", "Inspect every image."),
        ]
        for index in range(3):
            attachment = file_attachment(
                artifact_id=f"art-{index}",
                kind=FileAttachmentKind.IMAGE,
                filename=f"image-{index}.png",
                content_type="image/png",
                size_bytes=1_000_000,
            )
            messages.extend(
                [
                    Message.tool_call(
                        tool_call_id=f"call-{index}",
                        tool_name="read_image",
                        arguments={"artifact_id": f"art-{index}"},
                    ),
                    Message.tool_result(
                        tool_call_id=f"call-{index}",
                        tool_name="read_image",
                        content="attached",
                        artifacts=[attachment],
                    ),
                ]
            )
        request = ContextRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=messages,
            step=4,
        )

        retained_compactor = RecordingCompactor()
        retained = await CheckpointCompactionContextPolicy(
            compactor=retained_compactor,
            compact_after_estimated_context_tokens=125_070,
            max_recent_context_tokens=100_000,
            max_attachment_results=3,
        ).build_with_checkpoint(request, checkpoint=None)

        assert len(retained_compactor.requests) == 1
        assert retained.checkpoint is not None

        stripped_compactor = RecordingCompactor()
        stripped = await CheckpointCompactionContextPolicy(
            compactor=stripped_compactor,
            compact_after_estimated_context_tokens=1_000,
            max_recent_context_tokens=500,
            max_attachment_results=0,
        ).build_with_checkpoint(request, checkpoint=None)

        assert stripped_compactor.requests == []
        assert stripped.checkpoint is None
        assert "cayu.file_attachment.v1" not in json.dumps(
            [message.model_dump(mode="json") for message in stripped.messages]
        )

    asyncio.run(run())


def test_size_based_compaction_reduces_context_when_fixed_overhead_exceeds_target() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-fixed-overhead",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        messages = [
            Message.text("system", "s" * 2_000),
            Message.text("user", "ORIGINAL_TASK"),
            Message.text("assistant", "OLD_REMOVABLE_CONTEXT " + "x" * 4_000),
            Message.text("user", "recent request"),
        ]
        request = ContextRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=messages,
            step=2,
        )

        compactor = RecordingCompactor()
        result = await CheckpointCompactionContextPolicy(
            compactor=compactor,
            compact_after_estimated_context_tokens=1_800,
            max_recent_context_tokens=250,
            reserved_output_tokens=100,
        ).build_with_checkpoint(request, checkpoint=None)

        projected = json.dumps([message.model_dump(mode="json") for message in result.messages])
        assert len(compactor.requests) == 1
        assert result.checkpoint is not None
        assert "ORIGINAL_TASK" in projected
        assert "OLD_REMOVABLE_CONTEXT" not in projected

        unsafe_compactor = RecordingCompactor()
        with pytest.raises(ContextBuildError, match="configured size bounds") as error:
            await CheckpointCompactionContextPolicy(
                compactor=unsafe_compactor,
                compact_after_estimated_context_tokens=600,
                max_recent_context_tokens=250,
                reserved_output_tokens=100,
            ).build_with_checkpoint(request, checkpoint=None)
        assert len(unsafe_compactor.requests) == 1
        assert error.value.checkpoint is not None

    asyncio.run(run())


def test_size_based_compaction_fails_closed_and_checkpoints_oversized_summary() -> None:
    class OversizedSummaryCompactor(RecordingCompactor):
        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.requests.append(request)
            return CompactionResult(
                summary="z" * 5_000,
                covered_message_count=len(request.messages),
                represented_existing_summary_sha256=(
                    hashlib.sha256(request.existing_summary.encode("utf-8")).hexdigest()
                    if request.existing_summary is not None
                    else None
                ),
            )

    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-oversized-summary",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        messages = [
            Message.text("system", "system"),
            Message.text("user", "ORIGINAL_TASK"),
            Message.text("assistant", "old context " + "x" * 5_000),
            Message.text("assistant", "latest observation"),
        ]
        request = ContextRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=messages,
            step=2,
        )
        first_compactor = OversizedSummaryCompactor()

        with pytest.raises(ContextBuildError, match="configured size bounds") as first_error:
            await CheckpointCompactionContextPolicy(
                compactor=first_compactor,
                compact_after_estimated_context_tokens=1_000,
                max_recent_context_tokens=800,
                reserved_output_tokens=100,
            ).build_with_checkpoint(request, checkpoint=None)

        checkpoint = first_error.value.checkpoint
        assert checkpoint is not None
        assert checkpoint["context_compaction"]["summary"] == "z" * 5_000
        assert len(first_compactor.requests) == 1

        restarted_compactor = OversizedSummaryCompactor()
        with pytest.raises(ContextBuildError, match="configured size bounds") as restarted_error:
            await CheckpointCompactionContextPolicy(
                compactor=restarted_compactor,
                compact_after_estimated_context_tokens=1_000,
                max_recent_context_tokens=800,
                reserved_output_tokens=100,
            ).build_with_checkpoint(
                request.model_copy(update={"force_bounded_compaction": True}),
                checkpoint=checkpoint,
            )
        assert len(restarted_compactor.requests) == 1
        assert restarted_compactor.requests[0].existing_summary == "z" * 5_000
        exhausted_checkpoint = restarted_error.value.checkpoint
        assert exhausted_checkpoint is not None
        assert exhausted_checkpoint["context_compaction"]["compacted_transcript_cursor"] == len(
            messages
        )

        exhausted_compactor = OversizedSummaryCompactor()
        with pytest.raises(ContextBuildError, match="configured size bounds"):
            await CheckpointCompactionContextPolicy(
                compactor=exhausted_compactor,
                compact_after_estimated_context_tokens=1_000,
                max_recent_context_tokens=800,
                reserved_output_tokens=100,
            ).build_with_checkpoint(
                request.model_copy(update={"force_bounded_compaction": True}),
                checkpoint=exhausted_checkpoint,
            )
        assert exhausted_compactor.requests == []

    asyncio.run(run())


def test_size_based_compaction_honors_bounded_overflow_force_below_trigger() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-bounded-overflow",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        compactor = RecordingCompactor()

        result = await CheckpointCompactionContextPolicy(
            compactor=compactor,
            compact_after_estimated_context_tokens=10_000,
            max_recent_context_tokens=8_000,
            reserved_output_tokens=1_000,
        ).build_with_checkpoint(
            ContextRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("system", "system"),
                    Message.text("user", "ORIGINAL_TASK"),
                    Message.text("assistant", "compact despite the local estimate"),
                    Message.text("assistant", "latest observation"),
                ],
                step=2,
                force_bounded_compaction=True,
            ),
            checkpoint=None,
        )

        assert len(compactor.requests) == 1
        assert compactor.requests[0].force_bounded_compaction is True
        assert result.checkpoint is not None
        assert "ORIGINAL_TASK" in json.dumps(
            [message.model_dump(mode="json") for message in result.messages]
        )

    asyncio.run(run())


def test_size_based_compaction_keeps_the_only_recent_tool_round_visible() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-only-recent-tool-round",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        compactor = RecordingCompactor()
        messages = [
            Message.text("user", "ORIGINAL_TASK"),
            Message.tool_call(
                tool_call_id="recent-call",
                tool_name="search",
                arguments={"query": "latest"},
            ),
            Message.tool_result(
                tool_call_id="recent-call",
                tool_name="search",
                content="latest result",
            ),
        ]

        result = await CheckpointCompactionContextPolicy(
            compactor=compactor,
            compact_after_estimated_context_tokens=10_000,
            max_recent_context_tokens=5_000,
        ).build_with_checkpoint(
            ContextRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
                force_compaction=True,
            ),
            checkpoint=None,
        )

        assert compactor.requests == []
        assert result.checkpoint is None
        assert [message.model_dump(mode="json") for message in result.messages] == [
            message.model_dump(mode="json") for message in messages
        ]

    asyncio.run(run())


def test_size_based_compaction_compacts_an_oversized_latest_tool_round_atomically() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-oversized-latest-tool-round",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        compactor = RecordingCompactor()
        calls = [
            ToolCallPart(
                tool_call_id=f"inspect-{index}",
                tool_name="product_inspect",
                arguments={"candidate_id": f"candidate-{index}"},
            )
            for index in range(12)
        ]
        results = [
            ToolResultPart(
                tool_call_id=f"inspect-{index}",
                tool_name="product_inspect",
                content=f"candidate-{index}-evidence:" + "x" * 12_000,
            )
            for index in range(12)
        ]
        messages = [
            Message.text("user", "ORIGINAL_TASK"),
            Message.tool_call(calls=calls),
            Message.tool_result(results=results),
        ]

        result = await CheckpointCompactionContextPolicy(
            compactor=compactor,
            compact_after_estimated_context_tokens=10_000,
            max_recent_context_tokens=8_000,
            reserved_output_tokens=1_000,
        ).build_with_checkpoint(
            ContextRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
            ),
            checkpoint=None,
        )

        assert len(compactor.requests) == 1
        compacted = compactor.requests[0].messages
        assert [message.model_dump(mode="json") for message in compacted] == [
            message.model_dump(mode="json") for message in messages[1:]
        ]
        assert result.checkpoint is not None
        assert result.checkpoint["context_compaction"]["compacted_transcript_cursor"] == len(
            messages
        )
        projected = json.dumps([message.model_dump(mode="json") for message in result.messages])
        assert "ORIGINAL_TASK" in projected
        assert "durable compact summary" in projected
        assert "candidate-0-evidence" not in projected

    asyncio.run(run())


def test_size_based_compaction_checkpoint_survives_sqlite_restart(tmp_path) -> None:
    async def run() -> None:
        database = tmp_path / "size-based-compaction-restart.sqlite"
        store = SQLiteSessionStore(database)
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="size-based-sqlite-restart",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        messages = [
            Message.text("system", "Solve precisely."),
            Message.text("user", "ORIGINAL_TASK"),
            Message.text("assistant", "old answer " + "x" * 4_000),
            Message.text("assistant", "recent observation"),
        ]
        await store.append_transcript_messages(session.id, messages)
        first_compactor = RecordingCompactor()
        first = await CheckpointCompactionContextPolicy(
            compactor=first_compactor,
            compact_after_estimated_context_tokens=600,
            max_recent_context_tokens=250,
            reserved_output_tokens=100,
        ).build_with_checkpoint(
            ContextRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        await store.checkpoint(session.id, first.checkpoint)
        await store.close()

        reopened = SQLiteSessionStore(database)
        restarted_compactor = RecordingCompactor()
        try:
            await reopened.append_transcript_messages(
                session.id,
                [Message.text("user", "follow-up after restart")],
            )
            reloaded_session = await reopened.load(session.id)
            assert reloaded_session is not None
            reloaded_messages = await reopened.load_transcript(session.id)
            reloaded_checkpoint = await reopened.load_checkpoint(session.id)
            assert reloaded_checkpoint is not None

            restarted = await CheckpointCompactionContextPolicy(
                compactor=restarted_compactor,
                compact_after_estimated_context_tokens=600,
                max_recent_context_tokens=250,
                reserved_output_tokens=100,
            ).build_with_checkpoint(
                ContextRequest(
                    session=reloaded_session,
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=reloaded_messages,
                    step=3,
                ),
                checkpoint=reloaded_checkpoint,
            )

            projected = json.dumps(
                [message.model_dump(mode="json") for message in restarted.messages]
            )
            assert restarted_compactor.requests == []
            assert restarted.checkpoint is None
            assert "durable compact summary" in projected
            assert "ORIGINAL_TASK" in projected
            assert "recent observation" in projected
            assert "follow-up after restart" in projected
            assert "old answer " + "x" * 100 not in projected
        finally:
            await reopened.close()

    asyncio.run(run())


class _LongLoopEvidenceTool(Tool):
    spec = ToolSpec(
        name="collect_evidence",
        description="Collect one large deterministic evidence record.",
        input_schema={
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
            "additionalProperties": False,
        },
        effect=ToolEffect.NONE,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        index = int(args["index"])
        return ToolResult(
            content=f"evidence-{index}:" + "x" * 4_000,
            structured={"index": index, "payload": "x" * 4_000},
        )


class _LongToolLoopProvider(ModelProvider):
    name = "long-tool-loop"
    execution_profile_identity = ExecutionProfileBehaviorIdentity(
        name="tests.long-tool-loop-provider",
        behavior_version="1",
        implementation_version="1",
    )

    def __init__(self, rounds: int) -> None:
        self.rounds = rounds
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        index = len(self.requests) - 1
        if index < self.rounds:
            yield ModelStreamEvent.tool_call(
                id=f"evidence-call-{index}",
                name="collect_evidence",
                arguments={"index": index},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("finished after compacted tool loop")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _WideToolRoundProvider(ModelProvider):
    name = "wide-tool-round"
    execution_profile_identity = ExecutionProfileBehaviorIdentity(
        name="tests.wide-tool-round-provider",
        behavior_version="1",
        implementation_version="1",
    )

    def __init__(self, tool_calls: int) -> None:
        self.tool_calls = tool_calls
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        if len(self.requests) == 1:
            for index in range(self.tool_calls):
                yield ModelStreamEvent.tool_call(
                    id=f"evidence-call-{index}",
                    name="collect_evidence",
                    arguments={"index": index},
                )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("continued after compacting the wide tool round")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def test_size_based_compaction_runs_through_a_real_long_tool_loop() -> None:
    async def run() -> None:
        original_task = "ORIGINAL-LONG-TOOL-TASK:" + " preserve-me" * 80
        provider = _LongToolLoopProvider(rounds=20)
        compactor = RecordingCompactor()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                workflow_tool_names=("collect_evidence",),
            ),
            tools=[_LongLoopEvidenceTool()],
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                compact_after_estimated_context_tokens=4_000,
                max_recent_context_tokens=2_500,
                reserved_output_tokens=500,
            ),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="real-long-tool-loop-compaction",
                    messages=[Message.text("user", original_task)],
                    max_steps=32,
                )
            )
        ]

        assert events[-1].type == EventType.SESSION_COMPLETED
        assert len(provider.requests) == 21
        assert len(compactor.requests) >= 1
        for request in provider.requests:
            projected = json.dumps(
                [message.model_dump(mode="json") for message in request.messages]
            )
            assert original_task in projected

    asyncio.run(run())


def test_size_based_compaction_continues_after_a_wide_oversized_tool_round() -> None:
    async def run() -> None:
        original_task = "ORIGINAL-WIDE-TOOL-TASK"
        provider = _WideToolRoundProvider(tool_calls=12)
        compactor = RecordingCompactor()
        app = CayuApp(
            enable_logging=False,
            config=CayuConfig(tool_execution=ToolExecutionConfig(max_parallel_tool_calls=12)),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                workflow_tool_names=("collect_evidence",),
            ),
            tools=[_LongLoopEvidenceTool()],
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                compact_after_estimated_context_tokens=5_000,
                max_recent_context_tokens=3_000,
                reserved_output_tokens=500,
            ),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="wide-tool-round-compaction",
                    messages=[Message.text("user", original_task)],
                    max_steps=4,
                )
            )
        ]

        assert events[-1].type == EventType.SESSION_COMPLETED
        assert len(provider.requests) == 2
        assert len(compactor.requests) == 1
        assert len(compactor.requests[0].messages) == 2
        compacted_round = json.dumps(
            [message.model_dump(mode="json") for message in compactor.requests[0].messages]
        )
        assert compacted_round.count('"type": "tool_call"') == 12
        assert compacted_round.count('"type": "tool_result"') == 12
        continued_projection = json.dumps(
            [message.model_dump(mode="json") for message in provider.requests[1].messages]
        )
        assert original_task in continued_projection
        assert "durable compact summary" in continued_projection
        assert "evidence-0:" not in continued_projection

    asyncio.run(run())


def test_compact_session_preserves_transcript_and_replays_original_outcome() -> None:
    async def run() -> None:
        store = NoFullHistoryReplayStore()
        compactor = RecordingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_compact",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("system", "Be precise."),
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-explicit-1",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
            instructions="Keep the decisions and file names.",
        )

        first = [event async for event in app.compact_session(request)]
        replay = [event async for event in app.compact_session(request)]
        durable_events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]

        assert [event.type for event in first] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]
        assert [event.id for event in replay] == [event.id for event in first]
        assert [event.id for event in first] == await _public_ids_for_private_event_ids(
            store,
            created.id,
            [event.id for event in durable_events],
        )
        assert len({event.payload["model_step_id"] for event in first}) == 1
        assert all("model_attempt_id" not in event.payload for event in first)
        assert len(compactor.requests) == 1
        assert compactor.requests[0].instructions == "Keep the decisions and file names."
        assert await store.load_transcript(created.id) == transcript
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"] == {
            "version": 2,
            "summary": "durable compact summary",
            "compacted_transcript_cursor": 3,
            "metadata": {"compactor": "RecordingCompactor", "mode": "deterministic"},
        }
        for event in first:
            assert "summary" not in event.payload
            assert "instructions" not in event.payload
            assert event.payload["operation_id"]
            assert event.payload["reason"] == "application_requested"

    asyncio.run(run())


def test_compact_session_instructions_change_the_governing_profile_without_publication() -> None:
    async def run() -> tuple[list[Event], list[Event]]:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )

        async def compact(session_id: str, instructions: str) -> list[Event]:
            created = await _create_profiled_session(
                app,
                store,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            transcript = [
                Message.text("user", "old request"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current request"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key=f"compact-{session_id}",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                        instructions=instructions,
                    )
                )
            ]

        return (
            await compact("sess_compaction_profile_a", "Preserve decisions."),
            await compact("sess_compaction_profile_b", "Preserve file names."),
        )

    first, second = asyncio.run(run())

    first_profiles = {event.payload.get("execution_profile_fingerprint") for event in first}
    second_profiles = {event.payload.get("execution_profile_fingerprint") for event in second}
    assert len(first_profiles) == 1
    assert len(second_profiles) == 1
    assert None not in first_profiles
    assert None not in second_profiles
    assert first_profiles != second_profiles
    assert (
        first[0].payload["instruction_digest"] == hashlib.sha256(b"Preserve decisions.").hexdigest()
    )
    serialized = str([event.model_dump(mode="json") for event in (*first, *second)])
    assert "Preserve decisions." not in serialized
    assert "Preserve file names." not in serialized


def test_compact_session_redacts_instructions_before_provider_and_durable_boundaries() -> None:
    secret = "explicit-compaction-instruction-canary"

    async def run() -> tuple[RecordingCompactor, list[Event], dict[str, object] | None]:
        store = InMemorySessionStore()
        compactor = RecordingCompactor()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_boundary",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-boundary-1",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    instructions=f"Preserve decisions but never expose {secret}.",
                )
            )
        ]
        return compactor, events, await store.load_checkpoint(created.id)

    compactor, events, checkpoint = asyncio.run(run())

    assert compactor.requests[0].instructions == (
        f"Preserve decisions but never expose {REDACTED_SECRET}."
    )
    assert secret not in str([event.model_dump(mode="json") for event in events])
    assert secret not in str(checkpoint)


def test_compact_session_rejects_secret_policy_checkpoint_before_publication() -> None:
    from cayu.runtime.context import ContextBuildResult

    secret = "explicit-policy-checkpoint-secret-canary"

    class SecretCheckpointPolicy(CheckpointCompactionContextPolicy):
        async def build_with_checkpoint(
            self,
            request: ContextRequest,
            *,
            checkpoint: dict[str, Any] | None,
        ) -> ContextBuildResult:
            del checkpoint
            return ContextBuildResult(
                messages=[Message.text("assistant", f"unsafe result {secret}")],
                checkpoint={"context_compaction": {"summary": secret}},
                checkpoint_event_payload={
                    "compacted_transcript_cursor": len(request.messages),
                    "custom_detail": secret,
                },
            )

    async def run() -> tuple[
        ValueError,
        list[Event],
        dict[str, object] | None,
        list[Event],
    ]:
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=SecretCheckpointPolicy(
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_secret_checkpoint",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        observed: list[Event] = []
        with pytest.raises(ValueError, match="contains a workload secret") as exc_info:
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="secret-checkpoint",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                observed.append(event)
        return (
            exc_info.value,
            observed,
            await store.load_checkpoint(created.id),
            await store.load_events(created.id),
        )

    error, observed, checkpoint, durable_events = asyncio.run(run())

    assert all(event.type is not EventType.SESSION_CHECKPOINTED for event in observed)
    assert secret not in repr(observed)
    assert secret not in repr(checkpoint)
    assert secret not in repr(durable_events)
    traceback = error.__traceback__
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("\\", "/")
        if "/src/cayu/" in filename:
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next


@pytest.mark.parametrize("secret_location", ["key", "value", "nested_typed_key"])
def test_compact_session_rejects_secret_checkpoint_event_payload_before_publication(
    secret_location: str,
) -> None:
    from cayu.runtime.context import ContextBuildResult

    secret = (
        "checkpoint"
        if secret_location == "nested_typed_key"
        else "explicit-policy-event-payload-secret-canary"
    )

    class SecretEventPayloadPolicy(CheckpointCompactionContextPolicy):
        def __init__(self) -> None:
            super().__init__(
                max_user_turns=1,
                compact_after_messages=1,
            )
            self.result: ContextBuildResult | None = None

        async def build_with_checkpoint(
            self,
            request: ContextRequest,
            *,
            checkpoint: dict[str, Any] | None,
        ) -> ContextBuildResult:
            del checkpoint
            if secret_location == "nested_typed_key":
                unsafe_payload = {"custom": {"checkpoint": "safe"}}
            else:
                unsafe_field = secret if secret_location == "key" else "custom_detail"
                unsafe_value = "safe" if secret_location == "key" else secret
                unsafe_payload = {unsafe_field: unsafe_value}
            self.result = ContextBuildResult(
                messages=[Message.text("assistant", f"unsafe result {secret}")],
                checkpoint={"context_compaction": {"summary": "safe summary"}},
                checkpoint_event_payload={
                    "compacted_transcript_cursor": len(request.messages),
                    **unsafe_payload,
                },
            )
            return self.result

    async def run() -> tuple[
        ValueError,
        SecretEventPayloadPolicy,
        dict[str, object] | None,
        list[Event],
    ]:
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        policy = SecretEventPayloadPolicy()
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=policy,
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_explicit_secret_event_{secret_location}",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        with pytest.raises(ValueError, match="contains a workload secret") as exc_info:
            async for _ in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key=f"secret-event-{secret_location}",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass
        return (
            exc_info.value,
            policy,
            await store.load_checkpoint(created.id),
            await store.load_events(created.id),
        )

    error, policy, checkpoint, durable_events = asyncio.run(run())

    assert checkpoint == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
    assert secret not in repr(durable_events)
    assert policy.result is not None
    assert policy.result.messages == []
    assert policy.result.checkpoint == {}
    assert policy.result.checkpoint_event_payload == {}
    if secret_location == "nested_typed_key":
        return
    traceback = error.__traceback__
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("\\", "/")
        if "/src/cayu/" in filename:
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next


@pytest.mark.parametrize(
    "failure_checkpoint",
    [
        None,
        {"context_compaction": {"summary": "safe summary"}},
    ],
)
def test_compact_session_discards_secret_checkpoint_event_payload_from_failure(
    failure_checkpoint: dict[str, Any] | None,
) -> None:
    secret = "explicit-failure-event-payload-secret-canary"

    class FailingCheckpointPolicy(CheckpointCompactionContextPolicy):
        async def build_with_checkpoint(
            self,
            request: ContextRequest,
            *,
            checkpoint: dict[str, Any] | None,
        ):
            del request, checkpoint
            raise ContextBuildError(
                "context build failed",
                compaction_telemetry=[],
                checkpoint=failure_checkpoint,
                checkpoint_event_payload={
                    "checkpoint": "context_compaction",
                    "custom_detail": secret,
                },
                cause=RuntimeError("safe policy failure"),
            )

    async def run() -> tuple[ContextBuildError, dict[str, object] | None, list[Event]]:
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=FailingCheckpointPolicy(
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_secret_failure_event_payload",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(ContextBuildError) as exc_info:
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="secret-failure-event-payload",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass
        return (
            exc_info.value,
            await store.load_checkpoint(created.id),
            await store.load_events(created.id),
        )

    error, checkpoint, durable_events = asyncio.run(run())

    assert error.checkpoint is None
    assert error.checkpoint_event_payload is None
    assert secret not in repr(error.__dict__)
    traceback = error.__traceback__
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("\\", "/")
        if "/src/cayu/" in filename:
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next
    assert secret not in repr(checkpoint)
    assert secret not in repr(durable_events)


def test_compact_session_fences_expired_recovery_owner_before_retry() -> None:
    async def run() -> None:
        now = datetime(2026, 7, 18, tzinfo=UTC)
        session_id = "sess_compact_expired_recovery_owner"

        class InterleavedTakeoverStore(InMemorySessionStore):
            invocation_lifecycle_command_version = 1

            def __init__(self) -> None:
                super().__init__()
                self.takeover_started = asyncio.Event()
                self.allow_takeover = asyncio.Event()

            async def fence_run_and_transform_checkpoint(self, *args, **kwargs):
                self.takeover_started.set()
                await self.allow_takeover.wait()
                return await super().fence_run_and_transform_checkpoint(*args, **kwargs)

        store = InterleavedTakeoverStore()
        compactor = RecordingCompactor()
        app = CayuApp(session_store=store, enable_logging=False, clock=lambda: now)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(session_id, transcript)
        await store.update_status(session_id, SessionStatus.COMPLETED)
        stale_owner_ready = asyncio.Event()
        release_stale_owner = asyncio.Event()

        async def stale_recovery_owner() -> None:
            fenced = await store.fence_stalled_run(
                session_id,
                statuses={SessionStatus.COMPLETED},
                inactive_for_seconds=0,
            )
            assert fenced is not None

            def add_expired_claim(_session, checkpoint):
                updated = {} if checkpoint is None else dict(checkpoint)
                updated["incomplete_session_recovery_claim"] = {
                    "version": 1,
                    "claim_id": "stale-recovery-owner",
                    "claimed_at": (now - timedelta(minutes=10)).isoformat(),
                    "claim_expires_at": (now - timedelta(minutes=5)).isoformat(),
                }
                return updated

            await store.transform_checkpoint(session_id, add_expired_claim)
            stale_owner_ready.set()
            await store.takeover_started.wait()
            await store.update_metadata(
                session_id,
                {"stale_owner_write_before_atomic_takeover": True},
            )
            store.allow_takeover.set()
            await release_stale_owner.wait()
            try:
                with pytest.raises(SessionRunFenced):
                    await store.update_metadata(session_id, {"stale_owner_write": True})
            finally:
                await store.release_run_fence(session_id)

        stale_owner_task = asyncio.create_task(stale_recovery_owner())
        await asyncio.wait_for(stale_owner_ready.wait(), timeout=5)
        owned = await store.load(session_id)
        assert owned is not None
        stale_epoch = owned.run_epoch

        with pytest.raises(
            ValueError,
            match="fenced an expired incomplete-session recovery owner",
        ):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=session_id,
                    idempotency_key="compact-after-expired-recovery",
                    expected_run_epoch=stale_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pytest.fail("Compaction emitted an event before fencing the stale owner.")

        after_takeover = await store.load(session_id)
        assert after_takeover is not None
        assert after_takeover.run_epoch > stale_epoch
        assert after_takeover.metadata["stale_owner_write_before_atomic_takeover"] is True
        assert after_takeover.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=())
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "incomplete_session_recovery_claim" not in checkpoint
        assert compactor.requests == []

        release_stale_owner.set()
        await asyncio.wait_for(stale_owner_task, timeout=5)

        compacted = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=session_id,
                    idempotency_key="compact-after-expired-recovery",
                    expected_run_epoch=after_takeover.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]
        assert compacted[-1].type == EventType.SESSION_CHECKPOINTED
        assert len(compactor.requests) == 1

    asyncio.run(run())


def test_compact_session_replays_original_outcome_after_session_advances() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        compactor = RecordingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider())
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_advanced_replay",
                messages=[Message.text("user", "create only")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-before-resume",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        first = [event async for event in app.compact_session(request)]
        resume_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=created.id,
                    messages=[Message.text("user", "advance the durable session")],
                )
            )
        ]
        advanced = await store.load(created.id)
        assert advanced is not None
        assert advanced.run_epoch > request.expected_run_epoch
        assert len(await store.load_transcript(created.id)) > request.expected_transcript_cursor
        assert EventType.SESSION_COMPLETED in [event.type for event in resume_events]

        replay = [event async for event in app.compact_session(request)]
        operation_sequences = {public_event_sequence(event.id) for event in first}
        durable_operation_events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
            if record.sequence in operation_sequences
        ]

        assert [event.id for event in replay] == [event.id for event in first]
        assert [event.id for event in first] == await _public_ids_for_private_event_ids(
            store,
            created.id,
            [event.id for event in durable_operation_events],
        )
        assert len(compactor.requests) == 1

    asyncio.run(run())


def test_compact_session_replays_legacy_terminal_record_before_later_pending_state() -> None:
    from cayu.runtime import _approval_support as approval_support
    from cayu.runtime.approvals import PendingToolApproval, PendingToolCallApproval

    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_legacy_replay_pending",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-legacy-replay-pending",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=0,
        )
        replay_events = [
            Event(
                id="legacy-compaction-completed",
                type=EventType.CONTEXT_COMPACTION_COMPLETED,
                session_id=created.id,
            ),
            Event(
                id="legacy-session-checkpointed",
                type=EventType.SESSION_CHECKPOINTED,
                session_id=created.id,
            ),
        ]
        await store.append_events(created.id, replay_events)
        pending = PendingToolApproval(
            approval_id="approval-safe",
            tool_round_id=f"tround_{'3' * 32}",
            model_step_id=f"mstep_{'1' * 32}",
            model_attempt_id=f"matt_{'2' * 32}",
            tool_call_id="call-safe",
            tool_name="safe-tool",
            agent_name="assistant",
            publish_arguments=True,
            tool_calls=[
                PendingToolCallApproval(
                    tool_call_id="call-safe",
                    tool_name="safe-tool",
                )
            ],
        )
        await store.transform_checkpoint(
            created.id,
            lambda _session, _checkpoint: {
                "session_operations": {
                    "version": 1,
                    "active_operation_id": None,
                    "records": {
                        request.idempotency_key: {
                            "status": "completed",
                            "request_digest": session_engine_module._compact_session_request_digest(
                                request
                            ),
                            "event_ids": [event.id for event in replay_events],
                        }
                    },
                },
                approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY: pending.model_dump(
                    mode="json"
                ),
            },
        )
        assert await store.load_session_operation(created.id, request.idempotency_key) is None
        replay = [event async for event in app.compact_session(request)]

        assert [event.id for event in replay] == await _public_ids_for_private_event_ids(
            store,
            created.id,
            [event.id for event in replay_events],
        )

    asyncio.run(run())


def test_compact_session_rejects_unobserved_adversarial_completion_telemetry() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        compactor = AdversarialTelemetryCompactor()
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_private_events",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(
            RuntimeError,
            match="Compaction completion was observed outside its provider dispatch",
        ):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-private-events",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    instructions="private caller instructions",
                )
            ):
                pass

        events = await store.load_events(created.id)

        serialized_events = json.dumps(
            [event.model_dump(mode="json") for event in events],
            sort_keys=True,
        )
        for secret in (
            "private summary text",
            "metadata summary leak",
            "metadata instruction leak",
            "model event summary leak",
            "model event instruction leak",
            "opaque continuation leak",
            "private caller instructions",
            "m" * 1_000,
            "p" * 1_000,
        ):
            assert secret not in serialized_events
        assert len(serialized_events.encode("utf-8")) < 10_000
        assert all(event.payload["mode"] == "bounded" for event in events)
        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert events[0].payload["model_step_id"] == events[1].payload["model_step_id"]
        assert "model_attempt_id" not in events[1].payload

    asyncio.run(run())


def test_compact_session_rejects_second_completion_for_one_provider_dispatch() -> None:
    class DuplicatingWrapperCompactor(ContextCompactor):
        def __init__(self, provider: ModelProvider) -> None:
            self.provider = provider

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            result = await ModelCompactor(
                provider=self.provider,
                model="summary-model",
            ).compact(request)
            forged = dict(result.model_completed_payloads[0])
            forged.pop("compaction_attempt_id", None)
            return result.model_copy(
                update={
                    "model_completed_payloads": [
                        *result.model_completed_payloads,
                        forged,
                    ]
                }
            )

    async def run() -> None:
        store = InMemorySessionStore()
        provider = UsageCompactionProvider()
        app = CayuApp(
            session_store=store,
            request_footprint=RequestFootprintConfig(enabled=False),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=DuplicatingWrapperCompactor(provider),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_duplicate_dispatch_completion",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(
            ValueError,
            match="Compaction provider dispatch produced conflicting completion identities",
        ):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="explicit-duplicate-dispatch-completion",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        events = await store.load_events(created.id)
        assert provider.calls == 1
        assert sum(event.type == EventType.MODEL_COMPLETED for event in events) == 1
        assert events[-1].type == EventType.CONTEXT_COMPACTION_FAILED

    asyncio.run(run())


def test_compact_session_generator_exit_preserves_completed_hierarchy_usage() -> None:
    class AbandonSecondHierarchyCallProvider(ModelProvider):
        name = "abandon-hierarchy"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            if self.calls == 2:
                raise GeneratorExit("abandon hierarchy after first completion")
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )

    async def run() -> None:
        session_id = "sess_explicit_hierarchy_generator_exit"
        store = InMemorySessionStore()
        provider = AbandonSecondHierarchyCallProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(session_id, transcript)
        completed = await store.update_status(session_id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=session_id,
            idempotency_key="explicit-generator-exit",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        with pytest.raises(GeneratorExit, match="abandon hierarchy after first completion"):
            async for _event in app.compact_session(request):
                pass

        events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=session_id, limit=100))
        ]
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 2
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 7
        assert completions[1].payload["compaction_outcome"] == "unfinished_stream"
        assert completions[1].payload["usage_unavailable_reason"]
        failures = [event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED]
        assert len(failures) == 1
        assert failures[0].payload["error_type"] == "GeneratorExit"
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint
        operation = await store.load_session_operation(session_id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "failed"

    asyncio.run(run())


def test_compact_session_generator_exit_propagates_abandonment_accounting_failure() -> None:
    class FailingAbandonmentPublishStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            events = kwargs.get("events", [])
            if any(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events):
                raise RuntimeError("simulated abandonment accounting failure")
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    class AbandonSecondHierarchyCallProvider(ModelProvider):
        name = "abandon-hierarchy-accounting-failure"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            if self.calls == 2:
                raise GeneratorExit("abandon hierarchy before accounting failure")
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )

    async def run() -> None:
        session_id = "sess_explicit_hierarchy_abandonment_accounting_failure"
        store = FailingAbandonmentPublishStore()
        provider = AbandonSecondHierarchyCallProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(session_id, transcript)
        completed = await store.update_status(session_id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=session_id,
            idempotency_key="explicit-generator-exit-accounting-failure",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        with pytest.raises(
            RuntimeError,
            match="simulated abandonment accounting failure",
        ) as exc_info:
            async for _event in app.compact_session(request):
                pass

        assert isinstance(exc_info.value.__cause__, GeneratorExit)
        assert exc_info.value.__cause__.args == ("abandon hierarchy before accounting failure",)
        assert any(
            "accounting failure is authoritative" in note
            for note in getattr(exc_info.value, "__notes__", ())
        )
        events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=session_id, limit=100))
        ]
        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
        ]
        assert events[3].payload["model"] == "summary-model"
        assert events[3].payload["usage"] == {"input_tokens": 5, "output_tokens": 2}
        assert events[3].payload["purpose"] == "context_compaction"
        assert events[6].payload["compaction_outcome"] == "unfinished_stream"
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint
        operation = await store.load_session_operation(session_id, request.idempotency_key)
        assert operation is None

    asyncio.run(run())


def test_compact_session_generator_exit_keeps_accounting_failure_authoritative_when_heartbeat_fails(
    monkeypatch,
) -> None:
    class FailingHeartbeatAndAbandonmentStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.fail_heartbeat_reconciliation = False
            self.heartbeat_failed = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if self.fail_heartbeat_reconciliation and kwargs.get("events") == []:
                self.fail_heartbeat_reconciliation = False
                self.heartbeat_failed.set()
                raise RuntimeError("Session compaction operation ownership changed during renewal.")
            if any(
                event.type == EventType.CONTEXT_COMPACTION_FAILED
                for event in kwargs.get("events", [])
            ):
                self.fail_heartbeat_reconciliation = True
                await asyncio.wait_for(self.heartbeat_failed.wait(), timeout=5)
                raise RuntimeError("simulated abandonment accounting failure")
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    class AbandonAfterHeartbeatFailureProvider(ModelProvider):
        name = "abandon-after-heartbeat-failure"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            if self.calls == 2:
                raise GeneratorExit("abandon after heartbeat failure")
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        session_id = "sess_explicit_abandonment_accounting_and_heartbeat_failure"
        store = FailingHeartbeatAndAbandonmentStore()
        provider = AbandonAfterHeartbeatFailureProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(session_id, transcript)
        completed = await store.update_status(session_id, SessionStatus.COMPLETED)

        with pytest.raises(
            RuntimeError,
            match="simulated abandonment accounting failure",
        ) as exc_info:
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=session_id,
                    idempotency_key="explicit-generator-exit-accounting-and-heartbeat-failure",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert any(
            "accounting failure is authoritative" in note
            for note in getattr(exc_info.value, "__notes__", ())
        )
        assert isinstance(exc_info.value.__cause__, BaseExceptionGroup)
        heartbeat_failures = exc_info.value.__cause__.exceptions
        assert len(heartbeat_failures) == 1
        assert isinstance(heartbeat_failures[0], RuntimeError)
        assert "ownership changed" in str(heartbeat_failures[0])
        assert isinstance(exc_info.value.__cause__.__cause__, GeneratorExit)
        assert exc_info.value.__cause__.__cause__.args == ("abandon after heartbeat failure",)

    asyncio.run(run())


def test_compact_session_cancellation_preserves_completed_hierarchy_usage() -> None:
    class CancellationBoundaryStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failure_publish_started = asyncio.Event()
            self.allow_failure_publish = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            events = kwargs.get("events", [])
            if any(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events):
                self.failure_publish_started.set()
                await self.allow_failure_publish.wait()
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    class CancelSecondHierarchyCallProvider(ModelProvider):
        name = "cancel-explicit-hierarchy"

        def __init__(self) -> None:
            self.calls = 0
            self.second_started = asyncio.Event()

        async def stream(self, request: ModelRequest):
            self.calls += 1
            if self.calls == 2:
                self.second_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )

    async def run() -> None:
        session_id = "sess_explicit_hierarchy_cancelled"
        store = CancellationBoundaryStore()
        provider = CancelSecondHierarchyCallProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(session_id, transcript)
        completed = await store.update_status(session_id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=session_id,
            idempotency_key="explicit-cancelled-hierarchy",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        async def consume() -> None:
            async for _event in app.compact_session(request):
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(provider.second_started.wait(), timeout=5)
        task.cancel("cancel explicit hierarchy after first completion")
        await asyncio.wait_for(store.failure_publish_started.wait(), timeout=5)
        task.cancel("later cancellation during failure publication")
        store.allow_failure_publish.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel explicit hierarchy after first completion",)
        events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=session_id, limit=100))
        ]
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 2
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 7
        assert completions[1].payload["compaction_outcome"] == "cancelled"
        assert completions[1].payload["usage_unavailable_reason"]
        failures = [event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED]
        assert len(failures) == 1
        assert failures[0].payload["error_type"] == "CancelledError"
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint
        operation = await store.load_session_operation(session_id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "failed"

    asyncio.run(run())


def test_compact_session_claim_blocks_concurrent_resume() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        compactor = BlockingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider())
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_resume_race",
                messages=[Message.text("user", "create only")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        compact_events = []

        async def compact() -> None:
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-race-1",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                compact_events.append(event)

        compact_task = asyncio.create_task(compact())
        await compactor.started.wait()
        try:
            with pytest.raises(RuntimeError, match="active durable operation"):
                async for _event in app.resume(
                    ResumeRequest(
                        session_id=created.id,
                        messages=[Message.text("user", "race")],
                    )
                ):
                    pass
            with pytest.raises(RuntimeError, match="active durable operation"):
                async for _event in app.fork_session(
                    ForkSessionRequest(
                        source_session_id=created.id,
                        session_id="sess_compact_race_fork",
                    )
                ):
                    pass
        finally:
            compactor.release.set()
            await compact_task

        assert [event.type for event in compact_events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]

    asyncio.run(run())


def test_compact_session_rejects_an_equivalent_unexpired_concurrent_claim() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        compactor = BlockingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_same_key_race",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-same-key-race",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )
        first_events = []

        async def compact() -> None:
            async for event in app.compact_session(request):
                first_events.append(event)

        first_task = asyncio.create_task(compact())
        await compactor.started.wait()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                async for _event in app.compact_session(request):
                    pass
        finally:
            compactor.release.set()
            await first_task

        assert [event.type for event in first_events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]

    asyncio.run(run())


@pytest.mark.parametrize("expiry_boundary", ["commit", "acknowledgement"])
def test_compact_session_expired_initial_claim_never_enters_compactor(
    expiry_boundary: str,
) -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}

        class ExpiringInitialClaimStore(InMemorySessionStore):
            invocation_lifecycle_command_version = 1

            async def publish_session_operation_guarded_with_store_time(
                self,
                session_id: str,
                **kwargs,
            ):
                events = kwargs.get("events", [])
                if [event.type for event in events] != [EventType.CONTEXT_COMPACTION_STARTED]:
                    return await super().publish_session_operation_guarded_with_store_time(
                        session_id,
                        **kwargs,
                    )
                commit_time_guard = kwargs["commit_time_guard"]
                if expiry_boundary == "commit":

                    def expire_then_guard(_commit_at: datetime) -> None:
                        now["value"] = accepted_at + timedelta(minutes=6)
                        commit_time_guard(now["value"])

                    kwargs["commit_time_guard"] = expire_then_guard
                result = await super().publish_session_operation_guarded_with_store_time(
                    session_id,
                    **kwargs,
                )
                if expiry_boundary == "acknowledgement":
                    now["value"] = accepted_at + timedelta(minutes=6)
                return result

        store = ExpiringInitialClaimStore(ownership_clock=lambda: now["value"])
        compactor = RecordingCompactor()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_compact_initial_claim_{expiry_boundary}",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(
            session_engine_module.SessionCompactionAttemptSuperseded,
            match="(?:initial publication|operation claim).*expired",
        ):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key=f"compact-initial-claim-{expiry_boundary}",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert compactor.requests == []
        durable_events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        assert EventType.CONTEXT_COMPACTION_COMPLETED not in {
            event.type for event in durable_events
        }
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is None or "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_heartbeats_claim_during_blocked_provider_dispatch(
    monkeypatch,
) -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = InMemorySessionStore(ownership_clock=lambda: now["value"])
        provider = BlockingCompactionProvider()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_provider_claim_heartbeat",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-provider-claim-heartbeat",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        async def collect() -> list[Event]:
            return [event async for event in app.compact_session(request)]

        first_task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=5)
        now["value"] = accepted_at + timedelta(minutes=4)

        first_renewal_expiry = accepted_at + timedelta(minutes=9)
        async with asyncio.timeout(5):
            while True:
                checkpoint = await store.load_checkpoint(created.id)
                assert checkpoint is not None
                record = checkpoint["session_operations"]["records"][request.idempotency_key]
                if datetime.fromisoformat(record["claim_expires_at"]) >= first_renewal_expiry:
                    break
                await asyncio.sleep(0)

        now["value"] = accepted_at + timedelta(minutes=6)
        expected_expiry = accepted_at + timedelta(minutes=11)
        async with asyncio.timeout(5):
            while True:
                checkpoint = await store.load_checkpoint(created.id)
                assert checkpoint is not None
                record = checkpoint["session_operations"]["records"][request.idempotency_key]
                if datetime.fromisoformat(record["claim_expires_at"]) >= expected_expiry:
                    break
                await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="already running"):
            async for _event in app.compact_session(request):
                pass
        assert provider.calls == 1

        provider.release.set()
        events = await first_task
        assert provider.calls == 1
        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert sum(event.type == EventType.MODEL_COMPLETED for event in events) == 1

    asyncio.run(run())


def test_compact_session_claim_heartbeat_loss_cancels_provider_and_fences_publication(
    monkeypatch,
) -> None:
    class ClaimStealingStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.stolen = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if not self.stolen and kwargs.get("events") == []:
                self.stolen = True

                def steal(_session, checkpoint):
                    assert checkpoint is not None
                    updated = dict(checkpoint)
                    operations = dict(updated["session_operations"])
                    records = dict(operations["records"])
                    key, record = next(iter(records.items()))
                    records[key] = {**record, "current_attempt_id": "replacement-attempt"}
                    operations["records"] = records
                    updated["session_operations"] = operations
                    return updated

                await self.transform_checkpoint(session_id, steal)
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = ClaimStealingStore()
        provider = BlockingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_heartbeat_fenced",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(RuntimeError, match="superseded"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-claim-heartbeat-fenced",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert store.stolen
        assert provider.calls == 1
        assert provider.cancelled.is_set()
        durable_events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        completions = [event for event in durable_events if event.type == EventType.MODEL_COMPLETED]
        assert len(completions) == 1
        assert completions[0].payload["compaction_outcome"] == "cancelled"
        assert completions[0].payload["usage_unavailable_reason"]
        assert EventType.SESSION_CHECKPOINTED not in {event.type for event in durable_events}

    asyncio.run(run())


def test_compact_session_claim_heartbeat_retries_transient_store_failure(monkeypatch) -> None:
    class TransientHeartbeatStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.heartbeat_calls = 0
            self.renewed = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if kwargs.get("events") == []:
                self.heartbeat_calls += 1
                if self.heartbeat_calls == 1:
                    raise ConnectionError("transient heartbeat write failure")
                result = await super().publish_session_operation_guarded_with_store_time(
                    session_id, **kwargs
                )
                self.renewed.set()
                return result
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_RETRY_SECONDS",
            0.01,
        )
        store = TransientHeartbeatStore()
        provider = BlockingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_heartbeat_retry",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-claim-heartbeat-retry",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=5)
        await asyncio.wait_for(store.renewed.wait(), timeout=5)
        assert store.heartbeat_calls >= 2
        assert not task.done()

        provider.release.set()
        events = await task
        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert provider.calls == 1

    asyncio.run(run())


def test_compact_session_claim_heartbeat_reconciles_lost_renewal_acknowledgement(
    monkeypatch,
) -> None:
    accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
    now = {"value": accepted_at}

    class LostRenewalAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__(ownership_clock=lambda: now["value"])
            self.acknowledgement_lost = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            result = await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )
            if kwargs.get("events") == [] and not self.acknowledgement_lost.is_set():
                self.acknowledgement_lost.set()
                raise ConnectionError("renewal acknowledgement lost after commit")
            return result

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = LostRenewalAcknowledgementStore()
        provider = BlockingCompactionProvider()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        reconciliation_finished = asyncio.Event()
        reconcile_claim = app._session_engine._reconcile_compaction_operation_claim_before_deadline

        async def observe_reconciliation(**kwargs):
            result = await reconcile_claim(**kwargs)
            if store.acknowledgement_lost.is_set():
                reconciliation_finished.set()
            return result

        monkeypatch.setattr(
            app._session_engine,
            "_reconcile_compaction_operation_claim_before_deadline",
            observe_reconciliation,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_heartbeat_lost_ack",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-claim-heartbeat-lost-ack",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        async def collect() -> list[Event]:
            return [event async for event in app.compact_session(request)]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=5)
        now["value"] = accepted_at + timedelta(minutes=4)
        await asyncio.wait_for(store.acknowledgement_lost.wait(), timeout=5)
        await asyncio.wait_for(reconciliation_finished.wait(), timeout=5)
        now["value"] = accepted_at + timedelta(minutes=6)

        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        record = checkpoint["session_operations"]["records"][request.idempotency_key]
        assert datetime.fromisoformat(record["claim_expires_at"]) == (
            accepted_at + timedelta(minutes=9)
        )
        assert now["value"] > accepted_at + session_engine_module._SESSION_OPERATION_CLAIM_LEASE
        assert not task.done()
        assert not provider.cancelled.is_set()

        provider.release.set()
        events = await task
        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert provider.calls == 1
        operation = await store.load_session_operation(created.id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "completed"

    asyncio.run(run())


def test_compact_session_stops_when_renewal_acknowledgement_exceeds_lease_deadline(
    monkeypatch,
) -> None:
    accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
    now = {"value": accepted_at}

    class DelayedRenewalAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__(ownership_clock=lambda: now["value"])
            self.renewal_committed = asyncio.Event()
            self.release_acknowledgement = asyncio.Event()
            self.delayed = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            result = await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )
            if kwargs.get("events") == [] and not self.delayed:
                self.delayed = True
                self.renewal_committed.set()
                await self.release_acknowledgement.wait()
            return result

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_LEASE",
            timedelta(milliseconds=500),
        )
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.2,
        )
        store = DelayedRenewalAcknowledgementStore()
        provider = BlockingCompactionProvider()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_delayed_renewal_acknowledgement",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-delayed-renewal-acknowledgement",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=5)
        await asyncio.wait_for(store.renewal_committed.wait(), timeout=5)
        now["value"] = accepted_at + timedelta(seconds=1)
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            60.0,
        )
        try:
            with pytest.raises(RuntimeError, match="not confirmed before its lease deadline"):
                await asyncio.wait_for(task, timeout=0.4)
        finally:
            store.release_acknowledgement.set()
            if not task.done():
                await asyncio.gather(task, return_exceptions=True)

        assert provider.calls == 1
        assert provider.cancelled.is_set()
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        operation = checkpoint["session_operations"]["records"][
            "compact-delayed-renewal-acknowledgement"
        ]
        assert operation["status"] == "running"
        assert datetime.fromisoformat(operation["claim_expires_at"]) <= now["value"]

    asyncio.run(run())


def test_compaction_claim_heartbeat_cancellation_observes_concurrent_renewal_failure(
    monkeypatch,
) -> None:
    accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at,
        )
        engine = app._session_engine
        heartbeat_task: asyncio.Task[None] | None = None

        async def reconcile_claim(**_kwargs) -> datetime:
            return accepted_at + timedelta(minutes=5)

        async def renew_claim(**_kwargs) -> datetime:
            assert heartbeat_task is not None
            heartbeat_task.cancel("cancel heartbeat while renewal fails")
            raise RuntimeError("renewal failed concurrently with cancellation")

        monkeypatch.setattr(
            engine,
            "_reconcile_compaction_operation_claim_expiry",
            reconcile_claim,
        )
        monkeypatch.setattr(engine, "_renew_compaction_operation_claim", renew_claim)
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_renewal_cancellation_race",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        request = CompactSessionRequest(
            session_id=session.id,
            idempotency_key="compaction-renewal-cancellation-race",
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )
        claim_expires_at = accepted_at + timedelta(minutes=5)
        task = asyncio.create_task(
            engine._heartbeat_compaction_operation_claim(
                session=session,
                request=request,
                operation_id="operation-id",
                attempt_id="attempt-id",
                claim_expires_at=claim_expires_at,
                stop=asyncio.Event(),
                state=_test_compaction_heartbeat_state(claim_expires_at),
            )
        )
        heartbeat_task = task

        with pytest.raises(
            asyncio.CancelledError,
            match="cancel heartbeat while renewal fails",
        ) as exc_info:
            await task

        assert task.cancelling() == 1
        assert task.cancelled()
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == ("renewal failed concurrently with cancellation")
        assert any(
            "claim renewal also failed during cancellation" in note
            for note in getattr(exc_info.value, "__notes__", ())
        )

    asyncio.run(run())


def test_compaction_claim_heartbeat_cancellation_observes_concurrent_reconciliation_failure(
    monkeypatch,
) -> None:
    accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at,
        )
        engine = app._session_engine
        heartbeat_task: asyncio.Task[None] | None = None

        async def reconcile_claim(**_kwargs) -> datetime:
            assert heartbeat_task is not None
            heartbeat_task.cancel("cancel heartbeat while reconciliation fails")
            raise session_engine_module.SessionCompactionAttemptSuperseded(
                "ownership changed during concurrent reconciliation"
            )

        monkeypatch.setattr(
            engine,
            "_reconcile_compaction_operation_claim_expiry",
            reconcile_claim,
        )
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_reconciliation_cancellation_race",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        request = CompactSessionRequest(
            session_id=session.id,
            idempotency_key="compaction-reconciliation-cancellation-race",
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )
        claim_expires_at = accepted_at + timedelta(minutes=5)
        task = asyncio.create_task(
            engine._heartbeat_compaction_operation_claim(
                session=session,
                request=request,
                operation_id="operation-id",
                attempt_id="attempt-id",
                claim_expires_at=claim_expires_at,
                stop=asyncio.Event(),
                state=_test_compaction_heartbeat_state(claim_expires_at),
            )
        )
        heartbeat_task = task

        with pytest.raises(
            asyncio.CancelledError,
            match="cancel heartbeat while reconciliation fails",
        ) as exc_info:
            await task

        assert task.cancelling() == 1
        assert task.cancelled()
        assert isinstance(
            exc_info.value.__cause__,
            session_engine_module.SessionCompactionAttemptSuperseded,
        )
        assert "ownership changed" in str(exc_info.value.__cause__)
        assert any(
            "claim reconciliation also failed during cancellation" in note
            for note in getattr(exc_info.value, "__notes__", ())
        )

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["reconciliation", "renewal"])
def test_compaction_claim_heartbeat_converts_concurrent_child_cancellation_to_failure(
    stage: str,
    monkeypatch,
) -> None:
    accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at,
        )
        engine = app._session_engine
        heartbeat_task: asyncio.Task[None] | None = None

        async def cancel_child() -> datetime:
            assert heartbeat_task is not None
            child_task = asyncio.current_task()
            assert child_task is not None
            child_task.add_done_callback(
                lambda _completed: heartbeat_task.cancel(f"stop heartbeat during {stage}")
            )
            child_task.cancel(f"cancel {stage} child")
            await asyncio.sleep(0)
            raise AssertionError("cancelled child resumed unexpectedly")

        async def reconcile_claim(**_kwargs) -> datetime:
            if stage == "reconciliation":
                return await cancel_child()
            return accepted_at + timedelta(minutes=5)

        async def renew_claim(**_kwargs) -> datetime:
            if stage == "renewal":
                return await cancel_child()
            raise AssertionError("unexpected renewal")

        monkeypatch.setattr(
            engine,
            "_reconcile_compaction_operation_claim_expiry",
            reconcile_claim,
        )
        monkeypatch.setattr(engine, "_renew_compaction_operation_claim", renew_claim)
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_compaction_{stage}_child_cancellation_race",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        request = CompactSessionRequest(
            session_id=session.id,
            idempotency_key=f"compaction-{stage}-child-cancellation-race",
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )
        claim_expires_at = accepted_at + timedelta(minutes=5)
        task = asyncio.create_task(
            engine._heartbeat_compaction_operation_claim(
                session=session,
                request=request,
                operation_id="operation-id",
                attempt_id="attempt-id",
                claim_expires_at=claim_expires_at,
                stop=asyncio.Event(),
                state=_test_compaction_heartbeat_state(claim_expires_at),
            )
        )
        heartbeat_task = task

        with pytest.raises(
            asyncio.CancelledError, match=f"stop heartbeat during {stage}"
        ) as exc_info:
            await task

        assert task.cancelling() == 1
        assert task.cancelled()
        operational_failure = exc_info.value.__cause__
        assert isinstance(operational_failure, RuntimeError)
        assert (
            str(operational_failure)
            == f"Session compaction claim {stage} was cancelled without caller cancellation."
        )
        child_cancellation = operational_failure.__cause__
        assert isinstance(child_cancellation, asyncio.CancelledError)
        assert child_cancellation.args == (f"cancel {stage} child",)

    asyncio.run(run())


def test_claimed_compaction_surfaces_failure_carried_by_cancelled_heartbeat() -> None:
    async def run() -> None:
        heartbeat_started = asyncio.Event()

        async def heartbeat() -> None:
            heartbeat_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise cancellation from RuntimeError("heartbeat ownership changed")

        heartbeat_task = asyncio.create_task(heartbeat())
        await heartbeat_started.wait()
        heartbeat_task.cancel("stop failed heartbeat")
        await asyncio.gather(heartbeat_task, return_exceptions=True)

        operation_called = False

        async def operation():
            nonlocal operation_called
            operation_called = True
            raise AssertionError("operation must not start after heartbeat failure")

        with pytest.raises(RuntimeError, match="heartbeat ownership changed"):
            await session_engine_module._run_while_session_operation_claimed(
                operation,
                heartbeat_task=heartbeat_task,
            )

        assert not operation_called
        assert heartbeat_task.cancelling() == 1
        assert heartbeat_task.cancelled()

    asyncio.run(run())


def test_claimed_compaction_observes_cancelled_heartbeat_cause_only_once() -> None:
    async def run() -> None:
        heartbeat_started = asyncio.Event()
        finish_heartbeat = asyncio.Event()

        async def heartbeat() -> None:
            heartbeat_started.set()
            await finish_heartbeat.wait()
            raise asyncio.CancelledError("heartbeat stopped") from RuntimeError(
                "heartbeat ownership changed during operation startup"
            )

        heartbeat_task = asyncio.create_task(heartbeat())
        await heartbeat_started.wait()
        finish_heartbeat.set()

        operation_called = False

        async def operation():
            nonlocal operation_called
            operation_called = True
            raise AssertionError("operation must not start after heartbeat failure")

        with pytest.raises(
            RuntimeError,
            match="heartbeat ownership changed during operation startup",
        ):
            await session_engine_module._run_while_session_operation_claimed(
                operation,
                heartbeat_task=heartbeat_task,
            )

        assert not operation_called
        assert heartbeat_task.cancelled()

    asyncio.run(run())


def test_compact_session_caller_cancellation_does_not_wait_for_uncertain_claim_commit(
    monkeypatch,
) -> None:
    class PostGuardStalledStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.guard_passed = asyncio.Event()
            self.release_renewal = asyncio.Event()
            self.renewal_finished = asyncio.Event()
            self.delayed = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if not self.delayed and kwargs.get("events") == []:
                self.delayed = True
                kwargs["commit_guard"]()
                self.guard_passed.set()
                try:
                    await self.release_renewal.wait()
                    return await super().publish_session_operation_guarded_with_store_time(
                        session_id,
                        **kwargs,
                    )
                finally:
                    self.renewal_finished.set()
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = PostGuardStalledStore()
        provider = BlockingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_cancel_uncertain_claim_commit",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-cancel-uncertain-claim-commit",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=5)
        await asyncio.wait_for(store.guard_passed.wait(), timeout=5)
        task.cancel("cancel during uncertain claim commit")

        with pytest.raises(
            asyncio.CancelledError,
            match="cancel during uncertain claim commit",
        ):
            await asyncio.wait_for(task, timeout=1)

        # Durable cancellation cleanup consumes the delivered request before
        # re-raising the same caller-visible cancellation signal.
        assert task.cancelling() == 0
        assert task.cancelled()
        assert provider.cancelled.is_set()
        assert not store.renewal_finished.is_set()

        store.release_renewal.set()
        await asyncio.wait_for(store.renewal_finished.wait(), timeout=5)

    asyncio.run(run())


def test_compact_session_claim_heartbeat_cannot_revive_an_expired_lease(monkeypatch) -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = InMemorySessionStore(ownership_clock=lambda: now["value"])
        provider = BlockingCompactionProvider()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_heartbeat_expired",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-claim-heartbeat-expired",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=5)
        now["value"] = accepted_at + timedelta(minutes=6)
        with pytest.raises(RuntimeError, match="expired before (?:renewal|reconciliation)"):
            await asyncio.wait_for(task, timeout=5)

        assert provider.calls == 1
        assert provider.cancelled.is_set()
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is None or "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_stalled_claim_renewal_is_bounded_by_lease_deadline(
    monkeypatch,
) -> None:
    class CancellableCompactor(ContextCompactor):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        def provider_budget_identity(self, _session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    class StalledHeartbeatStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.heartbeat_started = asyncio.Event()
            self.heartbeat_cancelled = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if kwargs.get("events") == []:
                self.heartbeat_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.heartbeat_cancelled.set()
                    raise
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_LEASE",
            timedelta(milliseconds=100),
        )
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = StalledHeartbeatStore()
        compactor = CancellableCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_heartbeat_stall",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-claim-heartbeat-stall",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(compactor.started.wait(), timeout=5)
        await asyncio.wait_for(store.heartbeat_started.wait(), timeout=5)
        with pytest.raises(RuntimeError, match="not confirmed before its lease deadline"):
            await asyncio.wait_for(task, timeout=5)

        assert store.heartbeat_cancelled.is_set()
        assert compactor.cancelled.is_set()
        durable_events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        assert EventType.SESSION_CHECKPOINTED not in {event.type for event in durable_events}

    asyncio.run(run())


def test_compact_session_stalled_claim_reconciliation_is_bounded_by_lease_deadline(
    monkeypatch,
) -> None:
    class StalledReconciliationStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.block_reconciliation = False
            self.reconciliation_started = asyncio.Event()
            self.reconciliation_cancelled = asyncio.Event()

        async def transform_checkpoint_with_store_time(
            self, session_id: str, checkpoint_transform
        ) -> None:
            if self.block_reconciliation:
                self.reconciliation_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.reconciliation_cancelled.set()
                    raise
            await super().transform_checkpoint_with_store_time(
                session_id,
                checkpoint_transform,
            )

    class BlockingCompactorWithReconciliation(ContextCompactor):
        def __init__(self, store: StalledReconciliationStore) -> None:
            self.store = store
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        def provider_budget_identity(self, _session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.store.block_reconciliation = True
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_LEASE",
            timedelta(milliseconds=100),
        )
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = StalledReconciliationStore()
        compactor = BlockingCompactorWithReconciliation(store)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_stalled_claim_reconciliation",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-stalled-claim-reconciliation",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(compactor.started.wait(), timeout=5)
        await asyncio.wait_for(store.reconciliation_started.wait(), timeout=5)
        with pytest.raises(RuntimeError, match="reconciliation was not confirmed"):
            await asyncio.wait_for(task, timeout=1)

        assert store.reconciliation_cancelled.is_set()
        assert compactor.cancelled.is_set()
        assert EventType.SESSION_CHECKPOINTED not in {
            event.type for event in await store.load_events(created.id)
        }

    asyncio.run(run())


@pytest.mark.parametrize(
    "commit_started_before_stall",
    [False, True],
    ids=["before-commit-guard", "after-commit-guard"],
)
def test_sqlite_stalled_claim_renewal_cannot_keep_work_running_after_deadline(
    monkeypatch,
    tmp_path,
    commit_started_before_stall: bool,
) -> None:
    class CancellableCompactor(ContextCompactor):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        def provider_budget_identity(self, _session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    class StalledCommitGuardSQLiteStore(SQLiteSessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(
            self,
            work_started: asyncio.Event,
            *,
            ownership_clock: Callable[[], datetime],
        ) -> None:
            super().__init__(
                tmp_path / "stalled-renewal.sqlite",
                ownership_clock=ownership_clock,
            )
            self.work_started = work_started
            self.guard_started = threading.Event()
            self.release_guard = threading.Event()
            self.failure_publication_started = asyncio.Event()
            self.failure_publication_finished = asyncio.Event()
            self.delayed_renewal = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if any(
                event.type == EventType.CONTEXT_COMPACTION_FAILED
                for event in kwargs.get("events", [])
            ):
                self.failure_publication_started.set()
                try:
                    return await super().publish_session_operation_guarded_with_store_time(
                        session_id,
                        **kwargs,
                    )
                finally:
                    self.failure_publication_finished.set()
            commit_guard = kwargs.get("commit_guard")
            if (
                self.work_started.is_set()
                and not self.delayed_renewal
                and kwargs.get("events") == []
                and commit_guard is not None
            ):
                self.delayed_renewal = True

                def delayed_commit_guard() -> None:
                    if commit_started_before_stall:
                        commit_guard()
                    self.guard_started.set()
                    if not self.release_guard.wait(timeout=5):
                        raise TimeoutError("test did not release the renewal commit guard")
                    if not commit_started_before_stall:
                        commit_guard()

                kwargs["commit_guard"] = delayed_commit_guard
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_LEASE",
            timedelta(milliseconds=100),
        )
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        compactor = CancellableCompactor()
        store = StalledCommitGuardSQLiteStore(
            compactor.started,
            ownership_clock=lambda: now["value"],
        )
        try:
            app = CayuApp(
                session_store=store,
                enable_logging=False,
                clock=lambda: now["value"],
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            created = await _create_profiled_session(
                app,
                store,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_sqlite_stalled_renewal",
                    messages=[],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            transcript = [
                Message.text("user", "old request"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current request"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)

            async def collect() -> list[Event]:
                return [
                    event
                    async for event in app.compact_session(
                        CompactSessionRequest(
                            session_id=created.id,
                            idempotency_key="compact-sqlite-stalled-renewal",
                            expected_run_epoch=completed.run_epoch,
                            expected_transcript_cursor=len(transcript),
                        )
                    )
                ]

            task = asyncio.create_task(collect())
            await asyncio.wait_for(compactor.started.wait(), timeout=5)
            await asyncio.wait_for(
                asyncio.to_thread(store.guard_started.wait),
                timeout=5,
            )
            now["value"] = accepted_at + timedelta(seconds=1)
            await asyncio.wait_for(compactor.cancelled.wait(), timeout=5)

            with pytest.raises(RuntimeError, match="not confirmed before its lease deadline"):
                await asyncio.wait_for(task, timeout=1)

            assert not store.failure_publication_started.is_set()
            assert not store.failure_publication_finished.is_set()
            store.release_guard.set()
            await asyncio.wait_for(store.failure_publication_started.wait(), timeout=5)
            await asyncio.wait_for(store.failure_publication_finished.wait(), timeout=5)

            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            record = checkpoint["session_operations"]["records"]["compact-sqlite-stalled-renewal"]
            assert datetime.fromisoformat(record["claim_expires_at"]) == (
                accepted_at + timedelta(milliseconds=100)
            )
            assert "context_compaction" not in checkpoint
            assert EventType.SESSION_CHECKPOINTED not in {
                event.type for event in await store.load_events(created.id)
            }
        finally:
            store.release_guard.set()
            await store.close()

    asyncio.run(run())


def test_sqlite_caller_cancellation_does_not_wait_for_stalled_claim_write(
    monkeypatch,
    tmp_path,
) -> None:
    class CancellableCompactor(ContextCompactor):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        def provider_budget_identity(self, _session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    class StalledClaimWriteSQLiteStore(SQLiteSessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, work_started: asyncio.Event) -> None:
            super().__init__(tmp_path / "cancel-stalled-renewal.sqlite")
            self.work_started = work_started
            self.guard_started = threading.Event()
            self.release_guard = threading.Event()
            self.failure_publication_finished = asyncio.Event()
            self.delayed_renewal = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if any(
                event.type == EventType.CONTEXT_COMPACTION_FAILED
                for event in kwargs.get("events", [])
            ):
                try:
                    return await super().publish_session_operation_guarded_with_store_time(
                        session_id,
                        **kwargs,
                    )
                finally:
                    self.failure_publication_finished.set()
            commit_guard = kwargs.get("commit_guard")
            if (
                self.work_started.is_set()
                and not self.delayed_renewal
                and kwargs.get("events") == []
                and commit_guard is not None
            ):
                self.delayed_renewal = True

                def delayed_commit_guard() -> None:
                    commit_guard()
                    self.guard_started.set()
                    if not self.release_guard.wait(timeout=5):
                        raise TimeoutError("test did not release the renewal commit guard")

                kwargs["commit_guard"] = delayed_commit_guard
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        compactor = CancellableCompactor()
        store = StalledClaimWriteSQLiteStore(compactor.started)
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            created = await _create_profiled_session(
                app,
                store,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_sqlite_cancel_stalled_claim_write",
                    messages=[],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            transcript = [
                Message.text("user", "old request"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current request"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)

            async def collect() -> list[Event]:
                return [
                    event
                    async for event in app.compact_session(
                        CompactSessionRequest(
                            session_id=created.id,
                            idempotency_key="compact-sqlite-cancel-stalled-claim-write",
                            expected_run_epoch=completed.run_epoch,
                            expected_transcript_cursor=len(transcript),
                        )
                    )
                ]

            task = asyncio.create_task(collect())
            await asyncio.wait_for(compactor.started.wait(), timeout=5)
            await asyncio.wait_for(
                asyncio.to_thread(store.guard_started.wait),
                timeout=5,
            )
            task.cancel("cancel while claim commit is stalled")

            with pytest.raises(
                asyncio.CancelledError,
                match="cancel while claim commit is stalled",
            ):
                await asyncio.wait_for(task, timeout=1)

            assert task.cancelled()
            assert compactor.cancelled.is_set()
            assert not store.failure_publication_finished.is_set()
            store.release_guard.set()
            await asyncio.wait_for(store.failure_publication_finished.wait(), timeout=5)
            assert EventType.CONTEXT_COMPACTION_FAILED in {
                event.type for event in await store.load_events(created.id)
            }
        finally:
            store.release_guard.set()
            await store.close()

    asyncio.run(run())


def test_compact_session_expired_claim_cannot_publish_terminal_checkpoint(monkeypatch) -> None:
    class StalledHeartbeatStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, *, ownership_clock: Callable[[], datetime]) -> None:
            super().__init__(ownership_clock=ownership_clock)
            self.heartbeat_started = asyncio.Event()
            self.heartbeat_cancelled = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if kwargs.get("events") == []:
                self.heartbeat_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.heartbeat_cancelled.set()
                    raise
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        store_now = {"value": accepted_at}
        worker_now = {"value": accepted_at}
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = StalledHeartbeatStore(ownership_clock=lambda: store_now["value"])
        compactor = BlockingCompactor()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: worker_now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_terminal_after_claim_expiry",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-terminal-after-claim-expiry",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(compactor.started.wait(), timeout=5)
        await asyncio.wait_for(store.heartbeat_started.wait(), timeout=5)
        store_now["value"] = accepted_at + timedelta(minutes=6)
        compactor.release.set()
        with pytest.raises(RuntimeError, match="expired before terminal publication"):
            await asyncio.wait_for(task, timeout=5)

        assert store.heartbeat_cancelled.is_set()
        durable_events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        assert EventType.SESSION_CHECKPOINTED not in {event.type for event in durable_events}
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint
        operation = checkpoint["session_operations"]["records"][
            "compact-terminal-after-claim-expiry"
        ]
        assert operation["status"] == "running"
        assert datetime.fromisoformat(operation["claim_expires_at"]) <= store_now["value"]

        recovered_events = await collect()
        assert recovered_events[-1].type == EventType.SESSION_CHECKPOINTED

    asyncio.run(run())


def test_compact_session_failure_publication_cannot_terminalize_after_claim_expiry() -> None:
    class FailOnceCompactor(ContextCompactor):
        def __init__(self) -> None:
            self.calls = 0

        def provider_budget_identity(self, _session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("compaction failed before delayed publication")
            return CompactionResult(
                summary="recovered summary",
                covered_message_count=len(request.messages),
            )

    class DelayedFailureStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, *, ownership_clock: Callable[[], datetime]) -> None:
            super().__init__(ownership_clock=ownership_clock)
            self.failure_publication_started = asyncio.Event()
            self.release_failure_publication = asyncio.Event()
            self.delayed = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if not self.delayed and any(
                event.type == EventType.CONTEXT_COMPACTION_FAILED
                for event in kwargs.get("events", [])
            ):
                self.delayed = True
                self.failure_publication_started.set()
                await self.release_failure_publication.wait()
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}
        store = DelayedFailureStore(ownership_clock=lambda: now["value"])
        compactor = FailOnceCompactor()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_failure_publication_after_claim_expiry",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-failure-after-claim-expiry",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        async def collect() -> list[Event]:
            return [event async for event in app.compact_session(request)]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(store.failure_publication_started.wait(), timeout=5)
        now["value"] = accepted_at + timedelta(minutes=6)
        store.release_failure_publication.set()
        with pytest.raises(RuntimeError, match="failed before delayed publication"):
            await asyncio.wait_for(task, timeout=5)

        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        operation = checkpoint["session_operations"]["records"][request.idempotency_key]
        assert operation["status"] == "running"
        assert datetime.fromisoformat(operation["claim_expires_at"]) <= now["value"]

        recovered_events = await collect()
        assert recovered_events[-1].type == EventType.SESSION_CHECKPOINTED
        assert compactor.calls == 2

    asyncio.run(run())


def test_compact_session_terminal_publication_wins_blocked_heartbeat_race(monkeypatch) -> None:
    class BlockingHeartbeatStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.heartbeat_started = asyncio.Event()
            self.heartbeat_cancelled = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if kwargs.get("events") == []:
                self.heartbeat_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.heartbeat_cancelled.set()
                    raise
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = BlockingHeartbeatStore()
        provider = BlockingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_heartbeat_terminal_race",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-claim-heartbeat-terminal-race",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=5)
        await asyncio.wait_for(store.heartbeat_started.wait(), timeout=5)
        provider.release.set()
        events = await asyncio.wait_for(task, timeout=5)

        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert store.heartbeat_cancelled.is_set()
        operation = await store.load_session_operation(
            created.id, "compact-claim-heartbeat-terminal-race"
        )
        assert operation is not None
        assert operation["status"] == "completed"

    asyncio.run(run())


def test_compact_session_claim_loss_waits_for_completed_dispatch_settlement(monkeypatch) -> None:
    class BlockingReconcileLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__()
            self.reconcile_started = asyncio.Event()
            self.release_reconcile = asyncio.Event()
            self.reconcile_cancelled = asyncio.Event()

        async def reconcile(self, **kwargs):
            self.reconcile_started.set()
            try:
                await self.release_reconcile.wait()
            except asyncio.CancelledError:
                self.reconcile_cancelled.set()
                raise
            return await super().reconcile(**kwargs)

    class ClaimLossDuringReconcileStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, ledger: BlockingReconcileLedger) -> None:
            super().__init__()
            self.ledger = ledger
            self.heartbeat_failed = asyncio.Event()
            self.stolen = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if not self.stolen and kwargs.get("events") == []:
                await self.ledger.reconcile_started.wait()
                self.stolen = True

                def steal(_session, checkpoint):
                    assert checkpoint is not None
                    updated = dict(checkpoint)
                    operations = dict(updated["session_operations"])
                    records = dict(operations["records"])
                    key, record = next(iter(records.items()))
                    records[key] = {**record, "current_attempt_id": "replacement-attempt"}
                    operations["records"] = records
                    updated["session_operations"] = operations
                    return updated

                await self.transform_checkpoint(session_id, steal)
                try:
                    return await super().publish_session_operation_guarded_with_store_time(
                        session_id,
                        **kwargs,
                    )
                finally:
                    self.heartbeat_failed.set()
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        ledger = BlockingReconcileLedger()
        store = ClaimLossDuringReconcileStore(ledger)
        provider = UsageCompactionProvider()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=100,
                            max_output_tokens=100,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_loss_during_settlement",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-claim-loss-during-settlement",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(ledger.reconcile_started.wait(), timeout=5)
        await asyncio.wait_for(store.heartbeat_failed.wait(), timeout=5)
        await asyncio.sleep(0)
        assert not task.done()
        assert not ledger.reconcile_cancelled.is_set()

        ledger.release_reconcile.set()
        with pytest.raises(RuntimeError, match="superseded"):
            await asyncio.wait_for(task, timeout=5)

        durable = await store.load_events(created.id)
        completions = [event for event in durable if event.type == EventType.MODEL_COMPLETED]
        reconciliations = [event for event in durable if event.type == EventType.BUDGET_RECONCILED]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 10
        assert len(reconciliations) == 1
        assert reconciliations[0].payload["actual_amount"] == "0.000012"
        record = next(iter(ledger._records.values()))
        assert record.status == "reconciled"
        assert record.actual_amount == Decimal("0.000012")
        assert provider.calls == 1

    asyncio.run(run())


def test_compact_session_claim_loss_retains_concurrent_result_telemetry(monkeypatch) -> None:
    class CompletionReportingProvider(ModelProvider):
        name = "reported-provider"

        def __init__(self) -> None:
            self.calls = 0
            self.result_ready = asyncio.Event()
            self.release_result = asyncio.Event()

        async def stream(self, request: ModelRequest):
            del request
            self.calls += 1
            self.result_ready.set()
            await self.release_result.wait()
            yield ModelStreamEvent.text_delta("completed before claim loss")
            yield ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 8, "output_tokens": 2},
                }
            )

    class CompletionReportingCompactor(ContextCompactor):
        def __init__(self) -> None:
            self.provider = CompletionReportingProvider()
            self.calls = 0
            self.result_ready = self.provider.result_ready
            self.release_result = self.provider.release_result

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.calls += 1
            return await ModelCompactor(
                provider=self.provider,
                model="summary-model",
                max_input_chars=100_000,
            ).compact(request)

    class ConcurrentClaimLossStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, compactor: CompletionReportingCompactor) -> None:
            super().__init__()
            self.compactor = compactor
            self.stolen = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if not self.stolen and kwargs.get("events") == []:
                await self.compactor.result_ready.wait()
                self.stolen = True

                def steal(_session, checkpoint):
                    assert checkpoint is not None
                    updated = dict(checkpoint)
                    operations = dict(updated["session_operations"])
                    records = dict(operations["records"])
                    key, record = next(iter(records.items()))
                    records[key] = {**record, "current_attempt_id": "replacement-attempt"}
                    operations["records"] = records
                    updated["session_operations"] = operations
                    return updated

                await self.transform_checkpoint(session_id, steal)
                self.compactor.release_result.set()
                await asyncio.sleep(0)
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        compactor = CompletionReportingCompactor()
        store = ConcurrentClaimLossStore(compactor)
        app = CayuApp(
            session_store=store,
            request_footprint=RequestFootprintConfig(enabled=False),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_loss_result_telemetry",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(RuntimeError, match="superseded"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-claim-loss-result-telemetry",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        durable = await store.load_events(created.id)
        completions = [event for event in durable if event.type == EventType.MODEL_COMPLETED]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 10
        assert EventType.SESSION_CHECKPOINTED not in {event.type for event in durable}
        assert compactor.calls == 1

    asyncio.run(run())


def test_compact_session_caller_cancellation_interrupts_blocked_claim_heartbeat(
    monkeypatch,
) -> None:
    class BlockingHeartbeatStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.heartbeat_started = asyncio.Event()
            self.heartbeat_cancelled = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if kwargs.get("events") == []:
                self.heartbeat_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.heartbeat_cancelled.set()
                    raise
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = BlockingHeartbeatStore()
        provider = BlockingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_heartbeat_cancel",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-claim-heartbeat-cancel",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=5)
        await asyncio.wait_for(store.heartbeat_started.wait(), timeout=5)
        task.cancel("cancel while claim heartbeat write is blocked")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.wait_for(task, timeout=5)

        assert exc_info.value.args == ("cancel while claim heartbeat write is blocked",)
        assert task.cancelled()
        assert provider.cancelled.is_set()
        assert store.heartbeat_cancelled.is_set()

    asyncio.run(run())


def test_compact_session_no_context_failure_is_replayable_without_provider_spend() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        compactor = RecordingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_no_context",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [Message.text("user", "only current request")]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-no-context",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )
        first = []

        with pytest.raises(ValueError, match="no complete older context"):
            async for event in app.compact_session(request):
                first.append(event)
        replay = [event async for event in app.compact_session(request)]

        assert [event.type for event in first] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert [event.id for event in replay] == [event.id for event in first]
        assert compactor.requests == []
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "session_operations" not in checkpoint
        operation = await store.load_session_operation(created.id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "failed"

    asyncio.run(run())


def test_compact_session_recovers_an_abandoned_accepted_operation() -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        store_now = {"value": accepted_at}
        store = InMemorySessionStore(ownership_clock=lambda: store_now["value"])
        first_app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            first_app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_recovery",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-recover-1",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        abandoned_stream = first_app.compact_session(request)
        abandoned_started = await anext(abandoned_stream)
        await abandoned_stream.aclose()

        store_now["value"] = accepted_at + timedelta(minutes=6)
        recovering_compactor = RecordingCompactor()
        recovered_app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at + timedelta(minutes=6),
        )
        recovered_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=recovering_compactor,
                max_user_turns=1,
            ),
        )
        recovered = [event async for event in recovered_app.compact_session(request)]
        replay = [event async for event in recovered_app.compact_session(request)]

        assert [event.type for event in recovered] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]
        assert [event.id for event in replay] == [
            abandoned_started.id,
            *[event.id for event in recovered],
        ]
        assert len(recovering_compactor.requests) == 1

    asyncio.run(run())


def test_compact_session_renews_operation_claim_between_provider_dispatches(monkeypatch) -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            1.0,
        )

        class DelayedAttemptPublicationStore(InMemorySessionStore):
            invocation_lifecycle_command_version = 1

            def __init__(self) -> None:
                super().__init__(ownership_clock=lambda: now["value"])
                self.delayed = False

            async def publish_session_operation_guarded_with_store_time(
                self, session_id: str, **kwargs
            ):
                events = kwargs["events"]
                if not self.delayed and any(
                    event.type == EventType.MODEL_COMPLETED for event in events
                ):
                    self.delayed = True
                    # The operation transform was already created at minute four,
                    # but the store does not execute it until thirty seconds later.
                    now["value"] = accepted_at + timedelta(minutes=4, seconds=30)
                return await super().publish_session_operation_guarded_with_store_time(
                    session_id, **kwargs
                )

        class MultiDispatchProvider(ModelProvider):
            name = "compaction-provider"

            def __init__(self) -> None:
                self.requests: list[ModelRequest] = []
                self.second_dispatch_started = asyncio.Event()
                self.release_second_dispatch = asyncio.Event()

            async def stream(self, request: ModelRequest):
                self.requests.append(request)
                call = len(self.requests)
                if call == 1:
                    # The first durable completion should renew the claim from
                    # minute four through minute nine.
                    now["value"] = accepted_at + timedelta(minutes=4)
                elif call == 2:
                    self.second_dispatch_started.set()
                    await self.release_second_dispatch.wait()
                yield ModelStreamEvent.text_delta(f"summary {call}")
                yield ModelStreamEvent.completed(
                    {
                        "model": request.model,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                )

        store = DelayedAttemptPublicationStore()
        provider = MultiDispatchProvider()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_renewal",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-claim-renewal",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        first_events: list[Event] = []

        async def compact() -> None:
            async for event in app.compact_session(request):
                first_events.append(event)

        first_task = asyncio.create_task(compact())
        await asyncio.wait_for(provider.second_dispatch_started.wait(), timeout=5)
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        record = checkpoint["session_operations"]["records"][request.idempotency_key]
        assert datetime.fromisoformat(record["updated_at"]) == accepted_at + timedelta(
            minutes=4, seconds=30
        )
        assert datetime.fromisoformat(record["claim_expires_at"]) == (
            accepted_at + timedelta(minutes=9, seconds=30)
        )

        # The original claim has expired, but the first attempt's atomic event
        # publication renewed it. The renewed lease must fence an overlapping
        # same-key retry while the second dispatch is in flight.
        now["value"] = accepted_at + timedelta(minutes=6)
        heartbeat_expiry = accepted_at + timedelta(minutes=11)
        async with asyncio.timeout(5):
            while True:
                checkpoint = await store.load_checkpoint(created.id)
                assert checkpoint is not None
                record = checkpoint["session_operations"]["records"][request.idempotency_key]
                if datetime.fromisoformat(record["claim_expires_at"]) >= heartbeat_expiry:
                    break
                await asyncio.sleep(0)
        assert not first_task.done()
        with pytest.raises(RuntimeError, match="already running"):
            async for _event in app.compact_session(request):
                pass
        assert len(provider.requests) == 2

        provider.release_second_dispatch.set()
        await first_task
        assert first_events[-1].type == EventType.SESSION_CHECKPOINTED

    asyncio.run(run())


def test_compact_session_heartbeat_timeout_honors_concurrent_publication_renewal(
    monkeypatch,
) -> None:
    class StalledFirstHeartbeatStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, *, ownership_clock: Callable[[], datetime]) -> None:
            super().__init__(ownership_clock=ownership_clock)
            self.heartbeat_started = asyncio.Event()
            self.release_heartbeat = asyncio.Event()
            self.stalled = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if not self.stalled and kwargs.get("events") == []:
                self.stalled = True
                self.heartbeat_started.set()
                await self.release_heartbeat.wait()
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    class PublishWhileHeartbeatStalledProvider(ModelProvider):
        name = "compaction-provider"

        def __init__(
            self,
            *,
            store: StalledFirstHeartbeatStore,
            advance_clock: Callable[[], None],
        ) -> None:
            self._store = store
            self._advance_clock = advance_clock
            self.calls = 0
            self.second_dispatch_started = asyncio.Event()
            self.release_second_dispatch = asyncio.Event()

        async def stream(self, request: ModelRequest):
            self.calls += 1
            if self.calls == 1:
                await self._store.heartbeat_started.wait()
                # Publish the first attempt late enough that its guarded claim
                # renewal extends beyond the original local lease deadline.
                await asyncio.sleep(1)
                self._advance_clock()
            else:
                self.second_dispatch_started.set()
                await self.release_second_dispatch.wait()
            yield ModelStreamEvent.text_delta(f"summary {self.calls}")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_LEASE",
            timedelta(seconds=2),
        )
        monkeypatch.setattr(
            session_engine_module,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = StalledFirstHeartbeatStore(ownership_clock=lambda: now["value"])
        provider = PublishWhileHeartbeatStalledProvider(
            store=store,
            advance_clock=lambda: now.update(value=accepted_at + timedelta(seconds=1)),
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_concurrent_publication_renewal",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-concurrent-publication-renewal",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(provider.second_dispatch_started.wait(), timeout=5)
            # This is past the original two-second deadline, but remains
            # inside the two-second lease measured from event publication.
            await asyncio.sleep(1.1)
            assert not task.done()
            assert provider.calls == 2

            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            record = checkpoint["session_operations"]["records"][
                "compact-concurrent-publication-renewal"
            ]
            assert datetime.fromisoformat(record["claim_expires_at"]) > (
                accepted_at + timedelta(seconds=2)
            )

            # Let the second attempt publish under later authoritative store
            # time while the old heartbeat store call is still stalled. Its
            # own dispatch point establishes the next conservative deadline.
            now["value"] = accepted_at + timedelta(seconds=2)
            provider.release_second_dispatch.set()
            async with asyncio.timeout(5):
                while True:
                    checkpoint = await store.load_checkpoint(created.id)
                    assert checkpoint is not None
                    record = checkpoint["session_operations"]["records"][
                        "compact-concurrent-publication-renewal"
                    ]
                    if datetime.fromisoformat(record["claim_expires_at"]) >= (
                        accepted_at + timedelta(seconds=4)
                    ):
                        break
                    await asyncio.sleep(0)
            store.release_heartbeat.set()
            events = await asyncio.wait_for(task, timeout=5)
            assert events[-1].type == EventType.SESSION_CHECKPOINTED
        finally:
            store.release_heartbeat.set()
            provider.release_second_dispatch.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_compact_session_recovery_fences_a_late_attempt_and_preserves_its_usage() -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        store_now = {"value": accepted_at}
        store = InMemorySessionStore(ownership_clock=lambda: store_now["value"])
        compactor = OverlappingCompactor()

        def configured_app(*, now: datetime) -> CayuApp:
            app = CayuApp(
                session_store=store,
                enable_logging=False,
                clock=lambda: now,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            return app

        first_app = configured_app(now=accepted_at)
        recovered_app = configured_app(now=accepted_at + timedelta(minutes=6))
        created = await _create_profiled_session(
            first_app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_overlap",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        first_request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-overlap-1",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
            requested_by=ResolutionActor(subject="operator-a"),
        )
        recovered_request = first_request.model_copy(
            update={"requested_by": ResolutionActor(subject="operator-b")}
        )
        first_events = []
        recovered_events = []

        async def collect_first() -> None:
            async for event in first_app.compact_session(first_request):
                first_events.append(event)

        async def collect_recovered() -> None:
            async for event in recovered_app.compact_session(recovered_request):
                recovered_events.append(event)

        first_task = asyncio.create_task(collect_first())
        await compactor.started[0].wait()
        store_now["value"] = accepted_at + timedelta(minutes=6)
        recovered_task = asyncio.create_task(collect_recovered())
        await compactor.started[1].wait()
        compactor.release[1].set()
        await recovered_task
        compactor.release[0].set()
        with pytest.raises(RuntimeError, match="superseded"):
            await first_task

        durable_events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        replay = [event async for event in recovered_app.compact_session(recovered_request)]

        assert (
            sum(event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in durable_events)
            == 1
        )
        assert sum(event.type == EventType.SESSION_CHECKPOINTED for event in durable_events) == 1
        assert sum(event.type == EventType.MODEL_COMPLETED for event in durable_events) == 2
        assert first_events[-1].type == EventType.CONTEXT_COMPACTION_FAILED
        assert first_events[-1].payload["error_type"] == "ContextBuildError"
        assert recovered_events[-1].type == EventType.SESSION_CHECKPOINTED
        assert [event.id for event in replay] == await _public_ids_for_private_event_ids(
            store,
            created.id,
            [event.id for event in durable_events],
        )
        operation_events = durable_events
        assert len({event.payload["operation_id"] for event in operation_events}) == 1
        assert len({event.payload["attempt_id"] for event in operation_events}) == 2
        footprint_events = [
            event for event in durable_events if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert footprint_events
        assert all("step" not in event.payload for event in footprint_events)
        model_started_events = [
            event for event in durable_events if event.type == EventType.MODEL_STARTED
        ]
        assert len(model_started_events) == len(footprint_events)
        for footprint_event in footprint_events:
            started_event = next(
                event
                for event in model_started_events
                if event.payload["model_attempt_id"] == footprint_event.payload["model_attempt_id"]
            )
            assert durable_events.index(footprint_event) < durable_events.index(started_event)
            assert (
                started_event.payload["operation_id"] == (footprint_event.payload["operation_id"])
            )
            assert started_event.payload["attempt_id"] == footprint_event.payload["attempt_id"]
        model_step_ids = {
            event.payload["model_step_id"]
            for event in durable_events
            if event.type
            in {
                EventType.REQUEST_FOOTPRINT_RECORDED,
                EventType.MODEL_COMPLETED,
                EventType.CONTEXT_COMPACTION_STARTED,
                EventType.CONTEXT_COMPACTION_COMPLETED,
                EventType.CONTEXT_COMPACTION_FAILED,
                EventType.SESSION_CHECKPOINTED,
            }
        }
        model_attempt_ids = {
            event.payload["model_attempt_id"]
            for event in durable_events
            if event.type == EventType.MODEL_COMPLETED
        }
        assert len(model_step_ids) == 1
        assert len(model_attempt_ids) == 2
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"]["summary"] == "summary from attempt 2"

    asyncio.run(run())


def test_compact_session_completion_publication_failure_releases_the_claim() -> None:
    async def run() -> None:
        store = FailingCompletionPublishStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_completion_failure",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-completion-failure",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )
        first = []

        with pytest.raises(RuntimeError, match="simulated completion publication failure"):
            async for event in app.compact_session(request):
                first.append(event)
        replay = [event async for event in app.compact_session(request)]
        checkpoint = await store.load_checkpoint(created.id)

        assert [event.type for event in first] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert [event.id for event in replay] == [event.id for event in first]
        assert checkpoint is not None
        assert "session_operations" not in checkpoint
        operation = await store.load_session_operation(created.id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "failed"

    asyncio.run(run())


def test_expired_compaction_claim_does_not_block_resume() -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        store_now = {"value": accepted_at}
        store = InMemorySessionStore(ownership_clock=lambda: store_now["value"])
        first_app = CayuApp(session_store=store, enable_logging=False, clock=lambda: accepted_at)
        first_app.register_provider(CompletingProvider())
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        created = await _create_profiled_session(
            first_app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_expired_compact_resume",
                messages=[Message.text("user", "create only")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=first_app,
            ),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        abandoned = first_app.compact_session(
            CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-expired-resume",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
        )
        await anext(abandoned)
        await abandoned.aclose()

        store_now["value"] = accepted_at + timedelta(minutes=6)
        resumed_app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at + timedelta(minutes=6),
        )
        resumed_app.register_provider(CompletingProvider())
        resumed_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        events = [
            event
            async for event in resumed_app.resume(
                ResumeRequest(
                    session_id=created.id,
                    messages=[Message.text("user", "continue")],
                )
            )
        ]

        assert EventType.SESSION_COMPLETED in [event.type for event in events]
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        operations = checkpoint["session_operations"]
        assert operations["active_operation_id"] is None
        assert operations["records"]["compact-expired-resume"]["status"] == "abandoned"

    asyncio.run(run())


def test_compaction_store_wait_ignores_application_clock_skew() -> None:
    async def run() -> None:
        store_now = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        application_now = datetime(2100, 1, 1, tzinfo=UTC)
        store = InMemorySessionStore(ownership_clock=lambda: store_now)
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: application_now,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_application_clock_skew",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-with-skewed-application-clock",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]
        operation = await store.load_session_operation(
            created.id,
            "compact-with-skewed-application-clock",
        )
        return events, operation

    events, operation = asyncio.run(run())

    assert events[-1].type is EventType.SESSION_CHECKPOINTED
    assert operation is not None
    assert operation["status"] == "completed"


def test_failed_attempt_evidence_extends_an_archived_abandoned_record() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_archived_abandoned_attempt",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transform = session_engine_module._fail_session_operation_checkpoint(
            idempotency_key="expired-operation",
            operation_id="operation-id",
            attempt_id="attempt-id",
            failed_event_id="failed-event-id",
            attempt_event_ids=["completion-event-id", "settlement-event-id"],
            error_type="ContextBuildError",
            clock=lambda: datetime(2026, 7, 14, 12, 6, tzinfo=UTC),
            on_terminalize=lambda _expires_at: None,
        )

        publication = transform(
            session,
            {},
            {
                "operation_id": "operation-id",
                "current_attempt_id": "attempt-id",
                "status": "abandoned",
                "event_ids": ["started-event-id"],
            },
        )

        assert publication.operation_records is not None
        archived = publication.operation_records["expired-operation"]
        assert archived["status"] == "abandoned"
        assert archived["event_ids"] == [
            "started-event-id",
            "completion-event-id",
            "settlement-event-id",
            "failed-event-id",
        ]

    asyncio.run(run())


def test_expired_compaction_claim_does_not_block_fork_or_a_new_key() -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        store_now = {"value": accepted_at}
        store = InMemorySessionStore(ownership_clock=lambda: store_now["value"])
        first_app = CayuApp(session_store=store, enable_logging=False, clock=lambda: accepted_at)
        first_app.register_provider(CompletingProvider())
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            first_app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_expired_compact_fork",
                messages=[Message.text("user", "create only")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=first_app,
            ),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        abandoned = first_app.compact_session(
            CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-expired-fork",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
        )
        await anext(abandoned)
        await abandoned.aclose()

        store_now["value"] = accepted_at + timedelta(minutes=6)
        recovered_app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at + timedelta(minutes=6),
        )
        recovered_app.register_provider(CompletingProvider())
        recovered_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        forked = [
            event
            async for event in recovered_app.fork_session(
                ForkSessionRequest(
                    source_session_id=created.id,
                    session_id="sess_after_expired_compact",
                )
            )
        ]
        compacted = [
            event
            async for event in recovered_app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-after-expired-key",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]

        assert forked[-1].type == EventType.SESSION_FORKED
        assert compacted[-1].type == EventType.SESSION_CHECKPOINTED
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "session_operations" not in checkpoint
        abandoned_operation = await store.load_session_operation(
            created.id,
            "compact-expired-fork",
        )
        completed_operation = await store.load_session_operation(
            created.id,
            "compact-after-expired-key",
        )
        assert abandoned_operation is not None
        assert abandoned_operation["status"] == "abandoned"
        assert completed_operation is not None
        assert completed_operation["status"] == "completed"

    asyncio.run(run())


def test_fork_inherits_compacted_context_without_source_operation_records() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider())
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_fork_source",
                messages=[Message.text("user", "create only")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        source = await store.update_status(created.id, SessionStatus.COMPLETED)
        idempotency_key = "compact-fork-independent-key"
        source_events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=source.id,
                    idempotency_key=idempotency_key,
                    expected_run_epoch=source.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]
        assert source_events[-1].type == EventType.SESSION_CHECKPOINTED

        fork_events = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id="sess_compact_fork_child",
                )
            )
        ]
        assert fork_events[-1].type == EventType.SESSION_FORKED
        child = await store.load("sess_compact_fork_child")
        assert child is not None
        child_checkpoint = await store.load_checkpoint(child.id)
        assert child_checkpoint is not None
        assert child_checkpoint["context_compaction"]["summary"] == "durable compact summary"
        assert "session_operations" not in child_checkpoint

        child_messages = [
            Message.text("user", "child request"),
            Message.text("assistant", "child answer"),
        ]
        await store.append_transcript_messages(child.id, child_messages)
        child_events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=child.id,
                    idempotency_key=idempotency_key,
                    expected_run_epoch=child.run_epoch,
                    expected_transcript_cursor=len(transcript) + len(child_messages),
                )
            )
        ]

        assert child_events[-1].type == EventType.SESSION_CHECKPOINTED
        child_checkpoint = await store.load_checkpoint(child.id)
        assert child_checkpoint is not None
        assert "session_operations" not in child_checkpoint
        child_operation = await store.load_session_operation(child.id, idempotency_key)
        assert child_operation is not None
        assert child_operation["status"] == "completed"

    asyncio.run(run())


def test_compact_session_failure_replays_without_spend_and_new_key_retries() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        compactor = FailOnceCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_failure",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        failed_request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-failed-1",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
            instructions="secret instructions",
        )
        failed_events = []
        with pytest.raises(RuntimeError, match="provider prompt echoed"):
            async for event in app.compact_session(failed_request):
                failed_events.append(event)

        replay = [event async for event in app.compact_session(failed_request)]
        assert [event.id for event in replay] == [event.id for event in failed_events]
        assert [event.type for event in replay] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert compactor.calls == 1
        assert "error" not in replay[-1].payload
        assert replay[-1].payload["requested_source_start"] == 0
        assert replay[-1].payload["requested_source_end"] == 2
        assert replay[-1].payload["represented_source_start"] == 0
        assert replay[-1].payload["represented_source_end"] == 0
        assert replay[-1].payload["represented_message_count"] == 0
        assert replay[-1].payload["coverage_mode"] == "failed"
        assert replay[-1].payload["chunk_count"] == 0
        assert replay[-1].payload["chunk_mode"] == "failed"
        assert "bounded_input" not in replay[-1].payload
        assert replay[-1].payload["compaction_failed"] is True
        assert "compacted_transcript_cursor" not in replay[-1].payload
        assert "result_transcript_cursor" not in replay[-1].payload
        assert "secret instructions" not in json.dumps(replay[-1].model_dump(mode="json"))

        retry = [
            event
            async for event in app.compact_session(
                failed_request.model_copy(update={"idempotency_key": "compact-retry-1"})
            )
        ]
        assert retry[-1].type == EventType.SESSION_CHECKPOINTED
        assert compactor.calls == 2

        with pytest.raises(ValueError, match="different request"):
            async for _event in app.compact_session(
                failed_request.model_copy(update={"instructions": "different"})
            ):
                pass
        assert compactor.calls == 2

    asyncio.run(run())


def test_sqlite_compaction_outcome_reconstructs_after_reopen(tmp_path) -> None:
    async def run() -> None:
        path = tmp_path / "compact-reconstruction.sqlite"
        store = SQLiteSessionStore(path)
        compactor = RecordingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_sqlite_reopen",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-sqlite-reopen-1",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )
        first = [event async for event in app.compact_session(request)]
        await store.close()

        reopened = SQLiteSessionStore(path)
        replay_compactor = RecordingCompactor()
        replay_app = CayuApp(session_store=reopened, enable_logging=False)
        replay_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=replay_compactor,
                max_user_turns=1,
            ),
        )
        try:
            replay = [event async for event in replay_app.compact_session(request)]
            assert [event.id for event in replay] == [event.id for event in first]
            assert replay_compactor.requests == []
            assert await reopened.load_transcript(created.id) == transcript
        finally:
            await reopened.close()

    asyncio.run(run())


def test_compact_session_attributes_provider_usage_and_honors_run_limits() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=UsageCompactionProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_usage",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        first = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-usage-1",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]
        usage_event = next(event for event in first if event.type == EventType.MODEL_COMPLETED)
        assert usage_event.payload["operation_id"]
        assert usage_event.payload["reason"] == "application_requested"
        assert usage_event.payload["usage_metrics"]["total_tokens"] == 10
        terminal_context_events = [
            event
            for event in first
            if event.type
            in {
                EventType.CONTEXT_COMPACTION_COMPLETED,
                EventType.SESSION_CHECKPOINTED,
            }
        ]
        assert usage_event.payload["model_attempt_id"] == PRIVATE_EVENT_AUTHORITY
        usage_sequence = public_event_sequence(usage_event.id)
        assert usage_sequence is not None
        private_usage_record = next(
            record
            for record in await store.query_events(
                EventQuery(
                    session_id=created.id,
                    after_sequence=usage_sequence - 1,
                    limit=1,
                )
            )
            if record.sequence == usage_sequence
        )
        assert private_usage_record.event.payload["model_attempt_id"].startswith("matt_")
        assert all(
            event.payload["model_step_id"] == usage_event.payload["model_step_id"]
            for event in terminal_context_events
        )
        assert all("model_attempt_id" not in event.payload for event in terminal_context_events)

        tail = [
            Message.text("user", "later request"),
            Message.text("assistant", "later answer"),
        ]
        await store.append_transcript_messages(created.id, tail)
        checkpoint_before_limit = await store.load_checkpoint(created.id)
        limited_events = []
        with pytest.raises(RuntimeError, match="limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-usage-limited",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript) + len(tail),
                    limits=RunLimits(max_total_tokens=5),
                )
            ):
                limited_events.append(event)

        assert [event.type for event in limited_events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        checkpoint_after_limit = await store.load_checkpoint(created.id)
        assert (
            checkpoint_after_limit["context_compaction"]
            == checkpoint_before_limit["context_compaction"]
        )

    asyncio.run(run())


@pytest.mark.parametrize("invalid_text", ["100\x00", "\ud800"], ids=["nul", "surrogate"])
@pytest.mark.parametrize(
    "invalid_primary_counter",
    [True, False],
    ids=["invalid-primary", "invalid-extra-field"],
)
def test_explicit_compaction_persists_undurable_usage_as_unpriced_evidence(
    invalid_text: str,
    invalid_primary_counter: bool,
) -> None:
    class UndurableUsageProvider(ModelProvider):
        name = "renamed-openai"
        usage_dialect = UsageDialect.OPENAI

        async def stream(self, request: ModelRequest):
            yield ModelStreamEvent.text_delta("provider summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": (
                        {
                            "input_tokens": invalid_text,
                            "output_tokens": 1,
                        }
                        if invalid_primary_counter
                        else {
                            "input_tokens": 10,
                            "output_tokens": 1,
                            "provider_note": invalid_text,
                        }
                    ),
                }
            )

    async def run() -> None:
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="renamed-openai",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=1_000_000,
                            max_output_tokens=0,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=UndurableUsageProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=(
                    "explicit-undurable-compaction-"
                    f"{invalid_primary_counter}-{ord(invalid_text[-1])}"
                ),
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed_session = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []
        with pytest.raises(ContextBuildError):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="explicit-undurable-compaction",
                    expected_run_epoch=completed_session.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)
        completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
        reserved = next(event for event in events if event.type == EventType.BUDGET_RESERVED)
        reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
        cost = await app.get_session_cost(created.id, pricing)

        assert "usage" not in completed.payload
        assert "usage_metrics" not in completed.payload
        assert completed.payload["usage_normalization_failed"] is True
        assert completed.payload["usage_unavailable_reason"] == (
            "invalid compaction usage telemetry"
        )
        assert cost.model_steps == 1
        assert cost.priced_model_steps == 0
        assert cost.unpriced_model_steps == 1
        assert reserved.payload["actual"] == "1"
        assert reconciled.payload["actual_amount"] == "1"
        assert reconciled.payload["released_amount"] == "0"
        assert reconciled.payload["reason"] == (
            "compaction completed without priced usage; charged reserved amount"
        )
        ledger_record = next(iter(ledger._records.values()))
        assert ledger_record.reason == (
            "compaction completed without priced usage; charged reserved amount"
        )
        assert events[-1].type == EventType.CONTEXT_COMPACTION_FAILED

    asyncio.run(run())


def test_compact_session_deduplicates_persisted_usage_for_session_run_limit() -> None:
    async def run() -> None:
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_session_limit_deduplication",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-session-limit-deduplication",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    limits=RunLimits(max_total_tokens=15, scope="session"),
                )
            )
        ]

        assert provider.calls == 1
        completions = [event for event in events if event.type == EventType.MODEL_COMPLETED]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 10
        assert sum(event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in events) == 1
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2

    asyncio.run(run())


def test_compact_session_preserves_usage_for_durably_invalid_summary() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider(summary="invalid\x00summary")
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_invalid_summary",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-invalid-summary",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )
        first = []

        with pytest.raises(RuntimeError, match="must not contain NUL characters"):
            async for event in app.compact_session(request):
                first.append(event)
        replay = [event async for event in app.compact_session(request)]

        assert [event.type for event in first] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_RECONCILED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert [event.id for event in replay] == [event.id for event in first]
        assert provider.calls == 1
        usage_event = first[5]
        assert usage_event.payload["compaction_outcome"] == "invalid_summary"
        assert usage_event.payload["usage_metrics"]["total_tokens"] == 10
        assert "compaction_attempt_id" not in usage_event.payload
        assert first[6].payload["actual_amount"] == "0.000012"
        assert first[7].payload["error_type"] == "ContextBuildError"
        usage = await app.get_session_usage(created.id)
        assert usage.model_steps == 1
        assert usage.usage.total_tokens == 10
        cost = await app.get_session_cost(created.id, pricing)
        assert cost.model_steps == 1
        assert cost.total_cost == Decimal("0.000012")
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint
        operation = await store.load_session_operation(created.id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "failed"
        assert [event.id for event in first] == await _public_ids_for_private_event_ids(
            store,
            created.id,
            operation["event_ids"],
        )

    asyncio.run(run())


def test_compact_session_admits_each_static_priced_hierarchy_dispatch() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        ledger = InMemoryBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.000035"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_static_hierarchy_dispatch_budget",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events: list[Event] = []

        with pytest.raises(RuntimeError, match="budget reservation failed"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="static-hierarchy-dispatch-budget",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert provider.calls == 1
        assert sum(event.type == EventType.BUDGET_RESERVED for event in events) == 1
        assert sum(event.type == EventType.BUDGET_RECONCILED for event in events) == 1
        assert sum(event.type == EventType.BUDGET_RESERVATION_FAILED for event in events) == 1
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 10
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_admits_each_hierarchy_dispatch_against_run_limits() -> None:
    async def run() -> None:
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_hierarchy_run_limit",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events: list[Event] = []

        with pytest.raises(RuntimeError, match="Compaction limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="explicit-hierarchy-run-limit",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    limits=RunLimits(max_total_tokens=10),
                )
            ):
                events.append(event)

        assert provider.calls == 1
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 10
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events) == 1
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_admits_each_hierarchy_dispatch_against_static_budget() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.000010"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_hierarchy_static_budget",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events: list[Event] = []

        with pytest.raises(RuntimeError, match="Compaction budget limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="explicit-hierarchy-static-budget",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert provider.calls == 1
        assert EventType.BUDGET_RESERVED not in {event.type for event in events}
        assert EventType.BUDGET_LIMIT_REACHED in {event.type for event in events}
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 10
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events) == 1
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_contextual_budget_counts_prior_hierarchy_dispatches() -> None:
    async def run() -> None:
        identity = bedrock_billing_identity(
            invoked_model="summary-model",
            source_region="us-east-1",
            resource_type="foundation_model",
        )
        provider = ContextualBedrockCompactionProvider(identity)
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="bedrock",
                    model="summary-model",
                    match="exact",
                    input_per_million=Decimal("3"),
                    output_per_million=Decimal("15"),
                    pricing_context={
                        "source_region": ("us-east-1",),
                        "service_tier": ("default",),
                    },
                ),
            )
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.0001"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_contextual_hierarchy_dispatch_budget",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events: list[Event] = []

        with pytest.raises(RuntimeError, match="budget limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="contextual-hierarchy-dispatch-budget",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert provider.calls == 2
        completions = [event for event in events if event.type == EventType.MODEL_COMPLETED]
        assert len(completions) == 2
        assert sum(event.payload["usage_metrics"]["total_tokens"] for event in completions) == 20
        assert EventType.BUDGET_LIMIT_REACHED in {event.type for event in events}
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_cancellation_reconciles_observed_completion_usage() -> None:
    class CompletedThenBlockingProvider(ModelProvider):
        name = "compaction-provider"

        def __init__(self) -> None:
            self.completed = asyncio.Event()
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            yield ModelStreamEvent.text_delta("provider summary")
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}})
            self.completed.set()
            await asyncio.Event().wait()

    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = CompletedThenBlockingProvider()
        ledger = InMemoryBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=100,
                            max_output_tokens=100,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_cancel_after_compaction_completion",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="cancel-after-compaction-completion",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await provider.completed.wait()
        task.cancel("cancel after completed")
        with pytest.raises(asyncio.CancelledError, match="cancel after completed"):
            await task

        durable = await store.load_events(created.id)
        completion = next(event for event in durable if event.type == EventType.MODEL_COMPLETED)
        reconciliation = next(
            event for event in durable if event.type == EventType.BUDGET_RECONCILED
        )
        assert completion.payload["usage_metrics"]["total_tokens"] == 10
        assert reconciliation.payload["actual_amount"] == "0.000012"
        record = next(iter(ledger._records.values()))
        assert record.actual_amount == Decimal("0.000012")
        assert record.status == "reconciled"

    asyncio.run(run())


@pytest.mark.parametrize("terminal_signal", ["caller_cancellation", "lease_loss"])
def test_compact_session_preserves_observed_usage_when_terminal_signal_replaces_child_failure(
    terminal_signal: str,
) -> None:
    class CompletedToolCallThenBlockingProvider(ModelProvider):
        name = "compaction-provider"

        def __init__(self) -> None:
            self.completed = asyncio.Event()
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            yield ModelStreamEvent.text_delta("unusable provider summary")
            yield ModelStreamEvent.tool_call(name="unexpected", arguments={})
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}})
            self.completed.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                # The provider's own failure turns the already-observed tool call into
                # an ordinary protocol failure while the outer signal stays primary.
                raise RuntimeError("provider failed during stream cleanup") from exc

    class LoseLeaseOnSecondHeartbeatLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=1)
            self.heartbeat_calls = 0

        async def heartbeat(self, *, reservation_id: str) -> bool:
            self.heartbeat_calls += 1
            if self.heartbeat_calls == 2:
                return False
            return await super().heartbeat(reservation_id=reservation_id)

    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = CompletedToolCallThenBlockingProvider()
        ledger = (
            LoseLeaseOnSecondHeartbeatLedger()
            if terminal_signal == "lease_loss"
            else InMemoryBudgetLedger(reservation_ttl_seconds=30)
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=100,
                            max_output_tokens=100,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_child_failure_{terminal_signal}",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key=f"child-failure-{terminal_signal}",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await provider.completed.wait()
        if terminal_signal == "caller_cancellation":
            task.cancel("cancel after completed tool call")
            with pytest.raises(
                asyncio.CancelledError,
                match="cancel after completed tool call",
            ):
                await task
            assert task.cancelling() == 0
            assert task.cancelled()
        else:
            with pytest.raises(ContextBuildError, match="lease was lost") as raised:
                await task
            assert isinstance(raised.value.__cause__, BudgetReservationLeaseLost)
            assert raised.value.__cause__.__dict__["completed_metadata"]["usage"] == {
                "input_tokens": 8,
                "output_tokens": 2,
            }

        durable = await store.load_events(created.id)
        completion_events = [event for event in durable if event.type == EventType.MODEL_COMPLETED]
        assert len(completion_events) == 1
        assert completion_events[0].payload["usage_metrics"]["total_tokens"] == 10
        assert completion_events[0].payload["compaction_outcome"] == (
            "cancelled_after_completion"
            if terminal_signal == "caller_cancellation"
            else "provider_error_after_completion"
        )
        reconciliations = [event for event in durable if event.type == EventType.BUDGET_RECONCILED]
        assert len(reconciliations) == 1
        assert reconciliations[0].payload["actual_amount"] == "0.000012"
        record = next(iter(ledger._records.values()))
        assert record.actual_amount == Decimal("0.000012")
        assert record.status == "reconciled"
        assert provider.calls == 1
        if isinstance(ledger, LoseLeaseOnSecondHeartbeatLedger):
            assert ledger.heartbeat_calls == 2

    asyncio.run(run())


def test_compact_session_cancellation_during_final_renewal_reconciles_actual_usage() -> None:
    class BlockingFinalRenewalLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=30)
            self.heartbeat_calls = 0
            self.final_renewal_started = asyncio.Event()

        async def heartbeat(self, *, reservation_id: str) -> bool:
            self.heartbeat_calls += 1
            if self.heartbeat_calls == 2:
                self.final_renewal_started.set()
                await asyncio.Event().wait()
            return await super().heartbeat(reservation_id=reservation_id)

    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        ledger = BlockingFinalRenewalLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=100,
                            max_output_tokens=100,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_cancel_during_final_compaction_renewal",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="cancel-during-final-compaction-renewal",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(ledger.final_renewal_started.wait(), timeout=5)
        task.cancel("cancel during final reservation renewal")
        with pytest.raises(
            asyncio.CancelledError,
            match="cancel during final reservation renewal",
        ):
            await task

        durable = await store.load_events(created.id)
        completion = next(event for event in durable if event.type == EventType.MODEL_COMPLETED)
        reconciliation = next(
            event for event in durable if event.type == EventType.BUDGET_RECONCILED
        )
        assert provider.calls == 1
        assert completion.payload["usage_metrics"]["total_tokens"] == 10
        assert reconciliation.payload["actual_amount"] == "0.000012"
        record = next(iter(ledger._records.values()))
        assert record.actual_amount == Decimal("0.000012")
        assert record.status == "reconciled"

    asyncio.run(run())


def test_compact_session_failure_after_completion_reconciles_actual_usage() -> None:
    class EventAfterCompletionProvider(ModelProvider):
        name = "compaction-provider"

        async def stream(self, request: ModelRequest):
            yield ModelStreamEvent.text_delta("provider summary")
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}})
            yield ModelStreamEvent.text_delta("invalid trailing event")

    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        ledger = InMemoryBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=100,
                            max_output_tokens=100,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=EventAfterCompletionProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_event_after_completion",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(
            RuntimeError,
            match="Compaction provider emitted event after completed: text_delta",
        ):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="event-after-compaction-completion",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        durable = await store.load_events(created.id)
        completion = next(event for event in durable if event.type == EventType.MODEL_COMPLETED)
        reconciliation = next(
            event for event in durable if event.type == EventType.BUDGET_RECONCILED
        )
        assert completion.payload["usage_metrics"]["total_tokens"] == 10
        assert reconciliation.payload["actual_amount"] == "0.000012"
        record = next(iter(ledger._records.values()))
        assert record.actual_amount == Decimal("0.000012")
        assert record.status == "reconciled"

    asyncio.run(run())


def test_compact_session_uses_frozen_canonical_billing_provider_without_identity() -> None:
    class GatewayProvider(UsageCompactionProvider):
        name = "gateway"
        billing_provider_name = "billco"

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            assert request.model == "summary-model"
            self.name = "poisoned\x00provider"
            self.billing_provider_name = "poisoned\ud800billing"
            return None

    async def run() -> None:
        provider = GatewayProvider()
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="billco",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_gateway_compaction_billing",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="gateway-compaction-billing",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]

        completion = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
        reserved = next(event for event in events if event.type == EventType.BUDGET_RESERVED)
        reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
        assert provider.calls == 1
        assert completion.payload["provider_name"] == "billco"
        assert completion.payload["usage_metrics"]["provider_name"] == "billco"
        assert reserved.payload["provider_name"] == "billco"
        assert reconciled.payload["actual_amount"] == "0.000012"
        assert provider.name == "poisoned\x00provider"
        assert provider.billing_provider_name == "poisoned\ud800billing"

    asyncio.run(run())


def test_compact_session_usage_counts_against_supplied_cost_budget() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=UsageCompactionProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_budget",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        events = []

        with pytest.raises(RuntimeError, match="budget limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-budget-1",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    budget_limits=(
                        BudgetLimit(
                            scope="run",
                            max_estimated_cost=Decimal("0.00001"),
                            pricing=pricing,
                        ),
                    ),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_LIMIT_REACHED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        budget_event = events[-2]
        assert budget_event.payload["actual"] == "0.000012"
        assert budget_event.payload["operation_id"]
        assert await store.load_transcript(created.id) == transcript
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_resolves_bedrock_identity_before_reservation_and_replays() -> None:
    async def run() -> None:
        identity = bedrock_billing_identity(
            invoked_model="global.anthropic.claude-sonnet-4-6",
            source_region="us-east-1",
            resource_type="inference_profile",
            profile_scope="global",
        )
        provider = ContextualBedrockCompactionProvider(identity)
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="bedrock",
                    model=identity.resource_id,
                    match="exact",
                    input_per_million=Decimal("3"),
                    output_per_million=Decimal("15"),
                    pricing_context={
                        "source_region": ("us-east-1",),
                        "service_tier": ("default",),
                    },
                ),
            )
        )
        ledger = InMemoryBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("10"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model=identity.resource_id),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_contextual_explicit_compaction",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="contextual-explicit-compaction",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        first = [event async for event in app.compact_session(request)]
        replay = [event async for event in app.compact_session(request)]

        assert provider.calls == 1
        assert [event.id for event in replay] == [event.id for event in first]
        reserved = next(event for event in first if event.type == EventType.BUDGET_RESERVED)
        reconciled = next(event for event in first if event.type == EventType.BUDGET_RECONCILED)
        completion = next(event for event in first if event.type == EventType.MODEL_COMPLETED)
        completed_identity = BillingIdentity.model_validate(completion.payload["billing_identity"])
        assert reserved.payload["billing_identity"] == identity.model_dump(mode="json")
        assert reconciled.payload["billing_identity"] == completion.payload["billing_identity"]
        assert reconciled.payload["actual_amount"] == "0.000054"
        assert next(iter(ledger._records.values())).billing_identity == completed_identity
        assert len([event for event in first if event.type == EventType.BUDGET_CHECKED]) == 1

    asyncio.run(run())


def test_compact_session_preserves_bedrock_identity_without_usage_metrics() -> None:
    async def run() -> None:
        identity = bedrock_billing_identity(
            invoked_model="global.anthropic.claude-sonnet-4-6",
            source_region="us-east-1",
            resource_type="inference_profile",
            profile_scope="global",
        )
        completed_identity = completed_bedrock_billing_identity(
            identity,
            effective_service_tier="default",
        )
        provider = ContextualBedrockCompactionProvider(identity, report_usage=False)
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="bedrock",
                    model=identity.resource_id,
                    match="exact",
                    input_per_million=Decimal("3"),
                    output_per_million=Decimal("15"),
                    pricing_context={
                        "source_region": ("us-east-1",),
                        "service_tier": ("default",),
                    },
                ),
            )
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model=identity.resource_id),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_contextual_compaction_without_usage",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="contextual-compaction-without-usage",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        first = [event async for event in app.compact_session(request)]
        replay = [event async for event in app.compact_session(request)]

        assert provider.calls == 1
        assert [event.id for event in replay] == [event.id for event in first]
        completion = next(event for event in first if event.type == EventType.MODEL_COMPLETED)
        assert completion.payload["billing_identity"] == completed_identity.model_dump(mode="json")
        assert "usage_metrics" not in completion.payload
        completion_sequence = public_event_sequence(completion.id)
        assert completion_sequence is not None
        stored_completion = next(
            record.event
            for record in await store.query_events(
                EventQuery(
                    session_id=created.id,
                    after_sequence=completion_sequence - 1,
                    limit=1,
                )
            )
            if record.sequence == completion_sequence
        )
        assert (
            stored_completion.payload["billing_identity"] == completion.payload["billing_identity"]
        )
        cost = await app.get_session_cost(created.id, pricing)
        assert cost.model_steps == 1
        assert cost.unpriced_model_steps == 1
        assert cost.line_items[0].billing_identity == completed_identity
        assert (
            cost.line_items[0].missing_pricing_reason
            == "model.completed event has no token usage metrics"
        )

    asyncio.run(run())


def test_compact_session_rejects_completion_billing_identity_rewrite() -> None:
    async def run() -> None:
        identity = bedrock_billing_identity(
            invoked_model="global.anthropic.claude-sonnet-4-6",
            source_region="us-east-1",
            resource_type="inference_profile",
            profile_scope="global",
        )

        class RewritingCompactionProvider(ContextualBedrockCompactionProvider):
            def billing_identity_for_completion(
                self,
                requested: BillingIdentity | None,
                payload: dict,
            ) -> BillingIdentity | None:
                assert requested == identity
                return identity.model_copy(
                    update={
                        "request_evidence": {
                            **identity.request_evidence,
                            "source_region": "eu-west-1",
                        }
                    }
                )

        provider = RewritingCompactionProvider(identity)
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model=identity.resource_id),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_rewritten_compaction_billing_identity",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(
            ContextBuildError,
            match="Model provider billing identity resolution failed",
        ) as exc_info:
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="rewritten-compaction-billing-identity",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert provider.calls == 1
        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        completion = events[3]
        assert completion.payload["billing_identity"] == identity.model_dump(mode="json")
        assert completion.payload["usage_metrics"]["input_tokens"] == 8
        assert completion.payload["usage_metrics"]["output_tokens"] == 2
        assert completion.payload["usage_metrics"]["total_tokens"] == 10
        assert completion.payload["compaction_outcome"] == "completion_observation_failed"
        assert "compaction_attempt_id" not in completion.payload
        failed = events[-1]
        assert failed.payload["error_type"] == type(exc_info.value).__name__
        assert "error" not in failed.payload
        assert exc_info.value.cause is not None
        assert "Model provider billing identity resolution failed" in str(exc_info.value)
        assert "Model provider billing identity resolution failed" in str(exc_info.value.cause)
        assert await store.load_transcript(created.id) == transcript
        completion_sequence = public_event_sequence(completion.id)
        assert completion_sequence is not None
        stored_completion = await store.query_events(
            EventQuery(
                session_id=created.id,
                after_sequence=completion_sequence - 1,
                limit=1,
            )
        )
        assert [record.sequence for record in stored_completion] == [completion_sequence]
        usage = await app.get_session_usage(created.id)
        assert usage.model_steps == 1
        assert usage.usage.input_tokens == 8
        assert usage.usage.output_tokens == 2
        assert usage.usage.total_tokens == 10

    asyncio.run(run())


def test_compact_session_rejects_unpriced_bedrock_identity_before_dispatch() -> None:
    async def run() -> None:
        identity = bedrock_billing_identity(
            invoked_model="global.anthropic.claude-sonnet-4-6",
            source_region="eu-west-1",
            resource_type="inference_profile",
            profile_scope="global",
        )
        provider = ContextualBedrockCompactionProvider(identity)
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="bedrock",
                    model=identity.resource_id,
                    match="exact",
                    input_per_million=Decimal("3"),
                    output_per_million=Decimal("15"),
                    pricing_context={
                        "source_region": ("us-east-1",),
                        "service_tier": ("default",),
                    },
                ),
            )
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("10"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model=identity.resource_id),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_unpriced_contextual_explicit_compaction",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events: list[Event] = []

        with pytest.raises(RuntimeError, match="budget limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="unpriced-contextual-explicit-compaction",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert provider.calls == 0
        assert EventType.BUDGET_RESERVED not in [event.type for event in events]
        assert EventType.BUDGET_LIMIT_REACHED in [event.type for event in events]
        assert events[-1].type == EventType.CONTEXT_COMPACTION_FAILED

    asyncio.run(run())


def test_compact_session_applies_app_policy_and_reserves_before_provider_work() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.00001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=8,
                            max_output_tokens=2,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_policy_reservation",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="reservation failed"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-policy-reservation",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVATION_FAILED,
            EventType.BUDGET_LIMIT_REACHED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert provider.calls == 0
        assert all(event.payload["operation_id"] for event in events[1:])
        profile_fingerprints = {
            event.payload.get("execution_profile_fingerprint") for event in events
        }
        assert None not in profile_fingerprints
        assert len(profile_fingerprints) == 1

    asyncio.run(run())


def test_compact_session_stops_on_an_exhausted_app_budget_before_provider_work() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.00001"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_exhausted_policy",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        await store.append_events(
            created.id,
            [
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=created.id,
                    agent_name="assistant",
                    payload={
                        "provider_name": "compaction-provider",
                        "model": "summary-model",
                        "usage_metrics": {
                            "input_tokens": 8,
                            "output_tokens": 2,
                            "total_tokens": 10,
                        },
                    },
                )
            ],
        )
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="budget limit reached"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-exhausted-policy",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_LIMIT_REACHED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert provider.calls == 0

    asyncio.run(run())


def test_compact_session_releases_partial_reservations_when_later_acquisition_fails() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        reservation = BudgetReservation(max_input_tokens=10, max_output_tokens=10)
        provider = UsageCompactionProvider()
        ledger = FailingSecondReservationBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_partial_reservation_failure",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="simulated reservation store failure"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-partial-reservation-failure",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    budget_limits=(
                        BudgetLimit(
                            scope="agent",
                            key="assistant",
                            max_estimated_cost=Decimal("0.001"),
                            pricing=pricing,
                            reservation=reservation,
                        ),
                    ),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RESERVATION_RELEASED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        reservation_sequence = public_event_sequence(events[2].id)
        assert reservation_sequence is not None
        reservation_record = next(
            record
            for record in await store.query_events(
                EventQuery(
                    session_id=created.id,
                    after_sequence=reservation_sequence - 1,
                    limit=1,
                )
            )
            if record.sequence == reservation_sequence
        )
        reservation_id = reservation_record.event.payload["reservation_id"]
        assert ledger.reserve_calls == 2
        assert ledger.release_calls == 1
        assert not await ledger.heartbeat(reservation_id=reservation_id)
        assert provider.calls == 0

    asyncio.run(run())


def test_compact_session_preserves_partial_cleanup_and_releases_remaining_reservations() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        reservation = BudgetReservation(max_input_tokens=10, max_output_tokens=10)
        provider = UsageCompactionProvider()
        ledger = FailingSecondReleaseBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                    BudgetLimit(
                        scope="agent",
                        key="assistant",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                    BudgetLimit(
                        scope="causal",
                        key="compact-partial-cleanup-causal",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_partial_cleanup_failure",
                causal_budget_id="compact-partial-cleanup-causal",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="simulated second release failure"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-partial-cleanup-failure",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    budget_limits=(
                        BudgetLimit(
                            scope="app",
                            max_estimated_cost=Decimal("0.000001"),
                            pricing=pricing,
                            reservation=reservation,
                        ),
                    ),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RESERVATION_FAILED,
            EventType.BUDGET_RESERVATION_RELEASED,
            EventType.BUDGET_RESERVATION_RELEASED,
            EventType.BUDGET_RESERVATION_RELEASED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert len(ledger.reservation_ids) == 3
        private_releases = await _private_events_for_public_events(
            store,
            created.id,
            events[8:11],
        )
        assert {event.payload["reservation_id"] for event in private_releases} == set(
            ledger.reservation_ids
        )
        assert ledger.release_calls == 4
        for reservation_id in ledger.reservation_ids:
            assert not await ledger.heartbeat(reservation_id=reservation_id)
        assert provider.calls == 0

    asyncio.run(run())


def test_compact_session_reconciles_reservations_and_replays_budget_events() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_policy_reconcile",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-policy-reconcile",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        first = [event async for event in app.compact_session(request)]
        replay = [event async for event in app.compact_session(request)]

        assert [event.type for event in first] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_RECONCILED,
            EventType.BUDGET_CHECKED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]
        assert [event.id for event in replay] == [event.id for event in first]
        assert provider.calls == 1
        reconciled = first[6]
        assert reconciled.payload["actual_amount"] == "0.000012"
        assert reconciled.payload["operation_id"]

    asyncio.run(run())


def test_compact_session_settles_usage_overflow_at_reserved_amount_once() -> None:
    class OverflowUsageCompactionProvider(ModelProvider):
        name = "compaction-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest):
            self.calls += 1
            yield ModelStreamEvent.text_delta("provider summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {
                        "input_tokens": MAX_DURABLE_JSON_INTEGER,
                        "output_tokens": MAX_DURABLE_JSON_INTEGER,
                    },
                }
            )

    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )
        provider = OverflowUsageCompactionProvider()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=60)
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_ledger=ledger,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_usage_overflow_reconcile",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-usage-overflow-reconcile",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )
        first: list[Event] = []
        with pytest.raises(ContextBuildError, match="integer_out_of_range"):
            async for event in app.compact_session(request):
                first.append(event)
        replay = [event async for event in app.compact_session(request)]

        events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        reservations = [event for event in events if event.type == EventType.BUDGET_RESERVED]
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        reconciliations = [event for event in events if event.type == EventType.BUDGET_RECONCILED]

        assert provider.calls == 1
        assert [event.id for event in replay] == [event.id for event in first]
        assert len(reservations) == 1
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics_rejected"] is True
        assert "usage" not in completions[0].payload
        assert "usage_metrics" not in completions[0].payload
        assert len(reconciliations) == 1
        assert reconciliations[0].payload["actual_amount"] == "0.00002"
        record = next(iter(ledger._records.values()))
        assert record.status == "reconciled"
        assert record.actual_amount == Decimal("0.00002")
        assert record.reason == (
            "compaction completed without priced usage; charged reserved amount"
        )
        assert not await ledger.heartbeat(reservation_id=reservations[0].payload["reservation_id"])

    asyncio.run(run())


def test_compact_session_reserves_prompt_cache_compactor_with_session_model() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="summary-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=PromptCacheCompactor(provider=provider),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_prompt_cache_reservation",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(
                provider_name="compaction-provider",
                model="summary-model",
            ),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-prompt-cache-reservation",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]

        assert EventType.BUDGET_RESERVED in [event.type for event in events]
        assert EventType.BUDGET_RECONCILED in [event.type for event in events]
        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert provider.calls == 1

    asyncio.run(run())


def test_compact_session_preserves_usage_when_final_reservation_renewal_fails() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        ledger = FinalRenewalFailureBudgetLedger()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_final_renewal",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="lease was lost"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-final-renewal",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_RECONCILED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert events[6].payload["actual_amount"] == "0.000012"
        assert provider.calls == 1
        assert ledger.heartbeat_calls == 2

    asyncio.run(run())


def test_compact_session_releases_reservation_when_initial_renewal_fails() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            budget_ledger=InitialRenewalFailureBudgetLedger(),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_initial_renewal",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="before provider dispatch"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-initial-renewal",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RESERVATION_RELEASED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert events[3].payload["actual_amount"] is None
        assert provider.calls == 0

    asyncio.run(run())


def test_compact_session_preserves_usage_returned_while_heartbeat_cancels() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        provider = CancellationCompletingProvider()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            budget_ledger=HeartbeatCancellationBudgetLedger(),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_heartbeat_cancellation",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        events = []

        with pytest.raises(RuntimeError, match="lease was lost"):
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-heartbeat-cancellation",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                events.append(event)

        assert [event.type for event in events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_RECONCILED,
            EventType.CONTEXT_COMPACTION_FAILED,
        ]
        assert events[6].payload["actual_amount"] == "0.000012"
        assert provider.calls == 1

    asyncio.run(run())


def test_compact_session_skips_provider_reservations_for_deterministic_compaction() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_deterministic_compaction_budget",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-deterministic-budget",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]

        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert not any(str(event.type).startswith("budget.") for event in events)

    asyncio.run(run())


def test_compact_session_rejects_unknown_compactor_budget_identity_before_work() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        compactor = UndeclaredProviderCompactor()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_unknown_compactor_budget_identity",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(RuntimeError, match="declare provider_budget_identity"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-unknown-budget-identity",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert compactor.calls == 0
        assert await store.load_checkpoint(created.id) is None

    asyncio.run(run())


def test_compact_session_rejects_provider_backed_deterministic_identity_claim() -> None:
    class NeverDispatchedProvider(ModelProvider):
        name = "compaction-provider"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest):
            self.requests.append(request)
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    class MisdeclaredModelCompactor(ModelCompactor):
        def provider_budget_identity(self, session) -> None:
            del session
            return None

    async def run() -> None:
        store = InMemorySessionStore()
        provider = NeverDispatchedProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=MisdeclaredModelCompactor(
                    provider=provider,
                    model="summary-model",
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_misdeclared_compactor_budget_identity",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(
            RuntimeError,
            match="cannot declare a deterministic budget identity",
        ):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-misdeclared-budget-identity",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                    limits=RunLimits(max_total_tokens=10),
                )
            ):
                pass

        assert provider.requests == []
        assert await store.load_checkpoint(created.id) is None

    asyncio.run(run())


def test_compact_session_sanitizes_failure_type_in_event_and_operation_record() -> None:
    unsafe_error_type = type("Unsafe\n" * 100, (BaseException,), {})

    class UnsafeFailureCompactor(ContextCompactor):
        def provider_budget_identity(self, session) -> None:
            del session
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            del request
            raise unsafe_error_type("compaction failed")

    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=UnsafeFailureCompactor(),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_explicit_unsafe_error_type",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-unsafe-error-type",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        events: list[Event] = []
        with pytest.raises(unsafe_error_type):
            async for event in app.compact_session(request):
                events.append(event)

        assert [event.type for event in events] == [EventType.CONTEXT_COMPACTION_STARTED]
        events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        failed = next(
            event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED
        )
        assert failed.payload["error_type"] == "BaseException"
        operation = await store.load_session_operation(created.id, request.idempotency_key)
        assert operation is not None
        assert operation["error_type"] == "BaseException"
        assert "\n" not in json.dumps(failed.model_dump(mode="json"))

    asyncio.run(run())


@pytest.mark.parametrize(
    "identity",
    [
        ("compaction-provider\x00suffix", "summary-model"),
        ("compaction-provider", "summary\nmodel"),
        ("compaction-provider", "summary-\ud800-model"),
        ("compaction-provider", "m" * 513),
    ],
)
def test_compact_session_rejects_unsafe_compactor_budget_identity_before_work(
    identity: tuple[str, str],
) -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        compactor = UnsafeProviderIdentityCompactor(identity)
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_unsafe_compactor_budget_identity",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="without control characters"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-unsafe-budget-identity",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert compactor.calls == 0
        assert await store.load_checkpoint(created.id) is None

    asyncio.run(run())


def test_compact_session_persists_reservation_lifecycle_before_provider_work() -> None:
    async def run() -> None:
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        store = InMemorySessionStore()
        provider = ReservationInspectingCompactionProvider(store)
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=10,
                        ),
                    ),
                )
            ),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_persisted_reservation",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        provider.session_id = created.id
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        stream = app.compact_session(
            CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-persisted-reservation",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
        )

        started = await anext(stream)
        checked = await anext(stream)
        reserved = await anext(stream)
        assert [started.type, checked.type, reserved.type] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
        ]
        assert provider.calls == 1
        assert provider.durable_types_before_dispatch == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
        ]
        durable_records_before_close = await store.query_events(
            EventQuery(session_id=created.id, limit=100)
        )
        durable_before_close = [record.event for record in durable_records_before_close]
        assert [
            public_event_id(record.sequence) for record in durable_records_before_close[:3]
        ] == [
            started.id,
            checked.id,
            reserved.id,
        ]
        assert [event.type for event in durable_before_close] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.BUDGET_RECONCILED,
            EventType.BUDGET_CHECKED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]

        await stream.aclose()

        durable_after_close = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        assert [event.id for event in durable_after_close] == [
            event.id for event in durable_before_close
        ]
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        record = await store.load_session_operation(
            created.id,
            "compact-persisted-reservation",
        )
        assert record is not None
        assert record["event_ids"] == [event.id for event in durable_after_close]
        assert record["status"] == "completed"

    asyncio.run(run())
