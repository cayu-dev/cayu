"""Explicit compaction publication faults, restart recovery and exact replay."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from tests.core._execution_profile_fixtures import (
    profiled_session_identity,
    versioned_test_provider_identity,
)
from tests.core._explicit_compaction_fixtures import (
    CompletingProvider,
    RecordingCompactor,
    _create_profiled_session,
    _public_ids_for_private_event_ids,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ResolutionActor
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation, InMemoryBudgetLedger
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextBuildError,
    ContextCompactor,
    ModelCompactor,
)
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.observability.events import EventSink, InMemoryEventSink
from cayu.providers.base import ModelProvider, ModelProviderError, ModelRequest, ModelStreamEvent
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime._compaction import explicit as _session_compaction
from cayu.runtime._event_projection import public_event_id, public_event_sequence
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import CompactSessionRequest, ResumeRequest, RunRequest
from cayu.storage.sqlite import SQLiteSessionStore


class NoFullHistoryReplayStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    async def load_events(self, session_id: str):
        raise AssertionError("compaction replay must use indexed event lookup")


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

        first = [event async for event in app.compact_session(request)]
        assert first[-1].type == EventType.SESSION_CHECKPOINTED

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
        assert [event.id for event in first] == [event.id for event in replay]
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
        await asyncio.wait_for(reconciliation_started.wait(), timeout=10)
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
            _session_compaction,
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
        # Startup includes profile/budget preparation; it is not the bounded
        # cancellation behavior measured after this publication barrier.
        await asyncio.wait_for(store.blocked.wait(), timeout=5)
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
            _session_compaction,
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
            _session_compaction,
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
        blocked = asyncio.create_task(store.blocked.wait())
        try:
            # This is a harness deadline, not the 10ms store deadline under test.
            # Surface an early operation failure instead of masking it as a
            # barrier timeout, and allow loaded CI workers time to reach it.
            done, _ = await asyncio.wait(
                {blocked, task}, timeout=10, return_when=asyncio.FIRST_COMPLETED
            )
            if task in done and not store.blocked.is_set():
                await task
            assert store.blocked.is_set(), "completion publication did not reach its barrier"
            with pytest.raises(ContextBuildError, match="bounded store wait"):
                await asyncio.wait_for(task, timeout=10)
        finally:
            # Even a failed assertion must release the cancellation-resistant
            # physical write so teardown cannot strand its owner.
            store.release.set()
            blocked.cancel()
            if not task.done():
                task.cancel()
            await asyncio.wait_for(asyncio.gather(task, blocked, return_exceptions=True), 10)
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
    from cayu.approvals.tools import PendingToolApproval, PendingToolCallApproval

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
                            "request_digest": _session_compaction._compact_session_request_digest(
                                request
                            ),
                            "event_ids": [event.id for event in replay_events],
                        }
                    },
                },
                pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY: pending.model_dump(
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
        transform = _session_compaction._fail_session_operation_checkpoint(
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
