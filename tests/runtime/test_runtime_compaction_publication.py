"""Compaction publication contracts."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from tests.core._execution_profile_fixtures import (
    interrupt_and_release_test_invocation,
    profiled_session_identity,
    run_request_with_registered_tool_ceiling,
)
from tests.core._session_operation_fault_harness import (
    CommitEvidence,
    FailBeforeTransform,
    MatchPolicy,
    PublicationBoundary,
    PublicationFaultActionKind,
    PublicationFaultOutcome,
    SessionOperationFaultHarness,
    SessionOperationFaultRule,
    SessionOperationSelector,
)
from tests.runtime.runtime_compaction_support import (
    _AUTOMATIC_COMPACTION_PROCESS_EXIT_CODE,
    ProcessLossAutomaticCompactionStore,
    RecordingCompactor,
    compaction_price_book,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    checkpoint_without_active_invocation_profile,
    collect_events,
    collect_resume_events,
)
from tests.runtime.runtime_session_support import (
    request_footprint_session_store as request_footprint_session_store,
)

import cayu
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from cayu.context.footprints import RequestFootprintConfig
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.observability.events import InMemoryEventSink
from cayu.providers.base import ModelProvider, ModelProviderError, ModelRequest, ModelStreamEvent
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime._compaction import automatic as _automatic_compaction
from cayu.runtime._event_projection import public_event_id
from cayu.runtime._model_completion_contracts import ModelCompletionManualRecoveryRequired
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus


@pytest.mark.parametrize(
    ("publication_type", "expected_compactor_calls"),
    [
        (EventType.CONTEXT_COMPACTION_STARTED, 0),
        (EventType.MODEL_COMPLETED, 1),
    ],
)
@pytest.mark.parametrize("release_reconciliation", [True, False])
def test_automatic_compaction_cancellation_during_publication_reconciliation_propagates(
    publication_type: EventType,
    expected_compactor_calls: int,
    release_reconciliation: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCompactionProvider(ModelProvider):
        name = "reconciliation-cancellation-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.calls += 1
            yield ModelStreamEvent.text_delta("partial")
            raise ModelProviderError(
                "provider failed",
                provider=self.name,
                retryable=True,
            )

    async def run() -> None:
        store = InMemorySessionStore()
        compaction_provider = FailingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                    retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        reconciliation_started = asyncio.Event()
        allow_reconciliation = asyncio.Event()
        original_emit_many = app._event_writer.emit_many
        original_complete_stage = store.complete_model_completion_stage
        original_is_persisted = app._event_writer.is_persisted

        async def fail_start_publication(session_id: str, events: list[Event]):
            if any(event.type == EventType.CONTEXT_COMPACTION_STARTED for event in events):
                raise RuntimeError("publication acknowledgement failed")
            return await original_emit_many(session_id, events)

        async def fail_completion_persistence(session_id: str, **kwargs):
            result = await original_complete_stage(session_id, **kwargs)
            if result.stage.purpose == "context-compaction":
                raise RuntimeError("publication acknowledgement failed")
            return result

        async def block_target_reconciliation(event: Event) -> bool:
            if event.type == publication_type:
                reconciliation_started.set()
                await allow_reconciliation.wait()
                return False
            return await original_is_persisted(event)

        if publication_type == EventType.CONTEXT_COMPACTION_STARTED:
            monkeypatch.setattr(app._event_writer, "emit_many", fail_start_publication)
        else:
            monkeypatch.setattr(
                store, "complete_model_completion_stage", fail_completion_persistence
            )
        monkeypatch.setattr(app._event_writer, "is_persisted", block_target_reconciliation)
        if not release_reconciliation:
            # Bound the blocked read after cancellation without shortening
            # ordinary publication or reconciliation during setup.
            monkeypatch.setattr(
                _automatic_compaction,
                "_CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S",
                0.01,
            )
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=f"sess_reconciliation_cancel_{publication_type.value}",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        await asyncio.wait_for(reconciliation_started.wait(), timeout=10)
        task.cancel("cancel during compaction publication reconciliation")
        assert task.cancelling() == 1
        if release_reconciliation:
            allow_reconciliation.set()
        # Observe completion without injecting a second cancellation during
        # termination telemetry. The short reconciliation deadline above is
        # the behavior under test; this guard allows loaded-runner cleanup.
        done, _pending = await asyncio.wait({task}, timeout=30)
        assert task in done
        with pytest.raises(
            asyncio.CancelledError,
            match="cancel during compaction publication reconciliation",
        ):
            await task

        assert task.cancelled()
        assert compaction_provider.calls == expected_compactor_calls

    asyncio.run(run())


@pytest.mark.parametrize("publication_delay", [0, 0.05])
def test_automatic_compaction_cancellation_does_not_wait_for_stalled_completion_sink(
    monkeypatch: pytest.MonkeyPatch,
    publication_delay: float,
) -> None:
    class CompletionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def complete_model_completion_stage(self, session_id: str, **kwargs):
            # Exercise publication slower than the old 10 ms setup deadline.
            await asyncio.sleep(publication_delay)
            return await super().complete_model_completion_stage(session_id, **kwargs)

    class BlockingCompletionSink(InMemoryEventSink):
        def __init__(self) -> None:
            super().__init__()
            self.blocked = asyncio.Event()
            self.release = asyncio.Event()
            self.cancellation_observed = asyncio.Event()
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
                            # Model a sink whose in-flight delivery cannot be
                            # interrupted even though Cayu's caller must be.
                            self.cancellation_observed.set()
            await super().emit(event)

    class CompletionProvider(ModelProvider):
        name = "automatic-stalled-completion-sink-provider"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        # Shorten only cancellation draining, not ordinary durable publication.
        monkeypatch.setattr(
            _automatic_compaction,
            "_CONTEXT_EVENT_STORE_WAIT_AFTER_CANCELLATION_TIMEOUT_S",
            0.01,
        )
        store = CompletionStore()
        sink = BlockingCompletionSink()
        app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=CompletionProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_automatic_stalled_completion_sink",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        try:
            # Reach the cancellation boundary using the normal publication timeout.
            # A 10 ms timeout during setup can abort compaction on a busy CI worker.
            await asyncio.wait_for(sink.blocked.wait(), timeout=10)
            durable = await store.load_events("sess_automatic_stalled_completion_sink")
            completions = [
                event
                for event in durable
                if event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
            ]
            assert len(completions) == 1

            task.cancel("cancel automatic compaction with stalled completion sink")
            assert task.cancelling() == 1
            await asyncio.wait_for(sink.cancellation_observed.wait(), timeout=1)
            done, _pending = await asyncio.wait({task}, timeout=1)
            # Prove caller cancellation completes before the non-cancellable sink.
            assert task in done
            with pytest.raises(
                asyncio.CancelledError,
                match="cancel automatic compaction with stalled completion sink",
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
                if delivered:
                    break
                await asyncio.sleep(0.01)
            completion_record = next(
                record
                for record in await store.query_events(
                    EventQuery(session_id="sess_automatic_stalled_completion_sink")
                )
                if record.event.id == completions[0].id
            )
            assert [event.id for event in delivered] == [
                public_event_id(completion_record.sequence)
            ]
            assert sink.completion_delivery_calls == 1
        finally:
            sink.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_automatic_compaction_late_completion_commit_keeps_one_accounting_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DelayedCompletionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.blocked = asyncio.Event()
            self.release = asyncio.Event()
            self.blocked_once = False
            self.cancellations = 0

        async def complete_model_completion_stage(self, session_id: str, **kwargs):
            publication = kwargs["publication"]
            if not self.blocked_once and any(
                event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
                for event in publication.events
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
            return await super().complete_model_completion_stage(session_id, **kwargs)

    class CompletionProvider(ModelProvider):
        name = "automatic-late-completion-provider"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        monkeypatch.setattr(
            _automatic_compaction,
            "_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S",
            0.01,
        )
        store = DelayedCompletionStore()
        sink = InMemoryEventSink()
        compaction_provider = CompletionProvider()
        app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_automatic_late_completion_commit",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        try:
            await asyncio.wait_for(store.blocked.wait(), timeout=5)
            await asyncio.sleep(0.05)
            assert not task.done()
        finally:
            store.release.set()
        events = await asyncio.wait_for(task, timeout=5)
        assert events[-1].type == EventType.SESSION_FAILED
        for _ in range(100):
            durable = await store.load_events("sess_automatic_late_completion_commit")
            completions = [
                event
                for event in durable
                if event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
            ]
            if completions:
                break
            await asyncio.sleep(0.01)

        assert len(completions) == 1
        sink_completions = [
            event
            for event in sink.events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        completion_records = [
            record
            for record in await store.query_events(
                EventQuery(session_id="sess_automatic_late_completion_commit")
            )
            if record.event.id == completions[0].id
        ]
        assert [event.id for event in sink_completions] == [
            public_event_id(completion_records[0].sequence)
        ]

    asyncio.run(run())


def test_automatic_compaction_late_start_commit_reuses_original_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DelayedStartStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.blocked = asyncio.Event()
            self.release = asyncio.Event()
            self.committed = asyncio.Event()
            self.blocked_once = False
            self.original_event_id: str | None = None
            self.cancellations = 0

        async def append_events(self, session_id: str, events: list[Event]) -> None:
            start = next(
                (event for event in events if event.type == EventType.CONTEXT_COMPACTION_STARTED),
                None,
            )
            if not self.blocked_once and start is not None:
                self.blocked_once = True
                self.original_event_id = start.id
                self.blocked.set()
                while not self.release.is_set():
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        # Match a physical store write that cannot be cancelled.
                        self.cancellations += 1
            await super().append_events(session_id, events)
            if start is not None and start.id == self.original_event_id:
                self.committed.set()

    class CountingCompactionProvider(ModelProvider):
        name = "automatic-late-start-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed({"model": request.model})

    async def run() -> None:
        monkeypatch.setattr(
            _automatic_compaction,
            "_AUTOMATIC_COMPACTION_PREDISPATCH_EVENT_STORE_WAIT_TIMEOUT_S",
            0.01,
        )
        store = DelayedStartStore()
        sink = InMemoryEventSink()
        compaction_provider = CountingCompactionProvider()
        app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        original_is_persisted = app._event_writer.is_persisted
        stale_start_checks = 0

        async def release_after_stale_start_check(event: Event) -> bool:
            nonlocal stale_start_checks
            persisted = await original_is_persisted(event)
            if (
                stale_start_checks < 1
                and event.type == EventType.CONTEXT_COMPACTION_STARTED
                and not persisted
            ):
                stale_start_checks += 1
                store.release.set()
                await store.committed.wait()
                # The reconciliation read is stale; the original physical
                # write commits immediately before the exact replay.
                return False
            return persisted

        app._event_writer.is_persisted = release_after_stale_start_check
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_automatic_late_start_commit",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )

        assert events[-1].type == EventType.SESSION_COMPLETED
        assert stale_start_checks == 1
        assert store.committed.is_set()
        assert store.cancellations >= 1
        assert compaction_provider.calls == 1
        durable = await store.load_events("sess_automatic_late_start_commit")
        starts = [event for event in durable if event.type == EventType.CONTEXT_COMPACTION_STARTED]
        assert [event.id for event in starts] == [store.original_event_id]
        assert EventType.CONTEXT_COMPACTION_FAILED not in {event.type for event in durable}
        sink_starts = [
            event for event in sink.events if event.type == EventType.CONTEXT_COMPACTION_STARTED
        ]
        start_records = [
            record
            for record in await store.query_events(
                EventQuery(session_id="sess_automatic_late_start_commit")
            )
            if record.event.id == store.original_event_id
        ]
        assert [event.id for event in sink_starts] == [public_event_id(start_records[0].sequence)]

    asyncio.run(run())


def test_automatic_compaction_retries_start_publication_with_causal_event() -> None:
    class CountingCompactionProvider(ModelProvider):
        name = "automatic-failed-start-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed({"model": request.model})

    async def run() -> None:
        store = InMemorySessionStore()
        compaction_provider = CountingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        original_emit_many = app._event_writer.emit_many
        rejected_start_id: str | None = None

        async def reject_first_start(session_id: str, events: list[Event]) -> list[Event]:
            nonlocal rejected_start_id
            start = next(
                (event for event in events if event.type == EventType.CONTEXT_COMPACTION_STARTED),
                None,
            )
            if rejected_start_id is None and start is not None:
                rejected_start_id = start.id
                raise ConnectionError("start publication failed before commit")
            return await original_emit_many(session_id, events)

        app._event_writer.emit_many = reject_first_start
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_automatic_failed_start_publication",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )

        assert events[-1].type == EventType.SESSION_COMPLETED
        assert compaction_provider.calls == 1
        durable = await store.load_events("sess_automatic_failed_start_publication")
        starts = [event for event in durable if event.type == EventType.CONTEXT_COMPACTION_STARTED]
        failures = [event for event in durable if event.type == EventType.CONTEXT_COMPACTION_FAILED]
        assert [event.id for event in starts] == [rejected_start_id]
        assert failures == []

    asyncio.run(run())


def test_automatic_compaction_exhausted_start_publication_is_typed_and_resumable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingCompactionProvider(ModelProvider):
        name = "automatic-exhausted-start-provider"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.calls += 1
            yield ModelStreamEvent.completed({})

    async def run() -> None:
        session_id = "sess_automatic_exhausted_start_publication"
        store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        compaction_provider = CountingCompactionProvider()
        await task_store.create_task(
            TaskCreate(
                task_id="task_automatic_exhausted_start_publication",
                type="compaction-recovery",
                assigned_agent_name="assistant",
            )
        )
        app = CayuApp(
            session_store=store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        attempts = 0
        original_emit_many = app._event_writer.emit_many

        async def reject_start(session_id: str, events: list[Event]) -> list[Event]:
            nonlocal attempts
            if all(event.type == EventType.CONTEXT_COMPACTION_STARTED for event in events):
                attempts += 1
                raise TimeoutError("start publication timed out before commit")
            return await original_emit_many(session_id, events)

        monkeypatch.setattr(
            _automatic_compaction,
            "_AUTOMATIC_COMPACTION_PREDISPATCH_PUBLICATION_ATTEMPTS",
            2,
        )
        monkeypatch.setattr(app._event_writer, "emit_many", reject_start)
        returned = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                task_id="task_automatic_exhausted_start_publication",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )

        assert attempts == 2
        assert compaction_provider.calls == 0
        failure = next(
            event for event in returned if event.type == EventType.CONTEXT_COMPACTION_FAILED
        )
        expected = {
            "phase": "start_publication",
            "reason": "publication_timeout",
            "retryable": True,
            "provider_dispatch_disposition": "not_dispatched",
            "recovery_action": "resume_session",
        }
        assert {key: failure.payload[key] for key in expected} == expected
        assert type(failure.payload["elapsed_ms"]) is int
        terminal = returned[-1]
        assert terminal.type == EventType.SESSION_FAILED
        assert terminal.payload["compaction_failure"] == {
            **expected,
            "elapsed_ms": failure.payload["elapsed_ms"],
        }
        task = await task_store.load_task("task_automatic_exhausted_start_publication")
        assert task is not None
        assert task.status == TaskStatus.FAILED
        assert task.error is not None
        assert task.error["compaction_failure"] == terminal.payload["compaction_failure"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "publication_type",
    [EventType.REQUEST_FOOTPRINT_RECORDED, EventType.MODEL_STARTED],
)
def test_automatic_compaction_retries_predispatch_publication_without_duplicate_provider_call(
    publication_type: EventType,
    request_footprint_session_store,
) -> None:
    class CompactionProvider(ModelProvider):
        name = "predispatch-publication-compactor"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed({"model": request.model})

    async def run() -> None:
        session_id = f"sess_automatic_{publication_type.value.replace('.', '_')}"
        compaction_provider = CompactionProvider()
        app = CayuApp(
            session_store=request_footprint_session_store,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="compaction-publication-key",
                fingerprint_key=SecretStr("p" * 32),
            ),
            enable_logging=False,
        )
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        rule = SessionOperationFaultRule(
            rule_id="compaction-predispatch",
            selector=SessionOperationSelector(
                session_id=session_id,
                event_types=frozenset({publication_type}),
            ),
            actions=(FailBeforeTransform(),),
            on_exhausted=MatchPolicy.DELEGATE,
        )
        async with SessionOperationFaultHarness(
            request_footprint_session_store,
            rules=(rule,),
            boundary=PublicationBoundary.EVENT_APPEND,
        ) as faults:
            returned = await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )

        assert returned[-1].type == EventType.SESSION_COMPLETED
        assert compaction_provider.calls == 1
        injected = [
            trace for trace in faults.trace if trace.matched_rule_id == "compaction-predispatch"
        ]
        assert len(injected) >= 2
        assert injected[0].action is PublicationFaultActionKind.FAIL_BEFORE_TRANSFORM
        assert injected[0].outcome is PublicationFaultOutcome.INJECTED_FAILURE
        assert injected[0].committed is CommitEvidence.NO
        assert injected[1].action is PublicationFaultActionKind.EXHAUSTED_DELEGATE
        assert injected[1].outcome is PublicationFaultOutcome.RETURNED
        assert injected[1].committed is CommitEvidence.YES
        durable = await request_footprint_session_store.load_events(session_id)
        target_events = [event for event in durable if event.type == publication_type]
        if publication_type == EventType.REQUEST_FOOTPRINT_RECORDED:
            target_events = [
                event
                for event in target_events
                if event.payload.get("request_variant") == "context_compaction"
            ]
        else:
            target_events = [
                event
                for event in target_events
                if event.payload.get("purpose") == "context_compaction"
            ]
        assert len(target_events) == 1

    asyncio.run(run())


def test_automatic_hierarchy_exhausted_next_start_requires_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingHierarchyProvider(ModelProvider):
        name = "automatic-hierarchy-publication-provider"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.text_delta(f"summary-{len(self.requests)}")
            yield ModelStreamEvent.completed({"model": request.model})

    async def run() -> None:
        session_id = "sess_automatic_hierarchy_next_start_failure"
        store = InMemorySessionStore()
        compaction_provider = CountingHierarchyProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compaction_provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        original_persist_exact_replay = app._event_writer.persist_exact_replay
        model_start_publications = 0

        async def fail_second_compactor_start(event: Event) -> Event:
            nonlocal model_start_publications
            if (
                event.type == EventType.MODEL_STARTED
                and event.payload.get("purpose") == "context_compaction"
            ):
                model_start_publications += 1
                if model_start_publications >= 2:
                    raise TimeoutError("next hierarchy model-start publication unavailable")
            return await original_persist_exact_replay(event)

        monkeypatch.setattr(
            _automatic_compaction,
            "_AUTOMATIC_COMPACTION_PREDISPATCH_PUBLICATION_ATTEMPTS",
            3,
        )
        monkeypatch.setattr(
            app._event_writer,
            "persist_exact_replay",
            fail_second_compactor_start,
        )
        returned = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "oversized " + "x" * 5000),
                    Message.text("user", "current"),
                ],
            ),
        )
        durable = await store.load_events(session_id)

        assert returned[-1].type == EventType.SESSION_FAILED
        assert len(compaction_provider.requests) == 1
        assert (
            sum(
                event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
                for event in durable
            )
            == 1
        )
        failure = next(
            event for event in durable if event.type == EventType.CONTEXT_COMPACTION_FAILED
        )
        expected = {
            "phase": "model_start_publication",
            "reason": "publication_timeout",
            "retryable": False,
            "provider_dispatch_disposition": "dispatched",
            "recovery_action": "reconcile_completion",
        }
        assert {key: failure.payload[key] for key in expected} == expected
        assert {
            key: returned[-1].payload["compaction_failure"][key] for key in expected
        } == expected

    asyncio.run(run())


@pytest.mark.parametrize("failure_point", ["before_commit", "after_commit"])
def test_automatic_compaction_checkpoint_and_evidence_publish_atomically(
    failure_point: str,
) -> None:
    class FailingAtomicCheckpointStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def publish_checkpoint_and_events(self, session_id: str, **kwargs):
            events = kwargs["events"]
            is_compaction_checkpoint = any(
                event.type == EventType.SESSION_CHECKPOINTED for event in events
            )
            if not is_compaction_checkpoint or self.failed:
                return await super().publish_checkpoint_and_events(session_id, **kwargs)
            self.failed = True
            if failure_point == "before_commit":
                raise ConnectionError("compaction checkpoint rejected before commit")
            await super().publish_checkpoint_and_events(session_id, **kwargs)
            raise ConnectionError("compaction checkpoint acknowledgement lost")

    async def run() -> None:
        session_id = f"sess_atomic_compaction_checkpoint_{failure_point}"
        store = FailingAtomicCheckpointStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=FakeProvider(
                        [
                            ModelStreamEvent.text_delta("summary"),
                            ModelStreamEvent.completed(
                                {
                                    "model": "summary-model",
                                    "usage": {"input_tokens": 5, "output_tokens": 2},
                                }
                            ),
                        ]
                    ),
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        returned = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
        durable = await store.load_events(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        completed_count = sum(
            event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in durable
        )
        checkpointed_count = sum(event.type == EventType.SESSION_CHECKPOINTED for event in durable)

        if failure_point == "after_commit":
            assert returned[-1].type == EventType.SESSION_COMPLETED
            assert completed_count == 1
            assert checkpointed_count == 1
            assert checkpoint is not None
            assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2
        else:
            assert returned[-1].type == EventType.SESSION_FAILED
            assert completed_count == 0
            assert checkpointed_count == 0
            assert checkpoint is None or "context_compaction" not in checkpoint

        # Provider usage is committed before the atomic checkpoint publication
        # attempt and is never duplicated by either failure branch.
        completions = [
            event
            for event in durable
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1

    asyncio.run(run())


def test_automatic_compaction_lost_checkpoint_ack_reconciles_effective_transform() -> None:
    class RuntimeStateChangingLostAckStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def publish_checkpoint_and_events(self, session_id: str, **kwargs):
            events = kwargs["events"]
            is_compaction_checkpoint = any(
                event.type == EventType.SESSION_CHECKPOINTED for event in events
            )
            if not is_compaction_checkpoint or self.failed:
                return await super().publish_checkpoint_and_events(session_id, **kwargs)
            self.failed = True

            def clear_runtime_owned_state(
                _session: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                updated = {} if checkpoint is None else dict(checkpoint)
                updated.pop("session_operations", None)
                return updated

            # Simulate a concurrent runtime-owned checkpoint update after the
            # context build read its source checkpoint but before the atomic
            # compaction publication applies its preserving transform.
            await super().transform_checkpoint(session_id, clear_runtime_owned_state)
            await super().publish_checkpoint_and_events(session_id, **kwargs)
            raise ConnectionError("compaction checkpoint acknowledgement lost")

    async def run() -> None:
        session_id = "sess_compaction_effective_transform_reconciliation"
        store = RuntimeStateChangingLostAckStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            FakeProvider(
                [
                    ModelStreamEvent.text_delta("final answer"),
                    ModelStreamEvent.completed({}),
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        transcript = [
            Message.text("user", "old"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(session_id, transcript)

        def seed_runtime_owned_state(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            updated = {} if checkpoint is None else dict(checkpoint)
            updated["session_operations"] = {
                "version": 1,
                "active_operation_id": None,
                "records": {},
            }
            return updated

        await store.transform_checkpoint(session_id, seed_runtime_owned_state)
        await store.update_status(session_id, SessionStatus.COMPLETED)

        events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "next")],
            ),
        )

        assert events[-1].type == EventType.SESSION_COMPLETED
        assert sum(event.type == EventType.SESSION_CHECKPOINTED for event in events) == 1
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "session_operations" not in checkpoint
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] > 0

    asyncio.run(run())


def test_automatic_compaction_internal_persistence_cancellation_is_not_caller_cancel() -> None:
    class InternallyCancelledCheckpointStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def publish_checkpoint_and_events(self, session_id: str, **kwargs):
            del session_id, kwargs
            raise asyncio.CancelledError("storage pool cancelled its write")

    async def run() -> None:
        session_id = "sess_compaction_internal_persistence_cancel"
        store = InternallyCancelledCheckpointStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        events = await task

        assert not task.cancelled()
        assert events[-1].type == EventType.SESSION_FAILED
        assert events[-1].payload["error_type"] == "RuntimeError"
        assert events[-1].payload["error"] == (
            "Context outcome persistence was cancelled without caller cancellation."
        )
        assert task.exception() is None
        assert checkpoint_without_active_invocation_profile(
            await store.load_checkpoint(session_id)
        ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}

    asyncio.run(run())


def test_automatic_compaction_lost_completion_ack_fails_closed_without_retry() -> None:
    class LostCompletionAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def promote_model_completion_stage(
            self,
            session_id: str,
            *,
            stage_id: str,
            expected_run_epoch: int,
        ):
            result = await super().promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=expected_run_epoch,
            )
            if not self.failed:
                self.failed = True
                raise ConnectionError("compaction completion acknowledgement lost")
            return result

    class RetryableFailureProvider(ModelProvider):
        name = "retryable-compaction"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise ModelProviderError(
                "provider overloaded",
                provider=self.name,
                status_code=503,
                retryable=True,
            )
            yield

    async def run() -> None:
        session_id = "sess_compaction_completion_lost_ack"
        store = LostCompletionAcknowledgementStore()
        provider = RetryableFailureProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )

        returned = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
        durable = await store.load_events(session_id)

        assert len(provider.requests) == 1
        completions = [
            event
            for event in durable
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["compaction_outcome"] == "provider_error"
        failures = [event for event in durable if event.type == EventType.CONTEXT_COMPACTION_FAILED]
        assert len(failures) == 1
        expected = {
            "phase": "completion_publication",
            "reason": "publication_failed",
            "retryable": False,
            "provider_dispatch_disposition": "unknown",
            "recovery_action": "reconcile_completion",
        }
        assert {key: failures[0].payload[key] for key in expected} == expected
        assert returned[-1].type == EventType.SESSION_FAILED
        assert {
            key: returned[-1].payload["compaction_failure"][key] for key in expected
        } == expected

    asyncio.run(run())


_AUTOMATIC_COMPACTION_PROCESS_SESSION_ID = "automatic-compaction-process-loss"


class ProcessLossSuccessfulCompactionProvider(ModelProvider):
    name = "fake"

    def __init__(self, marker: str) -> None:
        self._marker = Path(marker)

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="test.process-loss-successful-compaction-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        prior = int(self._marker.read_text(encoding="utf-8")) if self._marker.exists() else 0
        self._marker.write_text(str(prior + 1), encoding="utf-8")
        yield ModelStreamEvent.text_delta("durable summary")
        yield ModelStreamEvent.completed(
            {
                "model": request.model,
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }
        )


def _automatic_compaction_process_budget_policy() -> BudgetPolicy:
    return BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("1"),
                pricing=compaction_price_book(),
                reservation=BudgetReservation(
                    max_input_tokens=10,
                    max_output_tokens=10,
                ),
            ),
        )
    )


def _run_until_automatic_compaction_process_exit(
    database: str,
    ledger_database: str,
    marker: str,
    crash_point: str,
) -> None:
    async def run() -> None:
        store = ProcessLossAutomaticCompactionStore(
            database,
            crash_point=crash_point,
        )
        provider = ProcessLossSuccessfulCompactionProvider(marker)
        ledger = SQLiteBudgetLedger(ledger_database)
        app = CayuApp(
            session_store=store,
            budget_ledger=ledger,
            budget_policy=_automatic_compaction_process_budget_policy(),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=_AUTOMATIC_COMPACTION_PROCESS_SESSION_ID,
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            )
        ):
            pass
        raise AssertionError("the child did not reach the compaction crash point")

    asyncio.run(run())


@pytest.mark.parametrize("crash_point", ["completed", "promoted"])
def test_successful_automatic_compaction_process_loss_cannot_redispatch(
    tmp_path: Path,
    crash_point: str,
) -> None:
    database = tmp_path / f"automatic-compaction-{crash_point}.sqlite"
    ledger_database = tmp_path / f"automatic-compaction-{crash_point}-budget.sqlite"
    marker = tmp_path / f"automatic-compaction-{crash_point}-provider-count"
    repository_root = Path(__file__).resolve().parents[2]
    child_script = (
        "from tests.runtime.test_runtime_compaction_publication import "
        "_run_until_automatic_compaction_process_exit as run; "
        f"run({str(database)!r}, {str(ledger_database)!r}, "
        f"{str(marker)!r}, {crash_point!r})"
    )
    child_environment = os.environ.copy()
    existing_python_path = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            str(Path(cayu.__file__).resolve().parent.parent),
            existing_python_path,
        )
        if path
    )
    child = subprocess.run(
        [sys.executable, "-c", child_script],
        cwd=repository_root,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert child.returncode == _AUTOMATIC_COMPACTION_PROCESS_EXIT_CODE, child.stderr
    assert marker.read_text(encoding="utf-8") == "1"

    store = SQLiteSessionStore(database)
    active = asyncio.run(
        store.load_active_model_completion_stage(_AUTOMATIC_COMPACTION_PROCESS_SESSION_ID)
    )
    durable_events = asyncio.run(store.load_events(_AUTOMATIC_COMPACTION_PROCESS_SESSION_ID))
    completions = [
        event
        for event in durable_events
        if event.type is EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert not any(event.type is EventType.SESSION_CHECKPOINTED for event in durable_events)
    if crash_point == "completed":
        assert active is not None
        assert active.stage.state == "completed"
        assert completions == []
    else:
        assert active is None
        assert len(completions) == 1
        assert type(completions[0].payload.get("parent_model_step_id")) is str

    restarted_provider = ProcessLossSuccessfulCompactionProvider(str(marker))
    ledger = SQLiteBudgetLedger(ledger_database)
    restarted = CayuApp(
        session_store=store,
        budget_ledger=ledger,
        budget_policy=_automatic_compaction_process_budget_policy(),
        enable_logging=False,
    )
    restarted.register_provider(restarted_provider, default=True)
    restarted.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=restarted_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    recovery_failure: ModelCompletionManualRecoveryRequired | None = None
    try:
        asyncio.run(
            restarted.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=_AUTOMATIC_COMPACTION_PROCESS_SESSION_ID,
                    inactive_for_seconds=0,
                )
            )
        )
    except ModelCompletionManualRecoveryRequired as failure:
        recovery_failure = failure
    if crash_point == "completed":
        assert recovery_failure is not None

    recovered_events = asyncio.run(store.load_events(_AUTOMATIC_COMPACTION_PROCESS_SESSION_ID))
    recovered_completions = [
        event
        for event in recovered_events
        if event.type is EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert len(recovered_completions) == 1
    assert type(recovered_completions[0].payload.get("parent_model_step_id")) is str
    assert not any(event.type is EventType.SESSION_CHECKPOINTED for event in recovered_events)
    completed_model_step_id = recovered_completions[0].payload["model_step_id"]
    assert type(completed_model_step_id) is str
    completed_stage = asyncio.run(
        store.load_model_completion_stage(
            _AUTOMATIC_COMPACTION_PROCESS_SESSION_ID,
            f"{completed_model_step_id}:dispatch:0",
        )
    )
    assert completed_stage is not None
    assert len(completed_stage.reservation_ids) == 1
    reservation = asyncio.run(ledger.load_reservation(completed_stage.reservation_ids[0]))
    assert reservation is not None
    assert reservation.status == "reconciled"
    assert reservation.actual_amount == Decimal("0.000023")

    if recovery_failure is not None:
        asyncio.run(
            interrupt_and_release_test_invocation(
                store,
                _AUTOMATIC_COMPACTION_PROCESS_SESSION_ID,
            )
        )
    blocked_events = asyncio.run(
        collect_resume_events(
            restarted,
            ResumeRequest(
                session_id=_AUTOMATIC_COMPACTION_PROCESS_SESSION_ID,
                messages=[Message.text("user", "retry after process loss")],
            ),
        )
    )
    assert marker.read_text(encoding="utf-8") == "1"
    assert blocked_events[-1].type is EventType.SESSION_FAILED
    assert "no later durable context checkpoint" in blocked_events[-1].payload["error"]
    assert not any(
        event.type is EventType.MODEL_STARTED
        and event.payload.get("purpose") == "context_compaction"
        for event in blocked_events
    )
    asyncio.run(ledger.close())
    asyncio.run(store.close())
