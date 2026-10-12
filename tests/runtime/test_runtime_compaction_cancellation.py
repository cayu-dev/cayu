"""Compaction cancellation contracts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
from tests.runtime.runtime_compaction_support import compaction_price_book
from tests.runtime.runtime_provider_support import ContextOverflowProvider, FakeProvider
from tests.runtime.runtime_session_support import (
    _test_session,
    checkpoint_without_active_invocation_profile,
    collect_events,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    BudgetWindow,
    InMemoryBudgetStore,
)
from cayu.context.base import CheckpointCompactionContextPolicy, CompactionRequest, ModelCompactor
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import RunRequest


def test_automatic_hierarchical_cancellation_persists_completed_chunk_usage():
    class CancelSecondHierarchyCallProvider(ModelProvider):
        name = "cancel-hierarchy-accounting"

        def __init__(self) -> None:
            self.calls = 0
            self.second_started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
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
        session_id = "sess_hierarchy_cancelled_after_completion"
        store = InMemorySessionStore()
        compactor_provider = CancelSecondHierarchyCallProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compactor_provider,
                    model="summary-model",
                    max_input_chars=1000,
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
                    session_id=session_id,
                    messages=[
                        Message.text("user", "oversized " + "x" * 5000),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        await asyncio.wait_for(compactor_provider.second_started.wait(), timeout=10)
        task.cancel("cancel hierarchy after first completion")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel hierarchy after first completion",)
        events = await store.load_events(session_id)
        compaction_completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(compaction_completions) == 2
        assert compaction_completions[0].payload["usage_metrics"]["input_tokens"] == 5
        assert compaction_completions[0].payload["usage_metrics"]["output_tokens"] == 2
        assert compaction_completions[1].payload["compaction_outcome"] == "cancelled"
        assert compaction_completions[1].payload["usage_unavailable_reason"]
        failures = [event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED]
        assert len(failures) == 1
        assert failures[0].payload["chunk_count"] == 2
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is None or "context_compaction" not in checkpoint

    asyncio.run(run())


def test_automatic_hierarchical_cancellation_persists_each_completed_chunk():
    class CancelThirdHierarchyCallProvider(ModelProvider):
        name = "cancel-third-hierarchy-accounting"

        def __init__(self) -> None:
            self.calls = 0
            self.third_started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            if self.calls == 3:
                self.third_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            yield ModelStreamEvent.text_delta(f"fragment summary {self.calls}")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {
                        "input_tokens": self.calls,
                        "output_tokens": 1,
                    },
                }
            )

    async def run() -> None:
        session_id = "sess_hierarchy_cancelled_batch_persistence"
        store = InMemorySessionStore()
        budget_store = InMemoryBudgetStore()
        compactor_provider = CancelThirdHierarchyCallProvider()
        app = CayuApp(
            session_store=store,
            budget_store=budget_store,
            enable_logging=False,
        )
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compactor_provider,
                    model="summary-model",
                    max_input_chars=1000,
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
                    session_id=session_id,
                    messages=[
                        Message.text("user", "oversized " + "x" * 5000),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        await asyncio.wait_for(compactor_provider.third_started.wait(), timeout=10)
        task.cancel("cancel hierarchy after two completions")
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel hierarchy after two completions",)
        assert task.cancelled()
        events = await store.load_events(session_id)
        compaction_completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(compaction_completions) == 3
        assert [
            event.payload["usage_metrics"]["input_tokens"] for event in compaction_completions[:2]
        ] == [1, 2]
        assert compaction_completions[2].payload["compaction_outcome"] == "cancelled"
        assert compaction_completions[2].payload["usage_unavailable_reason"]
        failures = [event for event in events if event.type == EventType.CONTEXT_COMPACTION_FAILED]
        assert len(failures) == 1
        assert failures[0].payload["chunk_count"] == 3

    asyncio.run(run())


def test_automatic_compaction_stream_abandonment_preserves_success_accounting():
    async def run() -> None:
        session_id = "sess_compaction_success_stream_abandonment"
        store = InMemorySessionStore()
        compaction_provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }
                ),
            ]
        )
        main_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(main_provider, default=True)
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
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            )
        )
        while (await anext(stream)).type != EventType.CONTEXT_COMPACTION_STARTED:
            pass
        await stream.aclose()

        assert main_provider.requests == []
        events = await store.load_events(session_id)
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 7
        assert sum(event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in events) == 1
        assert sum(event.type == EventType.SESSION_CHECKPOINTED for event in events) == 1
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2

    asyncio.run(run())


def test_automatic_compaction_stream_abandonment_after_budget_event_keeps_outcome():
    async def run() -> None:
        session_id = "sess_compaction_budget_stream_abandonment"
        store = InMemorySessionStore()
        compaction_provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }
                ),
            ]
        )
        main_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=compaction_price_book(),
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=0,
                        ),
                    ),
                )
            ),
        )
        app.register_provider(main_provider, default=True)
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
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            )
        )
        while (await anext(stream)).type != EventType.BUDGET_RESERVED:
            pass
        await stream.aclose()

        assert len(compaction_provider.requests) == 1
        assert main_provider.requests == []
        events = await store.load_events(session_id)
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 7
        assert sum(event.type == EventType.BUDGET_RECONCILED for event in events) == 1
        assert sum(event.type == EventType.CONTEXT_COMPACTION_COMPLETED for event in events) == 1
        assert sum(event.type == EventType.SESSION_CHECKPOINTED for event in events) == 1
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == 2

    asyncio.run(run())


def test_automatic_compaction_stream_abandonment_preserves_failure_accounting():
    class FailSecondHierarchyCallProvider(ModelProvider):
        name = "fail-second-hierarchy-call"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second hierarchy call failed")
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )

    async def run() -> None:
        session_id = "sess_compaction_failure_stream_abandonment"
        store = InMemorySessionStore()
        compaction_provider = FailSecondHierarchyCallProvider()
        main_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(main_provider, default=True)
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
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[
                    Message.text("user", "oversized " + "x" * 5000),
                    Message.text("user", "current"),
                ],
            )
        )
        while (await anext(stream)).type != EventType.CONTEXT_COMPACTION_STARTED:
            pass
        await stream.aclose()

        assert main_provider.requests == []
        events = await store.load_events(session_id)
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
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events) == 1
        assert EventType.SESSION_CHECKPOINTED not in {event.type for event in events}
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is None or "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compaction_provider_cancellation_preserves_historical_task_cancellation() -> None:
    class CancellingProvider(ModelProvider):
        name = "provider-cancellation"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            yield ModelStreamEvent.text_delta("partial summary")
            raise asyncio.CancelledError("provider cancellation")

    async def run() -> None:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        try:
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await ModelCompactor(
                    provider=CancellingProvider(),
                    model="summary-model",
                ).compact(
                    CompactionRequest(
                        session=_test_session(),
                        agent=AgentSpec(name="assistant", model="fake-model"),
                        messages=[Message.text("user", "old request")],
                    )
                )
            assert exc_info.value.args == ("provider cancellation",)
            assert current.cancelling() == 1
        finally:
            current.uncancel()

    asyncio.run(run())


def test_automatic_compaction_termination_cleanup_preserves_historical_cancellation() -> None:
    class CancellingProvider(ModelProvider):
        name = "automatic-provider-cancellation"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            yield ModelStreamEvent.text_delta("partial summary")
            raise asyncio.CancelledError("provider cancellation")

    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=CancellingProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        current = asyncio.current_task()
        assert current is not None
        current.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        assert current.cancelling() == 1
        try:
            with pytest.raises(asyncio.CancelledError, match="provider cancellation"):
                await collect_events(
                    app,
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_termination_historical_cancellation",
                        messages=[
                            Message.text("user", "old"),
                            Message.text("assistant", "old answer"),
                            Message.text("user", "current"),
                        ],
                    ),
                )
            assert current.cancelling() == 1
        finally:
            current.uncancel()

        events = await store.load_events("sess_termination_historical_cancellation")
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["compaction_outcome"] == "cancelled"

    asyncio.run(run())


def test_overflow_compaction_termination_cleanup_preserves_historical_cancellation() -> None:
    class CancellingProvider(ModelProvider):
        name = "overflow-provider-cancellation"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            yield ModelStreamEvent.text_delta("partial summary")
            raise asyncio.CancelledError("overflow compactor cancellation")

    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ContextOverflowProvider(), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_overflow_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=CancellingProvider(),
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=1,
            ),
        )
        current = asyncio.current_task()
        assert current is not None
        current.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        assert current.cancelling() == 1
        try:
            with pytest.raises(
                asyncio.CancelledError,
                match="overflow compactor cancellation",
            ):
                await collect_events(
                    app,
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_overflow_historical_cancellation",
                        messages=[
                            Message.text("user", "old"),
                            Message.text("user", "current"),
                        ],
                    ),
                )
            assert current.cancelling() == 1
        finally:
            current.uncancel()

        events = await store.load_events("sess_overflow_historical_cancellation")
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["compaction_outcome"] == "cancelled"

    asyncio.run(run())


def test_automatic_compaction_ignores_handled_historical_task_cancellation():
    async def run() -> None:
        session_id = "sess_compaction_after_historical_cancellation"
        store = InMemorySessionStore()
        compaction_provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }
                ),
            ]
        )
        main_provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("answer"),
                ModelStreamEvent.completed({}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(main_provider, default=True)
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

        current = asyncio.current_task()
        assert current is not None
        current.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.sleep(0)
        assert exc_info.value.args == ("historical cancellation",)
        assert current.cancelling() == 1
        try:
            events = await collect_events(
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
            assert current.cancelling() == 1
        finally:
            current.uncancel()

        assert events[-1].type == EventType.SESSION_COMPLETED
        assert EventType.CONTEXT_COMPACTION_FAILED not in {event.type for event in events}
        assert len(compaction_provider.requests) == 1
        assert len(main_provider.requests) == 1

    asyncio.run(run())


def test_automatic_compaction_real_cancellation_during_start_publication_propagates():
    class BlockingCompactionStartStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.start_publication_started = asyncio.Event()
            self.allow_start_publication = asyncio.Event()

        async def append_events(self, session_id: str, events: list[Event]) -> None:
            if any(event.type == EventType.CONTEXT_COMPACTION_STARTED for event in events):
                self.start_publication_started.set()
                await self.allow_start_publication.wait()
            await super().append_events(session_id, events)

    async def run() -> None:
        session_id = "sess_cancel_compaction_start_publication"
        store = BlockingCompactionStartStore()
        compaction_provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("summary"),
                ModelStreamEvent.completed({}),
            ]
        )
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
        await asyncio.wait_for(store.start_publication_started.wait(), timeout=10)
        task.cancel("cancel during compaction start publication")
        await asyncio.sleep(0)
        assert not task.done()
        store.allow_start_publication.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel during compaction start publication",)
        assert task.cancelled()
        assert task.cancelling() == 0
        assert compaction_provider.requests == []
        durable = await store.load_events(session_id)
        assert sum(event.type == EventType.CONTEXT_COMPACTION_STARTED for event in durable) == 1
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in durable) == 1

    asyncio.run(run())


def test_automatic_compaction_cancellation_during_outcome_persistence_is_lossless():
    class BlockingContextBatchStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.context_batch_started = asyncio.Event()
            self.allow_context_batch = asyncio.Event()

        async def complete_model_completion_stage(self, session_id: str, **kwargs):
            publication = kwargs["publication"]
            if any(
                event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
                for event in publication.events
            ):
                self.context_batch_started.set()
                await self.allow_context_batch.wait()
            return await super().complete_model_completion_stage(session_id, **kwargs)

    async def run() -> None:
        session_id = "sess_compaction_outcome_persistence_cancellation"
        store = BlockingContextBatchStore()
        compaction_provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "summary-model",
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }
                ),
            ]
        )
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
        await asyncio.wait_for(store.context_batch_started.wait(), timeout=10)
        task.cancel("cancel during context outcome persistence")
        assert task.cancelling() == 1
        store.allow_context_batch.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel during context outcome persistence",)
        assert task.cancelled()
        events = await store.load_events(session_id)
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 7
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events) == 1
        assert EventType.CONTEXT_COMPACTION_COMPLETED not in {event.type for event in events}
        assert EventType.SESSION_CHECKPOINTED not in {event.type for event in events}
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is None or "context_compaction" not in checkpoint

    asyncio.run(run())


def test_automatic_compaction_real_cancellation_records_uncertain_dispatch() -> None:
    class BlockingPartialProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.partial_emitted = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.text_delta("partial summary")
            self.partial_emitted.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    async def run() -> None:
        session_id = "sess_cancelled_compaction_dispatch"
        store = InMemorySessionStore()
        budget_store = InMemoryBudgetStore()
        compactor_provider = BlockingPartialProvider()
        app = CayuApp(
            session_store=store,
            budget_store=budget_store,
            enable_logging=False,
        )
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compactor_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=2,
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
        await asyncio.wait_for(compactor_provider.partial_emitted.wait(), timeout=5)
        task.cancel("cancel compaction after provider dispatch")
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel compaction after provider dispatch",)
        assert task.cancelled()
        assert compactor_provider.cancelled.is_set()
        events = await store.load_events(session_id)
        completions = [
            event
            for event in events
            if event.type == EventType.MODEL_COMPLETED
            and event.payload.get("purpose") == "context_compaction"
        ]
        assert len(completions) == 1
        assert completions[0].payload["compaction_outcome"] == "cancelled"
        assert completions[0].payload["usage_unavailable_reason"]
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events) == 1
        assert checkpoint_without_active_invocation_profile(
            await store.load_checkpoint(session_id)
        ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
        usage = await app.get_session_usage(session_id)
        assert usage.model_steps == 1
        assert usage.usage.total_tokens == 0
        durable_budget_events = await budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
        assert [event.id for event in durable_budget_events] == [completions[0].id]

    asyncio.run(run())


def test_automatic_compaction_preserves_later_cancellation_during_termination_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingPartialProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.partial_emitted = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.text_delta("partial summary")
            self.partial_emitted.set()
            await asyncio.Event().wait()

    async def run() -> None:
        session_id = "sess_cancelled_compaction_termination_persistence"
        store = InMemorySessionStore()
        compactor_provider = BlockingPartialProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([ModelStreamEvent.completed({})]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compactor_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )

        termination_persistence_started = asyncio.Event()
        allow_termination_persistence = asyncio.Event()
        original_emit_many = app._event_writer.emit_many

        async def block_termination_persistence(
            emitted_session_id: str,
            events: list[Event],
        ) -> list[Event]:
            if any(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events):
                termination_persistence_started.set()
                await allow_termination_persistence.wait()
            return await original_emit_many(emitted_session_id, events)

        monkeypatch.setattr(app._event_writer, "emit_many", block_termination_persistence)
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
        await asyncio.wait_for(compactor_provider.partial_emitted.wait(), timeout=5)
        task.cancel("primary cancellation during provider dispatch")
        await asyncio.wait_for(termination_persistence_started.wait(), timeout=10)

        task.cancel("later cancellation during termination persistence")
        await asyncio.sleep(0)
        assert not task.done()
        allow_termination_persistence.set()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task

        def leaves(error: BaseException) -> list[BaseException]:
            if isinstance(error, BaseExceptionGroup):
                return [leaf for nested in error.exceptions for leaf in leaves(nested)]
            return [error]

        cancellations = [
            leaf for leaf in leaves(exc_info.value) if isinstance(leaf, asyncio.CancelledError)
        ]
        assert [cancellation.args for cancellation in cancellations] == [
            ("primary cancellation during provider dispatch",),
            ("later cancellation during termination persistence",),
        ]
        assert len(leaves(exc_info.value)) == 2
        assert task.cancelling() == 0
        assert not task.cancelled()

        events = await store.load_events(session_id)
        assert (
            sum(
                event.type == EventType.MODEL_COMPLETED
                and event.payload.get("purpose") == "context_compaction"
                for event in events
            )
            == 1
        )
        assert sum(event.type == EventType.CONTEXT_COMPACTION_FAILED for event in events) == 1
        assert len(compactor_provider.requests) == 1

    asyncio.run(run())
