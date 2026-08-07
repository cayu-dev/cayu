from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from cayu._exception_groups import iter_exception_tree
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.environments import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    EventQuery,
    IncompleteSessionRecoveryAction,
    IncompleteSessionsRecoveryRequest,
    InMemorySessionStore,
    InMemoryTaskStore,
    InteractionLifecyclePublicationRejected,
    LoopPolicy,
    ResumeRequest,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    SessionIdentity,
    SessionStatus,
    TaskCreate,
    TaskStatus,
    TerminalEventPublicationUncertain,
)
from cayu.runtime import sessions as sessions_module
from cayu.runtime.errors import (
    _is_runtime_interaction_lifecycle_publication_rejection,
    _runtime_interaction_lifecycle_publication_rejected,
)
from cayu.runtime.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    InteractionStatus,
    InteractionSummaryEvidence,
)


def test_interaction_summary_rejects_completion_before_start_after_timezone_normalization() -> None:
    started_at = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)

    with pytest.raises(ValidationError, match="completed_at must not precede started_at"):
        InteractionSummaryEvidence(
            status=InteractionStatus.COMPLETED,
            start_event_id="interaction-started",
            started_at=started_at.astimezone(timezone(timedelta(hours=2))),
            completed_at=started_at - timedelta(microseconds=1),
        )

    evidence = InteractionSummaryEvidence(
        status=InteractionStatus.COMPLETED,
        start_event_id="interaction-started",
        started_at=started_at.astimezone(timezone(timedelta(hours=-7))),
        completed_at=started_at,
    )
    assert evidence.started_at == evidence.completed_at == started_at

    with pytest.raises(
        ValidationError,
        match="completed_at must be present exactly for terminal interactions",
    ):
        InteractionSummaryEvidence(
            status=InteractionStatus.ACTIVE,
            start_event_id="interaction-started",
            started_at=started_at,
            completed_at=started_at,
        )
    with pytest.raises(
        ValidationError,
        match="completed_at must be present exactly for terminal interactions",
    ):
        InteractionSummaryEvidence(
            status=InteractionStatus.COMPLETED,
            start_event_id="interaction-started",
            started_at=started_at,
        )


def test_public_run_preserves_recoverable_state_when_interaction_clock_moves_backward() -> None:
    class BackwardClockCompletingProvider(ModelProvider):
        name = "backward-clock-completing"

        def __init__(self, wall: dict[str, datetime]) -> None:
            self.wall = wall

        async def stream(self, request: ModelRequest):
            del request
            self.wall["value"] -= timedelta(seconds=1)
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> None:
        wall = {"value": datetime(2026, 8, 6, 12, 0, tzinfo=UTC)}
        store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app = CayuApp(
            session_store=store,
            task_store=task_store,
            enable_logging=False,
            clock=lambda: wall["value"],
        )
        app.register_provider(BackwardClockCompletingProvider(wall), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_backward_interaction_clock"
        task_id = "task_backward_interaction_clock"
        await task_store.create_task(
            TaskCreate(task_id=task_id, type="respond", assigned_agent_name="assistant")
        )

        emitted: list[Event] = []
        with pytest.raises(InteractionLifecyclePublicationRejected) as raised:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    task_id=task_id,
                    messages=[Message.text("user", "start")],
                )
            ):
                emitted.append(event)

        assert raised.value.session_id == session_id
        assert raised.value.__cause__ is not None
        assert isinstance(raised.value.__cause__, ValidationError)
        assert "completed_at must not precede started_at" in str(raised.value.__cause__)
        assert EventType.MODEL_COMPLETED in {event.type for event in emitted}
        assert not {
            EventType.INTERACTION_COMPLETED,
            EventType.INTERACTION_FAILED,
            EventType.TURN_COMPLETED,
            EventType.SESSION_COMPLETED,
            EventType.SESSION_FAILED,
            EventType.TASK_FAILED,
        }.intersection(event.type for event in emitted)

        session = await store.load(session_id)
        task = await task_store.load_task(task_id)
        checkpoint = await store.load_checkpoint(session_id)
        assert session is not None
        assert session.status is SessionStatus.RUNNING
        assert task is not None
        assert task.status is TaskStatus.RUNNING
        assert checkpoint is None or "session_run_operation" not in checkpoint
        assert sessions_module._current_session_run_epoch(session_id) is None

        records = await store.query_events(EventQuery(session_id=session_id))
        interaction_id = next(
            record.event.interaction_id
            for record in records
            if record.event.type is EventType.INTERACTION_STARTED
        )
        assert interaction_id is not None
        assert [
            record.event.type
            for record in records
            if record.event.interaction_id == interaction_id
            and record.event.type in INTERACTION_LIFECYCLE_EVENT_TYPES
        ] == [EventType.INTERACTION_STARTED]

        wall["value"] += timedelta(seconds=1)
        page = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.RUNNING},
                limit=10,
            )
        )
        assert len(page.results) == 1
        assert page.results[0].actions == (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,)

        recovered_session = await store.load(session_id)
        recovered_task = await task_store.load_task(task_id)
        recovered_checkpoint = await store.load_checkpoint(session_id)
        recovered_records = await store.query_events(EventQuery(session_id=session_id))
        assert recovered_session is not None
        assert recovered_session.status is SessionStatus.INTERRUPTED
        assert recovered_task is not None
        assert recovered_task.status is TaskStatus.RUNNING
        assert recovered_checkpoint is None or "session_run_operation" not in recovered_checkpoint
        assert [
            record.event.type
            for record in recovered_records
            if record.event.interaction_id == interaction_id
            and record.event.type in INTERACTION_LIFECYCLE_EVENT_TYPES
        ] == [EventType.INTERACTION_STARTED, EventType.INTERACTION_INTERRUPTED]
        assert [
            record.event.type
            for record in recovered_records
            if record.event.type is EventType.SESSION_INTERRUPTED
        ] == [EventType.SESSION_INTERRUPTED]

    asyncio.run(run())


def test_run_failure_preserves_recoverable_state_when_interaction_clock_moves_backward() -> None:
    class BackwardClockFailingProvider(ModelProvider):
        name = "backward-clock-failing"

        def __init__(self, wall: dict[str, datetime]) -> None:
            self.wall = wall

        async def stream(self, request: ModelRequest):
            del request
            self.wall["value"] -= timedelta(seconds=1)
            raise RuntimeError("provider failed")
            yield  # pragma: no cover

    async def run() -> None:
        wall = {"value": datetime(2026, 8, 6, 12, 0, tzinfo=UTC)}
        store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app = CayuApp(
            session_store=store,
            task_store=task_store,
            enable_logging=False,
            clock=lambda: wall["value"],
        )
        app.register_provider(BackwardClockFailingProvider(wall), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_backward_clock_run_failure"
        task_id = "task_backward_clock_run_failure"
        await task_store.create_task(
            TaskCreate(task_id=task_id, type="respond", assigned_agent_name="assistant")
        )

        emitted: list[Event] = []
        with pytest.raises(RuntimeError, match="provider failed") as raised:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    task_id=task_id,
                    messages=[Message.text("user", "start")],
                )
            ):
                emitted.append(event)

        assert raised.value.__cause__ is not None
        assert not {
            EventType.INTERACTION_FAILED,
            EventType.TASK_FAILED,
            EventType.TURN_COMPLETED,
            EventType.SESSION_FAILED,
        }.intersection(event.type for event in emitted)
        session = await store.load(session_id)
        task = await task_store.load_task(task_id)
        assert session is not None
        assert session.status is SessionStatus.RUNNING
        assert task is not None
        assert task.status is TaskStatus.RUNNING
        assert sessions_module._current_session_run_epoch(session_id) is None

        wall["value"] += timedelta(seconds=2)
        page = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(statuses={SessionStatus.RUNNING}, limit=10)
        )
        assert page.results[0].actions == (IncompleteSessionRecoveryAction.FAILED,)
        recovered_session = await store.load(session_id)
        recovered_task = await task_store.load_task(task_id)
        assert recovered_session is not None
        assert recovered_session.status is SessionStatus.RUNNING
        assert recovered_task is not None
        assert recovered_task.status is TaskStatus.RUNNING

    asyncio.run(run())


def test_provider_failure_remains_authoritative_when_terminal_preflight_read_fails(
    monkeypatch,
) -> None:
    class FailingProvider(ModelProvider):
        name = "failing"

        async def stream(self, request: ModelRequest):
            del request
            raise RuntimeError("provider failed")
            yield  # pragma: no cover

    async def run() -> None:
        store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app = CayuApp(
            session_store=store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(FailingProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_terminal_preflight_read_failure"
        task_id = "task_terminal_preflight_read_failure"
        await task_store.create_task(
            TaskCreate(task_id=task_id, type="respond", assigned_agent_name="assistant")
        )
        original_query_events = store.query_events

        async def fail_interaction_start_read(query):
            if query.event_types == (EventType.INTERACTION_STARTED,):
                raise ConnectionError("interaction start read failed")
            return await original_query_events(query)

        monkeypatch.setattr(store, "query_events", fail_interaction_start_read)

        emitted: list[Event] = []
        with pytest.raises(RuntimeError, match="provider failed") as raised:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    task_id=task_id,
                    messages=[Message.text("user", "start")],
                )
            ):
                emitted.append(event)

        assert raised.value.__cause__ is not None
        causes = tuple(iter_exception_tree(raised.value.__cause__))
        read_failures = [cause for cause in causes if isinstance(cause, ConnectionError)]
        assert len(read_failures) == 1
        assert str(read_failures[0]) == "interaction start read failed"
        assert not {
            EventType.INTERACTION_FAILED,
            EventType.TASK_FAILED,
            EventType.TURN_COMPLETED,
            EventType.SESSION_FAILED,
        }.intersection(event.type for event in emitted)
        session = await store.load(session_id)
        task = await task_store.load_task(task_id)
        assert session is not None
        assert session.status is SessionStatus.RUNNING
        assert task is not None
        assert task.status is TaskStatus.RUNNING
        assert sessions_module._current_session_run_epoch(session_id) is None

    asyncio.run(run())


def test_setup_failure_preserves_recoverable_state_when_clock_moves_backward(
    monkeypatch,
) -> None:
    async def run() -> None:
        wall = {"value": datetime(2026, 8, 6, 12, 0, tzinfo=UTC)}
        store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app = CayuApp(
            session_store=store,
            task_store=task_store,
            enable_logging=False,
            clock=lambda: wall["value"],
        )
        app.register_provider(CompletingProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_backward_clock_setup_failure"
        task_id = "task_backward_clock_setup_failure"
        await task_store.create_task(
            TaskCreate(task_id=task_id, type="respond", assigned_agent_name="assistant")
        )

        async def fail_initial_transcript_publication(*args, **kwargs):
            del args, kwargs
            wall["value"] -= timedelta(seconds=1)
            raise RuntimeError("workspace setup failed")

        monkeypatch.setattr(
            store,
            "replace_initial_transcript_messages",
            fail_initial_transcript_publication,
        )

        with pytest.raises(RuntimeError, match="workspace setup failed"):
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    task_id=task_id,
                    messages=[Message.text("user", "start")],
                )
            ):
                pass

        session = await store.load(session_id)
        task = await task_store.load_task(task_id)
        assert session is not None
        assert session.status is SessionStatus.RUNNING
        assert task is not None
        assert task.status is TaskStatus.PENDING
        assert sessions_module._current_session_run_epoch(session_id) is None

        wall["value"] += timedelta(seconds=2)
        page = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(statuses={SessionStatus.RUNNING}, limit=10)
        )
        assert page.results[0].actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
            IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
        )
        recovered_session = await store.load(session_id)
        recovered_task = await task_store.load_task(task_id)
        assert recovered_session is not None
        assert recovered_session.status is SessionStatus.INTERRUPTED
        assert recovered_task is not None
        assert recovered_task.status is TaskStatus.PENDING

    asyncio.run(run())


def test_returned_factory_failure_preserves_state_when_clock_moves_backward() -> None:
    class BackwardClockFailingFactory(EnvironmentFactory):
        def __init__(self, wall: dict[str, datetime]) -> None:
            self.wall = wall

        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            del request
            self.wall["value"] -= timedelta(seconds=1)
            raise RuntimeError("factory failed")

    async def run() -> None:
        wall = {"value": datetime(2026, 8, 6, 12, 0, tzinfo=UTC)}
        store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app = CayuApp(
            session_store=store,
            task_store=task_store,
            enable_logging=False,
            clock=lambda: wall["value"],
        )
        app.register_provider(CompletingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            BackwardClockFailingFactory(wall),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_backward_clock_factory_failure"
        task_id = "task_backward_clock_factory_failure"
        await task_store.create_task(
            TaskCreate(task_id=task_id, type="respond", assigned_agent_name="assistant")
        )

        emitted: list[Event] = []
        with pytest.raises(RuntimeError, match="factory failed") as raised:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    task_id=task_id,
                    messages=[Message.text("user", "start")],
                )
            ):
                emitted.append(event)

        assert isinstance(raised.value.__cause__, InteractionLifecyclePublicationRejected)
        assert EventType.ENVIRONMENT_FACTORY_FAILED in {event.type for event in emitted}
        assert not {
            EventType.INTERACTION_FAILED,
            EventType.TASK_FAILED,
            EventType.TURN_COMPLETED,
            EventType.SESSION_FAILED,
        }.intersection(event.type for event in emitted)
        session = await store.load(session_id)
        task = await task_store.load_task(task_id)
        assert session is not None
        assert session.status is SessionStatus.RUNNING
        assert task is not None
        assert task.status is TaskStatus.RUNNING
        assert sessions_module._current_session_run_epoch(session_id) is None

        wall["value"] += timedelta(seconds=2)
        page = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(statuses={SessionStatus.RUNNING}, limit=10)
        )
        assert page.results[0].actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
            IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
        )
        recovered_session = await store.load(session_id)
        recovered_task = await task_store.load_task(task_id)
        assert recovered_session is not None
        assert recovered_session.status is SessionStatus.INTERRUPTED
        assert recovered_task is not None
        assert recovered_task.status is TaskStatus.RUNNING

    asyncio.run(run())


def test_returned_factory_failure_streams_terminal_hook_events() -> None:
    runtime: dict[str, CayuApp] = {}

    class FailingFactory(EnvironmentFactory):
        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            del request
            raise RuntimeError("factory failed")

    class EmittingFailureHook(RuntimeHook):
        async def after_session_failed(self, context: RuntimeHookContext) -> None:
            await runtime["app"].emit_event(
                Event(
                    type="custom.factory.failure.observed",
                    session_id=context.session.id,
                    payload={"session_id": context.session.id},
                )
            )

    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            runtime_hooks=[EmittingFailureHook()],
        )
        runtime["app"] = app
        app.register_provider(CompletingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            FailingFactory(),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_factory_failure_hook_event"

        emitted = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                )
            )
        ]

        assert [event.type for event in emitted].count("custom.factory.failure.observed") == 1
        assert EventType.SESSION_FAILED in {event.type for event in emitted}
        custom_event = next(
            event for event in emitted if event.type == "custom.factory.failure.observed"
        )
        assert custom_event.payload == {"session_id": session_id}

    asyncio.run(run())


def test_initial_factory_exception_group_uses_setup_failure_finalization() -> None:
    class GroupFailingFactory(EnvironmentFactory):
        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            del request
            raise ExceptionGroup(
                "factory failed",
                [RuntimeError("create failed"), ValueError("cleanup failed")],
            )

    async def run() -> None:
        store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app = CayuApp(
            session_store=store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(CompletingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            GroupFailingFactory(),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_factory_exception_group"
        task_id = "task_factory_exception_group"
        await task_store.create_task(
            TaskCreate(task_id=task_id, type="respond", assigned_agent_name="assistant")
        )

        emitted = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    task_id=task_id,
                    messages=[Message.text("user", "start")],
                )
            )
        ]

        assert [
            event.type
            for event in emitted
            if event.type
            in {
                EventType.TASK_FAILED,
                EventType.INTERACTION_FAILED,
                EventType.SESSION_FAILED,
            }
        ] == [
            EventType.TASK_FAILED,
            EventType.INTERACTION_FAILED,
            EventType.SESSION_FAILED,
        ]
        session = await store.load(session_id)
        task = await task_store.load_task(task_id)
        assert session is not None
        assert session.status is SessionStatus.FAILED
        assert task is not None
        assert task.status is TaskStatus.FAILED
        assert emitted[-1].payload["error_type"] == "ExceptionGroup"

    asyncio.run(run())


def test_backward_clock_stream_abandonment_keeps_lifecycle_nonterminal() -> None:
    async def run() -> None:
        wall = {"value": datetime(2026, 8, 6, 12, 0, tzinfo=UTC)}
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: wall["value"],
        )
        app.register_provider(CompletingProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_backward_clock_abandonment"
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "start")],
            )
        )

        assert (await anext(stream)).type is EventType.INTERACTION_STARTED
        assert (await anext(stream)).type is EventType.SESSION_STARTED
        wall["value"] -= timedelta(seconds=1)
        await stream.aclose()

        session = await store.load(session_id)
        records = await store.query_events(EventQuery(session_id=session_id))
        assert session is not None
        assert session.status is SessionStatus.RUNNING
        assert not {
            EventType.INTERACTION_INTERRUPTED,
            EventType.SESSION_INTERRUPTED,
        }.intersection(record.event.type for record in records)

        wall["value"] += timedelta(seconds=2)
        page = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(statuses={SessionStatus.RUNNING}, limit=10)
        )
        assert page.results[0].actions == (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,)
        recovered = await store.load(session_id)
        assert recovered is not None
        assert recovered.status is SessionStatus.INTERRUPTED

    asyncio.run(run())


def test_backward_clock_setup_stream_abandonment_preserves_generator_exit() -> None:
    async def run() -> None:
        wall = {"value": datetime(2026, 8, 6, 12, 0, tzinfo=UTC)}
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: wall["value"],
        )
        app.register_provider(CompletingProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_backward_clock_setup_abandonment"
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "start")],
            )
        )

        assert (await anext(stream)).type is EventType.INTERACTION_STARTED
        wall["value"] -= timedelta(seconds=1)
        await stream.aclose()

        session = await store.load(session_id)
        records = await store.query_events(EventQuery(session_id=session_id))
        assert session is not None
        assert session.status is SessionStatus.RUNNING
        assert not {
            EventType.INTERACTION_INTERRUPTED,
            EventType.SESSION_INTERRUPTED,
        }.intersection(record.event.type for record in records)

    asyncio.run(run())


def test_backward_clock_setup_cancellation_preserves_rejection_diagnostic() -> None:
    class BlockingFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.entered = asyncio.Event()

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            del request
            self.entered.set()
            await asyncio.Event().wait()
            raise AssertionError("Blocked factory unexpectedly resumed.")

    async def run() -> None:
        wall = {"value": datetime(2026, 8, 6, 12, 0, tzinfo=UTC)}
        store = InMemorySessionStore()
        factory = BlockingFactory()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: wall["value"],
        )
        app.register_provider(CompletingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_backward_clock_setup_cancellation"

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                )
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(factory.entered.wait(), timeout=5)
        wall["value"] -= timedelta(seconds=1)
        consumer.cancel()
        assert consumer.cancelling() == 1
        with pytest.raises(asyncio.CancelledError) as raised:
            await consumer
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 1
        assert raised.value.__cause__ is not None
        assert (
            sum(
                isinstance(error, InteractionLifecyclePublicationRejected)
                for error in iter_exception_tree(raised.value.__cause__)
            )
            == 1
        )

        session = await store.load(session_id)
        records = await store.query_events(EventQuery(session_id=session_id))
        assert session is not None
        assert session.status is SessionStatus.RUNNING
        assert not {
            EventType.INTERACTION_INTERRUPTED,
            EventType.SESSION_INTERRUPTED,
        }.intersection(record.event.type for record in records)

    asyncio.run(run())


def test_backward_clock_task_cancellation_preserves_cancellation_and_live_state() -> None:
    class BlockingProvider(ModelProvider):
        name = "blocking"

        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request: ModelRequest):
            del request
            self.entered.set()
            await self.release.wait()
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> None:
        wall = {"value": datetime(2026, 8, 6, 12, 0, tzinfo=UTC)}
        store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        provider = BlockingProvider()
        app = CayuApp(
            session_store=store,
            task_store=task_store,
            enable_logging=False,
            clock=lambda: wall["value"],
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_backward_clock_cancellation"
        task_id = "task_backward_clock_cancellation"
        await task_store.create_task(
            TaskCreate(task_id=task_id, type="respond", assigned_agent_name="assistant")
        )

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    task_id=task_id,
                    messages=[Message.text("user", "start")],
                )
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(provider.entered.wait(), timeout=5)
        wall["value"] -= timedelta(seconds=1)
        consumer.cancel()
        assert consumer.cancelling() == 1
        with pytest.raises(asyncio.CancelledError) as raised:
            await consumer
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 1
        assert raised.value.__cause__ is not None
        assert (
            sum(
                isinstance(error, InteractionLifecyclePublicationRejected)
                for error in iter_exception_tree(raised.value.__cause__)
            )
            == 1
        )

        session = await store.load(session_id)
        task = await task_store.load_task(task_id)
        records = await store.query_events(EventQuery(session_id=session_id))
        assert session is not None
        assert session.status is SessionStatus.RUNNING
        assert task is not None
        assert task.status is TaskStatus.RUNNING
        assert not {
            EventType.INTERACTION_INTERRUPTED,
            EventType.SESSION_INTERRUPTED,
        }.intersection(record.event.type for record in records)
        assert sessions_module._current_session_run_epoch(session_id) is None

    asyncio.run(run())


def test_extension_cannot_claim_interaction_lifecycle_publication_rejection() -> None:
    class ImpersonatingPolicy(LoopPolicy):
        async def before_stop(self, context):
            raise InteractionLifecyclePublicationRejected(
                session_id=context.session.id,
                interaction_id="caller-selected",
            )

    async def run() -> None:
        session_id = "sess_provider_lifecycle_rejection"
        task_id = "task_provider_lifecycle_rejection"
        store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app = CayuApp(
            session_store=store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(CompletingProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await task_store.create_task(
            TaskCreate(task_id=task_id, type="respond", assigned_agent_name="assistant")
        )

        emitted = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    task_id=task_id,
                    messages=[Message.text("user", "start")],
                    loop_policies=(ImpersonatingPolicy(),),
                )
            )
        ]

        session = await store.load(session_id)
        task = await task_store.load_task(task_id)
        assert session is not None
        assert session.status is SessionStatus.FAILED
        assert task is not None
        assert task.status is TaskStatus.FAILED
        assert task.error is not None
        assert task.error["type"] == "InteractionLifecyclePublicationRejected"
        assert [
            event.type
            for event in emitted
            if event.type
            in {
                EventType.INTERACTION_FAILED,
                EventType.TASK_FAILED,
                EventType.TURN_COMPLETED,
                EventType.SESSION_FAILED,
            }
        ] == [
            EventType.TASK_FAILED,
            EventType.INTERACTION_FAILED,
            EventType.TURN_COMPLETED,
            EventType.SESSION_FAILED,
        ]
        assert emitted[-1].payload["error_type"] == "InteractionLifecyclePublicationRejected"

    asyncio.run(run())


def test_runtime_interaction_rejection_provenance_cannot_be_retargeted() -> None:
    authority = object()
    rejection = _runtime_interaction_lifecycle_publication_rejected(
        session_id="source-session",
        interaction_id="source-interaction",
        runtime_authority=authority,
    )

    rejection.session_id = "target-session"
    rejection.interaction_id = "target-interaction"

    assert not _is_runtime_interaction_lifecycle_publication_rejection(
        rejection,
        session_id="target-session",
        interaction_id="target-interaction",
        runtime_authority=authority,
    )

    malformed_values = ((), 1, (authority,), (authority, "source-session", object()))
    for malformed in malformed_values:
        rejection._runtime_provenance = malformed  # type: ignore[assignment]
        assert not _is_runtime_interaction_lifecycle_publication_rejection(
            rejection,
            session_id="source-session",
            interaction_id="source-interaction",
            runtime_authority=authority,
        )

    del rejection._runtime_provenance
    assert not _is_runtime_interaction_lifecycle_publication_rejection(
        rejection,
        session_id="source-session",
        interaction_id="source-interaction",
        runtime_authority=authority,
    )


class CompletingProvider(ModelProvider):
    name = "completing"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class FailingProvider(ModelProvider):
    name = "failing"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        raise RuntimeError("provider failed")
        yield  # pragma: no cover


class CommitThenLoseInteractionTransitionAcknowledgementStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.lost_acknowledgement = False
        self.attempted_event_ids: list[str] = []

    async def publish_interaction_transition(
        self,
        session_id,
        *,
        event,
        from_statuses,
        to_status,
        only_if_no_queued_messages=False,
    ):
        self.attempted_event_ids.append(event.id)
        result = await super().publish_interaction_transition(
            session_id,
            event=event,
            from_statuses=from_statuses,
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
        )
        if not self.lost_acknowledgement:
            self.lost_acknowledgement = True
            raise ConnectionError("interaction transition acknowledgement lost")
        return result


class CommitThenLoseTerminalPublicationStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.armed = False
        self._unreadable_event_id: str | None = None
        self.publication_failure: ConnectionError | None = None
        self.reconciliation_failure: TimeoutError | None = None

    async def append_event(self, session_id: str, event: Event) -> None:
        await super().append_event(session_id, event)
        if self.armed and event.type == EventType.SESSION_COMPLETED:
            self.armed = False
            self._unreadable_event_id = event.id
            failure = ConnectionError("terminal append acknowledgement lost")
            self.publication_failure = failure
            raise failure

    async def query_events(self, query: EventQuery):
        if self._unreadable_event_id is not None and query.event_id == self._unreadable_event_id:
            self._unreadable_event_id = None
            failure = TimeoutError("terminal reconciliation unavailable")
            self.reconciliation_failure = failure
            raise failure
        return await super().query_events(query)


@pytest.mark.parametrize(
    ("provider", "expected_status", "expected_event_type"),
    [
        (CompletingProvider(), SessionStatus.COMPLETED, EventType.INTERACTION_COMPLETED),
        (FailingProvider(), SessionStatus.FAILED, EventType.INTERACTION_FAILED),
    ],
)
def test_runtime_reconstructs_interaction_transition_after_commit_acknowledgement_loss(
    provider: ModelProvider,
    expected_status: SessionStatus,
    expected_event_type: EventType,
) -> None:
    async def run() -> None:
        store = CommitThenLoseInteractionTransitionAcknowledgementStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"sess_{expected_status}",
                    messages=[Message.text("user", "run")],
                )
            )
        ]

        session = await store.load(f"sess_{expected_status}")
        assert session is not None
        assert session.status is expected_status
        assert store.lost_acknowledgement is True
        assert len(store.attempted_event_ids) == 2
        assert len(set(store.attempted_event_ids)) == 1
        durable = await store.load_events(session.id)
        assert sum(event.type == expected_event_type for event in durable) == 1
        assert sum(event.type == expected_event_type for event in events) == 1
        assert len(provider.requests) == 1  # type: ignore[attr-defined]

    asyncio.run(run())


def test_terminal_publication_uncertainty_preserves_durable_terminal_outcome() -> None:
    async def run() -> None:
        store = CommitThenLoseTerminalPublicationStore()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_terminal_publication_uncertain"

        initial_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                )
            )
        ]
        assert initial_events[-1].type == EventType.SESSION_COMPLETED

        store.armed = True
        with pytest.raises(TerminalEventPublicationUncertain) as raised:
            [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]

        session = await store.load(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        terminal_records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=(
                    EventType.SESSION_COMPLETED,
                    EventType.SESSION_FAILED,
                    EventType.SESSION_INTERRUPTED,
                ),
            )
        )

        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert checkpoint is not None
        run_operation = checkpoint["session_run_operation"]
        assert [record.event.type for record in terminal_records] == [
            EventType.SESSION_COMPLETED,
            EventType.SESSION_COMPLETED,
        ]
        assert (
            terminal_records[-1].event.payload["session_run_operation_id"]
            == run_operation["operation_id"]
        )
        assert len(provider.requests) == 2

        uncertainty = raised.value
        assert uncertainty.session_id == session_id
        assert uncertainty.event_id == terminal_records[-1].event.id
        assert uncertainty.event == terminal_records[-1].event
        assert uncertainty.__cause__ is uncertainty.failures
        assert uncertainty.failures.exceptions == (
            store.publication_failure,
            store.reconciliation_failure,
        )

    asyncio.run(run())


def test_initial_run_terminal_publication_uncertainty_preserves_only_completed_outcome() -> None:
    async def run() -> None:
        store = CommitThenLoseTerminalPublicationStore()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_initial_terminal_publication_uncertain"
        store.armed = True

        with pytest.raises(TerminalEventPublicationUncertain) as raised:
            [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "start")],
                    )
                )
            ]

        session = await store.load(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        terminal_records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=(
                    EventType.SESSION_COMPLETED,
                    EventType.SESSION_FAILED,
                    EventType.SESSION_INTERRUPTED,
                ),
            )
        )

        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert checkpoint is not None
        assert "session_run_operation" not in checkpoint
        assert [record.event.type for record in terminal_records] == [EventType.SESSION_COMPLETED]
        assert len(provider.requests) == 1
        assert raised.value.session_id == session_id
        assert raised.value.event_id == terminal_records[0].event.id
        assert raised.value.event == terminal_records[0].event

    asyncio.run(run())


def test_batch_recovery_attributes_repairs_before_terminal_reconciliation() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    provider = CompletingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    interaction_id = "interaction-batch-recovery"

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_batch_interaction",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name=provider.name, model="fake-model"),
        )
        await store.update_status("sess_batch_interaction", SessionStatus.RUNNING)
        started_at = datetime.now(UTC) - timedelta(seconds=1)
        started = Event(
            id="interaction-batch-recovery-start",
            type=EventType.INTERACTION_STARTED,
            session_id="sess_batch_interaction",
            interaction_id=interaction_id,
            timestamp=started_at,
            payload=InteractionSummaryEvidence(
                status=InteractionStatus.ACTIVE,
                start_event_id="interaction-batch-recovery-start",
                started_at=started_at,
            ).model_dump(mode="json"),
        )
        await store.append_event("sess_batch_interaction", started)
        page = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.RUNNING},
                limit=10,
            )
        )
        return page, await store.load_events("sess_batch_interaction")

    page, events = asyncio.run(setup_and_recover())

    assert page.results[0].actions == (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,)
    recovery_events = [event for event in events if event.id != "interaction-batch-recovery-start"]
    assert recovery_events
    assert all(
        event.interaction_id == interaction_id
        for event in recovery_events
        if event.type not in sessions_module.UNASSOCIATED_RUNTIME_EVENT_TYPES
    )
    assert all(
        event.interaction_id is None
        for event in recovery_events
        if event.type in sessions_module.UNASSOCIATED_RUNTIME_EVENT_TYPES
    )
    assert [event.type for event in events].count(EventType.INTERACTION_INTERRUPTED) == 1


def test_batch_recovery_fault_isolates_and_retries_interaction_reconciliation() -> None:
    class FailingInteractionReconciliationStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.interaction_failures_remaining = 2

        async def publish_interaction_transition(self, session_id: str, **kwargs):
            event = kwargs["event"]
            if (
                event.type == EventType.INTERACTION_INTERRUPTED
                and self.interaction_failures_remaining > 0
            ):
                self.interaction_failures_remaining -= 1
                raise ConnectionError("interaction transition unavailable")
            return await super().publish_interaction_transition(session_id, **kwargs)

    async def scenario() -> None:
        store = FailingInteractionReconciliationStore()
        app = CayuApp(session_store=store, enable_logging=False)
        provider = CompletingProvider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_batch_interaction_reconciliation_retry"
        interaction_id = "interaction-batch-reconciliation-retry"
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name=provider.name, model="fake-model"),
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        started_at = datetime.now(UTC) - timedelta(seconds=1)
        await store.append_event(
            session_id,
            Event(
                id="interaction-batch-reconciliation-retry-start",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
                timestamp=started_at,
                payload=InteractionSummaryEvidence(
                    status=InteractionStatus.ACTIVE,
                    start_event_id="interaction-batch-reconciliation-retry-start",
                    started_at=started_at,
                ).model_dump(mode="json"),
            ),
        )
        request = IncompleteSessionsRecoveryRequest(
            statuses={SessionStatus.INTERRUPTED},
            limit=10,
        )

        failed_page = await app.recover_incomplete_sessions(request)
        assert len(failed_page.results) == 1
        assert failed_page.results[0].actions == (IncompleteSessionRecoveryAction.FAILED,)
        failed_session = await store.load(session_id)
        assert failed_session is not None
        assert failed_session.status is SessionStatus.INTERRUPTED
        first_events = await store.load_events(session_id)
        assert [event.type for event in first_events].count(EventType.INTERACTION_INTERRUPTED) == 0

        retried_page = await app.recover_incomplete_sessions(request)
        assert retried_page.results == ()
        events = await store.load_events(session_id)
        assert [event.type for event in events].count(EventType.SESSION_INTERRUPTED) == 1
        assert [event.type for event in events].count(EventType.INTERACTION_INTERRUPTED) == 1

    asyncio.run(scenario())


def test_terminal_interaction_closes_attribution_before_session_finalization() -> None:
    class TerminalHook(RuntimeHook):
        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            await context.emit_custom_event(
                "custom.after-session-completed",
                payload={"session_id": context.session.id},
            )

    async def run() -> None:
        session_id = "sess_terminal_interaction_attribution"
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            runtime_hooks=[TerminalHook()],
            enable_logging=False,
        )
        app.register_provider(CompletingProvider())
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )

        events = await store.load_events(session_id)
        terminal_index = next(
            index
            for index, event in enumerate(events)
            if event.type == EventType.INTERACTION_COMPLETED
        )
        interaction_id = events[terminal_index].interaction_id
        assert interaction_id is not None
        assert all(event.interaction_id is None for event in events[terminal_index + 1 :])

        turn_completed = next(event for event in events if event.type == EventType.TURN_COMPLETED)
        assert turn_completed.payload["interaction_ids"] == [interaction_id]

    asyncio.run(run())


def test_run_stream_carries_interaction_context_across_consumer_tasks() -> None:
    async def run() -> None:
        session_id = "sess_cross_task_interaction_context"
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider())
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            )
        )
        events: list[Event] = []
        while True:
            try:
                events.append(await asyncio.create_task(anext(stream)))
            except StopAsyncIteration:
                break
            assert sessions_module._current_session_interaction_id(session_id) is None

        interaction_id = next(
            event.interaction_id for event in events if event.type == EventType.INTERACTION_STARTED
        )
        assert interaction_id is not None
        for event_type in (
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
            EventType.INTERACTION_COMPLETED,
        ):
            matching = [event for event in events if event.type == event_type]
            assert matching
            assert {event.interaction_id for event in matching} == {interaction_id}
        assert sessions_module._current_session_interaction_id(session_id) is None

    asyncio.run(run())


def test_in_memory_interaction_query_uses_interaction_scoped_candidates() -> None:
    async def run() -> None:
        session_id = "sess_interaction_candidate_index"
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider())
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
        interaction_id = next(
            event.interaction_id
            for event in await store.load_events(session_id)
            if event.type == EventType.INTERACTION_STARTED
        )
        assert interaction_id is not None
        await store.append_events(
            session_id,
            [
                Event(
                    id=f"evt_unrelated_{index}",
                    type="custom.unrelated",
                    session_id=session_id,
                    interaction_id="interaction-unrelated",
                )
                for index in range(32)
            ],
        )

        query = EventQuery(session_id=session_id, interaction_id=interaction_id)
        candidates = store._query_candidate_records(query, frozenset())
        records = await store.query_events(query)

        assert candidates
        assert len(candidates) == len(records)
        assert all(record.event.interaction_id == interaction_id for record in candidates)

    asyncio.run(run())


async def _collect_events(app: CayuApp, request: RunRequest) -> list:
    return [event async for event in app.run(request)]
