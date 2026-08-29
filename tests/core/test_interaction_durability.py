from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from cayu import SQLiteSessionStore
from cayu._exception_groups import exception_cause, iter_exception_tree
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.environments import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    EnqueueSessionMessageRequest,
    EventQuery,
    EventSink,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionsRecoveryRequest,
    InMemorySessionStore,
    InMemoryTaskStore,
    InteractionLifecyclePublicationRejected,
    InteractionTransitionReceiptResult,
    InteractionTransitionResult,
    InteractionTransitionSpec,
    InterruptSessionRequest,
    LoopPolicy,
    ResumeRequest,
    RunLimits,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    SessionIdentity,
    SessionMessageDeliveryMode,
    SessionRunFenced,
    SessionStatus,
    SessionStatusConflict,
    TaskCreate,
    TaskStatus,
    TerminalEventPublicationUncertain,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.runtime import _session_engine as session_engine_module
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


@pytest.mark.parametrize(
    ("field_name", "values"),
    [
        ("provider_names", ["openai", "openai"]),
        ("provider_names", ["openai", "anthropic", "openai"]),
        ("models", ["gpt-5", "gpt-5"]),
        ("models", ["gpt-5", "claude-opus-4", "gpt-5"]),
    ],
    ids=[
        "adjacent-providers",
        "non-adjacent-providers",
        "adjacent-models",
        "non-adjacent-models",
    ],
)
def test_interaction_summary_rejects_duplicate_provider_and_model_names(
    field_name: str,
    values: list[str],
) -> None:
    with pytest.raises(ValidationError, match=f"{field_name} must not contain duplicates"):
        InteractionSummaryEvidence.model_validate(
            {
                "status": InteractionStatus.ACTIVE,
                "start_event_id": "interaction-started",
                "started_at": datetime(2026, 8, 11, tzinfo=UTC),
                field_name: values,
            }
        )


def test_interaction_summary_preserves_distinct_provider_and_model_name_order() -> None:
    evidence = InteractionSummaryEvidence(
        status=InteractionStatus.ACTIVE,
        start_event_id="interaction-started",
        started_at=datetime(2026, 8, 11, tzinfo=UTC),
        provider_names=["anthropic", "openai"],
        models=["claude-opus-4", "gpt-5"],
    )

    assert evidence.provider_names == ["anthropic", "openai"]
    assert evidence.models == ["claude-opus-4", "gpt-5"]


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


class ApprovalLimitProvider(ModelProvider):
    name = "approval-limit"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        ordinal = len(self.requests)
        yield ModelStreamEvent.tool_call(
            id=f"call_{ordinal}",
            name="limited_side_effect",
            arguments={"ordinal": ordinal},
        )
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class LimitedSideEffectTool(Tool):
    spec = ToolSpec(
        name="limited_side_effect",
        description="Record one side effect after approval.",
        input_schema={
            "type": "object",
            "properties": {"ordinal": {"type": "integer"}},
            "required": ["ordinal"],
        },
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        self.calls.append(args)
        return ToolResult(content="recorded")


class RequireLimitedToolApprovalPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            reason=f"Approval required for {request.tool_name}.",
        )


class FailingProvider(ModelProvider):
    name = "failing"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        raise RuntimeError("provider failed")
        yield  # pragma: no cover


class CommitThenLoseInteractionTransitionAcknowledgementStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(
        self,
        *,
        lost_acknowledgements: int = 1,
        failure_factory: Callable[[], Exception] | None = None,
    ) -> None:
        super().__init__()
        self.remaining_lost_acknowledgements = lost_acknowledgements
        self.failure_factory = failure_factory
        self.attempted_events: list[Event] = []
        self.attempted_model_completion_stage_settlements = []

    async def publish_interaction_transition(
        self,
        session_id,
        *,
        event,
        from_statuses,
        to_status,
        only_if_no_queued_messages=False,
        model_completion_stage_settlement=None,
        expected_session_instance_id=None,
        expected_active_invocation_profile=None,
        expected_invocation_authority_state="active",
    ):
        self.attempted_events.append(event.model_copy(deep=True))
        self.attempted_model_completion_stage_settlements.append(model_completion_stage_settlement)
        result = await super().publish_interaction_transition(
            session_id,
            event=event,
            from_statuses=from_statuses,
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
            model_completion_stage_settlement=model_completion_stage_settlement,
            expected_session_instance_id=expected_session_instance_id,
            expected_active_invocation_profile=expected_active_invocation_profile,
            expected_invocation_authority_state=expected_invocation_authority_state,
        )
        if self.remaining_lost_acknowledgements > 0:
            self.remaining_lost_acknowledgements -= 1
            if self.failure_factory is not None:
                raise self.failure_factory()
            raise ConnectionError("interaction transition acknowledgement lost")
        return result


class CommitThenLoseTerminalPublicationStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

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


class RejectBeforeInteractionTransitionCommitStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(
        self,
        *,
        failure_type: type[Exception],
        failure_message: str,
        target_event_type: EventType = EventType.INTERACTION_COMPLETED,
    ) -> None:
        super().__init__()
        self.failure_type = failure_type
        self.failure_message = failure_message
        self.target_event_type = target_event_type
        self.attempted_events: list[Event] = []

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        event = kwargs["event"]
        if event.type == self.target_event_type:
            self.attempted_events.append(event.model_copy(deep=True))
            raise self.failure_type(self.failure_message)
        return await super().publish_interaction_transition(session_id, **kwargs)


class CommitAndBlockInteractionTransitionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, *, fail_after_release: bool) -> None:
        super().__init__()
        self.fail_after_release = fail_after_release
        self.committed = asyncio.Event()
        self.release = asyncio.Event()
        self.attempted_events: list[Event] = []
        self._blocked_once = False

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        event = kwargs["event"]
        if event.type == EventType.INTERACTION_COMPLETED:
            self.attempted_events.append(event.model_copy(deep=True))
        result = await super().publish_interaction_transition(session_id, **kwargs)
        if event.type == EventType.INTERACTION_COMPLETED and not self._blocked_once:
            self._blocked_once = True
            self.committed.set()
            await self.release.wait()
            if self.fail_after_release:
                raise ConnectionError("interaction transition acknowledgement lost")
        return result


class CommitThenLoseAndBlockInterruptedTransitionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.second_attempt_dispatched = asyncio.Event()
        self.release_second_attempt = asyncio.Event()
        self.attempts = 0

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        if kwargs["event"].type != EventType.INTERACTION_INTERRUPTED:
            return await super().publish_interaction_transition(session_id, **kwargs)
        self.attempts += 1
        if self.attempts == 1:
            await super().publish_interaction_transition(session_id, **kwargs)
            raise ConnectionError("interruption transition acknowledgement lost")
        if self.attempts == 2:
            self.second_attempt_dispatched.set()
            await self.release_second_attempt.wait()
        return await super().publish_interaction_transition(session_id, **kwargs)


class FailThenBlockInterruptedTransitionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.second_attempt_dispatched = asyncio.Event()
        self.release_second_attempt = asyncio.Event()
        self.attempts = 0

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        if kwargs["event"].type != EventType.INTERACTION_INTERRUPTED:
            return await super().publish_interaction_transition(session_id, **kwargs)
        self.attempts += 1
        if self.attempts == 1:
            raise ConnectionError("limit interruption rejected before commit")
        if self.attempts == 2:
            self.second_attempt_dispatched.set()
            await self.release_second_attempt.wait()
            raise TimeoutError("limit interruption replay rejected before commit")
        return await super().publish_interaction_transition(session_id, **kwargs)


class BlockingSiblingInteractionTransitionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, *, commit_before_release: bool) -> None:
        super().__init__()
        self.commit_before_release = commit_before_release
        self.dispatched = asyncio.Event()
        self.release = asyncio.Event()
        self.attempted_events: list[Event] = []
        self._blocked_once = False

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        event = kwargs["event"]
        if event.type != EventType.INTERACTION_FAILED or self._blocked_once:
            return await super().publish_interaction_transition(session_id, **kwargs)
        self._blocked_once = True
        self.attempted_events.append(event.model_copy(deep=True))
        result = None
        if self.commit_before_release:
            result = await super().publish_interaction_transition(session_id, **kwargs)
        self.dispatched.set()
        await self.release.wait()
        if not self.commit_before_release:
            raise ConnectionError("setup transition rejected before commit")
        if result is None:  # pragma: no cover - guarded by commit_before_release
            raise AssertionError("Committed transition produced no result.")
        return result


class QueueGuardedInteractionTransitionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.dispatched = asyncio.Event()
        self.release = asyncio.Event()
        self.attempted_events: list[Event] = []

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        event = kwargs["event"]
        if event.type == EventType.INTERACTION_COMPLETED:
            self.attempted_events.append(event.model_copy(deep=True))
            self.dispatched.set()
            await self.release.wait()
        return await super().publish_interaction_transition(session_id, **kwargs)


class CommitMutateThenLoseInteractionTransitionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.attempted_events: list[Event] = []
        self.attempted_from_statuses: list[set[SessionStatus]] = []
        self.attempted_event_objects: list[Event] = []
        self._lost_acknowledgement = False

    async def publish_interaction_transition(
        self,
        session_id: str,
        *,
        event: Event,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        only_if_no_queued_messages: bool = False,
        model_completion_stage_settlement=None,
        expected_session_instance_id: str | None = None,
        expected_active_invocation_profile=None,
        expected_invocation_authority_state="active",
    ):
        self.attempted_events.append(event.model_copy(deep=True))
        self.attempted_from_statuses.append(set(from_statuses))
        self.attempted_event_objects.append(event)
        result = await super().publish_interaction_transition(
            session_id,
            event=event,
            from_statuses=from_statuses,
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
            model_completion_stage_settlement=model_completion_stage_settlement,
            expected_session_instance_id=expected_session_instance_id,
            expected_active_invocation_profile=expected_active_invocation_profile,
            expected_invocation_authority_state=expected_invocation_authority_state,
        )
        if not self._lost_acknowledgement:
            self._lost_acknowledgement = True
            event.id = "mutated-after-commit"
            event.payload["mutated_after_commit"] = True
            from_statuses.clear()
            raise ConnectionError("interaction transition acknowledgement lost")
        return result


class CorruptThenReplayInteractionTransitionResultStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, corruption: str) -> None:
        super().__init__()
        self.corruption = corruption
        self.attempts = 0

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        result = await super().publish_interaction_transition(session_id, **kwargs)
        if kwargs["event"].type != EventType.INTERACTION_COMPLETED:
            return result
        self.attempts += 1
        if self.attempts != 1:
            return result
        if self.corruption == "wrong-type":
            return object()
        if self.corruption == "event":
            return result.model_copy(
                update={
                    "event": result.event.model_copy(update={"id": "conflicting-event"}),
                }
            )
        if self.corruption == "session":
            return result.model_copy(
                update={
                    "session": result.session.model_copy(update={"id": "other-session"}),
                }
            )
        if self.corruption == "status":
            return result.model_copy(
                update={
                    "session": result.session.model_copy(update={"status": SessionStatus.RUNNING}),
                }
            )
        if self.corruption == "queue-outcome":
            return result.model_copy(update={"status_changed": False})
        if self.corruption == "malformed":
            return InteractionTransitionResult.model_construct(
                session=result.session,
                event=result.event,
                status_changed="yes",
                replayed=False,
            )
        raise AssertionError(f"Unknown result corruption: {self.corruption}")


class CancelledCorruptInteractionTransitionResultStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.second_attempt_dispatched = asyncio.Event()
        self.release_second_attempt = asyncio.Event()
        self.attempts = 0

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        if kwargs["event"].type != EventType.INTERACTION_COMPLETED:
            return await super().publish_interaction_transition(session_id, **kwargs)
        self.attempts += 1
        if self.attempts == 1:
            await super().publish_interaction_transition(session_id, **kwargs)
            raise ConnectionError("completion acknowledgement lost")
        if self.attempts == 2:
            self.second_attempt_dispatched.set()
            await self.release_second_attempt.wait()
            result = await super().publish_interaction_transition(session_id, **kwargs)
            return result.model_copy(
                update={
                    "event": result.event.model_copy(update={"id": "conflicting-event"}),
                }
            )
        return await super().publish_interaction_transition(session_id, **kwargs)


class ThreadDispatchedInteractionTransitionStore(InMemorySessionStore):
    """Model a cancellation-opaque store mutation dispatched to a worker thread."""

    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.dispatched = asyncio.Event()
        self.release = threading.Event()
        self.cleanup_transition_started = asyncio.Event()
        self.attempted_events: list[Event] = []

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        event = kwargs["event"]
        if event.type == EventType.INTERACTION_FAILED:
            self.cleanup_transition_started.set()
        if event.type != EventType.INTERACTION_COMPLETED:
            return await super().publish_interaction_transition(session_id, **kwargs)

        self.attempted_events.append(event.model_copy(deep=True))
        loop = asyncio.get_running_loop()
        publish = super().publish_interaction_transition

        def dispatch_and_wait():
            loop.call_soon_threadsafe(self.dispatched.set)
            if not self.release.wait(timeout=5):
                raise TimeoutError("test store dispatch was not released")
            future = asyncio.run_coroutine_threadsafe(
                publish(session_id, **kwargs),
                loop,
            )
            return future.result(timeout=5)

        return await asyncio.to_thread(dispatch_and_wait)


class BlockingInteractionCompletionSink(EventSink):
    def __init__(self) -> None:
        self.completion_started = asyncio.Event()
        self.release = asyncio.Event()

    async def emit(self, event: Event) -> None:
        if event.type == EventType.INTERACTION_COMPLETED:
            self.completion_started.set()
            await self.release.wait()


class BlockingInteractionTransitionDiagnosticSink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []
        self.diagnostic_started = asyncio.Event()
        self.release = asyncio.Event()

    async def emit(self, event: Event) -> None:
        self.events.append(event.model_copy(deep=True))
        if event.type == EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED:
            self.diagnostic_started.set()
            await self.release.wait()


class ChildCancelledInteractionTransitionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        if kwargs["event"].type == EventType.INTERACTION_COMPLETED:
            self.attempts += 1
            if self.attempts == 1:
                raise asyncio.CancelledError("store child cancelled")
        return await super().publish_interaction_transition(session_id, **kwargs)


class GroupedChildCancelledInteractionTransitionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        if kwargs["event"].type == EventType.INTERACTION_COMPLETED:
            self.attempts += 1
            if self.attempts == 1:
                raise BaseExceptionGroup(
                    "store child failures",
                    [
                        ExceptionGroup(
                            "ordinary store child failure",
                            [ConnectionError("store child acknowledgement lost")],
                        ),
                        asyncio.CancelledError("store child cancelled"),
                    ],
                )
        return await super().publish_interaction_transition(session_id, **kwargs)


class CancelledInteractionTransitionFailureStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.failures: list[Exception] = [
            ConnectionError("acknowledgement lost"),
            PermissionError("connection replaced"),
        ]
        self.second_attempt_dispatched = asyncio.Event()
        self.release_second_attempt = asyncio.Event()
        self.attempts = 0

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        if kwargs["event"].type == EventType.INTERACTION_COMPLETED:
            failure = self.failures[self.attempts]
            self.attempts += 1
            if self.attempts == 2:
                self.second_attempt_dispatched.set()
                await self.release_second_attempt.wait()
            raise failure
        return await super().publish_interaction_transition(session_id, **kwargs)


class PostCommitCancelledSQLiteInteractionTransitionStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, path) -> None:
        super().__init__(path)
        self.failures: list[Exception] = [
            ConnectionError("acknowledgement lost after commit"),
            PermissionError("connection replaced during replay"),
        ]
        self.second_attempt_dispatched = asyncio.Event()
        self.release_second_attempt = asyncio.Event()
        self.attempts = 0

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        if kwargs["event"].type != EventType.INTERACTION_COMPLETED:
            return await super().publish_interaction_transition(session_id, **kwargs)
        self.attempts += 1
        if self.attempts == 1:
            await super().publish_interaction_transition(session_id, **kwargs)
            raise self.failures[0]
        if self.attempts == 2:
            self.second_attempt_dispatched.set()
            await self.release_second_attempt.wait()
            raise self.failures[1]
        return await super().publish_interaction_transition(session_id, **kwargs)


class PostSettlementFenceSQLiteInteractionTransitionStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, path) -> None:
        super().__init__(path)
        self.second_attempt_dispatched = asyncio.Event()
        self.release_second_attempt = asyncio.Event()
        self.receipt_loaded = asyncio.Event()
        self.release_receipt = asyncio.Event()
        self.attempts = 0

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        if kwargs["event"].type != EventType.INTERACTION_COMPLETED:
            return await super().publish_interaction_transition(session_id, **kwargs)
        self.attempts += 1
        if self.attempts == 1:
            await super().publish_interaction_transition(session_id, **kwargs)
            raise ConnectionError("completion acknowledgement lost before cancellation")
        if self.attempts == 2:
            self.second_attempt_dispatched.set()
            await self.release_second_attempt.wait()
        return await super().publish_interaction_transition(session_id, **kwargs)

    async def load_interaction_transition_receipt(self, session_id: str, **kwargs):
        receipt = await super().load_interaction_transition_receipt(session_id, **kwargs)
        self.receipt_loaded.set()
        await self.release_receipt.wait()
        return receipt


class PrunablePostCommitSQLiteInteractionTransitionStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, path) -> None:
        super().__init__(path)
        self.failures: list[Exception] = [
            ConnectionError("setup-failure acknowledgement lost after commit"),
            PermissionError("setup-failure replay acknowledgement lost"),
        ]
        self.first_attempt_dispatched = asyncio.Event()
        self.release_first_attempt = asyncio.Event()
        self.second_attempt_dispatched = asyncio.Event()
        self.release_second_attempt = asyncio.Event()
        self.receipt_lookup_started = asyncio.Event()
        self.release_receipt_lookup = asyncio.Event()
        self.attempts = 0

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        if kwargs["event"].type != EventType.INTERACTION_FAILED:
            return await super().publish_interaction_transition(session_id, **kwargs)
        self.attempts += 1
        if self.attempts == 1:
            self.first_attempt_dispatched.set()
            await self.release_first_attempt.wait()
            await super().publish_interaction_transition(session_id, **kwargs)
            raise self.failures[0]
        if self.attempts == 2:
            self.second_attempt_dispatched.set()
            await self.release_second_attempt.wait()
            raise self.failures[1]
        return await super().publish_interaction_transition(session_id, **kwargs)

    async def load_interaction_transition_receipt(self, session_id: str, **kwargs):
        self.receipt_lookup_started.set()
        await self.release_receipt_lookup.wait()
        return await super().load_interaction_transition_receipt(session_id, **kwargs)


class ConflictingInteractionTransitionReceiptStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.second_attempt_dispatched = asyncio.Event()
        self.release_second_attempt = asyncio.Event()
        self.attempts = 0

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        if kwargs["event"].type != EventType.INTERACTION_COMPLETED:
            return await super().publish_interaction_transition(session_id, **kwargs)
        self.attempts += 1
        if self.attempts == 1:
            await super().publish_interaction_transition(session_id, **kwargs)
            raise ConnectionError("completion acknowledgement lost")
        if self.attempts == 2:
            self.second_attempt_dispatched.set()
            await self.release_second_attempt.wait()
            raise PermissionError("completion replay acknowledgement lost")
        return await super().publish_interaction_transition(session_id, **kwargs)

    async def load_interaction_transition_receipt(self, session_id: str, **kwargs):
        receipt = await super().load_interaction_transition_receipt(session_id, **kwargs)
        if receipt is None:
            return None
        conflicting = InteractionTransitionSpec(
            event=receipt.transition.event,
            from_statuses=(SessionStatus.INTERRUPTING,),
            to_status=receipt.transition.to_status,
            only_if_no_queued_messages=(receipt.transition.only_if_no_queued_messages),
        )
        return InteractionTransitionReceiptResult(
            session=receipt.session,
            transition=conflicting,
            status_changed=receipt.status_changed,
        )


class IncoherentInteractionTransitionReceiptStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.second_attempt_dispatched = asyncio.Event()
        self.release_second_attempt = asyncio.Event()
        self.attempts = 0

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        if kwargs["event"].type != EventType.INTERACTION_COMPLETED:
            return await super().publish_interaction_transition(session_id, **kwargs)
        self.attempts += 1
        if self.attempts == 1:
            await super().publish_interaction_transition(session_id, **kwargs)
            raise ConnectionError("completion acknowledgement lost")
        if self.attempts == 2:
            self.second_attempt_dispatched.set()
            await self.release_second_attempt.wait()
            raise PermissionError("completion replay acknowledgement lost")
        return await super().publish_interaction_transition(session_id, **kwargs)

    async def load_interaction_transition_receipt(self, session_id: str, **kwargs):
        receipt = await super().load_interaction_transition_receipt(session_id, **kwargs)
        if receipt is None:
            return None
        inconsistent_session = receipt.session.model_copy(update={"status": SessionStatus.RUNNING})
        return InteractionTransitionReceiptResult.model_construct(
            session=inconsistent_session,
            transition=receipt.transition,
            status_changed=True,
            replayed=True,
        )


class DistinctInteractionTransitionFailureStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.nested_failures: list[Exception] = [
            TimeoutError("receipt read timed out"),
            ValueError("cleanup failed"),
        ]
        self.failures: list[Exception] = [
            ConnectionError("acknowledgement lost"),
            ExceptionGroup(
                "receipt and cleanup failed",
                self.nested_failures,
            ),
            PermissionError("connection replaced"),
        ]
        self.attempts = 0

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        if kwargs["event"].type == EventType.INTERACTION_COMPLETED:
            failure = self.failures[self.attempts]
            self.attempts += 1
            raise failure
        return await super().publish_interaction_transition(session_id, **kwargs)


class ReusedInteractionTransitionFailureStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.failure = ConnectionError("same acknowledgement failure")
        self.attempts = 0

    async def publish_interaction_transition(self, session_id: str, **kwargs):
        if kwargs["event"].type == EventType.INTERACTION_COMPLETED:
            self.attempts += 1
            raise self.failure
        return await super().publish_interaction_transition(session_id, **kwargs)


@pytest.mark.parametrize(
    ("provider_type", "expected_status", "expected_event_type"),
    [
        (CompletingProvider, SessionStatus.COMPLETED, EventType.INTERACTION_COMPLETED),
        (FailingProvider, SessionStatus.FAILED, EventType.INTERACTION_FAILED),
    ],
)
@pytest.mark.parametrize("lost_acknowledgements", [1, 2])
def test_runtime_reconstructs_interaction_transition_after_commit_acknowledgement_loss(
    provider_type: type[CompletingProvider] | type[FailingProvider],
    expected_status: SessionStatus,
    expected_event_type: EventType,
    lost_acknowledgements: int,
) -> None:
    async def run() -> None:
        store = CommitThenLoseInteractionTransitionAcknowledgementStore(
            lost_acknowledgements=lost_acknowledgements
        )
        provider = provider_type()
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
        assert store.remaining_lost_acknowledgements == 0
        assert len(store.attempted_events) == lost_acknowledgements + 1
        assert all(event == store.attempted_events[0] for event in store.attempted_events)
        durable = await store.load_events(session.id)
        assert sum(event.type == expected_event_type for event in durable) == 1
        assert sum(event.type == expected_event_type for event in events) == 1
        assert len(provider.requests) == 1
        attempted_settlements = store.attempted_model_completion_stage_settlements
        assert len(attempted_settlements) == lost_acknowledgements + 1
        if provider_type is FailingProvider:
            settlement_request = attempted_settlements[0]
            assert settlement_request is not None
            assert all(attempted == settlement_request for attempted in attempted_settlements)
            assert (
                settlement_request.disposition
                is sessions_module.ModelCompletionStageDisposition.PROVIDER_EFFECT_OUTCOME_UNKNOWN
            )
            assert await store.load_active_model_completion_stage(session.id) is None
            settlement = await store.load_model_completion_stage_settlement(
                session.id,
                settlement_request.stage_id,
            )
            assert settlement is not None
            assert settlement.stage_id == settlement_request.stage_id
            assert settlement.reason_code == "model_attempt_failed"
        else:
            assert all(attempted is None for attempted in attempted_settlements)

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure_factory",
    [
        pytest.param(
            lambda: ValueError("acknowledgement decoding failed"),
            id="value-error",
        ),
        pytest.param(
            lambda: ExceptionGroup(
                "acknowledgement and cleanup failed",
                [
                    ConnectionError("acknowledgement lost"),
                    ValueError("cleanup validation failed"),
                ],
            ),
            id="mixed-exception-group",
        ),
    ],
)
def test_runtime_replays_untyped_post_commit_acknowledgement_failures(
    failure_factory: Callable[[], Exception],
) -> None:
    async def run() -> None:
        store = CommitThenLoseInteractionTransitionAcknowledgementStore(
            lost_acknowledgements=2,
            failure_factory=failure_factory,
        )
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_untyped_post_commit_transition_failure"

        emitted = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            )
        ]

        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status == SessionStatus.COMPLETED
        assert emitted[-1].type == EventType.SESSION_COMPLETED
        assert len(store.attempted_events) == 3
        assert all(event == store.attempted_events[0] for event in store.attempted_events)
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_runtime_replay_uses_store_isolated_transition_snapshots() -> None:
    async def run() -> None:
        store = CommitMutateThenLoseInteractionTransitionStore()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_store_isolated_transition_replay"

        emitted = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            )
        ]

        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status == SessionStatus.COMPLETED
        assert emitted[-1].type == EventType.SESSION_COMPLETED
        assert len(store.attempted_events) == 2
        assert store.attempted_events[0] == store.attempted_events[1]
        assert store.attempted_from_statuses == [
            {SessionStatus.RUNNING},
            {SessionStatus.RUNNING},
        ]
        assert store.attempted_event_objects[0] is not store.attempted_event_objects[1]
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert len(provider.requests) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "corruption",
    ["wrong-type", "event", "session", "status", "queue-outcome", "malformed"],
)
def test_runtime_replays_invalid_interaction_transition_publication_results(
    corruption: str,
) -> None:
    async def run() -> None:
        store = CorruptThenReplayInteractionTransitionResultStore(corruption)
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = f"sess_invalid_transition_result_{corruption}"

        emitted = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            )
        ]

        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert store.attempts == 2
        assert emitted[-1].type is EventType.SESSION_COMPLETED
        assert sum(event.type is EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert sum(event.type is EventType.INTERACTION_COMPLETED for event in emitted) == 1
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_runtime_cancellation_reconciles_corrupt_transition_publication_result() -> None:
    async def run() -> None:
        store = CancelledCorruptInteractionTransitionResultStore()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_cancelled_corrupt_transition_result"

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(store.second_attempt_dispatched.wait(), timeout=5)
        consumer.cancel("cancel before corrupt transition result")
        assert consumer.cancelling() == 1
        await asyncio.sleep(0)
        assert consumer.done() is False
        store.release_second_attempt.set()

        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(consumer, timeout=5)

        assert raised.value.args == ("cancel before corrupt transition result",)
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 0
        cause = exception_cause(raised.value)
        assert cause is not None
        assert (
            sum(
                isinstance(error, RuntimeError) and "returned a conflicting event" in str(error)
                for error in iter_exception_tree(cause)
            )
            == 1
        )
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert store.attempts == 2
        assert sum(event.type is EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert sum(event.type is EventType.INTERACTION_INTERRUPTED for event in durable) == 0
        assert sum(event.type is EventType.SESSION_INTERRUPTED for event in durable) == 0
        diagnostics = [
            event
            for event in durable
            if event.type is EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED
        ]
        assert len(diagnostics) == 1
        assert diagnostics[0].payload["interaction_transition_failures"] == [
            {
                "error": "completion acknowledgement lost",
                "error_type": "ConnectionError",
            },
            {
                "error": "Interaction transition publication returned a conflicting event.",
                "error_type": "RuntimeError",
            },
        ]
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_runtime_fails_closed_after_bounded_post_commit_acknowledgement_loss() -> None:
    async def run() -> None:
        store = CommitThenLoseInteractionTransitionAcknowledgementStore(lost_acknowledgements=3)
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_exhausted_post_commit_transition_replay"

        emitted = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            )
        ]

        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.FAILED
        assert store.remaining_lost_acknowledgements == 0
        assert len(store.attempted_events) == 3
        assert all(event == store.attempted_events[0] for event in store.attempted_events)
        terminal_index = next(
            index
            for index, event in enumerate(durable)
            if event.type == EventType.INTERACTION_COMPLETED
        )
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert all(event.interaction_id is None for event in durable[terminal_index + 1 :])
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in emitted) == 0
        assert emitted[-1].type == EventType.SESSION_FAILED
        assert len(provider.requests) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    ("replay_window_seconds", "expected_attempts"),
    [(30.0, 3), (0.0, 1)],
)
def test_runtime_bounds_ambiguous_interaction_transition_replay(
    monkeypatch: pytest.MonkeyPatch,
    replay_window_seconds: float,
    expected_attempts: int,
) -> None:
    async def run() -> None:
        store = RejectBeforeInteractionTransitionCommitStore(
            failure_type=ConnectionError,
            failure_message="interaction transition unavailable",
        )
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = f"sess_bounded_transition_replay_{expected_attempts}"

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            )
        ]

        session = await store.load(session_id)
        assert session is not None
        assert session.status is SessionStatus.FAILED
        assert len(store.attempted_events) == expected_attempts
        assert all(event == store.attempted_events[0] for event in store.attempted_events)
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in events) == 0
        assert sum(event.type == EventType.INTERACTION_FAILED for event in events) == 1
        assert len(provider.requests) == 1

    monkeypatch.setattr(
        session_engine_module,
        "_INTERACTION_TRANSITION_REPLAY_WINDOW_SECONDS",
        replay_window_seconds,
    )
    asyncio.run(run())


def test_runtime_does_not_replay_authoritatively_rejected_interaction_transition() -> None:
    async def run() -> None:
        store = RejectBeforeInteractionTransitionCommitStore(
            failure_type=SessionStatusConflict,
            failure_message="transition rejected",
        )
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_authoritative_transition_rejection"

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            )
        ]

        session = await store.load(session_id)
        assert session is not None
        assert session.status is SessionStatus.FAILED
        assert len(store.attempted_events) == 1
        assert sum(event.type == EventType.INTERACTION_FAILED for event in events) == 1
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_runtime_cancellation_after_transition_commit_waits_for_store_settlement() -> None:
    async def run() -> None:
        store = CommitAndBlockInteractionTransitionStore(fail_after_release=False)
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_cancel_committed_transition"

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(store.committed.wait(), timeout=5)
        consumer.cancel("cancel after interaction transition commit")
        assert consumer.cancelling() == 1
        await asyncio.sleep(0)
        assert consumer.done() is False
        assert len(store.attempted_events) == 1
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 0
        assert len(store.attempted_events) == 1
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert sum(event.type == EventType.INTERACTION_INTERRUPTED for event in durable) == 0
        assert sum(event.type == EventType.SESSION_INTERRUPTED for event in durable) == 0
        assert len(provider.requests) == 1

    asyncio.run(run())


@pytest.mark.parametrize("commit_before_release", [False, True])
def test_setup_failure_transition_cancellation_is_reconciled_at_sibling_boundary(
    monkeypatch: pytest.MonkeyPatch,
    commit_before_release: bool,
) -> None:
    async def run() -> None:
        terminal_profiles: list[object | None] = []

        class RecordingInterruptedHook(RuntimeHook):
            async def after_session_interrupted(self, context: RuntimeHookContext) -> None:
                terminal_profiles.append(context.execution_profile)

        store = BlockingSiblingInteractionTransitionStore(
            commit_before_release=commit_before_release
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider(), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            runtime_hooks=[RecordingInterruptedHook()],
        )
        session_id = f"sess_cancel_setup_failure_{commit_before_release}"
        transition_profiles: list[object | None] = []
        original_publish_transition = app._session_engine._publish_sibling_interaction_transition

        async def capture_transition_profile(**kwargs):
            transition_profiles.append(kwargs.get("execution_profile"))
            return await original_publish_transition(**kwargs)

        monkeypatch.setattr(
            app._session_engine,
            "_publish_sibling_interaction_transition",
            capture_transition_profile,
        )

        async def fail_initial_transcript_publication(*args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError("workspace setup failed")

        monkeypatch.setattr(
            store,
            "replace_initial_transcript_messages",
            fail_initial_transcript_publication,
        )

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(store.dispatched.wait(), timeout=5)
        consumer.cancel("cancel setup failure transition")
        assert consumer.cancelling() == 1
        await asyncio.sleep(0)
        assert consumer.done() is False
        store.release.set()

        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(consumer, timeout=5)

        assert raised.value.args == ("cancel setup failure transition",)
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 0
        assert (
            session_engine_module._INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE
            not in raised.value.__dict__
        )
        assert (
            session_engine_module._INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE
            not in raised.value.__dict__
        )
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        expected_status = (
            SessionStatus.FAILED if commit_before_release else SessionStatus.INTERRUPTED
        )
        assert session.status is expected_status
        expected_interaction_type = (
            EventType.INTERACTION_FAILED
            if commit_before_release
            else EventType.INTERACTION_INTERRUPTED
        )
        assert sum(event.type == expected_interaction_type for event in durable) == 1, [
            str(event.type) for event in durable
        ]
        assert (
            sum(
                event.type
                == (
                    EventType.INTERACTION_INTERRUPTED
                    if commit_before_release
                    else EventType.INTERACTION_FAILED
                )
                for event in durable
            )
            == 0
        )
        assert len(store.attempted_events) == 1
        assert transition_profiles[0] is not None
        if commit_before_release:
            assert len(transition_profiles) == 1
            assert sum(event.type == EventType.SESSION_INTERRUPTED for event in durable) == 0
            assert terminal_profiles == []
        else:
            assert len(transition_profiles) == 2
            assert all(profile is transition_profiles[0] for profile in transition_profiles)
            interrupted = next(
                event for event in durable if event.type == EventType.SESSION_INTERRUPTED
            )
            assert interrupted.payload["interaction_transition_failures"] == [
                {
                    "error": "setup transition rejected before commit",
                    "error_type": "ConnectionError",
                }
            ]
            assert terminal_profiles == [transition_profiles[0]]
            assert terminal_profiles[0] is transition_profiles[0]

    asyncio.run(run())


@pytest.mark.parametrize("run_fence_lost", [False, True])
def test_interrupt_transition_cancellation_consumes_exact_settlement_handoff(
    run_fence_lost: bool,
) -> None:
    async def run() -> None:
        store = CommitThenLoseAndBlockInterruptedTransitionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = f"sess_interrupt_transition_cancellation_{run_fence_lost}"
        interaction_id = f"interaction-interrupt-cancellation-{run_fence_lost}"
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
            identity=SessionIdentity(provider_name="completing", model="fake-model"),
        )
        started_at = datetime.now(UTC) - timedelta(seconds=1)
        started = Event(
            id=f"event-interrupt-cancellation-started-{run_fence_lost}",
            type=EventType.INTERACTION_STARTED,
            session_id=session_id,
            interaction_id=interaction_id,
            timestamp=started_at,
            payload=InteractionSummaryEvidence(
                status=InteractionStatus.ACTIVE,
                start_event_id=f"event-interrupt-cancellation-started-{run_fence_lost}",
                started_at=started_at,
            ).model_dump(mode="json"),
        )
        await store.append_event(session_id, started)
        sessions_module._activate_session_interaction(session_id, interaction_id)
        sessions_module._activate_session_run_fence(session)

        async def interrupt() -> None:
            async for _ in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="operator cancellation settlement test",
                )
            ):
                pass

        try:
            consumer = asyncio.create_task(interrupt())
            await asyncio.wait_for(store.second_attempt_dispatched.wait(), timeout=5)
            consumer.cancel("cancel interrupted transition replay")
            assert consumer.cancelling() == 1
            await asyncio.sleep(0)
            assert consumer.done() is False
            if run_fence_lost:
                resumed = await store.transition_status(
                    session_id,
                    from_statuses={SessionStatus.INTERRUPTED},
                    to_status=SessionStatus.RUNNING,
                )
                assert resumed.run_epoch == session.run_epoch + 1
            store.release_second_attempt.set()

            with pytest.raises(asyncio.CancelledError) as raised:
                await asyncio.wait_for(consumer, timeout=5)

            assert raised.value.args == ("cancel interrupted transition replay",)
            assert consumer.cancelled() is True
            assert consumer.cancelling() == 0
            assert (
                session_engine_module._INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE
                not in raised.value.__dict__
            )
            assert (
                session_engine_module._INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE
                not in raised.value.__dict__
            )
            assert (
                app._session_engine._session_control.is_interruption_request_active(session_id)
                is False
            )
            durable = await store.load_events(session_id)
            persisted = await store.load(session_id)
            assert persisted is not None
            assert persisted.status is (
                SessionStatus.RUNNING if run_fence_lost else SessionStatus.INTERRUPTED
            )
            assert store.attempts == 2
            assert sum(event.type is EventType.INTERACTION_INTERRUPTED for event in durable) == 1
            assert sum(event.type is EventType.SESSION_INTERRUPTED for event in durable) == 0
            diagnostics = [
                event
                for event in durable
                if event.type is EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED
            ]
            assert len(diagnostics) == (0 if run_fence_lost else 1)
            if diagnostics:
                assert diagnostics[0].payload["interaction_transition_failures"] == [
                    {
                        "error": "interruption transition acknowledgement lost",
                        "error_type": "ConnectionError",
                    }
                ]
        finally:
            sessions_module._deactivate_session_interaction(session_id)
            sessions_module._deactivate_session_run_fence(session_id)

    asyncio.run(run())


async def _approval_limit_request(
    app: CayuApp,
    *,
    session_id: str,
) -> ToolApprovalRequest:
    initial_events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run both limited tools")],
                limits=RunLimits(max_tool_calls=1),
            )
        )
    ]
    approval_event = next(
        event for event in initial_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval = approval_event.payload["approval"]
    return ToolApprovalRequest(
        session_id=session_id,
        approval_id=approval["approval_id"],
        tool_round_id=approval["tool_round_id"],
        tool_call_id=approval["tool_call_id"],
        decision=ToolApprovalDecision.APPROVE,
        limits=RunLimits(max_tool_calls=1),
    )


def test_recovery_limit_cancellation_reconciles_precommit_transition_failures() -> None:
    async def run() -> None:
        store = FailThenBlockInterruptedTransitionStore()
        provider = ApprovalLimitProvider()
        tool = LimitedSideEffectTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireLimitedToolApprovalPolicy(),
        )
        session_id = "sess_recovery_limit_precommit_cancellation"
        request = await _approval_limit_request(app, session_id=session_id)
        baseline_event_ids = {event.id for event in await store.load_events(session_id)}

        async def resolve() -> None:
            async for _ in app.resolve_tool_approval(request):
                pass

        consumer = asyncio.create_task(resolve())
        await asyncio.wait_for(store.second_attempt_dispatched.wait(), timeout=5)
        consumer.cancel("cancel recovery limit before transition commit")
        assert consumer.cancelling() == 1
        await asyncio.sleep(0)
        assert consumer.done() is False
        store.release_second_attempt.set()

        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(consumer, timeout=5)

        assert raised.value.args == ("cancel recovery limit before transition commit",)
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 0
        assert (
            session_engine_module._INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE
            not in raised.value.__dict__
        )
        assert (
            session_engine_module._INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE
            not in raised.value.__dict__
        )
        cause = exception_cause(raised.value)
        assert cause is not None
        assert (
            sum(
                isinstance(error, ConnectionError)
                and str(error) == "limit interruption rejected before commit"
                for error in iter_exception_tree(cause)
            )
            == 1
        )
        assert (
            sum(
                isinstance(error, TimeoutError)
                and str(error) == "limit interruption replay rejected before commit"
                for error in iter_exception_tree(cause)
            )
            == 1
        )
        persisted = await store.load(session_id)
        durable = await store.load_events(session_id)
        recovery_events = [event for event in durable if event.id not in baseline_event_ids]
        assert persisted is not None
        assert persisted.status is SessionStatus.INTERRUPTED
        # Two ambiguous transition attempts settle before abandonment cleanup
        # publishes the distinct terminal transition under the same owner.
        assert store.attempts == 3
        assert len(provider.requests) == 2
        assert tool.calls == [{"ordinal": 1}]
        assert sum(event.type == EventType.SESSION_LIMIT_REACHED for event in recovery_events) == 1
        assert (
            sum(event.type == EventType.INTERACTION_INTERRUPTED for event in recovery_events) == 1
        )
        diagnostics = [
            event
            for event in recovery_events
            if event.type == EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED
        ]
        assert diagnostics == []
        interrupted = [
            event for event in recovery_events if event.type == EventType.SESSION_INTERRUPTED
        ]
        assert len(interrupted) == 1
        assert interrupted[0].payload["interaction_transition_failures"] == [
            {
                "error": "limit interruption rejected before commit",
                "error_type": "ConnectionError",
            },
            {
                "error": "limit interruption replay rejected before commit",
                "error_type": "TimeoutError",
            },
        ]

    asyncio.run(run())


@pytest.mark.parametrize("run_fence_lost", [False, True])
def test_recovery_limit_cancellation_consumes_committed_transition_handoff(
    run_fence_lost: bool,
) -> None:
    async def run() -> None:
        store = CommitThenLoseAndBlockInterruptedTransitionStore()
        provider = ApprovalLimitProvider()
        tool = LimitedSideEffectTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireLimitedToolApprovalPolicy(),
        )
        session_id = f"sess_recovery_limit_committed_cancellation_{run_fence_lost}"
        request = await _approval_limit_request(app, session_id=session_id)
        baseline_event_ids = {event.id for event in await store.load_events(session_id)}

        async def resolve() -> None:
            async for _ in app.resolve_tool_approval(request):
                pass

        consumer = asyncio.create_task(resolve())
        await asyncio.wait_for(store.second_attempt_dispatched.wait(), timeout=5)
        consumer.cancel("cancel recovery limit after transition commit")
        assert consumer.cancelling() == 1
        await asyncio.sleep(0)
        assert consumer.done() is False
        committed = await store.load(session_id)
        assert committed is not None
        assert committed.status is SessionStatus.INTERRUPTED
        if run_fence_lost:
            resumed = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.INTERRUPTED},
                to_status=SessionStatus.RUNNING,
            )
            assert resumed.run_epoch == committed.run_epoch + 1
        store.release_second_attempt.set()

        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(consumer, timeout=5)

        assert raised.value.args == ("cancel recovery limit after transition commit",)
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 0
        assert (
            session_engine_module._INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE
            not in raised.value.__dict__
        )
        assert (
            session_engine_module._INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE
            not in raised.value.__dict__
        )
        persisted = await store.load(session_id)
        durable = await store.load_events(session_id)
        recovery_events = [event for event in durable if event.id not in baseline_event_ids]
        assert persisted is not None
        assert persisted.status is (
            SessionStatus.RUNNING if run_fence_lost else SessionStatus.INTERRUPTED
        )
        assert store.attempts == 2
        assert len(provider.requests) == 2
        assert tool.calls == [{"ordinal": 1}]
        assert sum(event.type == EventType.SESSION_LIMIT_REACHED for event in recovery_events) == 1
        assert (
            sum(event.type == EventType.INTERACTION_INTERRUPTED for event in recovery_events) == 1
        )
        assert sum(event.type == EventType.SESSION_INTERRUPTED for event in recovery_events) == 0
        diagnostics = [
            event
            for event in recovery_events
            if event.type == EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED
        ]
        assert len(diagnostics) == (0 if run_fence_lost else 1)
        if diagnostics:
            assert diagnostics[0].payload["interaction_transition_failures"] == [
                {
                    "error": "interruption transition acknowledgement lost",
                    "error_type": "ConnectionError",
                }
            ]
        cause = exception_cause(raised.value)
        assert cause is not None
        assert (
            sum(
                isinstance(error, ConnectionError)
                and str(error) == "interruption transition acknowledgement lost"
                for error in iter_exception_tree(cause)
            )
            == 1
        )
        assert sum(isinstance(error, SessionRunFenced) for error in iter_exception_tree(cause)) == (
            1 if run_fence_lost else 0
        )

    asyncio.run(run())


def test_runtime_cancellation_during_transition_fanout_preserves_committed_outcome() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = CompletingProvider()
        sink = BlockingInteractionCompletionSink()
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_cancel_transition_fanout"

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(sink.completion_started.wait(), timeout=5)
        consumer.cancel("cancel interaction transition fanout")

        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(consumer, timeout=5)

        assert raised.value.args == ("cancel interaction transition fanout",)
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 1
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert sum(event.type == EventType.INTERACTION_INTERRUPTED for event in durable) == 0
        assert sum(event.type == EventType.SESSION_INTERRUPTED for event in durable) == 0
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_runtime_stream_close_after_completed_interaction_preserves_outcome() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_close_after_interaction_completion"
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            )
        )

        while True:
            event = await anext(stream)
            if event.type == EventType.INTERACTION_COMPLETED:
                break
        await stream.aclose()

        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert sum(event.type == EventType.INTERACTION_INTERRUPTED for event in durable) == 0
        assert sum(event.type == EventType.SESSION_INTERRUPTED for event in durable) == 0
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_runtime_cross_task_stream_close_after_completed_interaction_preserves_outcome() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_cross_task_close_after_interaction_completion"
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            )
        )

        async def advance_to_interaction_completion() -> Event:
            while True:
                event = await anext(stream)
                if event.type == EventType.INTERACTION_COMPLETED:
                    return event

        completed = await asyncio.create_task(advance_to_interaction_completion())
        assert completed.type is EventType.INTERACTION_COMPLETED
        close_task = asyncio.create_task(stream.aclose())
        await close_task

        assert close_task.cancelled() is False
        assert close_task.cancelling() == 0
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert sum(event.type == EventType.INTERACTION_INTERRUPTED for event in durable) == 0
        assert sum(event.type == EventType.SESSION_INTERRUPTED for event in durable) == 0
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_new_interaction_invocation_clears_prior_settlement_evidence() -> None:
    session_id = "sess_reused_recovery_interaction"
    interaction_id = "interaction_reused_after_recovery"

    sessions_module._activate_session_interaction(session_id, interaction_id)
    sessions_module._mark_session_interaction_settled(session_id, interaction_id)
    assert sessions_module._latest_session_invocation_interaction_is_settled(session_id) is True

    sessions_module._deactivate_session_interaction(session_id)
    sessions_module._activate_session_interaction(session_id, interaction_id)
    try:
        assert (
            sessions_module._latest_session_invocation_interaction_is_settled(session_id) is False
        )
    finally:
        sessions_module._deactivate_session_interaction(session_id)


def test_runtime_cancellation_preserves_queue_guarded_running_transition() -> None:
    async def run() -> None:
        store = QueueGuardedInteractionTransitionStore()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        accepting_process = CayuApp(session_store=store, enable_logging=False)
        session_id = "sess_cancel_queue_guarded_transition"

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(store.dispatched.wait(), timeout=5)
        await accepting_process.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id=session_id,
                idempotency_key="queue-before-completion",
                content="continue after this response",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
        )
        consumer.cancel("cancel queue-guarded transition")
        assert consumer.cancelling() == 1
        await asyncio.sleep(0)
        assert consumer.done() is False
        store.release.set()

        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(consumer, timeout=5)

        assert raised.value.args == ("cancel queue-guarded transition",)
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 0
        assert len(store.attempted_events) == 1
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.RUNNING
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert sum(event.type == EventType.INTERACTION_INTERRUPTED for event in durable) == 0
        assert sum(event.type == EventType.SESSION_INTERRUPTED for event in durable) == 0
        assert sum(event.type == EventType.SESSION_MESSAGE_QUEUED for event in durable) == 1
        assert sum(event.type == EventType.SESSION_MESSAGE_DELIVERED for event in durable) == 0
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_runtime_cancellation_clears_transition_handoff_after_run_fence_loss() -> None:
    async def run() -> None:
        store = QueueGuardedInteractionTransitionStore()
        task_store = InMemoryTaskStore()
        provider = CompletingProvider()
        app = CayuApp(
            session_store=store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_cancel_transition_run_fenced"
        task_id = "task_cancel_transition_run_fenced"
        await task_store.create_task(
            TaskCreate(
                task_id=task_id,
                type="respond",
                assigned_agent_name="assistant",
            )
        )
        fence_acquired = asyncio.Event()
        release_fence = asyncio.Event()

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    task_id=task_id,
                    messages=[Message.text("user", "run")],
                )
            ):
                pass

        async def transfer_fence() -> None:
            fenced = await store.fence_stalled_run(
                session_id,
                statuses={SessionStatus.RUNNING},
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
            assert fenced is not None
            fence_acquired.set()
            await release_fence.wait()
            await store.release_run_fence(session_id)

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(store.dispatched.wait(), timeout=5)
        consumer.cancel("cancel while transition loses run fence")
        assert consumer.cancelling() == 1
        fencer = asyncio.create_task(transfer_fence())
        await asyncio.wait_for(fence_acquired.wait(), timeout=5)
        store.release.set()
        try:
            with pytest.raises(asyncio.CancelledError) as raised:
                await asyncio.wait_for(consumer, timeout=5)

            assert raised.value.args == ("cancel while transition loses run fence",)
            assert consumer.cancelled() is True
            assert consumer.cancelling() == 0
            assert (
                session_engine_module._INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE
                not in raised.value.__dict__
            )
            assert (
                session_engine_module._INTERACTION_TRANSITION_RUN_FENCE_ATTRIBUTE
                not in raised.value.__dict__
            )
            assert len(store.attempted_events) == 1
            session = await store.load(session_id)
            task = await task_store.load_task(task_id)
            durable = await store.load_events(session_id)
            assert session is not None
            assert session.status is SessionStatus.RUNNING
            assert task is not None
            assert task.status is TaskStatus.RUNNING
            assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 0
            assert sum(event.type == EventType.INTERACTION_FAILED for event in durable) == 0
            assert sum(event.type == EventType.SESSION_INTERRUPTED for event in durable) == 0
            assert len(provider.requests) == 1
        finally:
            release_fence.set()
            await fencer

    asyncio.run(run())


def test_runtime_cancellation_fences_thread_dispatched_transition_until_settled() -> None:
    async def run() -> None:
        store = ThreadDispatchedInteractionTransitionStore()
        task_store = InMemoryTaskStore()
        provider = CompletingProvider()
        sink = BlockingInteractionCompletionSink()
        app = CayuApp(
            session_store=store,
            task_store=task_store,
            event_sinks=[sink],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_cancel_thread_dispatched_transition"
        task_id = "task_cancel_thread_dispatched_transition"
        await task_store.create_task(
            TaskCreate(
                task_id=task_id,
                type="respond",
                assigned_agent_name="assistant",
            )
        )

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    task_id=task_id,
                    messages=[Message.text("user", "run")],
                )
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(store.dispatched.wait(), timeout=5)
        consumer.cancel("cancel opaque transition dispatch")
        assert consumer.cancelling() == 1
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert consumer.done() is False
        assert store.cleanup_transition_started.is_set() is False
        session_before_settlement = await store.load(session_id)
        task_before_settlement = await task_store.load_task(task_id)
        assert session_before_settlement is not None
        assert session_before_settlement.status is SessionStatus.RUNNING
        assert task_before_settlement is not None
        assert task_before_settlement.status is TaskStatus.RUNNING
        assert sink.completion_started.is_set() is False

        store.release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(consumer, timeout=5)

        assert raised.value.args == ("cancel opaque transition dispatch",)
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 0
        assert sink.completion_started.is_set() is False
        assert store.cleanup_transition_started.is_set() is False
        assert len(store.attempted_events) == 1
        session = await store.load(session_id)
        task = await task_store.load_task(task_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert task is not None
        assert task.status is TaskStatus.RUNNING
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert sum(event.type == EventType.INTERACTION_FAILED for event in durable) == 0
        assert sum(event.type == EventType.INTERACTION_INTERRUPTED for event in durable) == 0
        assert sum(event.type == EventType.SESSION_INTERRUPTED for event in durable) == 0
        assert len(provider.requests) == 1

        recovery = asyncio.create_task(app.recover_persisted_event_side_effects())
        await asyncio.wait_for(sink.completion_started.wait(), timeout=5)
        sink.release.set()
        recovered = await asyncio.wait_for(recovery, timeout=5)
        assert [event.type for event in recovered] == [EventType.INTERACTION_COMPLETED]

    asyncio.run(run())


def test_runtime_cancellation_durably_records_settled_transition_failures() -> None:
    async def run() -> None:
        store = CancelledInteractionTransitionFailureStore()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_cancelled_transition_failures"
        observed_cancellations: list[asyncio.CancelledError] = []

        async def consume() -> None:
            try:
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "run")],
                    )
                ):
                    pass
            except asyncio.CancelledError as cancellation:
                observed_cancellations.append(cancellation)
                raise

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(store.second_attempt_dispatched.wait(), timeout=5)
        consumer.cancel("cancel after ambiguous transition failures")
        assert consumer.cancelling() == 1
        await asyncio.sleep(0)
        assert consumer.done() is False
        store.release_second_attempt.set()

        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(consumer, timeout=5)

        assert raised.value.args == ("cancel after ambiguous transition failures",)
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 0
        assert store.attempts == 2
        assert len(observed_cancellations) == 1
        cancellation_cause = exception_cause(observed_cancellations[0])
        assert cancellation_cause is not None
        assert all(
            sum(candidate is failure for candidate in iter_exception_tree(cancellation_cause)) == 1
            for failure in store.failures
        )
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 0
        assert sum(event.type == EventType.INTERACTION_INTERRUPTED for event in durable) == 1
        interrupted = next(
            event for event in durable if event.type == EventType.SESSION_INTERRUPTED
        )
        assert interrupted.payload["interaction_transition_failures"] == [
            {
                "error": "acknowledgement lost",
                "error_type": "ConnectionError",
            },
            {
                "error": "connection replaced",
                "error_type": "PermissionError",
            },
        ]
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_runtime_cancellation_preserves_interrupted_diagnostic_hook_failure() -> None:
    class CancellingInterruptedHook(RuntimeHook):
        def __init__(self) -> None:
            self.failure = asyncio.CancelledError("interrupted diagnostic hook cancelled")

        async def after_session_interrupted(self, context: RuntimeHookContext) -> None:
            del context
            raise self.failure

    async def run() -> None:
        store = CancelledInteractionTransitionFailureStore()
        provider = CompletingProvider()
        hook = CancellingInterruptedHook()
        app = CayuApp(
            session_store=store,
            runtime_hooks=[hook],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_cancelled_transition_diagnostic_hook"
        observed_cancellations: list[asyncio.CancelledError] = []

        async def consume() -> None:
            try:
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "run")],
                    )
                ):
                    pass
            except asyncio.CancelledError as cancellation:
                observed_cancellations.append(cancellation)
                raise

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(store.second_attempt_dispatched.wait(), timeout=5)
        consumer.cancel("cancel before interrupted diagnostic publication")
        assert consumer.cancelling() == 1
        await asyncio.sleep(0)
        assert consumer.done() is False
        store.release_second_attempt.set()

        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(consumer, timeout=5)

        assert raised.value.args == ("cancel before interrupted diagnostic publication",)
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 0
        assert len(observed_cancellations) == 1
        cancellation_cause = exception_cause(observed_cancellations[0])
        assert cancellation_cause is not None
        assert all(
            sum(candidate is failure for candidate in iter_exception_tree(cancellation_cause)) == 1
            for failure in [*store.failures, hook.failure]
        )
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED
        interrupted = next(
            event for event in durable if event.type == EventType.SESSION_INTERRUPTED
        )
        assert interrupted.payload["interaction_transition_failures"] == [
            {
                "error": "acknowledgement lost",
                "error_type": "ConnectionError",
            },
            {
                "error": "connection replaced",
                "error_type": "PermissionError",
            },
        ]
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 0
        assert sum(event.type == EventType.INTERACTION_INTERRUPTED for event in durable) == 1
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_runtime_cancellation_after_post_commit_failure_records_durable_diagnostics(
    tmp_path,
) -> None:
    async def run() -> None:
        path = tmp_path / "post-commit-transition-cancellation.sqlite"
        store = PostCommitCancelledSQLiteInteractionTransitionStore(path)
        provider = CompletingProvider()
        sink = BlockingInteractionTransitionDiagnosticSink()
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_cancelled_post_commit_transition"
        observed_cancellations: list[asyncio.CancelledError] = []

        async def consume() -> None:
            try:
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "run")],
                    )
                ):
                    pass
            except asyncio.CancelledError as cancellation:
                observed_cancellations.append(cancellation)
                raise

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(store.second_attempt_dispatched.wait(), timeout=5)
        consumer.cancel("cancel after committed transition acknowledgement loss")
        assert consumer.cancelling() == 1
        await asyncio.sleep(0)
        assert consumer.done() is False
        store.release_second_attempt.set()

        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(consumer, timeout=5)

        assert raised.value.args == ("cancel after committed transition acknowledgement loss",)
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 0
        assert store.attempts == 2
        assert len(observed_cancellations) == 1
        cancellation_cause = exception_cause(observed_cancellations[0])
        assert cancellation_cause is not None
        assert all(
            sum(candidate is failure for candidate in iter_exception_tree(cancellation_cause)) == 1
            for failure in store.failures
        )
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert sum(event.type == EventType.INTERACTION_INTERRUPTED for event in durable) == 0
        assert sum(event.type == EventType.SESSION_INTERRUPTED for event in durable) == 0
        diagnostics = [
            event
            for event in durable
            if event.type == EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED
        ]
        assert len(diagnostics) == 1
        assert diagnostics[0].interaction_id is None
        assert diagnostics[0].payload == {
            "transition_event_type": str(EventType.INTERACTION_COMPLETED),
            "interaction_transition_failures": [
                {
                    "error": "acknowledgement lost after commit",
                    "error_type": "ConnectionError",
                },
                {
                    "error": "connection replaced during replay",
                    "error_type": "PermissionError",
                },
            ],
        }
        assert sink.diagnostic_started.is_set() is False
        public_diagnostics = [
            event
            for event in sink.events
            if event.type == EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED
        ]
        assert public_diagnostics == []

        recovery = asyncio.create_task(app.recover_persisted_event_side_effects())
        await asyncio.wait_for(sink.diagnostic_started.wait(), timeout=5)
        assert consumer.cancelled() is True
        sink.release.set()
        recovered = await asyncio.wait_for(recovery, timeout=5)
        assert (
            sum(
                event.type == EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED
                for event in recovered
            )
            == 1
        )
        public_diagnostics = [
            event
            for event in sink.events
            if event.type == EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED
        ]
        assert len(public_diagnostics) == 1
        assert public_diagnostics[0].interaction_id is None
        assert public_diagnostics[0].payload == diagnostics[0].payload
        assert len(provider.requests) == 1
        await store.close()

        reopened = SQLiteSessionStore(path)
        try:
            reconstructed_session = await reopened.load(session_id)
            reconstructed_events = await reopened.load_events(session_id)
            assert reconstructed_session is not None
            assert reconstructed_session.status is SessionStatus.COMPLETED
            reconstructed_diagnostics = [
                event
                for event in reconstructed_events
                if event.type == EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED
            ]
            assert [event.model_dump(mode="json") for event in reconstructed_diagnostics] == [
                diagnostics[0].model_dump(mode="json")
            ]
        finally:
            await reopened.close()

    asyncio.run(run())


def test_sqlite_cancellation_fences_diagnostic_after_transition_settles(
    tmp_path,
) -> None:
    async def run() -> None:
        store = PostSettlementFenceSQLiteInteractionTransitionStore(
            tmp_path / "post-settlement-diagnostic-fence.sqlite"
        )
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_post_settlement_diagnostic_fence"

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            ):
                pass

        consumer = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(store.second_attempt_dispatched.wait(), timeout=5)
            consumer.cancel("cancel before post-settlement receipt reconciliation")
            assert consumer.cancelling() == 1
            await asyncio.sleep(0)
            assert consumer.done() is False
            store.release_second_attempt.set()
            await asyncio.wait_for(store.receipt_loaded.wait(), timeout=5)

            completed = await store.load(session_id)
            assert completed is not None
            assert completed.status is SessionStatus.COMPLETED
            resumed = await store.transition_status(
                session_id,
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
            )
            assert resumed.run_epoch == completed.run_epoch + 1
            store.release_receipt.set()

            with pytest.raises(asyncio.CancelledError) as raised:
                await asyncio.wait_for(consumer, timeout=5)

            assert raised.value.args == ("cancel before post-settlement receipt reconciliation",)
            assert consumer.cancelled() is True
            assert consumer.cancelling() == 0
            assert (
                session_engine_module._INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE
                not in raised.value.__dict__
            )
            cause = exception_cause(raised.value)
            assert cause is not None
            assert (
                sum(isinstance(error, SessionRunFenced) for error in iter_exception_tree(cause))
                == 1
            )
            persisted = await store.load(session_id)
            durable = await store.load_events(session_id)
            assert persisted is not None
            assert persisted.status is SessionStatus.RUNNING
            assert store.attempts == 2
            assert sum(event.type is EventType.INTERACTION_COMPLETED for event in durable) == 1
            assert sum(event.type is EventType.INTERACTION_INTERRUPTED for event in durable) == 0
            assert sum(event.type is EventType.SESSION_INTERRUPTED for event in durable) == 0
            assert (
                sum(
                    event.type is EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED
                    for event in durable
                )
                == 0
            )
            assert len(provider.requests) == 1
        finally:
            store.release_second_attempt.set()
            store.release_receipt.set()
            if not consumer.done():
                consumer.cancel()
                with contextlib.suppress(BaseException):
                    await consumer
            await store.release_run_fence(session_id)
            await store.close()

    asyncio.run(run())


def test_sqlite_cancellation_reconciles_pruned_setup_failure_transition_receipt(
    tmp_path,
) -> None:
    class FailingFactory(EnvironmentFactory):
        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            del request
            raise RuntimeError("factory failed before model dispatch")

    async def run() -> None:
        store = PrunablePostCommitSQLiteInteractionTransitionStore(
            tmp_path / "pruned-transition-cancellation.sqlite"
        )
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            FailingFactory(),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        accepting_process = CayuApp(session_store=store, enable_logging=False)
        session_id = "sess_cancelled_pruned_setup_failure_transition"

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(store.first_attempt_dispatched.wait(), timeout=5)
        store.release_first_attempt.set()
        await asyncio.wait_for(store.second_attempt_dispatched.wait(), timeout=5)
        consumer.cancel("cancel before pruned receipt reconciliation")
        assert consumer.cancelling() == 1
        await asyncio.sleep(0)
        assert consumer.done() is False
        store.release_second_attempt.set()
        await asyncio.wait_for(store.receipt_lookup_started.wait(), timeout=5)

        durable_before_prune = await store.load_events(session_id)
        failed = next(
            event for event in durable_before_prune if event.type is EventType.INTERACTION_FAILED
        )
        recovered = await accepting_process.recover_persisted_event_side_effects()
        assert sum(event.type is EventType.INTERACTION_FAILED for event in recovered) == 1
        await store.prune_events(
            before=datetime.now(UTC) + timedelta(seconds=1),
            session_id=session_id,
        )
        assert failed.id not in {event.id for event in await store.load_events(session_id)}

        store.release_receipt_lookup.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(consumer, timeout=5)

        assert raised.value.args == ("cancel before pruned receipt reconciliation",)
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 0
        assert store.attempts == 2
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.FAILED
        assert sum(event.type is EventType.INTERACTION_INTERRUPTED for event in durable) == 0
        assert sum(event.type is EventType.SESSION_INTERRUPTED for event in durable) == 0
        diagnostics = [
            event
            for event in durable
            if event.type is EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED
        ]
        assert len(diagnostics) == 1
        assert diagnostics[0].payload["interaction_transition_failures"] == [
            {
                "error": "setup-failure acknowledgement lost after commit",
                "error_type": "ConnectionError",
            },
            {
                "error": "setup-failure replay acknowledgement lost",
                "error_type": "PermissionError",
            },
        ]
        receipt = await store.load_interaction_transition_receipt(
            session_id,
            transition=InteractionTransitionSpec(
                event=failed,
                from_statuses=(SessionStatus.INTERRUPTING, SessionStatus.RUNNING),
                to_status=SessionStatus.FAILED,
            ),
        )
        assert receipt is not None
        assert receipt.transition.event == failed
        assert receipt.status_changed is True
        assert receipt.session.status is SessionStatus.FAILED
        assert provider.requests == []
        await store.close()

    asyncio.run(run())


def test_runtime_cancellation_rejects_conflicting_complete_transition_receipt() -> None:
    async def run() -> None:
        store = ConflictingInteractionTransitionReceiptStore()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_conflicting_complete_transition_receipt"

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(store.second_attempt_dispatched.wait(), timeout=5)
        consumer.cancel("cancel before conflicting receipt readback")
        assert consumer.cancelling() == 1
        await asyncio.sleep(0)
        assert consumer.done() is False
        store.release_second_attempt.set()

        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(consumer, timeout=5)

        assert raised.value.args == ("cancel before conflicting receipt readback",)
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 0
        assert (
            session_engine_module._INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE
            not in raised.value.__dict__
        )
        cause = exception_cause(raised.value)
        assert cause is not None
        conflicts = [
            error
            for error in iter_exception_tree(cause)
            if isinstance(error, RuntimeError)
            and "conflicts with its durable receipt" in str(error)
        ]
        assert len(conflicts) == 1
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert store.attempts == 2
        assert sum(event.type is EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert sum(event.type is EventType.INTERACTION_INTERRUPTED for event in durable) == 0
        assert sum(event.type is EventType.SESSION_INTERRUPTED for event in durable) == 0
        assert (
            sum(
                event.type is EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED
                for event in durable
            )
            == 0
        )
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_runtime_cancellation_rejects_incoherent_transition_receipt_result() -> None:
    async def run() -> None:
        store = IncoherentInteractionTransitionReceiptStore()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_incoherent_complete_transition_receipt"

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(store.second_attempt_dispatched.wait(), timeout=5)
        consumer.cancel("cancel before incoherent receipt readback")
        assert consumer.cancelling() == 1
        await asyncio.sleep(0)
        assert consumer.done() is False
        store.release_second_attempt.set()

        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(consumer, timeout=5)

        assert raised.value.args == ("cancel before incoherent receipt readback",)
        assert consumer.cancelled() is True
        assert consumer.cancelling() == 0
        assert (
            session_engine_module._INTERACTION_TRANSITION_CANCELLATION_OUTCOME_ATTRIBUTE
            not in raised.value.__dict__
        )
        cause = exception_cause(raised.value)
        assert cause is not None
        invalid_results = [
            error
            for error in iter_exception_tree(cause)
            if isinstance(error, ValidationError)
            and "changed status does not match its session" in str(error)
        ]
        assert len(invalid_results) == 1
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert store.attempts == 2
        assert sum(event.type is EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert sum(event.type is EventType.INTERACTION_INTERRUPTED for event in durable) == 0
        assert sum(event.type is EventType.SESSION_INTERRUPTED for event in durable) == 0
        assert (
            sum(
                event.type is EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED
                for event in durable
            )
            == 0
        )
        assert len(provider.requests) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "store_type",
    [
        pytest.param(ChildCancelledInteractionTransitionStore, id="direct"),
        pytest.param(GroupedChildCancelledInteractionTransitionStore, id="exception-group"),
    ],
)
def test_runtime_replays_child_originated_transition_cancellation_as_failure(
    store_type,
) -> None:
    async def run() -> None:
        store = store_type()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_child_cancelled_transition"

        async def consume() -> list[Event]:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "run")],
                    )
                )
            ]

        consumer = asyncio.create_task(consume())
        events = await consumer

        assert consumer.cancelled() is False
        assert consumer.cancelling() == 0
        assert store.attempts == 2
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 1
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_runtime_preserves_all_exhausted_transition_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_groups: list[ExceptionGroup] = []
    build_terminal_failure = session_engine_module._interaction_transition_replay_failure

    def capture_terminal_failure(failures: list[Exception]) -> Exception:
        failure = build_terminal_failure(failures)
        if isinstance(failure, ExceptionGroup):
            captured_groups.append(failure)
        return failure

    monkeypatch.setattr(
        session_engine_module,
        "_interaction_transition_replay_failure",
        capture_terminal_failure,
    )

    async def run() -> None:
        store = DistinctInteractionTransitionFailureStore()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_distinct_transition_replay_failures"

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            )
        ]

        assert store.attempts == 3
        assert captured_groups[-1].exceptions == tuple(store.failures)
        assert all(
            sum(candidate is failure for candidate in iter_exception_tree(captured_groups[-1])) == 1
            for failure in [*store.failures, *store.nested_failures]
        )
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.FAILED
        assert events[-1].type is EventType.SESSION_FAILED
        assert events[-1].payload["interaction_transition_failures"] == [
            {
                "error": "acknowledgement lost",
                "error_type": "ConnectionError",
            },
            {
                "error": "receipt and cleanup failed (2 sub-exceptions)",
                "error_type": "ExceptionGroup",
                "children": [
                    {
                        "error": "receipt read timed out",
                        "error_type": "TimeoutError",
                    },
                    {
                        "error": "cleanup failed",
                        "error_type": "ValueError",
                    },
                ],
            },
            {
                "error": "connection replaced",
                "error_type": "PermissionError",
            },
        ]
        durable_failure = next(event for event in durable if event.type is EventType.SESSION_FAILED)
        assert (
            durable_failure.payload["interaction_transition_failures"]
            == events[-1].payload["interaction_transition_failures"]
        )
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_runtime_records_each_replay_attempt_when_store_reuses_exception() -> None:
    async def run() -> None:
        store = ReusedInteractionTransitionFailureStore()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_reused_transition_replay_failure"

        representative = session_engine_module._interaction_transition_replay_failure(
            [store.failure, store.failure, store.failure]
        )
        assert isinstance(representative, ExceptionGroup)
        assert representative.exceptions == (store.failure,)
        assert session_engine_module._interaction_transition_replay_failures(representative) == (
            store.failure,
            store.failure,
            store.failure,
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                )
            )
        ]

        assert store.attempts == 3
        expected_diagnostic = {
            "error": "same acknowledgement failure",
            "error_type": "ConnectionError",
        }
        assert events[-1].type is EventType.SESSION_FAILED
        assert events[-1].payload["interaction_transition_failures"] == [
            expected_diagnostic,
            expected_diagnostic,
            expected_diagnostic,
        ]
        session = await store.load(session_id)
        durable = await store.load_events(session_id)
        assert session is not None
        assert session.status is SessionStatus.FAILED
        durable_failure = next(event for event in durable if event.type is EventType.SESSION_FAILED)
        assert durable_failure.payload["interaction_transition_failures"] == [
            expected_diagnostic,
            expected_diagnostic,
            expected_diagnostic,
        ]
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_runtime_replay_fails_closed_after_run_fence_transfer() -> None:
    async def run() -> None:
        store = CommitAndBlockInteractionTransitionStore(fail_after_release=True)
        task_store = InMemoryTaskStore()
        provider = CompletingProvider()
        app = CayuApp(
            session_store=store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_fenced_transition_replay"
        task_id = "task_fenced_transition_replay"
        await task_store.create_task(
            TaskCreate(
                task_id=task_id,
                type="respond",
                assigned_agent_name="assistant",
            )
        )
        fence_acquired = asyncio.Event()
        release_fence = asyncio.Event()

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    task_id=task_id,
                    messages=[Message.text("user", "run")],
                )
            ):
                pass

        async def transfer_fence() -> None:
            fenced = await store.fence_stalled_run(
                session_id,
                statuses={SessionStatus.COMPLETED},
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
            assert fenced is not None
            fence_acquired.set()
            await release_fence.wait()
            await store.release_run_fence(session_id)

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(store.committed.wait(), timeout=5)
        fencer = asyncio.create_task(transfer_fence())
        await asyncio.wait_for(fence_acquired.wait(), timeout=5)
        store.release.set()
        try:
            with pytest.raises(SessionRunFenced) as raised:
                await consumer
            assert isinstance(raised.value.__cause__, ConnectionError)
            assert len(store.attempted_events) == 2
            assert store.attempted_events[0] == store.attempted_events[1]
            durable = await store.load_events(session_id)
            task = await task_store.load_task(task_id)
            assert task is not None
            assert task.status == TaskStatus.RUNNING
            assert sum(event.type == EventType.INTERACTION_COMPLETED for event in durable) == 1
            assert sum(event.type == EventType.INTERACTION_FAILED for event in durable) == 0
            assert sum(event.type == EventType.TASK_FAILED for event in durable) == 0
            assert sum(event.type == EventType.TASK_COMPLETED for event in durable) == 0
        finally:
            release_fence.set()
            await fencer

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


def test_unassociated_runtime_event_allowlist_is_exhaustive() -> None:
    assert (
        frozenset(
            {
                EventType.TURN_COMPLETED,
                EventType.SESSION_COMPLETED,
                EventType.SESSION_FAILED,
                EventType.SESSION_INTERRUPTED,
                EventType.TASK_INTERRUPTED_HANDOFF,
                EventType.TASK_COMPLETION_RESULT_RESOLVED,
                EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED,
            }
        )
        == sessions_module.UNASSOCIATED_RUNTIME_EVENT_TYPES
    )


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
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.interaction_failures_remaining = 3
            self.interaction_attempts = 0

        async def publish_interaction_transition(self, session_id: str, **kwargs):
            event = kwargs["event"]
            if (
                event.type == EventType.INTERACTION_INTERRUPTED
                and self.interaction_failures_remaining > 0
            ):
                self.interaction_attempts += 1
                self.interaction_failures_remaining -= 1
                raise ConnectionError("interaction transition unavailable")
            if event.type == EventType.INTERACTION_INTERRUPTED:
                self.interaction_attempts += 1
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
        assert store.interaction_attempts == 3

        retried_page = await app.recover_incomplete_sessions(request)
        assert retried_page.results == ()
        events = await store.load_events(session_id)
        assert [event.type for event in events].count(EventType.SESSION_INTERRUPTED) == 1
        assert [event.type for event in events].count(EventType.INTERACTION_INTERRUPTED) == 1
        assert store.interaction_attempts == 4

    asyncio.run(scenario())


def test_cancelled_terminal_recovery_settles_interaction_reconciliation() -> None:
    class CancelledInteractionReconciliationStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.interaction_failures_remaining = 3
            self.interaction_attempts = 0
            self.second_attempt_dispatched = asyncio.Event()
            self.release_second_attempt = asyncio.Event()

        async def publish_interaction_transition(self, session_id: str, **kwargs):
            event = kwargs["event"]
            if event.type == EventType.INTERACTION_INTERRUPTED:
                self.interaction_attempts += 1
                if self.interaction_attempts == 2:
                    self.second_attempt_dispatched.set()
                    await self.release_second_attempt.wait()
                if self.interaction_failures_remaining > 0:
                    self.interaction_failures_remaining -= 1
                    raise ConnectionError("interaction transition unavailable")
            return await super().publish_interaction_transition(session_id, **kwargs)

    async def scenario() -> None:
        store = CancelledInteractionReconciliationStore()
        app = CayuApp(session_store=store, enable_logging=False)
        provider = CompletingProvider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_cancelled_terminal_interaction_reconciliation"
        interaction_id = "interaction-cancelled-terminal-reconciliation"
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
                id="interaction-cancelled-terminal-reconciliation-start",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
                timestamp=started_at,
                payload=InteractionSummaryEvidence(
                    status=InteractionStatus.ACTIVE,
                    start_event_id=("interaction-cancelled-terminal-reconciliation-start"),
                    started_at=started_at,
                ).model_dump(mode="json"),
            ),
        )

        recovery_task = asyncio.create_task(
            app.recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))
        )
        await asyncio.wait_for(store.second_attempt_dispatched.wait(), timeout=1)
        assert recovery_task.cancelling() == 0
        recovery_task.cancel("cancel terminal interaction reconciliation")
        assert recovery_task.cancelling() == 1
        store.release_second_attempt.set()
        with pytest.raises(
            asyncio.CancelledError,
            match="cancel terminal interaction reconciliation",
        ):
            await recovery_task
        assert recovery_task.cancelled() is True

        session = await store.load(session_id)
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED
        events = await store.load_events(session_id)
        assert [event.type for event in events].count(EventType.SESSION_INTERRUPTED) == 1
        assert [event.type for event in events].count(EventType.INTERACTION_INTERRUPTED) == 1
        assert store.interaction_attempts == 4

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
