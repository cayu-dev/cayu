"""Interruption contracts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

import pytest
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _assert_proxy_tool_effect_remains_unknown,
    _private_record_for_public_event,
    assert_only_model_step_publication_checkpoint,
    checkpoint_without_active_invocation_profile,
    collect_events,
    collect_interrupt_events,
    interruption_payload_without_request_id,
)
from tests.runtime.runtime_tool_support import EchoTool, SideEffectTool

import cayu.providers.deadlines as provider_deadlines_module
import cayu.runtime._session_control as session_control_module
import cayu.runtime.execution_profiles as execution_profiles_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.context.base import validate_context_messages
from cayu.events import Event, EventType
from cayu.messages import Message, ToolResultPart
from cayu.observability.events import EventSink
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import _tool_execution as tool_execution
from cayu.sessions._invocation_terminal_decision import (
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionIdentity, SessionStatus
from cayu.sessions.requests import InterruptSessionRequest, RunRequest
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.vaults.redaction import SecretRedactor


def test_interrupt_session_marks_pending_session_interrupted_and_emits_event():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, secret_redactor=SecretRedactor("-"))
    app.register_provider(FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]))
    app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_direct",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fakemodel"),
        )
        events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_interrupt_direct",
                    reason="operator requested stop",
                    metadata={"actor": "operator"},
                    requested_by=ResolutionActor(
                        subject="operator42",
                        tenant="tenant7",
                        source=ResolutionActorSource.REQUEST,
                        claims={"role": "operator"},
                    ),
                )
            )
        ]
        records = await store.query_events(
            EventQuery(
                session_id="sess_interrupt_direct",
                event_type=EventType.SESSION_INTERRUPTED,
            )
        )
        return events, await store.load("sess_interrupt_direct"), records

    events, session, records = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert [event.type for event in events] == [EventType.SESSION_INTERRUPTED]
    assert interruption_payload_without_request_id(events[0].payload) == {
        "reason": "operator requested stop",
        "metadata": {"actor": "operator"},
        "requested_by": {
            "subject": "operator42",
            "tenant": "tenant7",
            "source": "request",
        },
        "interruption_type": "operator_requested",
    }
    assert len(records) == 1
    assert (
        str(UUID(records[0].event.payload["interruption_request_id"]))
        == records[0].event.payload["interruption_request_id"]
    )


def test_interrupt_session_race_returns_existing_interrupt_event_without_duplicate():
    class PausingInterruptStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.transition_started: asyncio.Event | None = None
            self.allow_transition_return: asyncio.Event | None = None

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform=None,
            store_time_checkpoint_transform=None,
            result_checkpoint_transform=None,
        ) -> Session:
            session = await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                store_time_checkpoint_transform=store_time_checkpoint_transform,
                result_checkpoint_transform=result_checkpoint_transform,
            )
            if (
                session_id == "sess_interrupt_race_idempotent"
                and to_status == SessionStatus.INTERRUPTING
                and self.transition_started is not None
                and self.allow_transition_return is not None
            ):
                self.transition_started.set()
                await self.allow_transition_return.wait()
            return session

    store = PausingInterruptStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]))
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        store.transition_started = asyncio.Event()
        store.allow_transition_return = asyncio.Event()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_race_idempotent",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

        async def interrupt(reason: str) -> list[Event]:
            return [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(
                        session_id="sess_interrupt_race_idempotent",
                        reason=reason,
                    )
                )
            ]

        first_interrupt = asyncio.create_task(interrupt("first request"))
        await store.transition_started.wait()
        second_interrupt = asyncio.create_task(interrupt("second request"))
        await asyncio.sleep(0)
        store.allow_transition_return.set()
        first_events, second_events = await asyncio.gather(first_interrupt, second_interrupt)
        assert await app.drain_background_interruptions(timeout_s=1) is True
        third_events = await interrupt("idempotent retry after cascade completion")
        assert await app.drain_background_interruptions(timeout_s=1) is True
        stored_events = await store.load_events("sess_interrupt_race_idempotent")
        checkpoint = await store.load_checkpoint("sess_interrupt_race_idempotent")
        return first_events, second_events, third_events, stored_events, checkpoint

    first_events, second_events, third_events, stored_events, checkpoint = asyncio.run(run())

    assert [event.type for event in first_events] == [EventType.SESSION_INTERRUPTED]
    assert [event.type for event in second_events] == [EventType.SESSION_INTERRUPTED]
    assert [event.type for event in third_events] == [EventType.SESSION_INTERRUPTED]
    assert first_events[0].id == second_events[0].id == third_events[0].id
    assert first_events[0].payload["reason"] == "first request"
    stored_interrupt_events = [
        event for event in stored_events if event.type == EventType.SESSION_INTERRUPTED
    ]
    assert len(stored_interrupt_events) == 1
    assert (
        _private_record_for_public_event(
            asyncio.run(
                store.query_events(EventQuery(session_id="sess_interrupt_race_idempotent"))
            ),
            first_events[0],
        ).event
        == stored_interrupt_events[0]
    )
    assert checkpoint == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}


def test_new_interruption_does_not_reuse_terminal_event_from_before_resume():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    old_request_id = "11111111-1111-4111-8111-111111111111"
    new_request_id = "22222222-2222-4222-8222-222222222222"

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_after_resume",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_interrupt_after_resume", SessionStatus.INTERRUPTED)
        old_event = Event(
            type=EventType.SESSION_INTERRUPTED,
            session_id="sess_interrupt_after_resume",
            agent_name="assistant",
            payload={
                "reason": "first interruption",
                "interruption_type": "operator_requested",
                "interruption_request_id": old_request_id,
            },
        )
        await store.append_event("sess_interrupt_after_resume", old_event)
        await store.update_status("sess_interrupt_after_resume", SessionStatus.RUNNING)
        await store.checkpoint(
            "sess_interrupt_after_resume",
            {
                "pending_session_interrupt": {
                    "reason": "second interruption",
                    "interruption_type": "operator_requested",
                    "interruption_request_id": new_request_id,
                }
            },
        )
        session = await store.update_status(
            "sess_interrupt_after_resume",
            SessionStatus.INTERRUPTING,
        )
        events = [
            event
            async for event in app._session_finalization.handle_session_interrupted(
                session=session,
                registered_agent=app._get_registered_agent("assistant"),
                registered_environment=None,
                environment_name=None,
            )
        ]
        await app.drain_background_interruptions(timeout_s=1)
        return old_event, events, await store.load_events(session.id)

    old_event, events, stored_events = asyncio.run(run())

    assert events[-1].type == EventType.SESSION_INTERRUPTED
    assert events[-1].id != old_event.id
    assert events[-1].payload["reason"] == "second interruption"
    assert events[-1].payload["interruption_request_id"] == new_request_id
    assert [
        event.payload["interruption_request_id"]
        for event in stored_events
        if event.type == EventType.SESSION_INTERRUPTED
    ] == [old_request_id, new_request_id]


def test_run_stops_after_session_is_interrupted_before_tool_execution():
    store = InMemorySessionStore()
    side_effect = SideEffectTool()

    class InterruptingSink(EventSink):
        async def emit(self, event: Event) -> None:
            if event.type == EventType.MODEL_COMPLETED:
                await store.update_status(event.session_id, SessionStatus.INTERRUPTED)
                await store.append_event(
                    event.session_id,
                    Event(
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=event.session_id,
                        agent_name=event.agent_name,
                        environment_name=event.environment_name,
                        payload={"reason": "test interruption", "metadata": {}},
                    ),
                )

    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(session_store=store, event_sinks=[InterruptingSink()])
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[side_effect],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_before_tool",
                messages=[Message.text("user", "call tool")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_interrupt_before_tool"))

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert side_effect.calls == []
    assert len(provider.requests) == 1
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    assert EventType.TOOL_CALL_STARTED not in [event.type for event in events]


def test_run_interrupt_during_tool_execution_persists_completed_tool_event():
    store = InMemorySessionStore()

    class InterruptRequestingTool(Tool):
        spec = ToolSpec(
            name="side_effect",
            description="Commit a side effect while an interrupt lands.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            await store.update_status(ctx.session_id, SessionStatus.INTERRUPTING)
            return ToolResult(content="side effect committed")

    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[InterruptRequestingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_mid_tool",
                messages=[Message.text("user", "call tool")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_interrupt_mid_tool"))
    stored_events = asyncio.run(store.load_events("sess_interrupt_mid_tool"))
    transcript = asyncio.run(store.load_transcript("sess_interrupt_mid_tool"))

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert events[-1].type == EventType.SESSION_INTERRUPTED

    # The completed result was in hand before the interrupt raised, so the
    # terminal tool event must be persisted; recovery would otherwise report
    # the outcome as unknown and invite a retry of the side effect.
    completed_events = [
        event for event in stored_events if event.type == EventType.TOOL_CALL_COMPLETED
    ]
    assert len(completed_events) == 1
    assert completed_events[0].payload["tool_call_id"] == "call_1"
    assert completed_events[0].payload["result"]["content"] == "side effect committed"
    assert EventType.TOOL_CALL_FAILED not in [event.type for event in stored_events]

    # The transcript records the real outcome, not an interrupted placeholder.
    tool_parts = [
        part
        for message in transcript
        for part in message.content
        if isinstance(part, ToolResultPart)
    ]
    assert len(tool_parts) == 1
    assert tool_parts[0].tool_call_id == "call_1"
    assert tool_parts[0].content == "side effect committed"
    assert tool_parts[0].is_error is False


def test_run_interrupt_leaves_linked_task_running():
    store = InMemorySessionStore()
    tasks = InMemoryTaskStore()

    class InterruptingSink(EventSink):
        async def emit(self, event: Event) -> None:
            if event.type == EventType.MODEL_COMPLETED:
                await store.update_status(event.session_id, SessionStatus.INTERRUPTED)
                await store.append_event(
                    event.session_id,
                    Event(
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=event.session_id,
                        agent_name=event.agent_name,
                        environment_name=event.environment_name,
                        payload={"reason": "task interruption", "metadata": {}},
                    ),
                )

    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(id="call_1", name="echo", arguments={"text": "hi"}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(session_store=store, task_store=tasks, event_sinks=[InterruptingSink()])
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    async def run():
        task = await tasks.create_task(TaskCreate(type="run", title="interrupt me"))
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interrupt_task",
                    task_id=task.id,
                    messages=[Message.text("user", "hello")],
                )
            )
        ]
        return events, await tasks.load_task(task.id)

    events, task = asyncio.run(run())

    assert task is not None
    assert task.status == TaskStatus.RUNNING
    assert EventType.TASK_CANCELLED not in [event.type for event in events]
    assert events[-1].type == EventType.SESSION_INTERRUPTED


def test_run_interrupt_race_reuses_external_interrupt_event_without_duplicate():
    class PausingProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.provider_waiting: asyncio.Event | None = None
            self.allow_provider_complete: asyncio.Event | None = None

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.tool_call(id="call_1", name="echo", arguments={"text": "hi"})
            if self.provider_waiting is None or self.allow_provider_complete is None:
                raise AssertionError("PausingProvider test events were not initialized.")
            self.provider_waiting.set()
            await self.allow_provider_complete.wait()
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

    class PausingInterruptStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.transition_started: asyncio.Event | None = None
            self.allow_transition_return: asyncio.Event | None = None

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform=None,
            store_time_checkpoint_transform=None,
            result_checkpoint_transform=None,
        ) -> Session:
            session = await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                store_time_checkpoint_transform=store_time_checkpoint_transform,
                result_checkpoint_transform=result_checkpoint_transform,
            )
            if (
                session_id == "sess_run_interrupt_race"
                and to_status == SessionStatus.INTERRUPTING
                and self.transition_started is not None
                and self.allow_transition_return is not None
            ):
                self.transition_started.set()
                await self.allow_transition_return.wait()
            return session

    store = PausingInterruptStore()
    provider = PausingProvider()
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    async def run():
        store.transition_started = asyncio.Event()
        store.allow_transition_return = asyncio.Event()
        provider.provider_waiting = asyncio.Event()
        provider.allow_provider_complete = asyncio.Event()

        async def run_session() -> list[Event]:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_run_interrupt_race",
                        messages=[Message.text("user", "hello")],
                    )
                )
            ]

        async def interrupt_session() -> list[Event]:
            return [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(
                        session_id="sess_run_interrupt_race",
                        reason="external interrupt",
                    )
                )
            ]

        run_task = asyncio.create_task(run_session())
        await provider.provider_waiting.wait()

        interrupt_task = asyncio.create_task(interrupt_session())
        await store.transition_started.wait()
        provider.allow_provider_complete.set()
        await asyncio.sleep(0)
        store.allow_transition_return.set()

        run_events, interrupt_events = await asyncio.gather(run_task, interrupt_task)
        stored_records = await store.query_events(
            EventQuery(session_id="sess_run_interrupt_race", limit=100)
        )
        return run_events, interrupt_events, stored_records

    run_events, interrupt_events, stored_records = asyncio.run(run())
    stored_events = [record.event for record in stored_records]

    stored_interrupt_events = [
        event for event in stored_events if event.type == EventType.SESSION_INTERRUPTED
    ]
    assert len(stored_interrupt_events) == 1
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert [event.type for event in run_events[-2:]] == [
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert run_events[-2].payload["status"] == "interrupted"
    assert run_events[-1].id == interrupt_events[0].id
    assert (
        _private_record_for_public_event(stored_records, interrupt_events[0]).event.id
        == stored_interrupt_events[0].id
    )
    assert stored_interrupt_events[0].payload["reason"] == "external interrupt"
    stored_event_types = [event.type for event in stored_events]
    assert stored_event_types.count(EventType.TURN_COMPLETED) == 1
    assert stored_event_types.index(EventType.TURN_COMPLETED) < stored_event_types.index(
        EventType.SESSION_INTERRUPTED
    )


def test_interrupt_session_stops_in_flight_provider_stream():
    class BlockingProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.stream_started: asyncio.Event | None = None
            self.stream_cancelled: asyncio.Event | None = None
            self.never_complete: asyncio.Event | None = None

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if (
                self.stream_started is None
                or self.stream_cancelled is None
                or self.never_complete is None
            ):
                raise AssertionError("BlockingProvider test events were not initialized.")
            self.stream_started.set()
            try:
                await self.never_complete.wait()
            except asyncio.CancelledError:
                self.stream_cancelled.set()
                raise
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    provider = BlockingProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        provider.stream_started = asyncio.Event()
        provider.stream_cancelled = asyncio.Event()
        provider.never_complete = asyncio.Event()

        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interrupt_provider_stream",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await provider.stream_started.wait()
        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_interrupt_provider_stream",
                    reason="operator stop",
                )
            )
        ]
        await asyncio.wait_for(provider.stream_cancelled.wait(), timeout=1)
        run_events = await run_task
        stored_events = await app.session_store.load_events("sess_interrupt_provider_stream")
        return run_events, interrupt_events, stored_events

    run_events, interrupt_events, stored_events = asyncio.run(run())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    assert run_events[-1].id == interrupt_events[0].id
    assert (
        len([event for event in stored_events if event.type == EventType.SESSION_INTERRUPTED]) == 1
    )
    assert EventType.MODEL_COMPLETED not in [event.type for event in stored_events]


def test_interrupt_session_payload_is_durable_across_app_instances():
    class ReleasingProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.started: asyncio.Event | None = None
            self.release: asyncio.Event | None = None

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            if self.started is None or self.release is None:
                raise AssertionError("ReleasingProvider test events were not initialized.")
            self.started.set()
            await self.release.wait()
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    store = InMemorySessionStore()
    provider = ReleasingProvider()
    worker_app = CayuApp(session_store=store)
    api_app = CayuApp(session_store=store)
    for app in (worker_app, api_app):
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        provider.started = asyncio.Event()
        provider.release = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                worker_app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_cross_app_interrupt",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await provider.started.wait()

        interrupt_task = asyncio.create_task(
            collect_interrupt_events(
                api_app,
                InterruptSessionRequest(
                    session_id="sess_cross_app_interrupt",
                    reason="operator stop from api",
                    metadata={"actor": "operator"},
                    requested_by=ResolutionActor(
                        subject="api-operator",
                        tenant="tenant-7",
                        source=ResolutionActorSource.HTTP_AUTH,
                        claims={"scope": "sessions:interrupt"},
                    ),
                ),
            )
        )
        for _ in range(100):
            session = await store.load("sess_cross_app_interrupt")
            if session is not None and session.status == SessionStatus.INTERRUPTING:
                break
            await asyncio.sleep(0.01)
        provider.release.set()

        run_events, interrupt_events = await asyncio.gather(run_task, interrupt_task)
        assert await worker_app.drain_background_interruptions(timeout_s=1) is True
        assert await api_app.drain_background_interruptions(timeout_s=1) is True
        checkpoint = await store.load_checkpoint("sess_cross_app_interrupt")
        return run_events, interrupt_events, checkpoint

    run_events, interrupt_events, checkpoint = asyncio.run(run())

    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    assert interrupt_events == [run_events[-1]]
    assert interruption_payload_without_request_id(run_events[-1].payload) == {
        "reason": "operator stop from api",
        "metadata": {"actor": "operator"},
        "requested_by": {
            "subject": "api-operator",
            "tenant": "tenant-7",
            "source": "http_auth",
        },
        "interruption_type": "operator_requested",
    }
    assert checkpoint is not None
    if any(event.type is EventType.MODEL_COMPLETED for event in run_events):
        assert_only_model_step_publication_checkpoint(checkpoint)
    else:
        # The cross-app interruption can win before the blocked provider returns.
        assert set(checkpoint) == {
            CHECKPOINT_SCHEMA_VERSION_KEY,
            execution_profiles_module.ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
            INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
            SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
        }
    assert settled_invocation_terminal_decision_from_checkpoint(checkpoint) is not None
    assert checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
    assert (
        execution_profiles_module.active_invocation_execution_profile_from_checkpoint(checkpoint)
        is not None
    )


def test_interrupt_session_clears_payload_before_yielding_direct_terminal_event():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]))
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_direct_stream_closed",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        stream = app.interrupt_session(
            InterruptSessionRequest(
                session_id="sess_interrupt_direct_stream_closed",
                reason="operator stop",
            )
        )
        first_event = await anext(stream)
        await stream.aclose()
        checkpoint_after_event = await store.load_checkpoint("sess_interrupt_direct_stream_closed")
        assert await app.drain_background_interruptions(timeout_s=1) is True
        final_checkpoint = await store.load_checkpoint("sess_interrupt_direct_stream_closed")
        return first_event, checkpoint_after_event, final_checkpoint

    first_event, checkpoint_after_event, final_checkpoint = asyncio.run(run())

    assert first_event.type == EventType.SESSION_INTERRUPTED
    assert first_event.payload["reason"] == "operator stop"
    assert "pending_session_interrupt" not in checkpoint_after_event
    assert final_checkpoint == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}


def test_run_interrupt_clears_payload_before_yielding_active_terminal_event():
    class BlockingProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.started: asyncio.Event | None = None
            self.never_complete: asyncio.Event | None = None

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            if self.started is None or self.never_complete is None:
                raise AssertionError("BlockingProvider test events were not initialized.")
            self.started.set()
            await self.never_complete.wait()
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    store = InMemorySessionStore()
    provider = BlockingProvider()
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        provider.started = asyncio.Event()
        provider.never_complete = asyncio.Event()

        async def run_until_interrupted() -> Event:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interrupt_active_stream_closed",
                    messages=[Message.text("user", "hello")],
                )
            ):
                if event.type == EventType.SESSION_INTERRUPTED:
                    return event
            raise AssertionError("Run stream ended without session.interrupted.")

        run_task = asyncio.create_task(run_until_interrupted())
        await provider.started.wait()
        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_interrupt_active_stream_closed",
                    reason="operator stop",
                )
            )
        ]
        run_event = await run_task
        checkpoint_after_event = await store.load_checkpoint("sess_interrupt_active_stream_closed")
        assert await app.drain_background_interruptions(timeout_s=1) is True
        final_checkpoint = await store.load_checkpoint("sess_interrupt_active_stream_closed")
        return run_event, interrupt_events, checkpoint_after_event, final_checkpoint

    run_event, interrupt_events, checkpoint_after_event, final_checkpoint = asyncio.run(run())

    assert run_event.type == EventType.SESSION_INTERRUPTED
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert run_event.id == interrupt_events[0].id
    assert run_event.payload["reason"] == "operator stop"
    assert "pending_session_interrupt" not in checkpoint_after_event
    assert checkpoint_without_active_invocation_profile(final_checkpoint) == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION
    }


def test_interrupt_session_persists_payload_atomically_with_interrupting_status():
    class AtomicInterruptStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.checked_inside_transition = False

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform=None,
            store_time_checkpoint_transform=None,
            result_checkpoint_transform=None,
        ):
            session = await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                store_time_checkpoint_transform=store_time_checkpoint_transform,
                result_checkpoint_transform=result_checkpoint_transform,
            )
            checkpoint = await self.load_checkpoint(session_id)
            assert session.status == SessionStatus.INTERRUPTING
            assert checkpoint is not None
            assert checkpoint["pending_session_interrupt"]["reason"] == "operator stop"
            self.checked_inside_transition = True
            return session

    store = AtomicInterruptStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]))
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_atomic_interrupt_payload",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        _ = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_atomic_interrupt_payload",
                    reason="operator stop",
                )
            )
        ]
        return store.checked_inside_transition

    assert asyncio.run(run()) is True


def test_interrupt_session_checkpoint_failure_does_not_transition_status():
    class FailingAtomicInterruptStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform=None,
            store_time_checkpoint_transform=None,
            result_checkpoint_transform=None,
        ):
            raise RuntimeError("checkpoint unavailable")

    store = FailingAtomicInterruptStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]))
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_checkpoint_failure",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        with pytest.raises(RuntimeError, match="checkpoint unavailable"):
            _ = [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(
                        session_id="sess_interrupt_checkpoint_failure",
                        reason="operator stop",
                    )
                )
            ]
        return await store.load("sess_interrupt_checkpoint_failure")

    session = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.PENDING


def test_interrupt_session_cleans_request_marker_when_caller_is_cancelled(monkeypatch):
    monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS", 100)
    monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S", 0.01)

    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]))
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_cancel_interrupt_request",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_cancel_interrupt_request", SessionStatus.RUNNING)

        task = asyncio.create_task(
            collect_interrupt_events(
                app,
                InterruptSessionRequest(
                    session_id="sess_cancel_interrupt_request",
                    reason="operator stop",
                ),
            )
        )
        for _ in range(100):
            if app._session_control.is_interruption_request_active("sess_cancel_interrupt_request"):
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return app._session_control.is_interruption_request_active("sess_cancel_interrupt_request")

    assert asyncio.run(run()) is False


def test_interrupt_session_returns_terminal_event_when_provider_delays_cancellation(monkeypatch):
    monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS", 2)
    monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S", 0)

    class DelayedInterruptionProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.stream_started: asyncio.Event | None = None
            self.release_after_cancel: asyncio.Event | None = None

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            if self.stream_started is None or self.release_after_cancel is None:
                raise AssertionError(
                    "DelayedInterruptionProvider test events were not initialized."
                )
            self.stream_started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                await self.release_after_cancel.wait()
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    provider = DelayedInterruptionProvider()
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        existing_owners = set(provider_deadlines_module._PROVIDER_DEADLINE_AWAIT_OWNERS)
        provider.stream_started = asyncio.Event()
        provider.release_after_cancel = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_delayed_provider_interrupt",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await provider.stream_started.wait()

        with pytest.raises(TimeoutError, match="interruption is still finalizing"):
            _ = [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(
                        session_id="sess_delayed_provider_interrupt",
                        reason="operator stop",
                    )
                )
            ]
        events_before_release = await store.load_events("sess_delayed_provider_interrupt")
        assert set(provider_deadlines_module._PROVIDER_DEADLINE_AWAIT_OWNERS) - existing_owners
        provider.release_after_cancel.set()
        run_events = await run_task
        events_after_release = await store.load_events("sess_delayed_provider_interrupt")
        repeated_interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_delayed_provider_interrupt",
                    reason="operator stop",
                )
            )
        ]
        return (
            repeated_interrupt_events,
            events_before_release,
            run_events,
            events_after_release,
        )

    interrupt_events, events_before_release, run_events, events_after_release = asyncio.run(run())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert interrupt_events[0].payload["reason"] == "operator stop"
    early_terminals = [
        event for event in events_before_release if event.type is EventType.SESSION_INTERRUPTED
    ]
    durable_terminal = next(
        event for event in events_after_release if event.type is EventType.SESSION_INTERRUPTED
    )
    assert [event.id for event in early_terminals] in ([], [durable_terminal.id])
    assert [event.type for event in run_events[-2:]] == [
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert run_events[-2].payload["status"] == "interrupted"
    assert run_events[-1].id == interrupt_events[0].id
    event_types_after_release = [event.type for event in events_after_release]
    assert event_types_after_release.count(EventType.SESSION_INTERRUPTED) == 1
    assert event_types_after_release.count(EventType.TURN_COMPLETED) == 1
    assert EventType.MODEL_COMPLETED not in event_types_after_release
    assert event_types_after_release[-3:] == [
        EventType.TURN_COMPLETED,
        EventType.INTERACTION_INTERRUPTED,
        EventType.SESSION_INTERRUPTED,
    ]


def test_interrupt_session_does_not_finalize_unowned_running_session(monkeypatch):
    monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS", 2)
    monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S", 0)

    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]))
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_unowned_running_interrupt",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_unowned_running_interrupt", SessionStatus.RUNNING)

        with pytest.raises(TimeoutError, match="interruption is still finalizing"):
            _ = [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(
                        session_id="sess_unowned_running_interrupt",
                        reason="operator stop",
                    )
                )
            ]
        return (
            await store.load("sess_unowned_running_interrupt"),
            await store.load_events("sess_unowned_running_interrupt"),
        )

    session, events = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTING
    assert EventType.SESSION_INTERRUPTED not in [event.type for event in events]


def test_interrupt_session_transition_loser_reports_finalizing(monkeypatch):
    monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS", 2)
    monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S", 0)

    class LosingTransitionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform=None,
            store_time_checkpoint_transform=None,
            result_checkpoint_transform=None,
        ) -> Session:
            if session_id == "sess_interrupt_transition_loser_finalizing":
                await super().transition_status_and_checkpoint(
                    session_id,
                    from_statuses=from_statuses,
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=checkpoint_transform,
                    store_time_checkpoint_transform=store_time_checkpoint_transform,
                    result_checkpoint_transform=result_checkpoint_transform,
                )
                raise ValueError("lost transition")
            return await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                store_time_checkpoint_transform=store_time_checkpoint_transform,
                result_checkpoint_transform=result_checkpoint_transform,
            )

    store = LosingTransitionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]))
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_transition_loser_finalizing",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(
            "sess_interrupt_transition_loser_finalizing",
            SessionStatus.RUNNING,
        )
        with pytest.raises(TimeoutError, match="interruption is still finalizing"):
            _ = [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(
                        session_id="sess_interrupt_transition_loser_finalizing",
                        reason="operator stop",
                    )
                )
            ]
        return await store.load("sess_interrupt_transition_loser_finalizing")

    session = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTING


def test_interrupt_session_stops_in_flight_tool_call():
    from cayu.runtime._tool_effect_state import ToolEffectStateOwner

    class BlockingTool(Tool):
        spec = ToolSpec(
            name="blocking_tool",
            description="Block until cancelled.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.started: asyncio.Event | None = None
            self.cancelled: asyncio.Event | None = None
            self.never_complete: asyncio.Event | None = None

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            if self.started is None or self.cancelled is None or self.never_complete is None:
                raise AssertionError("BlockingTool test events were not initialized.")
            self.started.set()
            try:
                await self.never_complete.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return ToolResult(content="unexpected")

    tool = BlockingTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(id="call_1", name="blocking_tool", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    async def run():
        tool.started = asyncio.Event()
        tool.cancelled = asyncio.Event()
        tool.never_complete = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interrupt_tool_call",
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await tool.started.wait()
        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_interrupt_tool_call",
                    reason="operator stop",
                )
            )
        ]
        await asyncio.wait_for(tool.cancelled.wait(), timeout=1)
        run_events = await run_task
        stored_events = await app.session_store.load_events("sess_interrupt_tool_call")
        transcript = await app.session_store.load_transcript("sess_interrupt_tool_call")
        return run_events, interrupt_events, stored_events, transcript

    run_events, interrupt_events, stored_events, transcript = asyncio.run(run())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert interruption_payload_without_request_id(interrupt_events[0].payload) == {
        "reason": "operator stop",
        "metadata": {},
        "requested_by": None,
        "interruption_type": "operator_requested",
    }
    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    assert run_events[-1].id == interrupt_events[0].id
    assert (
        len([event for event in stored_events if event.type == EventType.SESSION_INTERRUPTED]) == 1
    )
    assert EventType.TOOL_CALL_COMPLETED not in [event.type for event in stored_events]
    failed_tool_events = [
        event for event in stored_events if event.type == EventType.TOOL_CALL_FAILED
    ]
    assert failed_tool_events == []
    unknown_events = [
        event for event in stored_events if event.type == EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
    ]
    assert len(unknown_events) == 1
    assert unknown_events[0].payload["tool_call_id"] == "call_1"
    session = asyncio.run(app.session_store.load("sess_interrupt_tool_call"))
    assert session is not None
    record = asyncio.run(
        ToolEffectStateOwner(app.session_store).resolve_call(
            session,
            tool_round_id=unknown_events[0].payload["tool_round_id"],
            tool_call_id="call_1",
        )
    )
    assert record is not None
    assert record.intent.idempotency_key == tool_execution.tool_idempotency_key(
        session_id="sess_interrupt_tool_call",
        tool_round_id=unknown_events[0].payload["tool_round_id"],
        tool_call_id="call_1",
    )
    stored_event_types = [event.type for event in stored_events]
    assert stored_event_types.index(
        EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
    ) < stored_event_types.index(EventType.SESSION_INTERRUPTED)
    validate_context_messages(transcript)
    asyncio.run(
        _assert_proxy_tool_effect_remains_unknown(app.session_store, "sess_interrupt_tool_call")
    )


def test_interrupt_session_suppresses_late_tool_events_while_finalizing(monkeypatch):
    monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS", 2)
    monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S", 0)

    class DelayedInterruptionTool(Tool):
        spec = ToolSpec(
            name="delayed_tool",
            description="Delay after cancellation.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.started: asyncio.Event | None = None
            self.release_after_cancel: asyncio.Event | None = None

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            if self.started is None or self.release_after_cancel is None:
                raise AssertionError("DelayedInterruptionTool test events were not initialized.")
            self.started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                await self.release_after_cancel.wait()
            return ToolResult(content="late result")

    tool = DelayedInterruptionTool()
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(id="call_1", name="delayed_tool", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    async def run():
        tool.started = asyncio.Event()
        tool.release_after_cancel = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_delayed_tool_interrupt",
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await tool.started.wait()
        with pytest.raises(TimeoutError, match="interruption is still finalizing"):
            _ = [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(
                        session_id="sess_delayed_tool_interrupt",
                        reason="operator stop",
                    )
                )
            ]
        events_before_release = await store.load_events("sess_delayed_tool_interrupt")
        tool.release_after_cancel.set()
        run_events = await run_task
        events_after_release = await store.load_events("sess_delayed_tool_interrupt")
        transcript = await store.load_transcript("sess_delayed_tool_interrupt")
        repeated_interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_delayed_tool_interrupt",
                    reason="operator stop",
                )
            )
        ]
        return (
            repeated_interrupt_events,
            events_before_release,
            run_events,
            events_after_release,
            transcript,
        )

    (
        interrupt_events,
        events_before_release,
        run_events,
        events_after_release,
        transcript,
    ) = asyncio.run(run())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert EventType.SESSION_INTERRUPTED not in [event.type for event in events_before_release]
    assert [event.type for event in run_events[-2:]] == [
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert run_events[-2].payload["status"] == "interrupted"
    assert run_events[-1].id == interrupt_events[0].id
    event_types_after_release = [event.type for event in events_after_release]
    assert event_types_after_release.count(EventType.SESSION_INTERRUPTED) == 1
    assert event_types_after_release.count(EventType.TURN_COMPLETED) == 1
    assert EventType.TOOL_CALL_COMPLETED not in event_types_after_release
    assert EventType.TOOL_CALL_FAILED not in event_types_after_release
    assert event_types_after_release.count(EventType.TOOL_EFFECT_OUTCOME_UNKNOWN) == 1
    # The interruption owner settles the interaction's recovery pause before
    # publishing the invocation summary and final session event.
    assert event_types_after_release[-3:] == [
        EventType.INTERACTION_PAUSED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    validate_context_messages(transcript)
    assert [message.role for message in transcript] == ["user"]
    checkpoint = asyncio.run(store.load_checkpoint("sess_delayed_tool_interrupt"))
    assert checkpoint is not None and checkpoint.get("pending_tool_round") is not None


def test_repeated_interrupt_waits_for_active_interruption_terminal_event():
    class BlockingSink(EventSink):
        def __init__(self) -> None:
            self.failed_seen: asyncio.Event | None = None
            self.release_failed_event: asyncio.Event | None = None

        async def emit(self, event: Event) -> None:
            # External interruption publishes uncertainty, never a fake terminal.
            if event.type == EventType.TOOL_EFFECT_OUTCOME_UNKNOWN:
                if self.failed_seen is None or self.release_failed_event is None:
                    raise AssertionError("BlockingSink test events were not initialized.")
                self.failed_seen.set()
                await self.release_failed_event.wait()

    class BlockingTool(Tool):
        spec = ToolSpec(
            name="blocking_tool",
            description="Block until cancelled.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.started: asyncio.Event | None = None
            self.never_complete: asyncio.Event | None = None

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            if self.started is None or self.never_complete is None:
                raise AssertionError("BlockingTool test events were not initialized.")
            self.started.set()
            await self.never_complete.wait()
            return ToolResult(content="unexpected")

    sink = BlockingSink()
    tool = BlockingTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(id="call_1", name="blocking_tool", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(event_sinks=[sink])
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    async def run():
        sink.failed_seen = asyncio.Event()
        sink.release_failed_event = asyncio.Event()
        tool.started = asyncio.Event()
        tool.never_complete = asyncio.Event()

        async def interrupt(reason: str) -> list[Event]:
            return [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(
                        session_id="sess_repeated_interrupt_waits",
                        reason=reason,
                    )
                )
            ]

        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_repeated_interrupt_waits",
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await tool.started.wait()

        first_interrupt = asyncio.create_task(interrupt("first interrupt"))
        await asyncio.wait_for(sink.failed_seen.wait(), timeout=10)
        second_interrupt = asyncio.create_task(interrupt("second interrupt"))
        await asyncio.sleep(0)
        sink.release_failed_event.set()
        first_events, second_events, run_events = await asyncio.gather(
            first_interrupt,
            second_interrupt,
            run_task,
        )
        stored_events = await app.session_store.load_events("sess_repeated_interrupt_waits")
        return first_events, second_events, run_events, stored_events

    first_events, second_events, run_events, stored_events = asyncio.run(run())

    assert [event.type for event in first_events] == [EventType.SESSION_INTERRUPTED]
    assert [event.type for event in second_events] == [EventType.SESSION_INTERRUPTED]
    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    assert first_events[0].id == second_events[0].id == run_events[-1].id
    assert interruption_payload_without_request_id(first_events[0].payload) == {
        "reason": "first interrupt",
        "metadata": {},
        "requested_by": None,
        "interruption_type": "operator_requested",
    }
    stored_event_types = [event.type for event in stored_events]
    assert stored_event_types.index(
        EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
    ) < stored_event_types.index(EventType.SESSION_INTERRUPTED)
    assert EventType.TOOL_CALL_FAILED not in stored_event_types
    assert EventType.TOOL_CALL_COMPLETED not in stored_event_types
    assert stored_event_types.count(EventType.SESSION_INTERRUPTED) == 1


@pytest.mark.parametrize("loser_observes_cleared_marker", [False, True])
def test_concurrent_interrupt_transition_loser_waits_for_terminal_event(
    loser_observes_cleared_marker: bool,
):
    class BlockingProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.stream_started: asyncio.Event | None = None
            self.never_complete: asyncio.Event | None = None

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            if self.stream_started is None or self.never_complete is None:
                raise AssertionError("BlockingProvider test events were not initialized.")
            self.stream_started.set()
            await self.never_complete.wait()
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class RacingInterruptStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.transition_waiters = 0
            self.transition_barrier: asyncio.Event | None = None

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform=None,
            store_time_checkpoint_transform=None,
            result_checkpoint_transform=None,
        ) -> Session:
            if (
                session_id == "sess_concurrent_interrupt_transition_loser"
                and to_status == SessionStatus.INTERRUPTING
                and self.transition_barrier is not None
            ):
                self.transition_waiters += 1
                if self.transition_waiters >= 2:
                    self.transition_barrier.set()
                await self.transition_barrier.wait()
                try:
                    return await super().transition_status_and_checkpoint(
                        session_id,
                        from_statuses=from_statuses,
                        to_status=to_status,
                        checkpoint_transform=checkpoint_transform,
                        store_time_checkpoint_transform=store_time_checkpoint_transform,
                        result_checkpoint_transform=result_checkpoint_transform,
                    )
                except ValueError:
                    if loser_observes_cleared_marker:
                        for _attempt in range(1000):
                            current = await self.load(session_id)
                            checkpoint = await self.load_checkpoint(session_id)
                            if (
                                current is not None
                                and current.status is SessionStatus.INTERRUPTED
                                and (
                                    checkpoint is None
                                    or "pending_session_interrupt" not in checkpoint
                                )
                            ):
                                break
                            await asyncio.sleep(0)
                        else:
                            raise AssertionError(
                                "Winning interruption did not clear its durable marker."
                            )
                    raise
            return await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                store_time_checkpoint_transform=store_time_checkpoint_transform,
                result_checkpoint_transform=result_checkpoint_transform,
            )

    store = RacingInterruptStore()
    provider = BlockingProvider()
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        store.transition_barrier = asyncio.Event()
        provider.stream_started = asyncio.Event()
        provider.never_complete = asyncio.Event()

        async def interrupt(reason: str) -> list[Event]:
            return [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(
                        session_id="sess_concurrent_interrupt_transition_loser",
                        reason=reason,
                    )
                )
            ]

        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_concurrent_interrupt_transition_loser",
                    messages=[Message.text("user", "start")],
                ),
            )
        )
        await provider.stream_started.wait()
        first_interrupt = asyncio.create_task(interrupt("first interrupt"))
        second_interrupt = asyncio.create_task(interrupt("second interrupt"))
        first_events, second_events, run_events = await asyncio.gather(
            first_interrupt,
            second_interrupt,
            run_task,
        )
        stored_events = await store.load_events("sess_concurrent_interrupt_transition_loser")
        return first_events, second_events, run_events, stored_events

    first_events, second_events, run_events, stored_events = asyncio.run(run())

    assert [event.type for event in first_events] == [EventType.SESSION_INTERRUPTED]
    assert [event.type for event in second_events] == [EventType.SESSION_INTERRUPTED]
    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    assert first_events[0].id == second_events[0].id == run_events[-1].id
    stored_interrupt_events = [
        event for event in stored_events if event.type == EventType.SESSION_INTERRUPTED
    ]
    assert len(stored_interrupt_events) == 1
    assert (
        _private_record_for_public_event(
            asyncio.run(
                store.query_events(
                    EventQuery(session_id="sess_concurrent_interrupt_transition_loser")
                )
            ),
            run_events[-1],
        ).event
        == stored_interrupt_events[0]
    )
