"""Hooks contracts."""

from __future__ import annotations

import asyncio

import pytest
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events
from tests.runtime.runtime_tool_support import EchoTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.observability.hooks import (
    RuntimeHook,
    RuntimeHookContext,
    RuntimeHookPhase,
    ToolCallHookContext,
)
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY, public_event_id
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tools.base import ToolResult


def test_cayu_app_agent_runtime_hooks_run_only_for_registered_agent():
    class RecordingHook(RuntimeHook):
        def __init__(self, name: str) -> None:
            self._name = name
            self.sessions: list[str] = []

        @property
        def name(self) -> str:
            return self._name

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            self.sessions.append(context.session.id)

    builder_hook = RecordingHook("builder_hook")
    reviewer_hook = RecordingHook("reviewer_hook")
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("builder done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("reviewer done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="builder", model="fake-model"),
        runtime_hooks=[builder_hook],
    )
    app.register_agent(
        AgentSpec(name="reviewer", model="fake-model"),
        runtime_hooks=[reviewer_hook],
    )

    builder_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="builder",
                session_id="sess_builder_hook",
                messages=[Message.text("user", "Build it.")],
            ),
        )
    )
    reviewer_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="reviewer",
                session_id="sess_reviewer_hook",
                messages=[Message.text("user", "Review it.")],
            ),
        )
    )

    assert builder_hook.sessions == ["sess_builder_hook"]
    assert reviewer_hook.sessions == ["sess_reviewer_hook"]
    assert [
        event.payload["hook_name"]
        for event in builder_events
        if event.type == EventType.HOOK_STARTED
    ] == ["builder_hook"]
    assert [
        event.payload["scope"] for event in builder_events if event.type == EventType.HOOK_STARTED
    ] == ["agent"]
    assert [
        event.payload["hook_name"]
        for event in reviewer_events
        if event.type == EventType.HOOK_STARTED
    ] == ["reviewer_hook"]


def test_cayu_app_runtime_hooks_run_app_scope_before_agent_scope():
    class RecordingHook(RuntimeHook):
        def __init__(self, name: str, calls: list[str]) -> None:
            self._name = name
            self._calls = calls

        @property
        def name(self) -> str:
            return self._name

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            self._calls.append(self.name)

    calls: list[str] = []
    app_hook = RecordingHook("app_hook", calls)
    agent_hook = RecordingHook("agent_hook", calls)
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(runtime_hooks=[app_hook])
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        runtime_hooks=[agent_hook],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_hook_order",
                messages=[Message.text("user", "Do it.")],
            ),
        )
    )

    assert calls == ["app_hook", "agent_hook"]
    assert [
        (event.payload["hook_name"], event.payload["scope"])
        for event in events
        if event.type == EventType.HOOK_STARTED
    ] == [
        ("app_hook", "app"),
        ("agent_hook", "agent"),
    ]
    assert [
        (event.payload["hook_name"], event.payload["scope"])
        for event in events
        if event.type == EventType.HOOK_COMPLETED
    ] == [
        ("app_hook", "app"),
        ("agent_hook", "agent"),
    ]
    terminal_hook_events = [
        event
        for event in events
        if event.type in {EventType.HOOK_STARTED, EventType.HOOK_COMPLETED, EventType.HOOK_FAILED}
    ]
    assert terminal_hook_events
    assert all(event.interaction_id is None for event in terminal_hook_events)


def test_terminal_runtime_hook_started_ack_loss_recovers_only_its_claimant():
    class CommitThenLoseHookStartedAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.lost_acknowledgement = False

        async def append_event(self, session_id: str, event: Event) -> None:
            await super().append_event(session_id, event)
            if event.type == EventType.HOOK_STARTED and not self.lost_acknowledgement:
                self.lost_acknowledgement = True
                raise ConnectionError("hook-start acknowledgement lost after commit")

    class RecordingHook(RuntimeHook):
        def __init__(self) -> None:
            self.calls = 0

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            self.calls += 1

    async def scenario() -> None:
        store = CommitThenLoseHookStartedAcknowledgementStore()
        hook = RecordingHook()
        app = CayuApp(session_store=store, runtime_hooks=[hook], enable_logging=False)
        app.register_provider(
            FakeProvider(
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_terminal_hook_started_ack_loss",
                messages=[Message.text("user", "Finish once.")],
            ),
        )

        assert store.lost_acknowledgement is True
        assert hook.calls == 1
        hook_events = [
            event
            for event in events
            if event.type in {EventType.HOOK_STARTED, EventType.HOOK_COMPLETED}
        ]
        assert [event.type for event in hook_events] == [
            EventType.HOOK_STARTED,
            EventType.HOOK_COMPLETED,
        ]
        stored_hook_events = [
            event
            for event in await store.load_events("sess_terminal_hook_started_ack_loss")
            if event.type in {EventType.HOOK_STARTED, EventType.HOOK_COMPLETED}
        ]
        assert stored_hook_events[0].id.endswith(":started")
        assert stored_hook_events[1].id.endswith(":completed")
        assert (
            stored_hook_events[0].payload["hook_invocation_id"]
            == (stored_hook_events[1].payload["hook_invocation_id"])
        )

    asyncio.run(scenario())


def test_terminal_runtime_hook_completed_ack_loss_is_not_reentered():
    class CommitThenLoseHookCompletedAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.lost_acknowledgement = False

        async def append_event(self, session_id: str, event: Event) -> None:
            await super().append_event(session_id, event)
            if event.type == EventType.HOOK_COMPLETED and not self.lost_acknowledgement:
                self.lost_acknowledgement = True
                raise ConnectionError("hook-completed acknowledgement lost after commit")

    class RecordingHook(RuntimeHook):
        def __init__(self) -> None:
            self.calls = 0

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            self.calls += 1

    async def scenario() -> None:
        store = CommitThenLoseHookCompletedAcknowledgementStore()
        hook = RecordingHook()
        app = CayuApp(session_store=store, runtime_hooks=[hook], enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_terminal_hook_completed_ack_loss",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        terminal_event = Event(
            type=EventType.SESSION_COMPLETED,
            session_id=session.id,
            agent_name=session.agent_name,
        )
        await store.append_event(session.id, terminal_event)
        registered_agent = app._get_registered_agent("assistant")
        ordered_hooks = tuple(
            (registered_hook, "app", index)
            for index, registered_hook in enumerate(app._runtime_hooks)
        )

        first_attempt = app._terminal_event_publication._run_runtime_hooks(
            phase=RuntimeHookPhase.AFTER_SESSION_COMPLETED,
            session=session,
            terminal_event=terminal_event,
            registered_agent=registered_agent,
            registered_environment=None,
            hooks=ordered_hooks,
        )
        first_events = [event async for event in first_attempt]

        retry_events = [
            event
            async for event in app._terminal_event_publication._run_runtime_hooks(
                phase=RuntimeHookPhase.AFTER_SESSION_COMPLETED,
                session=session,
                terminal_event=terminal_event,
                registered_agent=registered_agent,
                registered_environment=None,
                hooks=ordered_hooks,
            )
        ]

        assert store.lost_acknowledgement is True
        assert [event.type for event in first_events] == [
            EventType.HOOK_STARTED,
            EventType.HOOK_COMPLETED,
        ]
        assert retry_events == []
        assert hook.calls == 1
        hook_events = [
            event
            for event in await store.load_events(session.id)
            if event.type in {EventType.HOOK_STARTED, EventType.HOOK_COMPLETED}
        ]
        assert [event.type for event in hook_events] == [
            EventType.HOOK_STARTED,
            EventType.HOOK_COMPLETED,
        ]
        assert (
            hook_events[0].payload["hook_invocation_id"]
            == hook_events[1].payload["hook_invocation_id"]
        )

    asyncio.run(scenario())


def test_terminal_runtime_hook_cancellation_after_reservation_enters_hook_once():
    class PauseAfterHookReservationStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.reservation_committed = asyncio.Event()
            self.release_reservation = asyncio.Event()

        async def append_event(self, session_id: str, event: Event) -> None:
            await super().append_event(session_id, event)
            if event.type is EventType.HOOK_STARTED:
                self.reservation_committed.set()
                await self.release_reservation.wait()

    class RecordingHook(RuntimeHook):
        def __init__(self) -> None:
            self.calls = 0

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            self.calls += 1
            await asyncio.Event().wait()

    async def scenario() -> None:
        store = PauseAfterHookReservationStore()
        hook = RecordingHook()
        app = CayuApp(session_store=store, runtime_hooks=[hook], enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_terminal_hook_reservation_cancellation",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        terminal_event = Event(
            type=EventType.SESSION_COMPLETED,
            session_id=session.id,
            agent_name=session.agent_name,
        )
        await store.append_event(session.id, terminal_event)
        registered_agent = app._get_registered_agent("assistant")
        ordered_hooks = tuple(
            (registered_hook, "app", index)
            for index, registered_hook in enumerate(app._runtime_hooks)
        )
        first_attempt = app._terminal_event_publication._run_runtime_hooks(
            phase=RuntimeHookPhase.AFTER_SESSION_COMPLETED,
            session=session,
            terminal_event=terminal_event,
            registered_agent=registered_agent,
            registered_environment=None,
            hooks=ordered_hooks,
        )
        delivery = asyncio.create_task(anext(first_attempt))
        await store.reservation_committed.wait()
        delivery.cancel("cancel after the hook reservation committed")
        store.release_reservation.set()
        with pytest.raises(asyncio.CancelledError):
            await delivery

        assert hook.calls == 1
        retry_events = [
            event
            async for event in app._terminal_event_publication._run_runtime_hooks(
                phase=RuntimeHookPhase.AFTER_SESSION_COMPLETED,
                session=session,
                terminal_event=terminal_event,
                registered_agent=registered_agent,
                registered_environment=None,
                hooks=ordered_hooks,
            )
        ]
        assert retry_events == []
        hook_events = [
            event
            for event in await store.load_events(session.id)
            if event.type
            in {EventType.HOOK_STARTED, EventType.HOOK_COMPLETED, EventType.HOOK_FAILED}
        ]
        assert [event.type for event in hook_events] == [EventType.HOOK_STARTED]

    asyncio.run(scenario())


def test_terminal_runtime_hook_reservations_distinguish_duplicate_names_by_index():
    class SameNameHook(RuntimeHook):
        def __init__(self, label: str, calls: list[str]) -> None:
            self.label = label
            self.calls = calls

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            self.calls.append(self.label)

    async def scenario() -> None:
        calls: list[str] = []
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            runtime_hooks=[SameNameHook("first", calls), SameNameHook("second", calls)],
            enable_logging=False,
        )
        app.register_provider(
            FakeProvider(
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_duplicate_terminal_hook_names",
                messages=[Message.text("user", "Finish in order.")],
            ),
        )

        assert calls == ["first", "second"]
        started = [
            event
            for event in await store.load_events("sess_duplicate_terminal_hook_names")
            if event.type == EventType.HOOK_STARTED
        ]
        assert [event.payload["hook_name"] for event in started] == [
            "SameNameHook",
            "SameNameHook",
        ]
        assert [event.payload["hook_index"] for event in started] == [0, 1]
        assert len({event.id for event in started}) == 2

    asyncio.run(scenario())


def test_terminal_runtime_hook_replay_advances_past_settled_slots():
    class RecordingHook(RuntimeHook):
        def __init__(self, label: str, calls: list[str]) -> None:
            self.label = label
            self.calls = calls

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            self.calls.append(self.label)

    async def scenario() -> None:
        calls: list[str] = []
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            runtime_hooks=[RecordingHook("first", calls), RecordingHook("second", calls)],
            enable_logging=False,
        )
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_terminal_hook_settled_prefix",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        terminal_event = Event(
            type=EventType.SESSION_COMPLETED,
            session_id=session.id,
            agent_name=session.agent_name,
        )
        await store.append_event(session.id, terminal_event)
        registered_agent = app._get_registered_agent("assistant")
        ordered_hooks = tuple(
            (registered_hook, "app", index)
            for index, registered_hook in enumerate(app._runtime_hooks)
        )

        first_attempt = app._terminal_event_publication._run_runtime_hooks(
            phase=RuntimeHookPhase.AFTER_SESSION_COMPLETED,
            session=session,
            terminal_event=terminal_event,
            registered_agent=registered_agent,
            registered_environment=None,
            hooks=ordered_hooks,
        )
        assert (await anext(first_attempt)).type is EventType.HOOK_STARTED
        assert (await anext(first_attempt)).type is EventType.HOOK_COMPLETED
        await first_attempt.aclose()
        assert calls == ["first"]

        replay = [
            event
            async for event in app._terminal_event_publication._run_runtime_hooks(
                phase=RuntimeHookPhase.AFTER_SESSION_COMPLETED,
                session=session,
                terminal_event=terminal_event,
                registered_agent=registered_agent,
                registered_environment=None,
                hooks=ordered_hooks,
            )
        ]
        assert [event.type for event in replay] == [
            EventType.HOOK_STARTED,
            EventType.HOOK_COMPLETED,
        ]
        assert calls == ["first", "second"]

    asyncio.run(scenario())


def test_unsettled_terminal_runtime_hook_reservation_is_not_reentered_or_skipped():
    class RecordingHook(RuntimeHook):
        def __init__(
            self,
            label: str,
            calls: list[str],
            *,
            entered: asyncio.Event | None = None,
            release: asyncio.Event | None = None,
        ) -> None:
            self.label = label
            self.calls = calls
            self.entered = entered
            self.release = release

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            self.calls.append(self.label)
            if self.entered is not None:
                self.entered.set()
            if self.release is not None:
                await self.release.wait()

    async def scenario() -> None:
        calls: list[str] = []
        entered = asyncio.Event()
        release = asyncio.Event()
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            runtime_hooks=[
                RecordingHook("first", calls, entered=entered, release=release),
                RecordingHook("later", calls),
            ],
            enable_logging=False,
        )
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_unsettled_terminal_hook",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        terminal_event = Event(
            type=EventType.SESSION_COMPLETED,
            session_id=session.id,
            agent_name=session.agent_name,
        )
        await store.append_event(session.id, terminal_event)
        registered_agent = app._get_registered_agent("assistant")
        ordered_hooks = tuple(
            (registered_hook, "app", index)
            for index, registered_hook in enumerate(app._runtime_hooks)
        )

        first_attempt = app._terminal_event_publication._run_runtime_hooks(
            phase=RuntimeHookPhase.AFTER_SESSION_COMPLETED,
            session=session,
            terminal_event=terminal_event,
            registered_agent=registered_agent,
            registered_environment=None,
            hooks=ordered_hooks,
        )
        first_delivery = asyncio.create_task(anext(first_attempt))
        await entered.wait()
        first_delivery.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_delivery

        retry_events = [
            event
            async for event in app._terminal_event_publication._run_runtime_hooks(
                phase=RuntimeHookPhase.AFTER_SESSION_COMPLETED,
                session=session,
                terminal_event=terminal_event,
                registered_agent=registered_agent,
                registered_environment=None,
                hooks=ordered_hooks,
            )
        ]

        assert retry_events == []
        assert calls == ["first"]
        hook_events = [
            event
            for event in await store.load_events(session.id)
            if event.type
            in {EventType.HOOK_STARTED, EventType.HOOK_COMPLETED, EventType.HOOK_FAILED}
        ]
        assert [event.type for event in hook_events] == [EventType.HOOK_STARTED]
        assert hook_events[0].payload["hook_index"] == 0

    asyncio.run(scenario())


def test_cayu_app_after_tool_call_hook_observes_tool_result_and_emits_events():
    class ToolObservationHook(RuntimeHook):
        def __init__(self) -> None:
            self.tool_name: str | None = None
            self.tool_call_id: str | None = None
            self.arguments: dict | None = None
            self.result: ToolResult | None = None
            self.task_id: str | None = None
            self.tool_event_type: EventType | str | None = None
            self.result_mutation_rejected = False

        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            self.tool_name = context.tool_name
            self.tool_call_id = context.tool_call_id
            self.arguments = context.arguments
            self.result = context.result
            self.task_id = context.task_id
            self.tool_event_type = context.tool_event.type
            self.arguments["text"] = "mutated"
            if self.result.structured is not None:
                try:
                    self.result.structured["echoed"] = "mutated"  # type: ignore[index]
                except TypeError:
                    self.result_mutation_rejected = True
            await context.emit_custom_event(
                "custom.tool.observed",
                payload={
                    "tool_name": context.tool_name,
                    "tool_call_id": context.tool_call_id,
                    "content": context.result.content,
                },
            )

    store = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    hook = ToolObservationHook()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_echo",
                    name="echo",
                    arguments={"text": "from tool"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, task_store=tasks)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        runtime_hooks=[hook],
    )
    task = asyncio.run(tasks.create_task(TaskCreate(task_id="task_tool_hook", type="respond")))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_hook",
                task_id=task.id,
                messages=[Message.text("user", "Use the tool.")],
            ),
        )
    )

    assert hook.tool_name == "echo"
    assert hook.tool_call_id == "call_echo"
    assert hook.arguments == {"text": "mutated"}
    assert hook.result is not None
    assert hook.result.structured == {"agent": "assistant", "echoed": "from tool"}
    assert hook.result_mutation_rejected is True
    assert hook.task_id == "task_tool_hook"
    assert hook.tool_event_type == EventType.TOOL_CALL_COMPLETED
    assert [
        (event.payload["phase"], event.payload["hook_name"], event.payload["scope"])
        for event in events
        if event.type == EventType.HOOK_STARTED
    ] == [("after_tool_call", "ToolObservationHook", "agent")]
    assert events[-1].type == EventType.SESSION_COMPLETED

    stored_events = asyncio.run(store.load_events("sess_tool_hook"))
    # after_tool_call hooks now run before the result event is persisted (so a modify decision can
    # rewrite the transcript), so the result event lands last, after the hook lifecycle events.
    assert [
        event.type
        for event in stored_events
        if event.type
        in {
            EventType.TOOL_CALL_COMPLETED,
            EventType.HOOK_STARTED,
            "custom.tool.observed",
            EventType.HOOK_COMPLETED,
        }
    ] == [
        EventType.HOOK_STARTED,
        "custom.tool.observed",
        EventType.HOOK_COMPLETED,
        EventType.TOOL_CALL_COMPLETED,
    ]
    hook_completed = next(
        event for event in stored_events if event.type == EventType.HOOK_COMPLETED
    )
    tool_completed = next(
        event for event in stored_events if event.type == EventType.TOOL_CALL_COMPLETED
    )
    assert hook_completed.payload["tool_name"] == "echo"
    assert hook_completed.payload["tool_call_id"] == "call_echo"
    assert hook_completed.payload["phase"] == "after_tool_call"
    assert tool_completed.interaction_id is not None
    assert all(
        event.interaction_id == tool_completed.interaction_id
        for event in stored_events
        if event.type in {EventType.HOOK_STARTED, EventType.HOOK_COMPLETED}
    )

    transcript = asyncio.run(store.load_transcript("sess_tool_hook"))
    tool_result = transcript[2].content[0]
    assert tool_result.structured == {"agent": "assistant", "echoed": "from tool"}
    assert provider.requests[1].messages[-1].content[0].structured == {
        "agent": "assistant",
        "echoed": "from tool",
    }


def test_cayu_app_after_tool_call_hook_failure_does_not_stop_tool_round():
    class FailingToolHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            raise RuntimeError("tool hook broke")

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_echo",
                    name="echo",
                    arguments={"text": "from tool"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done after hook failure"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        runtime_hooks=[FailingToolHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_hook_failure",
                messages=[Message.text("user", "Use the tool.")],
            ),
        )
    )

    assert [
        event.type
        for event in events
        if event.type in {EventType.HOOK_STARTED, EventType.HOOK_FAILED}
    ] == [EventType.HOOK_STARTED, EventType.HOOK_FAILED]
    hook_failed = next(event for event in events if event.type == EventType.HOOK_FAILED)
    assert hook_failed.payload["phase"] == "after_tool_call"
    assert hook_failed.payload["tool_name"] == "echo"
    assert hook_failed.payload["tool_call_id"] == PRIVATE_EVENT_AUTHORITY
    assert hook_failed.payload["error"] == "tool hook broke"
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(provider.requests) == 2
    session = asyncio.run(store.load("sess_tool_hook_failure"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_runtime_hook_failure_is_recorded_without_rewriting_session_status():
    class FailingHook(RuntimeHook):
        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            raise RuntimeError("hook broke")

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("main task done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store, runtime_hooks=[FailingHook()])
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_hook_failure",
                messages=[Message.text("user", "Do the work.")],
            ),
        )
    )

    assert [event.type for event in events[-3:]] == [
        EventType.SESSION_COMPLETED,
        EventType.HOOK_STARTED,
        EventType.HOOK_FAILED,
    ]
    assert events[-1].payload["error"] == "hook broke"
    session = asyncio.run(store.load("sess_hook_failure"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_runtime_hook_can_emit_custom_events():
    class CustomEventHook(RuntimeHook):
        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            await context.emit_custom_event(
                "custom.knowledge.extracted",
                payload={"session_id": context.session.id},
            )

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("main task done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store, runtime_hooks=[CustomEventHook()])
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_hook_custom_event",
                messages=[Message.text("user", "Do the work.")],
            ),
        )
    )

    assert [event.type for event in events[-2:]] == [
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
    ]
    stored_events = asyncio.run(store.load_events("sess_hook_custom_event"))
    assert [event.type for event in stored_events[-3:]] == [
        EventType.HOOK_STARTED,
        "custom.knowledge.extracted",
        EventType.HOOK_COMPLETED,
    ]
    assert stored_events[-2].payload == {"session_id": "sess_hook_custom_event"}
    stored_records = asyncio.run(
        store.query_events(EventQuery(session_id="sess_hook_custom_event"))
    )
    custom_record = next(
        record for record in stored_records if record.event.type == "custom.knowledge.extracted"
    )
    assert events[-1].payload["actions"] == [
        {
            "type": "emit_custom_event",
            "payload": {
                "event_id": public_event_id(custom_record.sequence),
                "event_type": "custom.knowledge.extracted",
                "session_id": "sess_hook_custom_event",
            },
        }
    ]


def test_cayu_app_runtime_hook_rejects_non_custom_emitted_events():
    class BadEventHook(RuntimeHook):
        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            await context.emit_custom_event("session.started")

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("main task done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store, runtime_hooks=[BadEventHook()])
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_hook_bad_event",
                messages=[Message.text("user", "Do the work.")],
            ),
        )
    )

    assert [event.type for event in events[-3:]] == [
        EventType.SESSION_COMPLETED,
        EventType.HOOK_STARTED,
        EventType.HOOK_FAILED,
    ]
    assert events[-1].payload["error"] == (
        "Custom event types must use non-empty dot-separated segments in the 'custom.' namespace."
    )
