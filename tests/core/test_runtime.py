from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from cayu.core import AgentSpec, Event, EventType, Message, TextPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
)
from cayu.runtime import (
    CayuApp,
    EventSink,
    InMemoryEventSink,
    InMemorySessionStore,
    InMemoryTaskStore,
    RunRequest,
    SessionStatus,
    TaskCreate,
    TaskStatus,
)
from cayu.workspaces import Workspace, WorkspaceListResult, WorkspaceReadResult


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(
        self,
        events: list[ModelStreamEvent] | list[list[ModelStreamEvent]],
    ) -> None:
        if events and isinstance(events[0], list):
            self.event_batches = events  # type: ignore[assignment]
        else:
            self.event_batches = [events]  # type: ignore[list-item]
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        batch_index = len(self.requests) - 1
        if batch_index >= len(self.event_batches):
            raise AssertionError(
                f"No fake provider event batch for request {batch_index}"
            )
        for event in self.event_batches[batch_index]:
            yield event

    def to_event(
        self,
        stream_event: ModelStreamEvent,
        *,
        session_id: str,
        agent_name: str | None = None,
    ) -> Event:
        if stream_event.type == ModelStreamEventType.TEXT_DELTA:
            return Event(
                type=EventType.MODEL_TEXT_DELTA,
                session_id=session_id,
                agent_name=agent_name,
                payload={"delta": stream_event.delta},
            )
        if stream_event.type == ModelStreamEventType.COMPLETED:
            return Event(
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                agent_name=agent_name,
                payload=stream_event.payload,
            )
        if stream_event.type == ModelStreamEventType.ERROR:
            return Event(
                type=EventType.MODEL_ERROR,
                session_id=session_id,
                agent_name=agent_name,
                payload=stream_event.payload,
            )
        return Event(
            type=f"custom.provider.{stream_event.type}",
            session_id=session_id,
            agent_name=agent_name,
            payload=stream_event.payload,
        )


class MutatingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
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

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        async for event in super().stream(request):
            if len(self.requests) == 1:
                request.messages[0].content[0].text = "mutated by provider"
            yield event


class FailingEventSink(EventSink):
    async def emit(self, event: Event) -> None:
        raise RuntimeError("sink unavailable")


class MutatingEventSink(EventSink):
    async def emit(self, event: Event) -> None:
        event.payload["mutated"] = True


class RecordingEventSink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


class MetadataMutatingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
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

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        async for event in super().stream(request):
            if len(self.requests) == 1:
                request.options["agent_metadata"]["nested"]["value"] = "mutated"
            yield event


class EventReturningProvider(FakeProvider):
    def __init__(self, event: object) -> None:
        super().__init__(
            [
                ModelStreamEvent.text_delta("hello"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        self.event = event

    def to_event(
        self,
        stream_event: ModelStreamEvent,
        *,
        session_id: str,
        agent_name: str | None = None,
    ) -> Event:
        return self.event  # type: ignore[return-value]


class EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=args["text"],
            structured={"agent": ctx.agent_name, "echoed": args["text"]},
        )


class FailingTool(Tool):
    spec = ToolSpec(
        name="fail",
        description="Fail intentionally.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        raise RuntimeError("intentional tool failure")


class BlankFailingTool(Tool):
    spec = ToolSpec(
        name="blank_fail",
        description="Fail intentionally without a message.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        raise RuntimeError()


class BlankErrorResultTool(Tool):
    spec = ToolSpec(
        name="blank_error_result",
        description="Return an error result without a message.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(is_error=True)


class InvalidResultTool(Tool):
    spec = ToolSpec(
        name="invalid_result",
        description="Return the wrong result type.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return {"content": "not a ToolResult"}  # type: ignore[return-value]


class InvalidConstructedResultTool(Tool):
    spec = ToolSpec(
        name="invalid_constructed_result",
        description="Return a ToolResult that bypassed validation.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        class BadStructured(dict):
            def items(self):
                raise RuntimeError("tool result traversal should not run")

        return ToolResult.model_construct(
            content="ok",
            structured=BadStructured({"bad": "value"}),
            artifacts=[],
            is_error=False,
        )


class SideEffectTool(Tool):
    spec = ToolSpec(
        name="side_effect",
        description="Record execution.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content="recorded")


class ArgumentMutatingTool(Tool):
    spec = ToolSpec(
        name="mutate_args",
        description="Mutate call arguments.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        args["nested"]["value"] = "mutated"
        return ToolResult(content="mutated")


class ResultHoldingTool(Tool):
    spec = ToolSpec(
        name="hold_result",
        description="Keep returned result for later mutation.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        super().__init__()
        self.result: ToolResult | None = None

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.result = ToolResult(
            content="held",
            structured={"nested": {"value": "original"}},
            artifacts=[{"nested": {"value": "original"}}],
        )
        return self.result


class UpperTool(Tool):
    spec = ToolSpec(
        name="upper",
        description="Uppercase text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content=args["text"].upper())


async def collect_events(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


def test_cayu_app_rejects_invalid_runtime_dependencies():
    class StoreLike:
        pass

    class TaskStoreLike:
        pass

    class SinkLike:
        async def emit(self, event):
            pass

    with pytest.raises(TypeError, match="SessionStore"):
        CayuApp(session_store=StoreLike())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="TaskStore"):
        CayuApp(task_store=TaskStoreLike())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="EventSink"):
        CayuApp(event_sinks=[SinkLike()])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="event_sinks"):
        CayuApp(event_sinks=False)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="event_sinks"):
        CayuApp(event_sinks=0)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="event_sinks"):
        CayuApp(event_sinks="")  # type: ignore[arg-type]


def test_cayu_app_preserves_falsey_session_store_instance():
    class FalseySessionStore(InMemorySessionStore):
        def __bool__(self):
            return False

    store = FalseySessionStore()
    app = CayuApp(session_store=store)

    assert app.session_store is store


def test_cayu_app_run_rejects_invalid_request_type():
    app = CayuApp()

    async def run_invalid_request():
        return [event async for event in app.run({"agent_name": "assistant"})]  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="RunRequest"):
        asyncio.run(run_invalid_request())


def test_cayu_app_run_revalidates_constructed_run_request():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(
        FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    request = RunRequest.model_construct(
        agent_name="assistant",
        session_id="sess_invalid_request",
        messages=[Message.text("user", "hi")],
        metadata={},
        max_steps="2",
    )

    async def run_invalid_request():
        return [event async for event in app.run(request)]

    with pytest.raises(ValidationError):
        asyncio.run(run_invalid_request())

    assert asyncio.run(store.load("sess_invalid_request")) is None


def test_cayu_app_run_revalidates_constructed_request_messages():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(
        FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    request = RunRequest.model_construct(
        agent_name="assistant",
        session_id="sess_invalid_message",
        messages=[
            Message.model_construct(
                role="user",
                content=[TextPart.model_construct(text=" ")],
            )
        ],
        metadata={},
        max_steps=1,
    )

    async def run_invalid_request():
        return [event async for event in app.run(request)]

    with pytest.raises(ValidationError):
        asyncio.run(run_invalid_request())

    assert asyncio.run(store.load("sess_invalid_message")) is None


def test_cayu_app_runs_text_only_session_and_persists_events():
    store = InMemorySessionStore()
    sink = InMemoryEventSink()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store, event_sinks=[sink])
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_text",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[2].payload == {"delta": "hello"}
    assert provider.requests[0].model == "fake-model"
    assert provider.requests[0].messages[0].content[0].text == "hi"
    assert provider.requests[0].tools == []
    assert sink.events == events

    persisted = asyncio.run(store.load_events("sess_text"))
    session = asyncio.run(store.load("sess_text"))

    assert persisted == events
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_links_successful_run_to_task():
    session_store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )

    async def run_task_session() -> tuple[list[Event], object]:
        await task_store.create_task(
            TaskCreate(
                task_id="task_runtime_success",
                type="respond",
                assigned_agent_name="assistant",
            )
        )
        app = CayuApp(session_store=session_store, task_store=task_store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_task_success",
                task_id="task_runtime_success",
                messages=[Message.text("user", "hi")],
            ),
        )
        task = await task_store.load_task("task_runtime_success")
        return events, task

    events, task = asyncio.run(run_task_session())

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TASK_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TASK_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    assert task.session_id == "sess_task_success"
    assert task.result == {
        "session_id": "sess_task_success",
        "agent_name": "assistant",
        "environment_name": None,
    }
    assert events[1].payload["task_id"] == "task_runtime_success"
    assert events[1].payload["task_status"] == "running"
    assert events[5].payload["task_status"] == "completed"
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_cayu_app_fails_task_when_run_fails():
    session_store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    provider = FakeProvider([ModelStreamEvent.error("provider down")])

    async def run_task_session() -> tuple[list[Event], object, object]:
        await task_store.create_task(
            TaskCreate(task_id="task_runtime_failure", type="respond")
        )
        app = CayuApp(session_store=session_store, task_store=task_store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_task_failure",
                task_id="task_runtime_failure",
                messages=[Message.text("user", "hi")],
            ),
        )
        task = await task_store.load_task("task_runtime_failure")
        session = await session_store.load("sess_task_failure")
        return events, task, session

    events, task, session = asyncio.run(run_task_session())

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TASK_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TASK_FAILED,
        EventType.SESSION_FAILED,
    ]
    assert task is not None
    assert task.status == TaskStatus.FAILED
    assert task.session_id == "sess_task_failure"
    assert task.error == {
        "message": "provider down",
        "type": "RuntimeError",
        "session_id": "sess_task_failure",
    }
    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert events[-1].payload == {
        "error": "provider down",
        "error_type": "RuntimeError",
    }


def test_cayu_app_fails_session_clearly_when_task_store_is_missing():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_missing_task_store",
                task_id="task_without_store",
                messages=[Message.text("user", "hi")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_missing_task_store"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload == {
        "error": "task_store is required when RunRequest.task_id is set.",
        "error_type": "RuntimeError",
    }
    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert provider.requests == []


def test_cayu_app_does_not_fail_task_it_could_not_start():
    session_store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )

    async def run_task_session() -> tuple[list[Event], object]:
        await task_store.create_task(
            TaskCreate(task_id="task_claimed_elsewhere", type="respond")
        )
        await task_store.start_task(
            "task_claimed_elsewhere",
            session_id="other_session",
        )
        app = CayuApp(session_store=session_store, task_store=task_store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_task_claim_conflict",
                task_id="task_claimed_elsewhere",
                messages=[Message.text("user", "hi")],
            ),
        )
        task = await task_store.load_task("task_claimed_elsewhere")
        return events, task

    events, task = asyncio.run(run_task_session())

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.SESSION_FAILED,
    ]
    assert task is not None
    assert task.status == TaskStatus.RUNNING
    assert task.session_id == "other_session"
    assert events[-1].payload == {
        "error": (
            "Task task_claimed_elsewhere cannot transition to running "
            "from running"
        ),
        "error_type": "ValueError",
    }
    assert provider.requests == []


def test_cayu_app_uses_registered_provider_name_after_provider_mutation():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    provider.name = "mutated"  # type: ignore[assignment]

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_name_snapshot",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[1].type == EventType.MODEL_STARTED
    assert events[1].payload["provider"] == "fake"


def test_cayu_app_records_sink_failures_without_failing_session():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store, event_sinks=[FailingEventSink()])
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_sink_failure",
                messages=[Message.text("user", "hi")],
            ),
        )
    )
    persisted = asyncio.run(store.load_events("sess_sink_failure"))
    session = asyncio.run(store.load("sess_sink_failure"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert session is not None
    assert session.status == SessionStatus.COMPLETED

    sink_failures = [
        event for event in persisted if event.type == EventType.RUNTIME_SINK_FAILED
    ]
    assert len(sink_failures) == len(events)
    assert sink_failures[0].payload == {
        "sink": "FailingEventSink",
        "error": "sink unavailable",
        "error_type": "RuntimeError",
        "event_id": events[0].id,
        "event_type": EventType.SESSION_STARTED,
    }


def test_cayu_app_protects_returned_and_later_sink_events_from_sink_mutation():
    recorder = RecordingEventSink()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(event_sinks=[MutatingEventSink(), recorder])
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_sink_mutation",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[0].type == EventType.SESSION_STARTED
    assert events[0].payload == {"agent_name": "assistant"}
    assert recorder.events[0].payload == {"agent_name": "assistant"}


def test_cayu_app_executes_tool_call_and_records_result():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
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
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2
    assert provider.requests[0].tools == [
        {
            "name": "echo",
            "description": "Echo text.",
            "input_schema": EchoTool.spec.input_schema,
        }
    ]
    assert events[3].tool_name == "echo"
    assert events[3].payload == {
        "tool_call_id": "call_1",
        "arguments": {"text": "from tool"},
    }
    assert events[4].payload["result"]["content"] == "from tool"
    assert events[4].payload["result"]["structured"] == {
        "agent": "assistant",
        "echoed": "from tool",
    }
    assert provider.requests[1].messages[-2].role == "assistant"
    tool_call_part = provider.requests[1].messages[-2].content[0]
    assert tool_call_part.type == "tool_call"
    assert tool_call_part.tool_call_id == "call_1"
    assert tool_call_part.tool_name == "echo"
    assert tool_call_part.arguments == {"text": "from tool"}

    assert provider.requests[1].messages[-1].role == "tool"
    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.tool_call_id == "call_1"
    assert tool_result_part.tool_name == "echo"
    assert tool_result_part.content == "from tool"
    assert tool_result_part.structured == {
        "agent": "assistant",
        "echoed": "from tool",
    }
    assert tool_result_part.artifacts == []
    assert tool_result_part.is_error is False


def test_cayu_app_sends_agent_system_prompt_as_first_message():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt="You are careful.",
        )
    )
    request = RunRequest(
        agent_name="assistant",
        session_id="sess_system_prompt",
        messages=[Message.text("user", "hi")],
    )

    events = asyncio.run(collect_events(app, request))

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert [message.role for message in provider.requests[0].messages] == [
        "system",
        "user",
    ]
    assert provider.requests[0].messages[0].content[0].text == "You are careful."
    assert provider.requests[0].messages[1].content[0].text == "hi"
    assert [message.role for message in request.messages] == ["user"]


@pytest.mark.parametrize("system_prompt", [None, "", "   "])
def test_cayu_app_does_not_send_blank_agent_system_prompt(system_prompt):
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt=system_prompt,
        )
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_blank_system_prompt_{system_prompt!r}",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert [message.role for message in provider.requests[0].messages] == ["user"]


def test_cayu_app_returns_tool_failure_to_model():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="fail",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[FailingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_failure",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_tool_failure"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_FAILED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2
    tool_call_part = provider.requests[1].messages[-2].content[0]
    assert tool_call_part.type == "tool_call"
    assert tool_call_part.tool_call_id == "call_1"
    assert tool_call_part.tool_name == "fail"

    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.is_error is True
    assert tool_result_part.content == "intentional tool failure"
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_returns_nonblank_tool_failure_when_exception_message_is_blank():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="blank_fail",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[BlankFailingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_blank_tool_failure",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[4].type == EventType.TOOL_CALL_FAILED
    assert events[4].payload["result"]["content"] == (
        "RuntimeError: tool execution failed"
    )

    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.is_error is True
    assert tool_result_part.content == "RuntimeError: tool execution failed"


def test_cayu_app_returns_nonblank_tool_failure_when_tool_error_content_is_blank():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="blank_error_result",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[BlankErrorResultTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_blank_tool_error_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[4].type == EventType.TOOL_CALL_FAILED
    assert events[4].payload["result"]["content"] == (
        "Tool returned an error without details."
    )

    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.is_error is True
    assert tool_result_part.content == "Tool returned an error without details."


def test_cayu_app_returns_clear_tool_failure_for_invalid_result_type():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="invalid_result",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[InvalidResultTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_invalid_tool_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[4].type == EventType.TOOL_CALL_FAILED
    assert events[4].payload["result"]["content"] == (
        "Tool returned invalid result type: dict. Expected ToolResult."
    )

    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.is_error is True
    assert tool_result_part.content == (
        "Tool returned invalid result type: dict. Expected ToolResult."
    )


def test_cayu_app_returns_clear_tool_failure_for_invalid_constructed_result():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="invalid_constructed_result",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[InvalidConstructedResultTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_invalid_constructed_tool_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_invalid_constructed_tool_result"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_FAILED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[4].payload["result"]["is_error"] is True
    assert events[4].payload["result"]["content"] == (
        "`structured` must contain JSON-compatible values."
    )

    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.is_error is True
    assert tool_result_part.content == (
        "`structured` must contain JSON-compatible values."
    )
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_keeps_text_and_tool_calls_in_one_assistant_turn():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("I will check. "),
                ModelStreamEvent.tool_call(
                    id="call_1",
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
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_mixed_assistant_turn",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]

    assistant_message = provider.requests[1].messages[-2]
    assert assistant_message.role == "assistant"
    assert [part.type for part in assistant_message.content] == [
        "text",
        "tool_call",
    ]
    assert assistant_message.content[0].text == "I will check. "
    assert assistant_message.content[1].tool_call_id == "call_1"
    assert assistant_message.content[1].tool_name == "echo"

    tool_result_message = provider.requests[1].messages[-1]
    assert tool_result_message.role == "tool"
    assert tool_result_message.content[0].tool_call_id == "call_1"


def test_cayu_app_preserves_whitespace_text_deltas():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("Hello"),
                ModelStreamEvent.text_delta(" "),
                ModelStreamEvent.text_delta("world"),
                ModelStreamEvent.tool_call(
                    id="call_1",
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
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_text_delta_whitespace",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assistant_message = provider.requests[1].messages[-2]
    assert assistant_message.role == "assistant"
    assert assistant_message.content[0].text == "Hello world"


def test_cayu_app_protects_tool_call_arguments_from_tool_mutation():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="mutate_args",
                    arguments={"nested": {"value": "original"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ArgumentMutatingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_mutating_tool_args",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert events[3].type == EventType.TOOL_CALL_STARTED
    assert events[3].payload["arguments"] == {"nested": {"value": "original"}}

    assistant_message = provider.requests[1].messages[-2]
    tool_call_part = assistant_message.content[0]
    assert tool_call_part.type == "tool_call"
    assert tool_call_part.arguments == {"nested": {"value": "original"}}


def test_cayu_app_protects_tool_result_messages_from_result_mutation():
    store = InMemorySessionStore()
    tool = ResultHoldingTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="hold_result",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_mutating_tool_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert tool.result is not None

    tool.result.structured["nested"]["value"] = "mutated"
    tool.result.artifacts[0]["nested"]["value"] = "mutated"

    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.structured == {"nested": {"value": "original"}}
    assert tool_result_part.artifacts == [{"nested": {"value": "original"}}]


def test_cayu_app_protects_runtime_history_from_provider_mutation():
    provider = MutatingProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_mutation",
                messages=[Message.text("user", "original")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert provider.requests[0].messages[0].content[0].text == "mutated by provider"
    assert provider.requests[1].messages[0].content[0].text == "original"


def test_cayu_app_protects_agent_metadata_from_provider_mutation():
    provider = MetadataMutatingProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            metadata={"nested": {"value": "original"}},
        ),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_metadata_mutation",
                messages=[Message.text("user", "original")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert provider.requests[0].options["agent_metadata"] == {
        "nested": {"value": "mutated"}
    }
    assert provider.requests[1].options["agent_metadata"] == {
        "nested": {"value": "original"}
    }


def test_cayu_app_groups_multiple_tool_calls_and_results_in_history():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "one"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="upper",
                    arguments={"text": "two"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool(), UpperTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_multi_tool",
                messages=[Message.text("user", "use both tools")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]

    tool_call_message = provider.requests[1].messages[-2]
    assert tool_call_message.role == "assistant"
    assert [part.type for part in tool_call_message.content] == [
        "tool_call",
        "tool_call",
    ]
    assert [part.tool_call_id for part in tool_call_message.content] == [
        "call_1",
        "call_2",
    ]

    tool_result_message = provider.requests[1].messages[-1]
    assert tool_result_message.role == "tool"
    assert [part.type for part in tool_result_message.content] == [
        "tool_result",
        "tool_result",
    ]
    assert [part.content for part in tool_result_message.content] == ["one", "TWO"]


def test_cayu_app_fails_session_when_max_steps_exceeded():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "again"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_max_steps",
                messages=[Message.text("user", "loop")],
                max_steps=1,
            ),
        )
    )
    session = asyncio.run(store.load("sess_max_steps"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == "Maximum model steps exceeded: 1"
    assert session is not None
    assert session.status == SessionStatus.FAILED


def test_cayu_app_records_failed_session_for_invalid_tool_call_payload():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent(
                type=ModelStreamEventType.TOOL_CALL,
                payload={"name": "echo", "arguments": "not-an-object"},
            )
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_failed",
                messages=[Message.text("user", "bad call")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_failed"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error_type"] == "ValueError"
    assert session is not None
    assert session.status == SessionStatus.FAILED


def test_cayu_app_rejects_custom_tool_call_argument_containers():
    class BadArguments(dict):
        def items(self):
            raise RuntimeError("custom argument traversal should not run")

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.model_construct(
                type=ModelStreamEventType.TOOL_CALL,
                payload={"name": "echo", "arguments": BadArguments({"text": "hi"})},
            )
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_custom_tool_arguments",
                messages=[Message.text("user", "bad call")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error_type"] == "ValueError"
    assert events[-1].payload["error"] == (
        "`payload.arguments` must contain JSON-compatible values."
    )


@pytest.mark.parametrize(
    ("stream_event", "error_type", "error"),
    [
        (
            {"type": "completed"},
            "TypeError",
            "Model providers must yield ModelStreamEvent instances.",
        ),
        (
            ModelStreamEvent.model_construct(
                type=ModelStreamEventType.TEXT_DELTA,
                delta=123,
                payload={},
            ),
            "ValueError",
            "Model provider stream event delta must be a string.",
        ),
    ],
)
def test_cayu_app_validates_provider_stream_events_at_runtime_boundary(
    stream_event,
    error_type,
    error,
):
    store = InMemorySessionStore()
    provider = FakeProvider([stream_event])  # type: ignore[list-item]
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_bad_stream_event_{error_type}",
                messages=[Message.text("user", "bad stream event")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error_type"] == error_type
    assert events[-1].payload["error"] == error


def test_cayu_app_validates_provider_stream_event_payload_container():
    class BadPayload(dict):
        def get(self, key, default=None):
            raise RuntimeError("custom payload access should not run")

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.model_construct(
                type=ModelStreamEventType.ERROR,
                payload=BadPayload({"error": "provider failed"}),
            )
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_bad_stream_payload_container",
                messages=[Message.text("user", "bad stream payload")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error_type"] == "ValueError"
    assert events[-1].payload["error"] == (
        "Model provider stream event payload must be an object."
    )


def test_cayu_app_rejects_provider_stream_event_subclasses_before_attribute_access():
    class BadStreamEvent(ModelStreamEvent):
        def __getattribute__(self, name):
            if name == "type":
                raise RuntimeError("stream event type access should not run")
            return super().__getattribute__(name)

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            BadStreamEvent.model_construct(
                type=ModelStreamEventType.COMPLETED,
                payload={},
            )
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_bad_stream_event_subclass",
                messages=[Message.text("user", "bad stream event")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error_type"] == "TypeError"
    assert events[-1].payload["error"] == (
        "Model providers must yield ModelStreamEvent instances."
    )


@pytest.mark.parametrize(
    ("runtime_event", "error_type", "error"),
    [
        (
            {"type": "model.text.delta"},
            "TypeError",
            "Model providers must convert stream events to Event instances.",
        ),
        (
            Event.model_construct(
                type=EventType.MODEL_TEXT_DELTA,
                session_id="wrong_session",
                payload={"delta": "hello"},
            ),
            "ValueError",
            "Provider event session_id does not match current session.",
        ),
    ],
)
def test_cayu_app_validates_provider_runtime_events(
    runtime_event,
    error_type,
    error,
):
    store = InMemorySessionStore()
    provider = EventReturningProvider(runtime_event)
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_bad_runtime_event_{error_type}",
                messages=[Message.text("user", "bad runtime event")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error_type"] == error_type
    assert events[-1].payload["error"] == error


def test_cayu_app_validates_provider_runtime_event_payload_container():
    class BadPayload(dict):
        def items(self):
            raise RuntimeError("custom event payload traversal should not run")

    store = InMemorySessionStore()
    provider = EventReturningProvider(
        Event.model_construct(
            type=EventType.MODEL_TEXT_DELTA,
            session_id="sess_bad_runtime_event_payload",
            payload=BadPayload({"delta": "hello"}),
        )
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_bad_runtime_event_payload",
                messages=[Message.text("user", "bad runtime event payload")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error_type"] == "ValueError"
    assert events[-1].payload["error"] == (
        "`payload` must contain JSON-compatible values."
    )


def test_cayu_app_rejects_provider_runtime_event_subclasses_before_attribute_access():
    class BadEvent(Event):
        def __getattribute__(self, name):
            if name == "type":
                raise RuntimeError("runtime event type access should not run")
            return super().__getattribute__(name)

    store = InMemorySessionStore()
    provider = EventReturningProvider(
        BadEvent.model_construct(
            type=EventType.MODEL_COMPLETED,
            session_id="sess_bad_runtime_event_subclass",
            payload={},
        )
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_bad_runtime_event_subclass",
                messages=[Message.text("user", "bad runtime event")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error_type"] == "TypeError"
    assert events[-1].payload["error"] == (
        "Model providers must convert stream events to Event instances."
    )


def test_cayu_app_records_failed_session_for_blank_tool_call_name():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent(
                type=ModelStreamEventType.TOOL_CALL,
                payload={"name": "   ", "arguments": {}},
            )
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_blank_tool_name",
                messages=[Message.text("user", "bad call")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_blank_tool_name"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == (
        "Model tool call payload requires non-empty string `name`."
    )
    assert session is not None
    assert session.status == SessionStatus.FAILED


@pytest.mark.parametrize("tool_call_id", ["   ", 123])
def test_cayu_app_records_failed_session_for_invalid_tool_call_id(tool_call_id):
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent(
                type=ModelStreamEventType.TOOL_CALL,
                payload={"id": tool_call_id, "name": "echo", "arguments": {}},
            )
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_invalid_tool_id_{type(tool_call_id).__name__}",
                messages=[Message.text("user", "bad call")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == (
        "Model tool call payload requires non-empty string `id`."
    )


def test_cayu_app_records_failed_session_for_provider_error_event():
    store = InMemorySessionStore()
    provider = FakeProvider([ModelStreamEvent.error("provider failed")])
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_error",
                messages=[Message.text("user", "fail")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_provider_error"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == "provider failed"
    assert session is not None
    assert session.status == SessionStatus.FAILED


def test_cayu_app_does_not_execute_tool_when_provider_errors_after_tool_call():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={},
            ),
            ModelStreamEvent.error("provider failed after tool call"),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_error_after_tool_call",
                messages=[Message.text("user", "fail")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_provider_error_after_tool_call"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.SESSION_FAILED,
    ]
    assert tool.calls == []
    assert session is not None
    assert session.status == SessionStatus.FAILED


def test_cayu_app_fails_when_provider_emits_text_after_completed():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.completed({"finish_reason": "stop"}),
            ModelStreamEvent.text_delta("late"),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_text_after_completed",
                messages=[Message.text("user", "fail")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_text_after_completed"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == (
        "Model provider emitted event after completed: text_delta"
    )
    assert session is not None
    assert session.status == SessionStatus.FAILED


def test_cayu_app_fails_without_tool_execution_when_provider_emits_tool_after_completed():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.completed({"finish_reason": "stop"}),
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={},
            ),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_after_completed",
                messages=[Message.text("user", "fail")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_tool_after_completed"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == (
        "Model provider emitted event after completed: tool_call"
    )
    assert tool.calls == []
    assert session is not None
    assert session.status == SessionStatus.FAILED


def test_cayu_app_fails_session_when_provider_stream_ends_without_completion():
    store = InMemorySessionStore()
    provider = FakeProvider([ModelStreamEvent.text_delta("partial")])
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_missing_completion",
                messages=[Message.text("user", "fail")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_missing_completion"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == (
        "Model provider stream ended without a completed event."
    )
    assert session is not None
    assert session.status == SessionStatus.FAILED


def test_cayu_app_ignores_blank_text_deltas():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta(""),
            ModelStreamEvent.text_delta("   "),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_blank_deltas",
                messages=[Message.text("user", "hi")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_blank_deltas"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 1
    assert len(provider.requests[0].messages) == 1
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_in_memory_session_store_rejects_duplicate_session_ids():
    store = InMemorySessionStore()
    request = RunRequest(
        agent_name="assistant",
        session_id="sess_duplicate",
        messages=[Message.text("user", "hi")],
    )

    asyncio.run(store.create(request))

    with pytest.raises(ValueError, match="Session already exists"):
        asyncio.run(store.create(request))


def test_in_memory_session_store_create_rejects_invalid_request_type():
    store = InMemorySessionStore()

    with pytest.raises(TypeError, match="RunRequest"):
        asyncio.run(store.create({"agent_name": "assistant"}))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="RunRequest"):
        asyncio.run(store.create(None))  # type: ignore[arg-type]


def test_in_memory_session_store_revalidates_constructed_run_requests():
    class BadString(str):
        def __bool__(self):
            raise RuntimeError("session id bool should not run")

        def strip(self):
            raise RuntimeError("session id strip should not run")

    class BadMessages(list):
        def __iter__(self):
            raise RuntimeError("messages iteration should not run")

    store = InMemorySessionStore()

    session = asyncio.run(
        store.create(
            RunRequest.model_construct(
                agent_name="assistant",
                session_id=BadString("sess_bad"),
                messages=[Message.text("user", "hi")],
                metadata={},
                max_steps=1,
            )
        )
    )
    assert session.id == "sess_bad"
    assert type(session.id) is str

    with pytest.raises(ValueError, match="messages must be a list"):
        asyncio.run(
            store.create(
                RunRequest.model_construct(
                    agent_name="assistant",
                    session_id="sess_bad_messages",
                    messages=BadMessages([]),
                    metadata={},
                    max_steps=1,
                )
            )
        )

    assert asyncio.run(store.load("sess_bad")) is not None
    assert asyncio.run(store.load("sess_bad_messages")) is None


def test_in_memory_session_store_rejects_loading_events_for_missing_session():
    store = InMemorySessionStore()

    with pytest.raises(KeyError, match="Session not found"):
        asyncio.run(store.load_events("missing"))


@pytest.mark.parametrize("session_id", ["", " ", 123])
def test_in_memory_session_store_rejects_invalid_session_ids(session_id):
    store = InMemorySessionStore()
    asyncio.run(
        store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_invalid_id_target",
                messages=[Message.text("user", "hi")],
            )
        )
    )
    event = Event(
        type=EventType.SESSION_STARTED,
        session_id="sess_invalid_id_target",
    )

    for method in [
        store.load,
        store.load_events,
        lambda value: store.update_status(value, SessionStatus.COMPLETED),
        lambda value: store.append_event(value, event),
        lambda value: store.checkpoint(value, {"ok": True}),
        store.load_checkpoint,
    ]:
        with pytest.raises(ValueError, match="session_id"):
            asyncio.run(method(session_id))  # type: ignore[arg-type]


def test_in_memory_session_store_rejects_invalid_event_type_on_append():
    store = InMemorySessionStore()
    asyncio.run(
        store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_event_type",
                messages=[Message.text("user", "hi")],
            )
        )
    )

    with pytest.raises(TypeError, match="Event"):
        asyncio.run(store.append_event("sess_event_type", {"session_id": "sess_event_type"}))  # type: ignore[arg-type]


def test_in_memory_session_store_revalidates_constructed_events_on_append():
    class BadPayload(dict):
        def items(self):
            raise RuntimeError("store event payload traversal should not run")

    store = InMemorySessionStore()
    asyncio.run(
        store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_constructed_event",
                messages=[Message.text("user", "hi")],
            )
        )
    )

    with pytest.raises(ValueError, match="`id` cannot be blank"):
        asyncio.run(
            store.append_event(
                "sess_constructed_event",
                Event.model_construct(
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id="sess_constructed_event",
                    id=" ",
                    payload={},
                ),
            )
        )

    with pytest.raises(ValueError, match="JSON-compatible"):
        asyncio.run(
            store.append_event(
                "sess_constructed_event",
                Event.model_construct(
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id="sess_constructed_event",
                    payload=BadPayload({"delta": "hello"}),
                ),
            )
        )


def test_in_memory_session_store_isolates_request_metadata():
    store = InMemorySessionStore()
    metadata = {"nested": {"value": "original"}}
    request = RunRequest(
        agent_name="assistant",
        session_id="sess_metadata_isolation",
        messages=[Message.text("user", "hi")],
        metadata=metadata,
    )

    asyncio.run(store.create(request))
    metadata["nested"]["value"] = "mutated"
    session = asyncio.run(store.load("sess_metadata_isolation"))

    assert session is not None
    assert session.metadata == {"nested": {"value": "original"}}


@pytest.mark.parametrize("status", ["bad", "completed", 123])
def test_in_memory_session_store_rejects_invalid_status_values(status):
    store = InMemorySessionStore()
    asyncio.run(
        store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_status_validation",
                messages=[Message.text("user", "hi")],
            )
        )
    )

    with pytest.raises(ValueError, match="SessionStatus"):
        asyncio.run(store.update_status("sess_status_validation", status))  # type: ignore[arg-type]


def test_in_memory_session_store_checkpoints_json_state_and_isolates_it():
    store = InMemorySessionStore()
    asyncio.run(
        store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_checkpoint",
                messages=[Message.text("user", "hi")],
            )
        )
    )
    state = {"nested": {"value": "original"}}

    asyncio.run(store.checkpoint("sess_checkpoint", state))
    state["nested"]["value"] = "mutated"
    loaded = asyncio.run(store.load_checkpoint("sess_checkpoint"))

    assert loaded == {"nested": {"value": "original"}}

    with pytest.raises(ValueError, match="JSON-compatible"):
        asyncio.run(store.checkpoint("sess_checkpoint", {"bad": object()}))

    with pytest.raises(ValueError, match="dictionary"):
        asyncio.run(store.checkpoint("sess_checkpoint", []))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="dictionary"):
        asyncio.run(store.checkpoint("sess_checkpoint", "bad"))  # type: ignore[arg-type]


def test_cayu_app_rejects_duplicate_agents_and_missing_registrations():
    app = CayuApp()
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(ValueError, match="Agent already registered"):
        app.register_agent(AgentSpec(name="assistant", model="other-model"))

    with pytest.raises(RuntimeError, match="No model provider"):
        app.get_provider()

    with pytest.raises(KeyError, match="Agent not registered"):
        app.get_agent("missing")


def test_cayu_app_rejects_invalid_agent_registration_inputs():
    class ToolLike:
        name = "tool_like"
        description = "Not actually a Tool."
        schema = {}

    class BadString(str):
        def strip(self):
            raise RuntimeError("strip should not run")

    class BadMetadata(dict):
        def items(self):
            raise RuntimeError("agent metadata traversal should not run")

    class BadSchema(dict):
        def items(self):
            raise RuntimeError("tool schema traversal should not run")

    app = CayuApp()

    with pytest.raises(TypeError, match="AgentSpec"):
        app.register_agent({"name": "assistant"})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="`name` cannot be blank"):
        app.register_agent(
            AgentSpec.model_construct(
                name=" ",
                model="fake-model",
                metadata={},
            )
        )

    with pytest.raises(ValueError, match="JSON-compatible"):
        app.register_agent(
            AgentSpec.model_construct(
                name="bad_metadata",
                model="fake-model",
                metadata=BadMetadata({"bad": "value"}),
            )
        )

    with pytest.raises(TypeError, match="Tool"):
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[ToolLike()],  # type: ignore[list-item]
        )

    blank_tool = EchoTool()
    blank_tool.spec = ToolSpec.model_construct(name=" ", description="")
    with pytest.raises(ValueError, match="`name` cannot be blank"):
        app.register_agent(
            AgentSpec(name="bad_tool_name", model="fake-model"),
            tools=[blank_tool],
        )

    bad_string_tool = EchoTool()
    bad_string_tool.spec = ToolSpec.model_construct(
        name=BadString("bad_string_tool"),
        description="",
    )
    with pytest.raises(ValueError, match="must be a string"):
        app.register_agent(
            AgentSpec(name="bad_tool_string_name", model="fake-model"),
            tools=[bad_string_tool],
        )

    bad_schema_tool = EchoTool()
    bad_schema = ToolSpec.model_construct(name="bad_schema", description="")
    object.__setattr__(bad_schema, "_input_schema", BadSchema({"type": "object"}))
    bad_schema_tool.spec = bad_schema
    with pytest.raises(ValueError, match="JSON-compatible"):
        app.register_agent(
            AgentSpec(name="bad_tool_schema", model="fake-model"),
            tools=[bad_schema_tool],
        )

    class BadString(str):
        def __deepcopy__(self, memo):
            raise RuntimeError("tool schema scalar deepcopy should not run")

    bad_scalar_schema_tool = EchoTool()
    bad_scalar_schema = ToolSpec.model_construct(
        name="bad_scalar_schema",
        description="",
    )
    object.__setattr__(
        bad_scalar_schema,
        "_input_schema",
        {"bad": BadString("value")},
    )
    bad_scalar_schema_tool.spec = bad_scalar_schema
    with pytest.raises(ValueError, match="JSON-compatible"):
        app.register_agent(
            AgentSpec(name="bad_tool_scalar_schema", model="fake-model"),
            tools=[bad_scalar_schema_tool],
        )

    with pytest.raises(TypeError, match="Agent tools"):
        app.register_agent(
            AgentSpec(name="tools_false", model="fake-model"),
            tools=False,  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="Agent tools"):
        app.register_agent(
            AgentSpec(name="tools_zero", model="fake-model"),
            tools=0,  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="Agent tools"):
        app.register_agent(
            AgentSpec(name="tools_empty_string", model="fake-model"),
            tools="",  # type: ignore[arg-type]
        )


def test_cayu_app_rejects_blank_agent_lookup_name():
    app = CayuApp()
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(ValueError, match="agent.name"):
        app.get_agent("")

    with pytest.raises(ValueError, match="agent.name"):
        app.get_agent(" ")


def test_cayu_app_rejects_blank_provider_name():
    class BlankProvider(FakeProvider):
        name = " "

    app = CayuApp()

    with pytest.raises(ValueError, match="provider.name"):
        app.register_provider(BlankProvider([]))


def test_cayu_app_rejects_invalid_provider_registration_inputs():
    class ProviderLike:
        name = "fake_like"

    class BadString(str):
        def strip(self):
            raise RuntimeError("strip should not run")

    class BadNameProvider(FakeProvider):
        name = BadString("bad_provider")

    app = CayuApp()

    with pytest.raises(TypeError, match="ModelProvider"):
        app.register_provider(ProviderLike())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="bool"):
        app.register_provider(FakeProvider([]), default="false")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be a string"):
        app.register_provider(BadNameProvider([]))


def test_cayu_app_rejects_blank_provider_lookup_name():
    app = CayuApp()
    app.register_provider(FakeProvider([]), default=True)

    with pytest.raises(ValueError, match="provider.name"):
        app.get_provider("")

    with pytest.raises(ValueError, match="provider.name"):
        app.get_provider(" ")


def test_cayu_app_registers_and_selects_default_environment():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local", metadata={"kind": "dev"})),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_default_environment",
                messages=[Message.text("user", "hi")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_default_environment"))

    assert events[0].payload == {"agent_name": "assistant"}
    assert events[0].environment_name == "local"
    assert events[1].environment_name == "local"
    assert events[-1].environment_name == "local"
    assert provider.requests[0].options["environment_metadata"] == {"kind": "dev"}
    assert session is not None
    assert session.environment_name == "local"
    assert app.get_environment().spec.name == "local"
    assert app.get_environment("local").spec.metadata == {"kind": "dev"}


def test_cayu_app_runs_with_explicit_environment():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="docker", metadata={"kind": "isolated"}))
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                environment_name="docker",
                session_id="sess_explicit_environment",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[0].environment_name == "docker"
    assert provider.requests[0].options["environment_metadata"] == {
        "kind": "isolated"
    }


def test_cayu_app_runs_without_environment_when_none_registered():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_no_environment",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[0].environment_name is None
    assert events[-1].environment_name is None
    assert "environment_name" not in events[0].payload
    assert "environment_name" not in events[-1].payload
    assert provider.requests[0].options["environment_metadata"] == {}

    with pytest.raises(RuntimeError, match="No environment registered"):
        app.get_environment()


def test_cayu_app_rejects_unknown_environment_for_run():
    app = CayuApp()
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(KeyError, match="Environment not registered"):
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    environment_name="missing",
                    messages=[Message.text("user", "hi")],
                ),
            )
        )


def test_cayu_app_includes_environment_on_failed_session_event():
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="echo",
                arguments={"text": "hello"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                max_steps=1,
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].environment_name == "local"
    assert events[-1].payload["error_type"] == "RuntimeError"


def test_cayu_app_tags_all_runtime_events_with_environment():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "hello"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events
    assert {event.environment_name for event in events} == {"local"}
    assert all("environment_name" not in event.payload for event in events)


def test_cayu_app_overrides_provider_event_environment():
    provider = EventReturningProvider(
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="sess_provider_environment",
            environment_name="provider_wrong",
            payload={"finish_reason": "stop"},
        )
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_environment",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events
    assert {event.environment_name for event in events} == {"local"}


def test_cayu_app_rejects_invalid_environment_lookup_name():
    app = CayuApp()
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)

    with pytest.raises(ValueError, match="environment.name"):
        app.get_environment("")

    with pytest.raises(ValueError, match="environment.name"):
        app.get_environment(" ")


def test_cayu_app_isolates_registered_environment_shell():
    class MemoryWorkspace(Workspace):
        def __init__(self, workspace_id: str) -> None:
            self.id = workspace_id

        async def read_bytes(
            self,
            path: str,
            *,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            return WorkspaceReadResult(content=b"", total_bytes=0)

        async def write_bytes(self, path: str, content: bytes) -> None:
            return None

        async def list(
            self,
            pattern: str = "**/*",
            *,
            limit: int | None = None,
        ) -> WorkspaceListResult:
            return WorkspaceListResult(paths=(), total_count=0)

    app = CayuApp()
    original_workspace = MemoryWorkspace("workspace_original")
    environment = Environment(
        EnvironmentSpec(name="local", metadata={"kind": "dev"}),
        workspace=original_workspace,
    )

    app.register_environment(environment, default=True)

    environment.spec = EnvironmentSpec(name="mutated", metadata={"kind": "mutated"})
    environment.workspace = MemoryWorkspace("workspace_mutated")

    registered = app.get_environment()
    registered.spec.metadata["kind"] = "returned"
    registered.environment.workspace = MemoryWorkspace("workspace_returned")

    registered_again = app.get_environment()

    assert registered_again.spec.name == "local"
    assert registered_again.spec.metadata == {"kind": "dev"}
    assert registered_again.environment.workspace is original_workspace


def test_cayu_app_rejects_invalid_environment_registration_inputs():
    class EnvironmentLike:
        spec = EnvironmentSpec(name="fake")

    class EnvironmentSubclass(Environment):
        pass

    class BadString(str):
        def strip(self):
            raise RuntimeError("strip should not run")

    class BadMetadata(dict):
        def items(self):
            raise RuntimeError("environment metadata traversal should not run")

    app = CayuApp()

    with pytest.raises(TypeError, match="Environment"):
        app.register_environment(EnvironmentLike())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Environment"):
        app.register_environment(EnvironmentSubclass(EnvironmentSpec(name="subclass")))

    with pytest.raises(TypeError, match="bool"):
        app.register_environment(
            Environment(EnvironmentSpec(name="local")),
            default="false",  # type: ignore[arg-type]
        )

    bad_name_environment = Environment(EnvironmentSpec(name="bad_name"))
    bad_name_environment.spec = EnvironmentSpec.model_construct(
        name=BadString("bad"),
        metadata={},
    )
    with pytest.raises(ValueError, match="must be a string"):
        app.register_environment(bad_name_environment)

    bad_metadata_environment = Environment(EnvironmentSpec(name="bad_metadata"))
    bad_metadata_environment.spec = EnvironmentSpec.model_construct(
        name="bad_metadata",
        metadata=BadMetadata({"bad": "value"}),
    )
    with pytest.raises(ValueError, match="JSON-compatible"):
        app.register_environment(bad_metadata_environment)

    with pytest.raises(ValueError, match="Environment already registered"):
        app.register_environment(Environment(EnvironmentSpec(name="local")))
        app.register_environment(Environment(EnvironmentSpec(name="local")))


def test_environment_rejects_invalid_bound_services():
    class WorkspaceLike:
        id = "workspace"

    class RunnerLike:
        isolation = "fake"

    class VaultLike:
        pass

    with pytest.raises(TypeError, match="workspace"):
        Environment(
            EnvironmentSpec(name="workspace_like"),
            workspace=WorkspaceLike(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="runner"):
        Environment(
            EnvironmentSpec(name="runner_like"),
            runner=RunnerLike(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="vault"):
        Environment(
            EnvironmentSpec(name="vault_like"),
            vault=VaultLike(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="mcp_servers"):
        Environment(
            EnvironmentSpec(name="bad_mcp_servers"),
            mcp_servers="not iterable specs",  # type: ignore[arg-type]
        )


def test_cayu_app_isolates_registered_agent_state():
    app = CayuApp()
    spec = AgentSpec(name="assistant", model="fake-model")
    tool = EchoTool()

    app.register_agent(spec, tools=[tool])
    spec.model = "mutated"

    registered = app.get_agent("assistant")
    registered.tools["other"] = tool

    assert registered.spec.model == "fake-model"
    assert app.get_agent("assistant").tools.keys() == {"echo"}


def test_cayu_app_isolates_returned_registered_tool_declarations():
    app = CayuApp()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    registered = app.get_agent("assistant")
    registered.tools["echo"].schema["properties"]["text"]["type"] = "integer"

    assert app.get_agent("assistant").tools["echo"].schema == EchoTool.spec.input_schema


def test_cayu_app_freezes_tool_declarations_at_registration():
    store = InMemorySessionStore()
    tool = EchoTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
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
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    tool.spec = ToolSpec(
        name="mutated",
        description="Mutated.",
        input_schema={"type": "object", "properties": {}},
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_frozen_tool",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert provider.requests[0].tools == [
        {
            "name": "echo",
            "description": "Echo text.",
            "input_schema": EchoTool.spec.input_schema,
        }
    ]
    assert events[3].type == EventType.TOOL_CALL_STARTED
    assert events[3].tool_name == "echo"
    assert events[4].type == EventType.TOOL_CALL_COMPLETED
