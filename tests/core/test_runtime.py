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
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextCompactor,
    ContextPolicy,
    ContextRequest,
    Dispatcher,
    DispatchHandle,
    DispatchRequest,
    DispatchStatus,
    EventSink,
    ForkSessionRequest,
    InMemoryEventSink,
    InMemorySessionStore,
    InMemoryTaskStore,
    MessageWindowContextPolicy,
    ModelCompactor,
    RecentTurnsContextPolicy,
    ResumeRequest,
    RunRequest,
    Session,
    SessionIdentity,
    SessionStatus,
    StaticToolPolicy,
    TaskCreate,
    TaskStatus,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    default_compaction_prompt,
    trim_context_messages,
    trim_context_turns,
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
            raise AssertionError(f"No fake provider event batch for request {batch_index}")
        for event in self.event_batches[batch_index]:
            yield event


class OtherProvider(FakeProvider):
    name = "other"


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


class RecordingCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.requests: list[CompactionRequest] = []

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.requests.append(request)
        compacted_text = "|".join(message.content[0].text for message in request.messages)
        if request.existing_summary is not None:
            summary = f"{request.existing_summary}|{compacted_text}"
        else:
            summary = compacted_text
        return CompactionResult(
            summary=summary,
            metadata={"request_count": len(self.requests)},
        )


class FailingCompactor(ContextCompactor):
    async def compact(self, request: CompactionRequest) -> CompactionResult:
        raise RuntimeError("compaction unavailable")


class FailingApprovalCloseStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.failed_close_once = False

    async def append_transcript_messages_and_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint: dict,
    ) -> None:
        if not self.failed_close_once:
            self.failed_close_once = True
            raise RuntimeError("approval close unavailable")
        await super().append_transcript_messages_and_checkpoint(
            session_id,
            messages,
            checkpoint,
        )


class FailingTerminalToolEventStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.failed_terminal_once = False

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        if not self.failed_terminal_once and any(
            event.type == EventType.TOOL_CALL_COMPLETED for event in events
        ):
            self.failed_terminal_once = True
            raise RuntimeError("terminal tool event unavailable")
        await super().append_events(session_id, events)


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


class RequireApprovalPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            reason=f"Approval required for {request.tool_name}.",
            metadata={"scope": "human"},
        )


class SideEffectApprovalPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name == "side_effect":
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason=f"Approval required for {request.tool_name}.",
                metadata={"scope": "human"},
            )
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


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


async def collect_resume_events(app: CayuApp, request: ResumeRequest) -> list[Event]:
    return [event async for event in app.resume(request)]


async def collect_fork_events(app: CayuApp, request: ForkSessionRequest) -> list[Event]:
    return [event async for event in app.fork_session(request)]


async def collect_dispatch_events(app: CayuApp, request: DispatchRequest) -> list[Event]:
    return [event async for event in app.dispatch_inline(request)]


async def submit_dispatch(app: CayuApp, request: DispatchRequest) -> DispatchHandle:
    return await app.dispatch(request)


async def collect_tool_approval_events(
    app: CayuApp,
    request: ToolApprovalRequest,
) -> list[Event]:
    return [event async for event in app.resolve_tool_approval(request)]


async def collect_tool_approval_recovery_events(
    app: CayuApp,
    request: ToolApprovalRecoveryRequest,
) -> list[Event]:
    return [event async for event in app.recover_tool_approval(request)]


def _test_session() -> Session:
    return Session(
        id="sess_context",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
    )


def _test_session_identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def test_cayu_app_rejects_invalid_runtime_dependencies():
    class StoreLike:
        pass

    class TaskStoreLike:
        pass

    class DispatcherLike:
        pass

    class SinkLike:
        async def emit(self, event):
            pass

    with pytest.raises(TypeError, match="SessionStore"):
        CayuApp(session_store=StoreLike())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="TaskStore"):
        CayuApp(task_store=TaskStoreLike())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Dispatcher"):
        CayuApp(dispatcher=DispatcherLike())  # type: ignore[arg-type]

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
    assert session.provider_name == "fake"
    assert session.model == "fake-model"
    assert session.runtime_name == "cayu"


def test_cayu_app_resumes_completed_session_from_stored_transcript():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_resume",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_resume",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[0].payload == {
        "agent_name": "assistant",
        "appended_messages": 1,
    }
    assert [message.content[0].text for message in provider.requests[1].messages] == [
        "first request",
        "first answer",
        "second request",
    ]

    transcript = asyncio.run(store.load_transcript("sess_resume"))
    assert [message.content[0].text for message in transcript] == [
        "first request",
        "first answer",
        "second request",
        "second answer",
    ]
    session = asyncio.run(store.load("sess_resume"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_forks_completed_session_and_preserves_source():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("fork answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_fork_source",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    fork_events = asyncio.run(
        collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_fork_source",
                session_id="sess_fork_child",
                metadata={"purpose": "alternate path"},
            ),
        )
    )

    assert [event.type for event in fork_events] == [EventType.SESSION_FORKED]
    assert fork_events[0].session_id == "sess_fork_child"
    assert fork_events[0].payload["source_session_id"] == "sess_fork_source"
    fork = asyncio.run(store.load("sess_fork_child"))
    source = asyncio.run(store.load("sess_fork_source"))
    assert fork is not None
    assert source is not None
    assert fork.parent_session_id == "sess_fork_source"
    assert fork.status == SessionStatus.COMPLETED
    assert fork.provider_name == source.provider_name == "fake"
    assert fork.model == source.model == "fake-model"
    assert fork.metadata == {"purpose": "alternate path"}

    fork_transcript = asyncio.run(store.load_transcript("sess_fork_child"))
    source_transcript = asyncio.run(store.load_transcript("sess_fork_source"))
    assert fork_transcript == source_transcript

    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_fork_child",
                messages=[Message.text("user", "continue fork")],
            ),
        )
    )
    assert [message.content[0].text for message in provider.requests[1].messages] == [
        "first request",
        "first answer",
        "continue fork",
    ]
    assert [message.content[0].text for message in source_transcript] == [
        "first request",
        "first answer",
    ]


def test_cayu_app_dispatches_existing_session_inline():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("dispatch answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_dispatch_source",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    dispatch_events = asyncio.run(
        collect_dispatch_events(
            app,
            DispatchRequest(
                session_id="sess_dispatch_source",
                dispatch_id="dispatch_1",
                messages=[Message.text("user", "run dispatched work")],
            ),
        )
    )

    assert [event.type for event in dispatch_events] == [
        EventType.SESSION_RESUMED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert dispatch_events[0].payload["dispatch_id"] == "dispatch_1"
    assert dispatch_events[0].payload["appended_messages"] == 1
    assert [message.content[0].text for message in provider.requests[1].messages] == [
        "first request",
        "first answer",
        "run dispatched work",
    ]
    session = asyncio.run(store.load("sess_dispatch_source"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_dispatch_returns_inline_handle():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("dispatch answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_dispatch_handle",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    handle = asyncio.run(
        submit_dispatch(
            app,
            DispatchRequest(
                session_id="sess_dispatch_handle",
                dispatch_id="dispatch_handle_1",
                messages=[Message.text("user", "run dispatched work")],
            ),
        )
    )

    assert handle == DispatchHandle(
        dispatch_id="dispatch_handle_1",
        session_id="sess_dispatch_handle",
        backend="inline",
        status=DispatchStatus.COMPLETED,
        metadata={"events": 5},
    )
    assert [message.content[0].text for message in provider.requests[1].messages] == [
        "first request",
        "first answer",
        "run dispatched work",
    ]
    session = asyncio.run(store.load("sess_dispatch_handle"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_dispatches_forked_session_with_task_linkage():
    store = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("fork dispatch answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, task_store=tasks)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_dispatch_fork_source",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_dispatch_fork_source",
                session_id="sess_dispatch_fork_child",
            ),
        )
    )
    task = asyncio.run(
        tasks.create_task(
            TaskCreate(
                type="follow_up",
                session_id="sess_dispatch_fork_child",
                assigned_agent_name="assistant",
                input={"objective": "summarize follow-up"},
            )
        )
    )

    dispatch_events = asyncio.run(
        collect_dispatch_events(
            app,
            DispatchRequest(
                session_id="sess_dispatch_fork_child",
                dispatch_id="dispatch_fork_1",
                task_id=task.id,
                messages=[Message.text("user", "run the forked follow-up")],
            ),
        )
    )

    assert [event.type for event in dispatch_events] == [
        EventType.SESSION_RESUMED,
        EventType.TASK_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TASK_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert dispatch_events[0].payload == {
        "agent_name": "assistant",
        "appended_messages": 1,
        "dispatch_id": "dispatch_fork_1",
        "task_id": task.id,
    }
    completed_task = asyncio.run(tasks.load_task(task.id))
    assert completed_task is not None
    assert completed_task.status == TaskStatus.COMPLETED
    assert completed_task.session_id == "sess_dispatch_fork_child"
    fork = asyncio.run(store.load("sess_dispatch_fork_child"))
    assert fork is not None
    assert fork.parent_session_id == "sess_dispatch_fork_source"
    assert fork.status == SessionStatus.COMPLETED


def test_cayu_app_dispatch_rejects_running_session():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_running_session() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_dispatch_running",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_dispatch_running", SessionStatus.RUNNING)

    asyncio.run(create_running_session())

    with pytest.raises(ValueError, match="status transition not allowed"):
        asyncio.run(
            collect_dispatch_events(
                app,
                DispatchRequest(
                    session_id="sess_dispatch_running",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )


def test_cayu_app_dispatch_requires_task_store_for_task_linkage():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_dispatch_no_task_store",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    with pytest.raises(RuntimeError, match="task_store is required"):
        asyncio.run(
            collect_dispatch_events(
                app,
                DispatchRequest(
                    session_id="sess_dispatch_no_task_store",
                    task_id="task_missing_store",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )
    session = asyncio.run(store.load("sess_dispatch_no_task_store"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_dispatch_uses_configured_dispatcher():
    class RecordingDispatcher(Dispatcher):
        def __init__(self) -> None:
            self.requests: list[DispatchRequest] = []

        async def submit(self, runtime, request: DispatchRequest) -> DispatchHandle:
            self.requests.append(request)
            return DispatchHandle(
                dispatch_id=request.dispatch_id,
                session_id=request.session_id,
                backend="recording",
                status=DispatchStatus.SUBMITTED,
                metadata={"queued": True},
            )

    dispatcher = RecordingDispatcher()
    app = CayuApp(dispatcher=dispatcher)

    handle = asyncio.run(
        submit_dispatch(
            app,
            DispatchRequest(
                session_id="sess_custom_dispatcher",
                dispatch_id="dispatch_custom_1",
                messages=[Message.text("user", "queued")],
            ),
        )
    )

    assert handle == DispatchHandle(
        dispatch_id="dispatch_custom_1",
        session_id="sess_custom_dispatcher",
        backend="recording",
        status=DispatchStatus.SUBMITTED,
        metadata={"queued": True},
    )
    assert [request.dispatch_id for request in dispatcher.requests] == ["dispatch_custom_1"]


def test_cayu_app_dispatch_rejects_mismatched_dispatch_handle():
    class MismatchedDispatcher(Dispatcher):
        async def submit(self, runtime, request: DispatchRequest) -> DispatchHandle:
            return DispatchHandle(
                dispatch_id="other_dispatch",
                session_id=request.session_id,
                backend="mismatched",
                status=DispatchStatus.SUBMITTED,
            )

    app = CayuApp(dispatcher=MismatchedDispatcher())

    with pytest.raises(ValueError, match="wrong request fields: dispatch_id"):
        asyncio.run(
            submit_dispatch(
                app,
                DispatchRequest(
                    session_id="sess_custom_dispatcher",
                    dispatch_id="dispatch_custom_1",
                    messages=[Message.text("user", "queued")],
                ),
            )
        )


def test_cayu_app_forks_partial_transcript_without_checkpoint():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_partial_fork_source",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(store.checkpoint("sess_partial_fork_source", {"context_compaction": {}}))

    asyncio.run(
        collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_partial_fork_source",
                session_id="sess_partial_fork_child",
                transcript_cursor=1,
                copy_checkpoint=False,
            ),
        )
    )

    fork_transcript = asyncio.run(store.load_transcript("sess_partial_fork_child"))
    assert [message.content[0].text for message in fork_transcript] == ["first request"]
    assert asyncio.run(store.load_checkpoint("sess_partial_fork_child")) is None

    with pytest.raises(ValueError, match="copy_checkpoint must be false"):
        asyncio.run(
            collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id="sess_partial_fork_source",
                    session_id="sess_invalid_partial_fork",
                    transcript_cursor=1,
                ),
            )
        )


def test_cayu_app_resume_uses_stored_provider_and_model_identity():
    class RecordingPolicy(ContextPolicy):
        def __init__(self) -> None:
            self.requests: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.requests.append(request)
            return request.messages

    store = InMemorySessionStore()
    initial_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("first answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    initial_app = CayuApp(session_store=store)
    initial_app.register_provider(initial_provider, default=True)
    initial_app.register_agent(AgentSpec(name="assistant", model="stored-model"))

    asyncio.run(
        collect_events(
            initial_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_stored_identity",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    resume_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("second answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    default_provider = OtherProvider(
        [
            ModelStreamEvent.text_delta("wrong provider"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    resumed_app = CayuApp(session_store=store)
    resumed_app.register_provider(resume_provider)
    resumed_app.register_provider(default_provider, default=True)
    context_policy = RecordingPolicy()
    resumed_app.register_agent(
        AgentSpec(name="assistant", model="new-default-model"),
        context_policy=context_policy,
    )

    asyncio.run(
        collect_resume_events(
            resumed_app,
            ResumeRequest(
                session_id="sess_stored_identity",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    assert len(resume_provider.requests) == 1
    assert resume_provider.requests[0].model == "stored-model"
    assert [request.agent.model for request in context_policy.requests] == ["stored-model"]
    assert default_provider.requests == []


def test_cayu_app_resume_model_updates_session_active_model():
    class RecordingPolicy(ContextPolicy):
        def __init__(self) -> None:
            self.requests: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.requests.append(request)
            return request.messages

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    context_policy = RecordingPolicy()
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="stored-model"),
        context_policy=context_policy,
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_model_update",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_model_update",
                messages=[Message.text("user", "second request")],
                model="upgraded-model",
            ),
        )
    )

    session_after_update = asyncio.run(store.load("sess_model_update"))
    assert session_after_update is not None
    assert session_after_update.model == "upgraded-model"
    assert second_events[1].type == EventType.MODEL_STARTED
    assert second_events[1].payload["model"] == "upgraded-model"

    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_model_update",
                messages=[Message.text("user", "third request")],
            ),
        )
    )

    assert [request.model for request in provider.requests] == [
        "stored-model",
        "upgraded-model",
        "upgraded-model",
    ]
    assert [request.agent.model for request in context_policy.requests] == [
        "stored-model",
        "upgraded-model",
        "upgraded-model",
    ]


def test_cayu_app_resume_rejects_active_sessions():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("should not run"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_running_session() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_running",
                messages=[Message.text("user", "hi")],
            ),
            identity=_test_session_identity(),
        )
        await store.update_status("sess_running", SessionStatus.RUNNING)
        await store.append_transcript_messages(
            "sess_running",
            [Message.text("user", "hi")],
        )

    asyncio.run(setup_running_session())

    with pytest.raises(ValueError, match="transition not allowed"):
        asyncio.run(
            collect_resume_events(
                app,
                ResumeRequest(
                    session_id="sess_running",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )

    assert provider.requests == []
    session = asyncio.run(store.load("sess_running"))
    assert session is not None
    assert session.status == SessionStatus.RUNNING


def test_cayu_app_resume_marks_session_failed_when_transcript_load_fails():
    class BrokenTranscriptStore(InMemorySessionStore):
        async def load_transcript(self, session_id: str) -> list[Message]:
            raise RuntimeError("transcript unavailable")

    store = BrokenTranscriptStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("should not run"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_completed_session() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_broken_transcript",
                messages=[Message.text("user", "hi")],
            ),
            identity=_test_session_identity(),
        )
        await store.update_status("sess_broken_transcript", SessionStatus.COMPLETED)

    asyncio.run(setup_completed_session())

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_broken_transcript",
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert provider.requests == []
    assert [event.type for event in events] == [EventType.SESSION_FAILED]
    assert events[0].payload == {
        "error": "transcript unavailable",
        "error_type": "RuntimeError",
    }
    session = asyncio.run(store.load("sess_broken_transcript"))
    assert session is not None
    assert session.status == SessionStatus.FAILED


def test_cayu_app_resume_uses_context_policy_and_preserves_full_transcript():
    class LastMessagePolicy(ContextPolicy):
        def __init__(self) -> None:
            self.seen_messages: list[list[str]] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.seen_messages.append([message.content[0].text for message in request.messages])
            return [request.messages[-1]]

    store = InMemorySessionStore()
    policy = LastMessagePolicy()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_resume_context",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_resume_context",
                messages=[Message.text("user", "second request")],
                metadata={"source": "resume-test"},
            ),
        )
    )

    assert policy.seen_messages[-1] == [
        "first request",
        "first answer",
        "second request",
    ]
    assert [message.content[0].text for message in provider.requests[1].messages] == [
        "second request"
    ]
    transcript = asyncio.run(store.load_transcript("sess_resume_context"))
    assert [message.content[0].text for message in transcript] == [
        "first request",
        "first answer",
        "second request",
        "second answer",
    ]


def test_cayu_app_resume_continues_tool_rounds():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("ready"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "from resumed tool"},
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

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_resume_tool",
                messages=[Message.text("user", "first")],
            ),
        )
    )

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_resume_tool",
                messages=[Message.text("user", "use tool")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert provider.requests[2].messages[-1].role == "tool"
    assert provider.requests[2].messages[-1].content[0].content == "from resumed tool"
    transcript = asyncio.run(store.load_transcript("sess_resume_tool"))
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


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
        await task_store.create_task(TaskCreate(task_id="task_runtime_failure", type="respond"))
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
        await task_store.create_task(TaskCreate(task_id="task_claimed_elsewhere", type="respond"))
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
        "error": ("Task task_claimed_elsewhere cannot transition to running from running"),
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

    sink_failures = [event for event in persisted if event.type == EventType.RUNTIME_SINK_FAILED]
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

    transcript = asyncio.run(store.load_transcript("sess_tool"))
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert transcript[0].content[0].text == "use the tool"
    assert transcript[1].content[0].type == "tool_call"
    assert transcript[2].content[0].type == "tool_result"
    assert transcript[3].content[0].text == "done"


def test_cayu_app_blocks_tool_call_before_execution_with_tool_policy():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("blocked handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=StaticToolPolicy(deny=["side_effect"]),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_blocked_tool",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert tool.calls == []
    blocked_event = events[4]
    assert blocked_event.tool_name == "side_effect"
    assert blocked_event.payload == {
        "tool_call_id": "call_1",
        "decision": "deny",
        "reason": "Tool denied by policy: side_effect",
        "metadata": {},
        "result": {
            "content": "Tool denied by policy: side_effect",
            "structured": {
                "decision": "deny",
                "reason": "Tool denied by policy: side_effect",
                "metadata": {},
            },
            "artifacts": [],
            "is_error": True,
        },
    }

    assert provider.requests[1].messages[-1].role == "tool"
    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.tool_call_id == "call_1"
    assert tool_result_part.tool_name == "side_effect"
    assert tool_result_part.content == "Tool denied by policy: side_effect"
    assert tool_result_part.is_error is True


def test_cayu_app_interrupts_session_when_tool_policy_requires_approval():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_approval",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.TOOL_CALL_APPROVAL_REQUESTED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert tool.calls == []

    approval = events[4].payload["approval"]
    assert approval["tool_call_id"] == "call_1"
    assert approval["tool_name"] == "side_effect"
    assert approval["arguments"] == {"value": "secret"}
    assert approval["agent_name"] == "assistant"
    assert approval["reason"] == "Approval required for side_effect."
    assert approval["metadata"] == {"scope": "human"}
    assert [call["tool_call_id"] for call in approval["tool_calls"]] == ["call_1"]

    session = asyncio.run(store.load("sess_tool_approval"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED

    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_approval"))
    assert checkpoint is not None
    assert checkpoint["pending_tool_approval"]["approval_id"] == approval["approval_id"]

    transcript = asyncio.run(store.load_transcript("sess_tool_approval"))
    assert [message.role for message in transcript] == ["user", "assistant"]
    assert transcript[-1].content[0].type == "tool_call"


def test_cayu_app_resolves_approved_tool_call_and_continues_session():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("approved handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_approval_allow",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = interrupt_events[4].payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_tool_approval_allow",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
                reason="approved by test",
                metadata={"reviewer": "test"},
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert tool.calls == [{"value": "secret"}]
    assert provider.requests[1].messages[-2].role == "assistant"
    assert provider.requests[1].messages[-1].role == "tool"
    assert provider.requests[1].messages[-1].content[0].content == "recorded"

    session = asyncio.run(store.load("sess_tool_approval_allow"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert asyncio.run(store.load_checkpoint("sess_tool_approval_allow")) == {}


def test_cayu_app_resolves_approved_multi_tool_round_in_order():
    store = InMemorySessionStore()
    side_effect = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "first"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="side_effect",
                    arguments={"value": "second"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("round handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool(), side_effect],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_multi_tool_approval",
                messages=[Message.text("user", "use both tools")],
            ),
        )
    )
    approval = interrupt_events[4].payload["approval"]
    assert [call["tool_call_id"] for call in approval["tool_calls"]] == [
        "call_1",
        "call_2",
    ]
    assert side_effect.calls == []

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_multi_tool_approval",
                approval_id=approval["approval_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert side_effect.calls == [{"value": "second"}]

    tool_result_message = provider.requests[1].messages[-1]
    assert tool_result_message.role == "tool"
    assert [part.tool_call_id for part in tool_result_message.content] == [
        "call_1",
        "call_2",
    ]
    assert [part.content for part in tool_result_message.content] == [
        "first",
        "recorded",
    ]


def test_cayu_app_resolves_denied_tool_call_and_continues_session():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("denial handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_approval_deny",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = interrupt_events[4].payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_tool_approval_deny",
                approval_id=approval_id,
                decision=ToolApprovalDecision.DENY,
                reason="not safe",
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert tool.calls == []
    assert provider.requests[1].messages[-1].role == "tool"
    tool_result = provider.requests[1].messages[-1].content[0]
    assert tool_result.type == "tool_result"
    assert tool_result.content == "Tool call denied by approval: not safe"
    assert tool_result.structured["denied_by_approval"] is True
    assert tool_result.structured["skipped_due_to_approval_denial"] is False
    assert tool_result.structured["tool_call_id"] == "call_1"
    assert tool_result.structured["tool_name"] == "side_effect"
    assert tool_result.is_error is True


def test_cayu_app_denied_multi_tool_round_marks_skipped_calls_explicitly():
    store = InMemorySessionStore()
    side_effect = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "first"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="side_effect",
                    arguments={"value": "second"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("denied round handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool(), side_effect],
        tool_policy=SideEffectApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_multi_tool_approval_deny",
                messages=[Message.text("user", "use both tools")],
            ),
        )
    )
    approval = interrupt_events[4].payload["approval"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_multi_tool_approval_deny",
                approval_id=approval["approval_id"],
                decision=ToolApprovalDecision.DENY,
                reason="not safe",
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert side_effect.calls == []

    tool_result_message = provider.requests[1].messages[-1]
    assert tool_result_message.role == "tool"
    assert [part.tool_call_id for part in tool_result_message.content] == [
        "call_1",
        "call_2",
    ]

    skipped_result = tool_result_message.content[0]
    assert skipped_result.content == (
        "Tool call skipped because approval was denied for the same tool round: not safe"
    )
    assert skipped_result.structured["denied_by_approval"] is False
    assert skipped_result.structured["skipped_due_to_approval_denial"] is True
    assert skipped_result.structured["denied_tool_call_id"] == "call_2"
    assert skipped_result.structured["denied_tool_name"] == "side_effect"

    denied_result = tool_result_message.content[1]
    assert denied_result.content == "Tool call denied by approval: not safe"
    assert denied_result.structured["denied_by_approval"] is True
    assert denied_result.structured["skipped_due_to_approval_denial"] is False
    assert denied_result.structured["tool_call_id"] == "call_2"
    assert denied_result.structured["tool_name"] == "side_effect"


def test_cayu_app_keeps_pending_approval_if_atomic_resolution_close_fails():
    store = FailingApprovalCloseStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("resumed after retry"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_atomic_close_failure",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = interrupt_events[4].payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_atomic_close_failure",
                approval_id=approval_id,
                decision=ToolApprovalDecision.DENY,
                reason="not safe",
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[-1].payload["approval"]["approval_id"] == approval_id
    assert events[-1].payload["error"] == "approval close unavailable"

    session = asyncio.run(store.load("sess_approval_atomic_close_failure"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    checkpoint = asyncio.run(store.load_checkpoint("sess_approval_atomic_close_failure"))
    assert checkpoint is not None
    assert checkpoint["pending_tool_approval"]["approval_id"] == approval_id

    transcript = asyncio.run(store.load_transcript("sess_approval_atomic_close_failure"))
    assert [message.role for message in transcript] == ["user", "assistant"]

    retry_events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_atomic_close_failure",
                approval_id=approval_id,
                decision=ToolApprovalDecision.DENY,
                reason="not safe",
            ),
        )
    )

    assert retry_events[-1].type == EventType.SESSION_COMPLETED
    assert asyncio.run(store.load_checkpoint("sess_approval_atomic_close_failure")) == {}
    assert provider.requests[1].messages[-2].role == "assistant"
    assert provider.requests[1].messages[-1].role == "tool"


def test_cayu_app_retries_approval_close_without_rerunning_completed_tool():
    store = FailingApprovalCloseStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("resumed after retry"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_approved_close_failure",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = interrupt_events[4].payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_approved_close_failure",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert tool.calls == [{"value": "secret"}]

    retry_events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_approved_close_failure",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert [event.type for event in retry_events] == [
        EventType.SESSION_RESUMED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert tool.calls == [{"value": "secret"}]
    assert provider.requests[1].messages[-1].role == "tool"
    assert provider.requests[1].messages[-1].content[0].content == "recorded"


def test_cayu_app_rejects_denial_retry_after_approved_tool_executed():
    store = FailingApprovalCloseStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_reject_conflicting_deny",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = interrupt_events[4].payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_reject_conflicting_deny",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    assert tool.calls == [{"value": "secret"}]

    retry_events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_reject_conflicting_deny",
                approval_id=approval_id,
                decision=ToolApprovalDecision.DENY,
                reason="changed mind",
            ),
        )
    )

    assert [event.type for event in retry_events] == [EventType.SESSION_INTERRUPTED]
    assert "cannot be retried as denied" in retry_events[0].payload["error"]
    assert tool.calls == [{"value": "secret"}]
    assert (
        asyncio.run(store.load_checkpoint("sess_approval_reject_conflicting_deny"))[
            "pending_tool_approval"
        ]["approval_id"]
        == approval_id
    )


def test_cayu_app_approval_recovery_ignores_unrelated_terminal_tool_events():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("resumed"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_scoped_recovery",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = interrupt_events[4].payload["approval"]["approval_id"]
    asyncio.run(
        store.append_event(
            "sess_approval_scoped_recovery",
            Event(
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="sess_approval_scoped_recovery",
                agent_name="assistant",
                tool_name="side_effect",
                payload={
                    "tool_call_id": "call_1",
                    "result": ToolResult(content="unrelated").model_dump(),
                },
            ),
        )
    )

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_scoped_recovery",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[2].payload["approval_id"] == approval_id
    assert events[3].payload["approval_id"] == approval_id
    assert tool.calls == [{"value": "secret"}]
    assert provider.requests[1].messages[-1].content[0].content == "recorded"


def test_cayu_app_requires_manual_recovery_for_started_tool_without_terminal_event():
    store = FailingTerminalToolEventStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered and continued"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_started_without_terminal",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = interrupt_events[4].payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_started_without_terminal",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert tool.calls == [{"value": "secret"}]

    retry_events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_started_without_terminal",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert [event.type for event in retry_events] == [EventType.SESSION_INTERRUPTED]
    assert retry_events[-1].payload["manual_recovery_required"] is True
    assert retry_events[-1].payload["tool_call_id"] == "call_1"
    assert tool.calls == [{"value": "secret"}]
    session = asyncio.run(store.load("sess_approval_started_without_terminal"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED

    recovery_events = asyncio.run(
        collect_tool_approval_recovery_events(
            app,
            ToolApprovalRecoveryRequest(
                session_id="sess_approval_started_without_terminal",
                approval_id=approval_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="side effect completed externally",
                structured={"source": "operator"},
                reason="operator confirmed the side effect completed",
            ),
        )
    )

    assert [event.type for event in recovery_events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert recovery_events[1].payload["approval_id"] == approval_id
    assert recovery_events[1].payload["manual_recovery"] is True
    assert recovery_events[1].payload["result"]["content"] == ("side effect completed externally")
    assert tool.calls == [{"value": "secret"}]
    assert provider.requests[1].messages[-1].role == "tool"
    assert provider.requests[1].messages[-1].content[0].content == (
        "side effect completed externally"
    )
    session = asyncio.run(store.load("sess_approval_started_without_terminal"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_recovery_does_not_append_terminal_event_without_session_claim():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recovery_claim_required",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = interrupt_events[4].payload["approval"]["approval_id"]
    asyncio.run(store.update_status("sess_recovery_claim_required", SessionStatus.RUNNING))

    with pytest.raises(ValueError, match="status transition not allowed"):
        asyncio.run(
            collect_tool_approval_recovery_events(
                app,
                ToolApprovalRecoveryRequest(
                    session_id="sess_recovery_claim_required",
                    approval_id=approval_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="confirmed externally",
                ),
            )
        )

    events = asyncio.run(store.load_events("sess_recovery_claim_required"))
    assert not any(event.payload.get("manual_recovery") is True for event in events)


def test_cayu_app_rejects_message_resume_with_pending_tool_approval():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_approval_resume_blocked",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    with pytest.raises(RuntimeError, match="pending tool approval"):
        asyncio.run(
            collect_resume_events(
                app,
                ResumeRequest(
                    session_id="sess_tool_approval_resume_blocked",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )


def test_cayu_app_forks_interrupted_session_with_pending_approval_checkpoint():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupted_fork_source",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = events[4].payload["approval"]["approval_id"]
    source_checkpoint = asyncio.run(store.load_checkpoint("sess_interrupted_fork_source"))
    assert source_checkpoint is not None
    source_checkpoint["pending_tool_approval"]["task_id"] = "source_task"
    asyncio.run(store.checkpoint("sess_interrupted_fork_source", source_checkpoint))

    fork_events = asyncio.run(
        collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_interrupted_fork_source",
                session_id="sess_interrupted_fork_child",
            ),
        )
    )

    assert [event.type for event in fork_events] == [EventType.SESSION_FORKED]
    fork = asyncio.run(store.load("sess_interrupted_fork_child"))
    assert fork is not None
    assert fork.status == SessionStatus.INTERRUPTED
    assert fork.parent_session_id == "sess_interrupted_fork_source"
    fork_checkpoint = asyncio.run(store.load_checkpoint("sess_interrupted_fork_child"))
    assert fork_checkpoint is not None
    assert fork_checkpoint["pending_tool_approval"]["approval_id"] == approval_id
    assert fork_checkpoint["pending_tool_approval"]["task_id"] is None

    source_checkpoint_after = asyncio.run(store.load_checkpoint("sess_interrupted_fork_source"))
    assert source_checkpoint_after is not None
    assert source_checkpoint_after["pending_tool_approval"]["task_id"] == "source_task"

    with pytest.raises(ValueError, match="without checkpoint state"):
        asyncio.run(
            collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id="sess_interrupted_fork_source",
                    session_id="sess_interrupted_fork_no_checkpoint",
                    copy_checkpoint=False,
                ),
            )
        )

    app.register_agent(AgentSpec(name="other", model="fake-model"))
    with pytest.raises(ValueError, match="different agent"):
        asyncio.run(
            collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id="sess_interrupted_fork_source",
                    session_id="sess_interrupted_fork_other_agent",
                    agent_name="other",
                ),
            )
        )


def test_in_memory_session_store_rejects_fork_status_mismatch():
    store = InMemorySessionStore()

    async def run_operations() -> None:
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_store_fork_status_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        with pytest.raises(ValueError, match="Fork status must match"):
            await store.create_fork(
                source_session_id=source.id,
                fork=Session(
                    id="sess_store_fork_status_child",
                    agent_name="assistant",
                    provider_name="fake",
                    model="fake-model",
                    parent_session_id=source.id,
                    status=SessionStatus.RUNNING,
                ),
                source_statuses={SessionStatus.COMPLETED},
                transcript_cursor=None,
                checkpoint_transform=None,
            )

    asyncio.run(run_operations())


def test_in_memory_session_store_rejects_fork_provider_mismatch():
    store = InMemorySessionStore()

    async def run_operations() -> None:
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_store_fork_provider_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        with pytest.raises(ValueError, match="Fork provider_name must match"):
            await store.create_fork(
                source_session_id=source.id,
                fork=Session(
                    id="sess_store_fork_provider_child",
                    agent_name="assistant",
                    provider_name="other",
                    model="fake-model",
                    parent_session_id=source.id,
                    status=SessionStatus.COMPLETED,
                ),
                source_statuses={SessionStatus.COMPLETED},
                transcript_cursor=None,
                checkpoint_transform=None,
            )

    asyncio.run(run_operations())


def test_in_memory_session_store_transforms_current_checkpoint_during_fork():
    store = InMemorySessionStore()

    async def run_operations() -> None:
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_store_fork_checkpoint_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        await store.checkpoint(source.id, {"version": 2})

        await store.create_fork(
            source_session_id=source.id,
            fork=Session(
                id="sess_store_fork_checkpoint_child",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                parent_session_id=source.id,
                status=SessionStatus.COMPLETED,
            ),
            source_statuses={SessionStatus.COMPLETED},
            transcript_cursor=None,
            checkpoint_transform=lambda _session, checkpoint: {
                "copied_version": checkpoint["version"] if checkpoint else None
            },
        )

        assert await store.load_checkpoint("sess_store_fork_checkpoint_child") == {
            "copied_version": 2
        }

    asyncio.run(run_operations())


def test_cayu_app_rejects_tool_approval_resolution_from_failed_status():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_failed_tool_approval",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = events[4].payload["approval"]["approval_id"]
    asyncio.run(store.update_status("sess_failed_tool_approval", SessionStatus.FAILED))

    with pytest.raises(ValueError, match="status transition not allowed"):
        asyncio.run(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id="sess_failed_tool_approval",
                    approval_id=approval_id,
                    decision=ToolApprovalDecision.APPROVE,
                ),
            )
        )

    with pytest.raises(ValueError, match="status transition not allowed"):
        asyncio.run(
            collect_tool_approval_recovery_events(
                app,
                ToolApprovalRecoveryRequest(
                    session_id="sess_failed_tool_approval",
                    approval_id=approval_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="confirmed externally",
                ),
            )
        )


def test_cayu_app_tool_policy_allowlist_blocks_unlisted_registered_tools():
    store = InMemorySessionStore()
    blocked_tool = SideEffectTool()
    allowed_tool = EchoTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "blocked"},
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
        tools=[allowed_tool, blocked_tool],
        tool_policy=StaticToolPolicy(allow=["echo"]),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_allowlist_tool_policy",
                messages=[Message.text("user", "use the blocked tool")],
            ),
        )
    )

    assert events[4].type == EventType.TOOL_CALL_BLOCKED
    assert events[4].payload["reason"] == "Tool not allowed by policy: side_effect"
    assert blocked_tool.calls == []


def test_cayu_app_tool_policy_receives_copied_tool_call_arguments():
    class RecordingToolPolicy(ToolPolicy):
        def __init__(self) -> None:
            self.requests: list[ToolPolicyRequest] = []

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            self.requests.append(request)
            request.arguments["nested"]["value"] = "mutated"
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    policy = RecordingToolPolicy()
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
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
        tools=[tool],
        tool_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_policy_argument_copy",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert policy.requests[0].session.id == "sess_policy_argument_copy"
    assert policy.requests[0].agent.name == "assistant"
    assert policy.requests[0].tool_name == "side_effect"
    assert policy.requests[0].tool_call_id == "call_1"
    assert policy.requests[0].environment_name is None
    assert policy.requests[0].workspace_id is None
    assert policy.requests[0].metadata == {}
    assert tool.calls == [{"nested": {"value": "original"}}]


def test_cayu_app_tool_policy_receives_workspace_identity():
    class RecordingToolPolicy(ToolPolicy):
        def __init__(self) -> None:
            self.requests: list[ToolPolicyRequest] = []

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            self.requests.append(request)
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    policy = RecordingToolPolicy()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
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
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=MemoryWorkspace("workspace_1"),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_policy_workspace",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert policy.requests[0].environment_name == "local"
    assert policy.requests[0].workspace_id == "workspace_1"


def test_cayu_app_tool_policy_receives_run_request_metadata_copy():
    class RecordingToolPolicy(ToolPolicy):
        def __init__(self) -> None:
            self.requests: list[ToolPolicyRequest] = []

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            self.requests.append(request)
            request.metadata["tenant"]["id"] = "mutated"
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    policy = RecordingToolPolicy()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
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
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_policy_run_metadata",
                messages=[Message.text("user", "use the tool")],
                metadata={"tenant": {"id": "tenant_1"}},
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert policy.requests[0].metadata == {"tenant": {"id": "mutated"}}
    session = asyncio.run(app.session_store.load("sess_policy_run_metadata"))
    assert session is not None
    assert session.metadata == {"tenant": {"id": "tenant_1"}}


def test_cayu_app_tool_policy_receives_resume_request_metadata_copy():
    class RecordingToolPolicy(ToolPolicy):
        def __init__(self) -> None:
            self.requests: list[ToolPolicyRequest] = []

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            self.requests.append(request)
            request.metadata["resume"]["id"] = "mutated"
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    policy = RecordingToolPolicy()
    tool = SideEffectTool()
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("ready"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
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
        tool_policy=policy,
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_policy_resume_metadata",
                messages=[Message.text("user", "first")],
                metadata={"original": {"id": "run"}},
            ),
        )
    )
    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_policy_resume_metadata",
                messages=[Message.text("user", "use the tool")],
                metadata={"resume": {"id": "resume_1"}},
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert policy.requests[0].metadata == {"resume": {"id": "mutated"}}
    session = asyncio.run(store.load("sess_policy_resume_metadata"))
    assert session is not None
    assert session.metadata == {"original": {"id": "run"}}


def test_cayu_app_tool_policy_blocked_event_uses_default_reason_when_omitted():
    class ReasonlessDenyPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY)

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=ReasonlessDenyPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_policy_default_reason",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    blocked_event = events[4]
    assert blocked_event.type == EventType.TOOL_CALL_BLOCKED
    assert blocked_event.payload["reason"] == "Tool call denied by policy."
    assert blocked_event.payload["result"]["content"] == "Tool call denied by policy."


def test_cayu_app_fails_session_when_tool_policy_raises_before_execution():
    class FailingToolPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            raise RuntimeError("policy unavailable")

    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=FailingToolPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_policy_raises",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert tool.calls == []
    assert events[-1].payload == {
        "error": "policy unavailable",
        "error_type": "RuntimeError",
    }


def test_cayu_app_fails_session_when_tool_policy_returns_invalid_result():
    class InvalidToolPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            return {"decision": "deny"}  # type: ignore[return-value]

    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=InvalidToolPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_policy_invalid_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_FAILED
    assert tool.calls == []
    assert events[-1].payload == {
        "error": "Tool policies must return ToolPolicyResult instances. Received dict.",
        "error_type": "TypeError",
    }


def test_cayu_app_context_policy_can_trim_model_facing_messages():
    class LastMessagePolicy(ContextPolicy):
        def __init__(self) -> None:
            self.seen: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.seen.append(request)
            request.messages[0].content[0].text = "mutated inside policy"
            return [request.messages[-1]]

    policy = LastMessagePolicy()
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("trimmed"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_trim_context",
                messages=[
                    Message.text("user", "old context"),
                    Message.text("user", "current request"),
                ],
                metadata={"source": "test"},
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(policy.seen) == 1
    assert policy.seen[0].session.id == "sess_trim_context"
    assert policy.seen[0].agent.name == "assistant"
    assert policy.seen[0].step == 1
    assert policy.seen[0].metadata == {"source": "test"}
    assert [message.content[0].text for message in provider.requests[0].messages] == [
        "current request"
    ]

    transcript = asyncio.run(store.load_transcript("sess_trim_context"))
    assert [message.content[0].text for message in transcript] == [
        "old context",
        "current request",
        "trimmed",
    ]


def test_cayu_app_context_policy_can_replace_tool_results_for_model_only():
    class CompactToolResultPolicy(ContextPolicy):
        async def build(self, request: ContextRequest) -> list[Message]:
            compacted: list[Message] = []
            for message in request.messages:
                if message.role == "tool":
                    compacted.append(
                        Message.tool_result(
                            tool_call_id=message.content[0].tool_call_id,
                            tool_name=message.content[0].tool_name,
                            content="[tool result compacted]",
                            structured={"compacted": True},
                        )
                    )
                else:
                    compacted.append(message)
            return compacted

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "large tool output"},
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
        context_policy=CompactToolResultPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_tool_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    provider_tool_result = provider.requests[1].messages[-1].content[0]
    assert provider_tool_result.content == "[tool result compacted]"
    assert provider_tool_result.structured == {"compacted": True}

    transcript = asyncio.run(store.load_transcript("sess_compact_tool_result"))
    stored_tool_result = transcript[2].content[0]
    assert stored_tool_result.content == "large tool output"
    assert stored_tool_result.structured == {
        "agent": "assistant",
        "echoed": "large tool output",
    }


def test_cayu_app_fails_cleanly_when_context_policy_returns_invalid_output():
    class BadPolicy(ContextPolicy):
        async def build(self, request: ContextRequest):
            return tuple(request.messages)

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("should not run"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=BadPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_bad_context_policy",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert provider.requests == []
    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.SESSION_FAILED,
    ]
    assert (
        events[-1].payload["error"]
        == "ContextPolicy.build() must return a list of Message instances."
    )
    transcript = asyncio.run(store.load_transcript("sess_bad_context_policy"))
    assert [message.content[0].text for message in transcript] == ["hi"]


def test_cayu_app_rejects_context_policy_that_cuts_through_tool_round():
    class OrphanToolResultPolicy(ContextPolicy):
        async def build(self, request: ContextRequest) -> list[Message]:
            return [request.messages[-1]]

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "large tool output"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("should not run"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        context_policy=OrphanToolResultPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_orphan_tool_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert "tool results without preceding assistant tool calls" in events[-1].payload["error"]


def test_builtin_context_policies_trim_recent_history_safely():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current"),
    ]

    turns_policy = RecentTurnsContextPolicy(max_user_turns=1)
    messages_policy = MessageWindowContextPolicy(max_messages=2)

    async def build_contexts() -> tuple[list[Message], list[Message]]:
        request = ContextRequest(
            session=_test_session(),
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=messages,
            step=1,
        )
        return (
            await turns_policy.build(request),
            await messages_policy.build(request),
        )

    turns_context, messages_context = asyncio.run(build_contexts())

    assert [message.role for message in turns_context] == ["system", "user"]
    assert [message.content[0].text for message in turns_context] == [
        "You are careful.",
        "current",
    ]
    assert [message.role for message in messages_context] == ["system", "user"]
    assert [message.content[0].text for message in messages_context] == [
        "You are careful.",
        "current",
    ]


def test_model_compactor_summarizes_context_with_provider():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("summary "),
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"usage": {"output_tokens": 2}}),
        ]
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        options={"anthropic": {"max_tokens": 512}},
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("user", "old request"),
                    Message.text("assistant", "old answer"),
                ],
                existing_summary="previous context",
            )
        )
    )

    assert result.summary == "summary done"
    assert result.metadata == {
        "compactor": "ModelCompactor",
        "provider": "fake",
        "model": "summary-model",
        "input_truncated": False,
        "max_input_chars": 120000,
        "completed": {"usage": {"output_tokens": 2}},
    }
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.model == "summary-model"
    assert request.tools == []
    assert request.options == {"anthropic": {"max_tokens": 512}}
    assert [message.role for message in request.messages] == ["system", "user"]
    prompt = request.messages[1].content[0].text
    assert "Existing summary:\nprevious context" in prompt
    assert "user: old request" in prompt
    assert "assistant: old answer" in prompt


def test_model_compactor_rejects_tool_calls_from_compaction_model():
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="echo",
                arguments={"text": "bad"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    compactor = ModelCompactor(provider=provider, model="summary-model")

    with pytest.raises(RuntimeError, match="must not call tools"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )


def test_model_compactor_fails_on_provider_error():
    provider = FakeProvider([ModelStreamEvent.error("provider unavailable")])
    compactor = ModelCompactor(provider=provider, model="summary-model")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )


def test_model_compactor_bounds_large_compaction_input():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("bounded summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        max_input_chars=1000,
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "x" * 5000)],
                existing_summary="important previous summary",
            )
        )
    )

    prompt = provider.requests[0].messages[1].content[0].text
    assert len(prompt) == 1000
    assert prompt.startswith("Summarize the transcript below so a future agent step can continue")
    assert "Existing summary:\nimportant previous summary" in prompt
    assert "Transcript to compact:\n[compaction transcript clipped" in prompt
    assert result.metadata["input_truncated"] is True
    assert result.metadata["max_input_chars"] == 1000


def test_model_compactor_accepts_custom_prompt_builder():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("custom summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    seen_requests: list[CompactionRequest] = []

    def prompt_builder(request: CompactionRequest) -> str:
        seen_requests.append(request)
        return f"custom prompt for {request.session.id}"

    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        prompt_builder=prompt_builder,
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert result.summary == "custom summary"
    assert len(seen_requests) == 1
    assert provider.requests[0].messages[1].content[0].text == ("custom prompt for sess_context")


def test_model_compactor_accepts_exported_default_prompt_builder():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("default summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        prompt_builder=default_compaction_prompt,
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert result.summary == "default summary"
    assert "Transcript to compact:\nuser: old request" in (
        provider.requests[0].messages[1].content[0].text
    )


def test_cayu_app_checkpoint_compacts_model_context_without_rewriting_transcript():
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt="You are careful.",
        ),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction",
                messages=[
                    Message.text("user", "old one"),
                    Message.text("assistant", "old answer one"),
                    Message.text("user", "old two"),
                    Message.text("assistant", "old answer two"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[1].payload == {
        "checkpoint": "context_compaction",
        "compactor": "RecordingCompactor",
        "compacted_transcript_cursor": 5,
        "previous_compacted_transcript_cursor": 1,
        "newly_compacted_message_count": 4,
        "recent_message_count": 1,
    }
    assert events[2].payload == {
        "checkpoint": "context_compaction",
        "compactor": "RecordingCompactor",
        "compacted_transcript_cursor": 5,
        "previous_compacted_transcript_cursor": 1,
        "newly_compacted_message_count": 4,
        "recent_message_count": 1,
        "summary_chars": len("old one|old answer one|old two|old answer two"),
        "metadata": {"request_count": 1},
    }
    assert "summary" not in events[2].payload
    assert events[3].payload == {
        "checkpoint": "context_compaction",
        "compacted_transcript_cursor": 5,
        "previous_compacted_transcript_cursor": 1,
        "newly_compacted_message_count": 4,
        "recent_message_count": 1,
    }
    assert len(compactor.requests) == 1
    assert [message.content[0].text for message in compactor.requests[0].messages] == [
        "old one",
        "old answer one",
        "old two",
        "old answer two",
    ]

    provider_context = provider.requests[0].messages
    assert [message.role for message in provider_context] == [
        "system",
        "user",
        "user",
    ]
    assert provider_context[1].content[0].text == (
        "Previous session context summary:\nold one|old answer one|old two|old answer two"
    )
    assert provider_context[2].content[0].text == "current"

    transcript = asyncio.run(store.load_transcript("sess_compaction"))
    assert [message.content[0].text for message in transcript] == [
        "You are careful.",
        "old one",
        "old answer one",
        "old two",
        "old answer two",
        "current",
        "final answer",
    ]
    checkpoint = asyncio.run(store.load_checkpoint("sess_compaction"))
    assert checkpoint == {
        "context_compaction": {
            "version": 1,
            "summary": "old one|old answer one|old two|old answer two",
            "compacted_transcript_cursor": 5,
            "metadata": {"request_count": 1},
        }
    }


def test_cayu_app_checkpoint_compaction_can_use_model_compactor():
    store = InMemorySessionStore()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
                options={"anthropic": {"max_tokens": 512}},
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_model_compaction",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(compactor_provider.requests) == 1
    assert compactor_provider.requests[0].model == "summary-model"
    assert compactor_provider.requests[0].tools == []
    assert compactor_provider.requests[0].options == {"anthropic": {"max_tokens": 512}}
    assert events[1].payload == {
        "checkpoint": "context_compaction",
        "compactor": "ModelCompactor",
        "compacted_transcript_cursor": 2,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 2,
        "recent_message_count": 1,
    }
    assert events[2].payload == {
        "checkpoint": "context_compaction",
        "compactor": "ModelCompactor",
        "compacted_transcript_cursor": 2,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 2,
        "recent_message_count": 1,
        "summary_chars": len("model summary"),
        "metadata": {
            "compactor": "ModelCompactor",
            "provider": "fake",
            "model": "summary-model",
            "input_truncated": False,
            "max_input_chars": 120000,
            "completed": {"finish_reason": "stop"},
        },
    }
    assert "model summary" not in str(events[2].payload)

    provider_context = runtime_provider.requests[0].messages
    assert [message.role for message in provider_context] == ["user", "user"]
    assert provider_context[0].content[0].text == (
        "Previous session context summary:\nmodel summary"
    )
    assert provider_context[1].content[0].text == "current"

    checkpoint = asyncio.run(store.load_checkpoint("sess_model_compaction"))
    assert checkpoint == {
        "context_compaction": {
            "version": 1,
            "summary": "model summary",
            "compacted_transcript_cursor": 2,
            "metadata": {
                "compactor": "ModelCompactor",
                "provider": "fake",
                "model": "summary-model",
                "input_truncated": False,
                "max_input_chars": 120000,
                "completed": {"finish_reason": "stop"},
            },
        }
    }


def test_cayu_app_emits_compaction_failed_event_before_session_failure():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=FailingCompactor(),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_failed",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_FAILED,
        EventType.SESSION_FAILED,
    ]
    assert events[2].payload == {
        "checkpoint": "context_compaction",
        "compactor": "FailingCompactor",
        "compacted_transcript_cursor": 2,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 2,
        "recent_message_count": 1,
        "error": "compaction unavailable",
        "error_type": "RuntimeError",
    }
    assert events[3].payload == {
        "error": "compaction unavailable",
        "error_type": "RuntimeError",
    }
    assert provider.requests == []
    assert asyncio.run(store.load_checkpoint("sess_compaction_failed")) is None


def test_cayu_app_emits_compaction_events_before_checkpoint_failure():
    class BrokenCheckpointStore(InMemorySessionStore):
        async def checkpoint(self, session_id: str, state: dict) -> None:
            raise RuntimeError("checkpoint unavailable")

    store = BrokenCheckpointStore()
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(runtime_provider, default=True)
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

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_checkpoint_failure_after_compaction",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[2].payload == {
        "checkpoint": "context_compaction",
        "compactor": "ModelCompactor",
        "compacted_transcript_cursor": 2,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 2,
        "recent_message_count": 1,
        "summary_chars": len("model summary"),
        "metadata": {
            "compactor": "ModelCompactor",
            "provider": "fake",
            "model": "summary-model",
            "input_truncated": False,
            "max_input_chars": 120000,
            "completed": {"finish_reason": "stop"},
        },
    }
    assert events[3].payload == {
        "error": "checkpoint unavailable",
        "error_type": "RuntimeError",
    }
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []


def test_cayu_app_checkpoint_compaction_ignores_cursor_without_valid_summary():
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    session = asyncio.run(
        store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_bad_checkpoint_pair",
                messages=[],
            ),
            identity=_test_session_identity(),
        )
    )
    asyncio.run(store.update_status(session.id, SessionStatus.COMPLETED))
    asyncio.run(
        store.append_transcript_messages(
            session.id,
            [
                Message.text("user", "old"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current"),
            ],
        )
    )
    asyncio.run(
        store.checkpoint(
            session.id,
            {
                "context_compaction": {
                    "version": 1,
                    "summary": 123,
                    "compacted_transcript_cursor": 2,
                    "metadata": {},
                }
            },
        )
    )

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session.id,
                messages=[Message.text("user", "follow up")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert [message.content[0].text for message in compactor.requests[0].messages] == [
        "old",
        "old answer",
        "current",
    ]
    assert compactor.requests[0].existing_summary is None
    assert events[3].payload == {
        "checkpoint": "context_compaction",
        "compacted_transcript_cursor": 3,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 3,
        "recent_message_count": 1,
    }


def test_cayu_app_checkpoint_compaction_ignores_summary_without_valid_cursor():
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    session = asyncio.run(
        store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_bad_checkpoint_cursor",
                messages=[],
            ),
            identity=_test_session_identity(),
        )
    )
    asyncio.run(store.update_status(session.id, SessionStatus.COMPLETED))
    asyncio.run(
        store.append_transcript_messages(
            session.id,
            [
                Message.text("user", "old"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current"),
            ],
        )
    )
    asyncio.run(
        store.checkpoint(
            session.id,
            {
                "context_compaction": {
                    "version": 1,
                    "summary": "stale summary",
                    "compacted_transcript_cursor": "bad",
                    "metadata": {},
                }
            },
        )
    )

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session.id,
                messages=[Message.text("user", "follow up")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert [message.content[0].text for message in compactor.requests[0].messages] == [
        "old",
        "old answer",
        "current",
    ]
    assert compactor.requests[0].existing_summary is None
    assert events[3].payload == {
        "checkpoint": "context_compaction",
        "compacted_transcript_cursor": 3,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 3,
        "recent_message_count": 1,
    }


def test_cayu_app_resume_uses_checkpointed_compaction_summary():
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_resume_compaction",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )
    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_resume_compaction",
                messages=[Message.text("user", "follow up")],
            ),
        )
    )

    assert [event.type for event in resume_events] == [
        EventType.SESSION_RESUMED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(compactor.requests) == 2
    assert [message.content[0].text for message in compactor.requests[1].messages] == [
        "current",
        "first answer",
    ]
    assert compactor.requests[1].existing_summary == "old|old answer"

    provider_context = provider.requests[1].messages
    assert [message.role for message in provider_context] == ["user", "user"]
    assert provider_context[0].content[0].text == (
        "Previous session context summary:\nold|old answer|current|first answer"
    )
    assert provider_context[1].content[0].text == "follow up"

    transcript = asyncio.run(store.load_transcript("sess_resume_compaction"))
    assert [message.content[0].text for message in transcript] == [
        "old",
        "old answer",
        "current",
        "first answer",
        "follow up",
        "second answer",
    ]


def test_trim_context_messages_preserves_complete_tool_rounds():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old"),
        Message.tool_call(
            tool_call_id="call_1",
            tool_name="echo",
            arguments={"text": "value"},
        ),
        Message.tool_result(
            tool_call_id="call_1",
            tool_name="echo",
            content="value",
        ),
        Message.text("assistant", "done"),
    ]

    trimmed = trim_context_messages(messages, max_messages=2)

    assert [message.role for message in trimmed] == ["system", "assistant"]
    assert trimmed[0].content[0].text == "You are careful."
    assert trimmed[1].content[0].text == "done"


def test_trim_context_turns_keeps_last_user_turns_with_tool_rounds():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old one"),
        Message.text("assistant", "old answer"),
        Message.text("user", "old two"),
        Message.tool_call(
            tool_call_id="call_old",
            tool_name="echo",
            arguments={"text": "old"},
        ),
        Message.tool_result(
            tool_call_id="call_old",
            tool_name="echo",
            content="old",
        ),
        Message.text("assistant", "old tool answer"),
        Message.text("user", "current one"),
        Message.tool_call(
            tool_call_id="call_current",
            tool_name="echo",
            arguments={"text": "current"},
        ),
        Message.tool_result(
            tool_call_id="call_current",
            tool_name="echo",
            content="current",
        ),
        Message.text("assistant", "current answer"),
    ]

    trimmed = trim_context_turns(messages, max_user_turns=1)

    assert [message.role for message in trimmed] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert trimmed[0].content[0].text == "You are careful."
    assert trimmed[1].content[0].text == "current one"
    assert trimmed[2].content[0].tool_call_id == "call_current"
    assert trimmed[3].content[0].tool_call_id == "call_current"
    assert trimmed[4].content[0].text == "current answer"


def test_trim_context_turns_keeps_all_messages_when_there_are_no_user_turns():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("assistant", "hello"),
    ]

    trimmed = trim_context_turns(messages, max_user_turns=1)

    assert [message.content[0].text for message in trimmed] == [
        "You are careful.",
        "hello",
    ]


def test_trim_context_turns_can_drop_system_messages_when_requested():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current"),
        Message.text("assistant", "current answer"),
    ]

    trimmed = trim_context_turns(
        messages,
        max_user_turns=1,
        preserve_system=False,
    )

    assert [message.role for message in trimmed] == ["user", "assistant"]
    assert [message.content[0].text for message in trimmed] == [
        "current",
        "current answer",
    ]


def test_trim_context_turns_can_drop_system_messages_without_trimming_turns():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "current"),
        Message.text("assistant", "current answer"),
    ]

    trimmed = trim_context_turns(
        messages,
        max_user_turns=10,
        preserve_system=False,
    )

    assert [message.role for message in trimmed] == ["user", "assistant"]
    assert [message.content[0].text for message in trimmed] == [
        "current",
        "current answer",
    ]


def test_trim_context_messages_can_drop_system_messages_without_trimming():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "current"),
    ]

    trimmed = trim_context_messages(
        messages,
        max_messages=10,
        preserve_system=False,
    )

    assert [message.role for message in trimmed] == ["user"]
    assert trimmed[0].content[0].text == "current"


def test_trim_context_messages_uses_limit_after_dropping_system_messages():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("system", "Use concise answers."),
        Message.text("user", "old"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current"),
    ]

    trimmed = trim_context_messages(
        messages,
        max_messages=3,
        preserve_system=False,
    )

    assert [message.role for message in trimmed] == ["user", "assistant", "user"]
    assert [message.content[0].text for message in trimmed] == [
        "old",
        "old answer",
        "current",
    ]


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
    transcript = asyncio.run(app.session_store.load_transcript("sess_system_prompt"))
    assert [message.role for message in transcript] == [
        "system",
        "user",
        "assistant",
    ]


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
    assert events[4].payload["result"]["content"] == ("RuntimeError: tool execution failed")

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
    assert events[4].payload["result"]["content"] == ("Tool returned an error without details.")

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
    assert tool_result_part.content == ("`structured` must contain JSON-compatible values.")
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
                ModelStreamEvent.text_delta(" After that."),
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
        "text",
    ]
    assert assistant_message.content[0].text == "I will check. "
    assert assistant_message.content[1].tool_call_id == "call_1"
    assert assistant_message.content[1].tool_name == "echo"
    assert assistant_message.content[2].text == " After that."

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
    assert provider.requests[0].options["agent_metadata"] == {"nested": {"value": "mutated"}}
    assert provider.requests[1].options["agent_metadata"] == {"nested": {"value": "original"}}


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
    assert events[-1].payload["error"] == ("Model provider stream event payload must be an object.")


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
    assert events[-1].payload["error"] == ("Model providers must yield ModelStreamEvent instances.")


def test_cayu_app_runtime_owns_model_event_identity():
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
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt="You are careful.",
        )
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_model_event_identity",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    model_events = [
        event
        for event in events
        if event.type
        in {
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
        }
    ]
    assert [event.type for event in model_events] == [
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
    ]
    assert {event.session_id for event in model_events} == {"sess_model_event_identity"}
    assert {event.agent_name for event in model_events} == {"assistant"}
    assert {event.environment_name for event in model_events} == {"local"}


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
    assert events[-1].payload["error"] == ("Model provider stream ended without a completed event.")
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

    asyncio.run(store.create(request, identity=_test_session_identity()))

    with pytest.raises(ValueError, match="Session already exists"):
        asyncio.run(store.create(request, identity=_test_session_identity()))


def test_in_memory_session_store_create_rejects_invalid_request_type():
    store = InMemorySessionStore()

    with pytest.raises(TypeError, match="RunRequest"):
        asyncio.run(store.create({"agent_name": "assistant"}, identity=_test_session_identity()))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="RunRequest"):
        asyncio.run(store.create(None, identity=_test_session_identity()))  # type: ignore[arg-type]


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
            ),
            identity=_test_session_identity(),
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
                ),
                identity=_test_session_identity(),
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
            ),
            identity=_test_session_identity(),
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
            ),
            identity=_test_session_identity(),
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
            ),
            identity=_test_session_identity(),
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

    asyncio.run(store.create(request, identity=_test_session_identity()))
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
            ),
            identity=_test_session_identity(),
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
            ),
            identity=_test_session_identity(),
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

    with pytest.raises(TypeError, match="context_policy"):
        app.register_agent(
            AgentSpec(name="bad_context_policy", model="fake-model"),
            context_policy=object(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="tool_policy"):
        app.register_agent(
            AgentSpec(name="bad_tool_policy", model="fake-model"),
            tool_policy=object(),  # type: ignore[arg-type]
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


def test_resume_request_rejects_blank_model():
    with pytest.raises(ValueError, match="model"):
        ResumeRequest(
            session_id="sess_blank_model",
            messages=[Message.text("user", "hi")],
            model=" ",
        )


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
    assert provider.requests[0].options["environment_metadata"] == {"kind": "isolated"}


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


def test_cayu_app_model_events_use_runtime_environment():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
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
    model_events = [
        event
        for event in events
        if event.type
        in {
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
        }
    ]
    assert {event.environment_name for event in model_events} == {"local"}


def test_cayu_app_rejects_invalid_environment_lookup_name():
    app = CayuApp()
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)

    with pytest.raises(ValueError, match="environment.name"):
        app.get_environment("")

    with pytest.raises(ValueError, match="environment.name"):
        app.get_environment(" ")


def test_cayu_app_isolates_registered_environment_shell():
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
    class MarkerPolicy(ContextPolicy):
        async def build(self, request: ContextRequest) -> list[Message]:
            return request.messages

    app = CayuApp()
    spec = AgentSpec(name="assistant", model="fake-model")
    tool = EchoTool()

    app.register_agent(spec, tools=[tool], context_policy=MarkerPolicy())
    spec.model = "mutated"

    registered = app.get_agent("assistant")
    registered.tools["other"] = tool

    assert registered.spec.model == "fake-model"
    assert not hasattr(registered, "context_policy")
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
