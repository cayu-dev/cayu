from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

import pytest
from pydantic import ValidationError

import cayu.runtime.app as runtime_app_module
from cayu.artifacts import file_attachment
from cayu.core import AgentSpec, Event, EventType, Message, TextPart, ToolCallPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
)
from cayu.runners import RunnerCancelledError
from cayu.runtime import (
    BeforeStopContext,
    BeforeStopDecision,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    BudgetWindow,
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
    EventQuery,
    EventSink,
    ForkSessionRequest,
    InMemoryBudgetLedger,
    InMemoryEventSink,
    InMemorySessionStore,
    InMemoryTaskStore,
    InterruptSessionRequest,
    LoopPolicy,
    MessageWindowContextPolicy,
    ModelCompactor,
    ModelPricing,
    PricingCatalog,
    RecentTurnsContextPolicy,
    ResumeRequest,
    RetryPolicy,
    RunLimits,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    Session,
    SessionIdentity,
    SessionQuery,
    SessionStatus,
    StaticToolPolicy,
    StructuredOutputSpec,
    TaskCreate,
    TaskStatus,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
    ToolCallHookContext,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    default_compaction_prompt,
    strip_old_file_attachments,
    trim_context_messages,
    trim_context_turns,
)
from cayu.runtime.context import validate_context_messages
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.tools import (
    SubagentExecutionMode,
    SubagentResultTool,
    SubagentSpec,
    SubagentTool,
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


class NativeStructuredOutputFakeProvider(FakeProvider):
    supports_native_structured_output = True


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


class DenyEchoRequireSideEffectApprovalPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name == "echo":
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason="echo is blocked",
            )
        if request.tool_name == "side_effect":
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason="side effect needs approval",
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


class ContinueBeforeStopPolicy(LoopPolicy):
    def __init__(self, message: str = "Please repair the final answer.") -> None:
        self.message = message
        self.calls = 0

    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        self.calls += 1
        if context.step == 1:
            return BeforeStopDecision.continue_with(
                Message.text("user", self.message),
                reason="needs repair",
            )
        return BeforeStopDecision.complete("second step is final")


class InterruptBeforeStopPolicy(LoopPolicy):
    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        return BeforeStopDecision.interrupt(
            "needs operator review",
            metadata={"classification": context.classification.type.value},
        )


class FailBeforeStopPolicy(LoopPolicy):
    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        return BeforeStopDecision.fail("completion gate failed")


async def collect_events(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


async def collect_resume_events(app: CayuApp, request: ResumeRequest) -> list[Event]:
    return [event async for event in app.resume(request)]


def fake_budget_limit(
    max_estimated_cost: str,
    *,
    scope: Literal["session", "run"] = "session",
    allow_unpriced: bool = False,
    window: BudgetWindow | str | None = None,
) -> BudgetLimit:
    return BudgetLimit(
        max_estimated_cost=Decimal(max_estimated_cost),
        window=BudgetWindow.all_time() if window is None else window,
        pricing=PricingCatalog(
            prices=(
                ModelPricing(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("10"),
                ),
            )
        ),
        scope=scope,
        allow_unpriced=allow_unpriced,
    )


async def collect_interrupt_events(app: CayuApp, request: InterruptSessionRequest) -> list[Event]:
    return [event async for event in app.interrupt_session(request)]


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

    class HookLike:
        pass

    class BlankNameHook(RuntimeHook):
        @property
        def name(self) -> str:
            return " "

    class SinkLike:
        async def emit(self, event):
            pass

    with pytest.raises(TypeError, match="SessionStore"):
        CayuApp(session_store=StoreLike())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="TaskStore"):
        CayuApp(task_store=TaskStoreLike())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Dispatcher"):
        CayuApp(dispatcher=DispatcherLike())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="RuntimeHook"):
        CayuApp(runtime_hooks=[HookLike()])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="runtime_hook.name"):
        CayuApp(runtime_hooks=[BlankNameHook()])

    app = CayuApp()
    with pytest.raises(TypeError, match="RuntimeHook"):
        app.register_agent(
            AgentSpec(name="invalid_hook_agent", model="fake-model"),
            runtime_hooks=[HookLike()],  # type: ignore[list-item]
        )

    with pytest.raises(ValueError, match="runtime_hook.name"):
        app.register_agent(
            AgentSpec(name="blank_hook_agent", model="fake-model"),
            runtime_hooks=[BlankNameHook()],
        )

    class ReservedStructuredOutputTool(Tool):
        spec = ToolSpec(
            name=STRUCTURED_OUTPUT_TOOL_NAME,
            description="Reserved.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="reserved")

    with pytest.raises(ValueError, match="reserved for structured output"):
        app.register_agent(
            AgentSpec(name="reserved_tool_agent", model="fake-model"),
            tools=[ReservedStructuredOutputTool()],
        )

    with pytest.raises(TypeError, match="EventSink"):
        CayuApp(event_sinks=[SinkLike()])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="event_sinks"):
        CayuApp(event_sinks=False)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="event_sinks"):
        CayuApp(event_sinks=0)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="event_sinks"):
        CayuApp(event_sinks="")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="max_file_attachment_bytes"):
        CayuApp(max_file_attachment_bytes=1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_file_attachment_bytes"):
        CayuApp(max_file_attachment_bytes=0)

    with pytest.raises(ValueError, match="max_total_file_attachment_bytes"):
        CayuApp(max_total_file_attachment_bytes=0)

    with pytest.raises(ValueError, match="max_file_attachments_per_request"):
        CayuApp(max_file_attachments_per_request=0)


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


def test_cayu_app_stops_on_token_limit_before_tool_side_effects():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={},
            ),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 4,
                        "total_tokens": 11,
                    },
                }
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
                session_id="sess_token_limit",
                messages=[Message.text("user", "do it")],
                limits=RunLimits(max_total_tokens=10),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[3].payload["limit"] == "total_tokens"
    assert events[3].payload["actual"] == 11
    assert events[3].payload["maximum"] == 10
    assert events[4].payload["reason"] == "limit_reached"
    assert events[5].payload["interruption_type"] == "limit_reached"
    assert tool.calls == []

    transcript = asyncio.run(store.load_transcript("sess_token_limit"))
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]
    session = asyncio.run(store.load("sess_token_limit"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_stops_on_token_limit_after_final_model_answer():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 6,
                        "output_tokens": 4,
                        "total_tokens": 10,
                    },
                }
            ),
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
                session_id="sess_final_token_limit",
                messages=[Message.text("user", "answer")],
                limits=RunLimits(max_total_tokens=10),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[4].payload["limit"] == "total_tokens"
    assert events[5].payload["interruption_type"] == "limit_reached"

    transcript = asyncio.run(store.load_transcript("sess_final_token_limit"))
    assert [message.role for message in transcript] == ["user", "assistant"]
    session = asyncio.run(store.load("sess_final_token_limit"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_stops_on_estimated_cost_limit_after_final_model_answer():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1000,
                        "output_tokens": 100,
                        "total_tokens": 1100,
                    },
                }
            ),
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
                session_id="sess_cost_limit",
                messages=[Message.text("user", "answer")],
                budget_limits=(fake_budget_limit("0.002"),),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[4].payload["limit"] == "estimated_cost"
    assert events[4].payload["maximum"] == "0.002"
    assert events[4].payload["actual"] == "0.002"
    assert events[4].payload["cost_summary"]["total_cost"] == "0.002"
    assert events[5].payload["interruption_type"] == "limit_reached"

    session = asyncio.run(store.load("sess_cost_limit"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_request_session_budget_uses_rolling_window():
    async def run():
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_rolling_request_budget",
                messages=[Message.text("user", "old")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_event(
            "sess_rolling_request_budget",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_rolling_request_budget",
                timestamp=datetime.now(UTC) - timedelta(seconds=120),
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_tokens": 1_000_000,
                        "output_tokens": 0,
                        "total_tokens": 1_000_000,
                    }
                },
            ),
        )
        await store.update_status("sess_rolling_request_budget", SessionStatus.COMPLETED)

        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("new answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000,
                        },
                    }
                ),
            ]
        )
        app = CayuApp(session_store=store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_rolling_request_budget",
                messages=[Message.text("user", "continue")],
                budget_limits=(
                    fake_budget_limit(
                        "0.50",
                        window=BudgetWindow.rolling(seconds=60),
                    ),
                ),
            ),
        )
        session = await store.load("sess_rolling_request_budget")
        return events, session, provider

    events, session, provider = asyncio.run(run())

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert len(provider.requests) == 1


def test_cayu_app_before_stop_policy_can_continue_with_durable_message():
    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("draft"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("final"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        policy = ContinueBeforeStopPolicy("Return a corrected final answer.")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_stop_continue",
                messages=[Message.text("user", "answer")],
                loop_policies=(policy,),
            ),
        )
        session = await store.load("sess_before_stop_continue")
        transcript = await store.load_transcript("sess_before_stop_continue")
        return events, session, transcript, provider, policy

    events, session, transcript, provider, policy = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert len(provider.requests) == 2
    assert policy.calls == 2
    assert any(event.type == "custom.loop.before_stop.selected" for event in events)
    assert [message.role for message in transcript] == ["user", "assistant", "user", "assistant"]
    assert transcript[2].content[0].text == "Return a corrected final answer."
    assert provider.requests[1].messages[-1].content[0].text == "Return a corrected final answer."


def test_cayu_app_before_stop_policy_can_interrupt_and_resume():
    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("needs review"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("resumed final"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        interrupted_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_stop_interrupt",
                messages=[Message.text("user", "answer")],
                loop_policies=(InterruptBeforeStopPolicy(),),
            ),
        )
        interrupted_session = await store.load("sess_before_stop_interrupt")
        resumed_events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_before_stop_interrupt",
                messages=[Message.text("user", "continue")],
            ),
        )
        resumed_session = await store.load("sess_before_stop_interrupt")
        return interrupted_events, interrupted_session, resumed_events, resumed_session, provider

    interrupted_events, interrupted_session, resumed_events, resumed_session, provider = (
        asyncio.run(run())
    )

    assert interrupted_session is not None
    assert interrupted_session.status == SessionStatus.INTERRUPTED
    assert interrupted_events[-1].type == EventType.SESSION_INTERRUPTED
    assert interrupted_events[-1].payload["reason"] == "needs operator review"
    assert resumed_session is not None
    assert resumed_session.status == SessionStatus.COMPLETED
    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert len(provider.requests) == 2


def test_cayu_app_before_stop_policy_can_fail_session():
    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("bad final"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_stop_fail",
                messages=[Message.text("user", "answer")],
                loop_policies=(FailBeforeStopPolicy(),),
            ),
        )
        session = await store.load("sess_before_stop_fail")
        return events, session

    events, session = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert events[-1].type == EventType.SESSION_FAILED
    assert "completion gate failed" in events[-1].payload["error"]


def test_cayu_app_before_stop_policy_order_uses_first_non_complete_decision():
    class CompletePolicy(LoopPolicy):
        async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
            return BeforeStopDecision.complete("looks okay")

    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("draft"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("final"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            loop_policies=(CompletePolicy(),),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            loop_policies=(CompletePolicy(),),
        )
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_stop_order",
                messages=[Message.text("user", "answer")],
                loop_policies=(ContinueBeforeStopPolicy("Continue from request."),),
            ),
        )
        selected = [event for event in events if event.type == "custom.loop.before_stop.selected"]
        completed = [event for event in events if event.type == "custom.loop.before_stop.completed"]
        return selected, completed, provider

    selected, completed, provider = asyncio.run(run())

    assert len(selected) == 1
    assert selected[0].payload["scope"] == "request"
    assert selected[0].payload["action"] == "continue"
    assert [event.payload["scope"] for event in completed[:3]] == ["app", "agent", "request"]
    assert len(provider.requests) == 2


def test_cayu_app_budget_limit_stops_before_tool_side_effects():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={},
            ),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "usage": {
                        "input_tokens": 1000,
                        "output_tokens": 100,
                        "total_tokens": 1100,
                    },
                }
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
                session_id="sess_cost_limit_tool",
                messages=[Message.text("user", "do it")],
                budget_limits=(fake_budget_limit("0.002"),),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[3].payload["limit"] == "estimated_cost"
    assert events[4].payload["reason"] == "limit_reached"
    assert tool.calls == []

    transcript = asyncio.run(store.load_transcript("sess_cost_limit_tool"))
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]


def test_cayu_app_tool_call_limit_allows_existing_result_then_blocks_next_tool():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"step": 1},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="side_effect",
                    arguments={"step": 2},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ),
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
                session_id="sess_tool_limit",
                messages=[Message.text("user", "do it")],
                limits=RunLimits(max_tool_calls=1),
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
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert tool.calls == [{"step": 1}]
    assert events[7].payload["limit"] == "tool_calls"
    assert events[7].payload["actual"] == 2
    assert events[8].payload["tool_call_id"] == "call_2"

    transcript = asyncio.run(store.load_transcript("sess_tool_limit"))
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]


def test_cayu_app_elapsed_limit_stops_between_tool_calls(monkeypatch):
    clock = {"value": 0.0}
    monkeypatch.setattr(runtime_app_module.time, "monotonic", lambda: clock["value"])

    class ElapsedTool(SideEffectTool):
        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            result = await super().run(ctx, args)
            clock["value"] = 1.0
            return result

    store = InMemorySessionStore()
    tool = ElapsedTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={"step": 1},
            ),
            ModelStreamEvent.tool_call(
                id="call_2",
                name="side_effect",
                arguments={"step": 2},
            ),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
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
                session_id="sess_elapsed_limit_between_tools",
                messages=[Message.text("user", "do it")],
                limits=RunLimits(max_elapsed_seconds=1),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert tool.calls == [{"step": 1}]
    assert events[5].payload["limit"] == "elapsed_seconds"
    assert events[6].payload["tool_call_id"] == "call_2"

    transcript = asyncio.run(store.load_transcript("sess_elapsed_limit_between_tools"))
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]
    tool_results = transcript[-1].content
    assert len(tool_results) == 2
    assert tool_results[0].tool_call_id == "call_1"
    assert tool_results[1].tool_call_id == "call_2"


def test_cayu_app_elapsed_limit_stops_after_policy_before_approval(monkeypatch):
    clock = {"value": 0.0}
    monkeypatch.setattr(runtime_app_module.time, "monotonic", lambda: clock["value"])

    class AdvancingApprovalPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            clock["value"] = 1.0
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason="Approval required.",
            )

    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={},
            ),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=AdvancingApprovalPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_elapsed_limit_before_approval",
                messages=[Message.text("user", "do it")],
                limits=RunLimits(max_elapsed_seconds=1),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert not any(event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED for event in events)
    assert tool.calls == []

    checkpoint = asyncio.run(store.load_checkpoint("sess_elapsed_limit_before_approval"))
    assert checkpoint is None


def test_cayu_app_resume_stops_before_model_when_persisted_budget_is_reached():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 6,
                            "output_tokens": 4,
                            "total_tokens": 10,
                        },
                    }
                ),
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
                session_id="sess_resume_limit",
                messages=[Message.text("user", "first")],
            ),
        )
    )

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_resume_limit",
                messages=[Message.text("user", "second")],
                limits=RunLimits(max_total_tokens=10),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert len(provider.requests) == 1
    assert events[1].payload["usage_summary"]["usage"]["total_tokens"] == 10

    session = asyncio.run(store.load("sess_resume_limit"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_run_scoped_resume_ignores_prior_session_usage():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 6,
                            "output_tokens": 4,
                            "total_tokens": 10,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ),
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
                session_id="sess_run_scope",
                messages=[Message.text("user", "first")],
            ),
        )
    )

    # The prior turn already consumed 10 total tokens. With scope="run" the
    # per-invocation delta starts at 0, so a resume must proceed normally even
    # though cumulative usage already meets the cap.
    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_run_scope",
                messages=[Message.text("user", "second")],
                limits=RunLimits(max_total_tokens=10, scope="run"),
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
    assert len(provider.requests) == 2


def test_cayu_app_budget_limit_fails_closed_when_model_step_is_unpriced():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ),
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
                session_id="sess_cost_limit_unpriced",
                messages=[Message.text("user", "answer")],
                budget_limits=(
                    BudgetLimit(
                        max_estimated_cost=Decimal("100"),
                        pricing=PricingCatalog(
                            prices=(
                                ModelPricing(
                                    provider_name="fake",
                                    model="other-model",
                                    input_per_million=Decimal("1"),
                                    output_per_million=Decimal("1"),
                                ),
                            )
                        ),
                    ),
                ),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert len(provider.requests) == 0
    assert events[1].payload["limit"] == "estimated_cost"
    assert "no matching pricing" in events[1].payload["message"]
    assert events[1].payload["cost_summary"]["unpriced_model_steps"] == 0


def test_cayu_app_budget_limit_allows_unpriced_steps_when_explicitly_configured():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ),
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
                session_id="sess_cost_limit_unpriced_allowed",
                messages=[Message.text("user", "answer")],
                budget_limits=(
                    BudgetLimit(
                        max_estimated_cost=Decimal("100"),
                        pricing=PricingCatalog(
                            prices=(
                                ModelPricing(
                                    provider_name="fake",
                                    model="other-model",
                                    input_per_million=Decimal("1"),
                                    output_per_million=Decimal("1"),
                                ),
                            )
                        ),
                        allow_unpriced=True,
                    ),
                ),
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


def test_cayu_app_app_budget_applies_across_sessions():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 0,
                            "total_tokens": 1,
                        },
                    }
                ),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_app_budget_first",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_app_budget_second",
                messages=[Message.text("user", "second")],
            ),
        )
    )
    second_session = asyncio.run(store.load("sess_app_budget_second"))

    assert [event.type for event in first_events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert second_events[1].payload["scope"] == "app"
    assert second_events[1].payload["actual"] == "1"
    assert second_events[2].payload["limit_reached"] is True
    assert second_session is not None
    assert second_session.status == SessionStatus.INTERRUPTED
    assert len(provider.requests) == 1


def test_cayu_app_request_app_budget_limit_applies_across_sessions():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("should not run"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_app_budget_first",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_app_budget_second",
                messages=[Message.text("user", "second")],
                budget_limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=fake_budget_limit("10").pricing,
                    ),
                ),
            ),
        )
    )

    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert second_events[1].payload["limit"] == "estimated_cost"
    assert second_events[1].payload["actual"] == "1"
    assert second_events[1].payload["cost_summary"]["total_cost"] == "1"
    assert len(provider.requests) == 1


def test_cayu_app_budget_reservation_reconciles_model_step():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 250_000,
                        "output_tokens": 0,
                        "total_tokens": 250_000,
                    },
                }
            )
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("2"),
                    pricing=fake_budget_limit("10").pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
        budget_ledger=InMemoryBudgetLedger(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_reservation",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_RESERVED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_RECONCILED,
        EventType.BUDGET_CHECKED,
        EventType.SESSION_COMPLETED,
    ]
    reserved = next(event for event in events if event.type == EventType.BUDGET_RESERVED)
    reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert reserved.payload["requested"] == "1"
    assert reserved.payload["actual"] == "1"
    assert reconciled.payload["actual_amount"] == "0.25"
    assert reconciled.payload["released_amount"] == "0.75"


def test_cayu_app_budget_reservation_stops_before_provider_when_capacity_is_unavailable():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 750_000,
                        "output_tokens": 0,
                        "total_tokens": 750_000,
                    },
                }
            )
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_reservation_first",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_reservation_second",
                messages=[Message.text("user", "second")],
            ),
        )
    )

    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_RESERVATION_FAILED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_INTERRUPTED,
    ]
    failed = next(
        event for event in second_events if event.type == EventType.BUDGET_RESERVATION_FAILED
    )
    assert failed.payload["accepted"] is False
    assert failed.payload["requested"] == "1"
    assert failed.payload["actual"] == "1.75"
    assert len(provider.requests) == 1


def test_cayu_app_causal_budget_is_shared_by_forked_sessions():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 750_000,
                        "output_tokens": 0,
                        "total_tokens": 750_000,
                    },
                }
            )
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="causal",
                    key="job_shared",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
        budget_ledger=InMemoryBudgetLedger(),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    parent_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_causal_budget_parent",
                causal_budget_id="job_shared",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    fork_events = asyncio.run(
        collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_causal_budget_parent",
                session_id="sess_causal_budget_child",
            ),
        )
    )
    child_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_causal_budget_child",
                messages=[Message.text("user", "continue")],
            ),
        )
    )
    child_session = asyncio.run(store.load("sess_causal_budget_child"))

    assert parent_events[-1].type == EventType.SESSION_COMPLETED
    assert fork_events[0].type == EventType.SESSION_FORKED
    assert fork_events[0].payload["causal_budget_id"] == "job_shared"
    assert [event.type for event in child_events] == [
        EventType.SESSION_RESUMED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_RESERVATION_FAILED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_INTERRUPTED,
    ]
    failed = next(
        event for event in child_events if event.type == EventType.BUDGET_RESERVATION_FAILED
    )
    assert failed.payload["scope"] == "causal"
    assert failed.payload["key"] == "job_shared"
    assert failed.payload["actual"] == "1.75"
    assert child_session is not None
    assert child_session.causal_budget_id == "job_shared"
    assert child_session.status == SessionStatus.INTERRUPTED
    assert len(provider.requests) == 1


def test_cayu_app_request_causal_budget_limit_applies_to_matching_causal_history():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("should not run"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_causal_budget_first",
                causal_budget_id="job_request",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_causal_budget_second",
                causal_budget_id="job_request",
                messages=[Message.text("user", "second")],
                budget_limits=(
                    BudgetLimit(
                        scope="causal",
                        key="job_request",
                        max_estimated_cost=Decimal("1"),
                        pricing=fake_budget_limit("10").pricing,
                    ),
                ),
            ),
        )
    )

    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert second_events[1].payload["actual"] == "1"
    assert len(provider.requests) == 1


def test_cayu_app_budget_reservation_is_released_when_model_step_fails():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [ModelStreamEvent.error("provider down")],
            [
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 0,
                            "total_tokens": 1,
                        },
                    }
                )
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    failed_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_reservation_failed",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    retry_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_reservation_retry",
                messages=[Message.text("user", "second")],
            ),
        )
    )

    assert EventType.BUDGET_RESERVATION_RELEASED in [event.type for event in failed_events]
    assert failed_events[-1].type == EventType.SESSION_FAILED
    assert EventType.BUDGET_RESERVED in [event.type for event in retry_events]
    assert retry_events[-1].type == EventType.SESSION_COMPLETED
    assert len(provider.requests) == 2


def test_cayu_app_agent_budget_only_applies_to_matching_agent():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("builder"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("researcher"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 0,
                            "total_tokens": 1,
                        },
                    }
                ),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="agent",
                    key="builder",
                    max_estimated_cost=Decimal("1"),
                    pricing=fake_budget_limit("10").pricing,
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="builder", model="fake-model"))
    app.register_agent(AgentSpec(name="researcher", model="fake-model"))

    builder_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="builder",
                session_id="sess_agent_budget_builder",
                messages=[Message.text("user", "builder")],
            ),
        )
    )
    researcher_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="researcher",
                session_id="sess_agent_budget_researcher",
                messages=[Message.text("user", "researcher")],
            ),
        )
    )

    assert EventType.BUDGET_CHECKED in [event.type for event in builder_events]
    assert EventType.BUDGET_CHECKED not in [event.type for event in researcher_events]
    assert researcher_events[-1].type == EventType.SESSION_COMPLETED
    assert len(provider.requests) == 2


def test_cayu_app_request_agent_budget_limit_applies_to_matching_agent_history():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("should not run"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="builder", model="fake-model"))

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="builder",
                session_id="sess_request_agent_budget_first",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="builder",
                session_id="sess_request_agent_budget_second",
                messages=[Message.text("user", "second")],
                budget_limits=(
                    BudgetLimit(
                        scope="agent",
                        key="builder",
                        max_estimated_cost=Decimal("1"),
                        pricing=fake_budget_limit("10").pricing,
                    ),
                ),
            ),
        )
    )

    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert second_events[1].payload["actual"] == "1"
    assert len(provider.requests) == 1


def test_cayu_app_budget_fails_closed_for_unpriced_model_steps():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("first"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ),
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("100"),
                    pricing=PricingCatalog(
                        prices=(
                            ModelPricing(
                                provider_name="fake",
                                model="other-model",
                                input_per_million=Decimal("1"),
                                output_per_million=Decimal("1"),
                            ),
                        )
                    ),
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_unpriced_first",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_unpriced_second",
                messages=[Message.text("user", "second")],
            ),
        )
    )

    assert [event.type for event in first_events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert "no matching pricing" in first_events[2].payload["message"]
    assert first_events[2].payload["unpriced_model_steps"] == 0
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert "no matching pricing" in second_events[2].payload["message"]
    assert second_events[2].payload["unpriced_model_steps"] == 0
    assert len(provider.requests) == 0


def test_cayu_app_run_scoped_budget_limit_ignores_prior_session_cost():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 100,
                            "total_tokens": 1100,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ),
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
                session_id="sess_run_scope_cost",
                messages=[Message.text("user", "first")],
            ),
        )
    )

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_run_scope_cost",
                messages=[Message.text("user", "second")],
                budget_limits=(fake_budget_limit("0.002", scope="run"),),
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
    assert len(provider.requests) == 2


def test_cayu_app_run_scoped_limit_still_trips_within_a_single_run():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"step": 1},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="side_effect",
                    arguments={"step": 2},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ),
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
                session_id="sess_run_scope_runaway",
                messages=[Message.text("user", "do it")],
                limits=RunLimits(max_tool_calls=1, scope="run"),
            ),
        )
    )

    assert EventType.SESSION_LIMIT_REACHED in [event.type for event in events]
    assert tool.calls == [{"step": 1}]


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
                labels={"owner": "org_123", "project": "feature_a"},
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
    assert fork.labels == source.labels == {"owner": "org_123", "project": "feature_a"}
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


def test_subagent_tool_runs_child_session_with_parent_and_causal_linkage():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "Review only the authentication changes.",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("auth review complete"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent received review"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    subagent_tool = SubagentTool(
        app,
        agents={
            "reviewer": SubagentSpec(
                agent_name="reviewer",
                description="Review delegated work.",
            )
        },
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[subagent_tool],
    )
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_subagent_parent",
                causal_budget_id="job_subagent",
                messages=[Message.text("user", "Implement and review auth.")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    child_sessions = asyncio.run(
        store.list_sessions(
            SessionQuery(
                parent_session_id="sess_subagent_parent",
            )
        )
    )
    assert len(child_sessions) == 1
    child = child_sessions[0]
    assert child.agent_name == "reviewer"
    assert child.parent_session_id == "sess_subagent_parent"
    assert child.causal_budget_id == "job_subagent"
    assert child.status == SessionStatus.COMPLETED
    assert child.metadata["subagent"] == {
        "agent": "reviewer",
        "agent_name": "reviewer",
        "context_mode": "task_only",
        "mode": "foreground",
        "parent_session_id": "sess_subagent_parent",
    }

    child_transcript = asyncio.run(store.load_transcript(child.id))
    assert [message.role for message in child_transcript] == ["user", "assistant"]
    assert child_transcript[0].content[0].text == "Review only the authentication changes."
    assert child_transcript[1].content[0].text == "auth review complete"

    parent_transcript = asyncio.run(store.load_transcript("sess_subagent_parent"))
    assert [message.role for message in parent_transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    tool_result = parent_transcript[2].content[0]
    assert tool_result.content == "auth review complete"
    assert tool_result.structured is not None
    assert tool_result.structured["child_session_id"] == child.id
    assert tool_result.structured["parent_session_id"] == "sess_subagent_parent"
    assert tool_result.structured["causal_budget_id"] == "job_subagent"

    assert [request.messages[0].content[0].text for request in provider.requests] == [
        "Implement and review auth.",
        "Review only the authentication changes.",
        "Implement and review auth.",
    ]
    assert provider.requests[2].messages[-1].role == "tool"


def test_subagent_tool_background_starts_child_without_waiting_for_completion():
    class BackgroundProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.child_model_started = asyncio.Event()
            self.release_child = asyncio.Event()
            self.parent_final_requested = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            first_text = request.messages[0].content[0].text
            if first_text == "parent task" and len(request.messages) == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_background_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "background review task",
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if first_text == "background review task":
                self.child_model_started.set()
                await self.release_child.wait()
                yield ModelStreamEvent.text_delta("background review complete")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            if first_text == "parent task" and request.messages[-1].role == "tool":
                self.parent_final_requested.set()
                yield ModelStreamEvent.text_delta("parent continued before child finished")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            raise AssertionError("Unexpected background subagent provider request.")

    async def run():
        store = InMemorySessionStore()
        provider = BackgroundProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[
                SubagentTool(
                    app,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.BACKGROUND,
                        )
                    },
                )
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

        parent_events = await collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_subagent_background_parent",
                causal_budget_id="job_subagent_background",
                messages=[Message.text("user", "parent task")],
            ),
        )
        assert parent_events[-1].type == EventType.SESSION_COMPLETED
        assert provider.parent_final_requested.is_set()

        child_sessions = await store.list_sessions(
            SessionQuery(parent_session_id="sess_subagent_background_parent")
        )
        assert len(child_sessions) == 1
        child = child_sessions[0]
        assert child.causal_budget_id == "job_subagent_background"
        assert child.status == SessionStatus.RUNNING

        parent_transcript = await store.load_transcript("sess_subagent_background_parent")
        tool_result = parent_transcript[2].content[0]
        assert tool_result.is_error is False
        assert tool_result.structured["mode"] == "background"
        assert tool_result.structured["status"] == "started"
        assert tool_result.structured["child_session_id"] == child.id

        await asyncio.wait_for(provider.child_model_started.wait(), timeout=1)
        provider.release_child.set()
        for _ in range(20):
            loaded_child = await store.load(child.id)
            if loaded_child is not None and loaded_child.status == SessionStatus.COMPLETED:
                return parent_events, child.id
            await asyncio.sleep(0)
        raise AssertionError("Background child session did not complete.")

    parent_events, child_session_id = asyncio.run(run())

    assert parent_events[-1].type == EventType.SESSION_COMPLETED
    assert child_session_id.startswith("sess_subagent_background_parent_subagent_")


def test_subagent_result_tool_waits_for_background_child_result():
    class BackgroundResultProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.child_model_started = asyncio.Event()
            self.release_child = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            first_text = request.messages[0].content[0].text
            if first_text == "parent task" and len(request.messages) == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_background_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "background review task",
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if first_text == "background review task":
                self.child_model_started.set()
                await self.release_child.wait()
                yield ModelStreamEvent.text_delta("background review complete")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            if first_text == "parent task" and len(request.messages) == 3:
                tool_result = request.messages[-1].content[0]
                child_session_id = tool_result.structured["child_session_id"]
                self.release_child.set()
                yield ModelStreamEvent.tool_call(
                    id="call_subagent_result",
                    name="subagent_result",
                    arguments={
                        "child_session_id": child_session_id,
                        "wait": True,
                        "timeout_s": 1,
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if first_text == "parent task" and len(request.messages) == 5:
                yield ModelStreamEvent.text_delta("parent used background result")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            raise AssertionError("Unexpected background result provider request.")

    async def run():
        store = InMemorySessionStore()
        provider = BackgroundResultProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[
                SubagentTool(
                    app,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.BACKGROUND,
                        )
                    },
                ),
                SubagentResultTool(store),
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

        parent_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="sess_subagent_background_result_parent",
                    messages=[Message.text("user", "parent task")],
                ),
            )
        )
        await asyncio.wait_for(provider.child_model_started.wait(), timeout=1)
        parent_events = await asyncio.wait_for(parent_task, timeout=2)
        parent_transcript = await store.load_transcript("sess_subagent_background_result_parent")
        child_sessions = await store.list_sessions(
            SessionQuery(parent_session_id="sess_subagent_background_result_parent")
        )
        return parent_events, parent_transcript, child_sessions

    parent_events, parent_transcript, child_sessions = asyncio.run(run())

    assert parent_events[-1].type == EventType.SESSION_COMPLETED
    assert len(child_sessions) == 1
    assert child_sessions[0].status == SessionStatus.COMPLETED
    result_tool_message = parent_transcript[4]
    result_tool_part = result_tool_message.content[0]
    assert result_tool_part.tool_name == "subagent_result"
    assert result_tool_part.content == "background review complete"
    assert result_tool_part.structured["retrieval_status"] == "ready"
    assert result_tool_part.structured["status"] == "completed"


def test_subagent_result_tool_can_wait_for_all_background_children():
    class MultiBackgroundProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.child_model_started = {"task a": asyncio.Event(), "task b": asyncio.Event()}
            self.release_children = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            first_text = request.messages[0].content[0].text
            if first_text == "parent task" and len(request.messages) == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_background_subagent_a",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "task a"},
                )
                yield ModelStreamEvent.tool_call(
                    id="call_background_subagent_b",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "task b"},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if first_text in {"task a", "task b"}:
                self.child_model_started[first_text].set()
                await self.release_children.wait()
                yield ModelStreamEvent.text_delta(f"{first_text} done")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            if first_text == "parent task" and len(request.messages) == 3:
                self.release_children.set()
                yield ModelStreamEvent.tool_call(
                    id="call_all_subagent_results",
                    name="subagent_result",
                    arguments={"all": True, "wait": True, "timeout_s": 1},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if first_text == "parent task" and len(request.messages) == 5:
                yield ModelStreamEvent.text_delta("parent used both background results")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            raise AssertionError("Unexpected multi-background provider request.")

    async def run():
        store = InMemorySessionStore()
        provider = MultiBackgroundProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[
                SubagentTool(
                    app,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.BACKGROUND,
                        )
                    },
                ),
                SubagentResultTool(store),
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

        parent_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="sess_subagent_background_all_parent",
                    messages=[Message.text("user", "parent task")],
                ),
            )
        )
        await asyncio.wait_for(provider.child_model_started["task a"].wait(), timeout=1)
        await asyncio.wait_for(provider.child_model_started["task b"].wait(), timeout=1)
        parent_events = await asyncio.wait_for(parent_task, timeout=2)
        parent_transcript = await store.load_transcript("sess_subagent_background_all_parent")
        child_sessions = await store.list_sessions(
            SessionQuery(parent_session_id="sess_subagent_background_all_parent")
        )
        return parent_events, parent_transcript, child_sessions

    parent_events, parent_transcript, child_sessions = asyncio.run(run())

    assert parent_events[-1].type == EventType.SESSION_COMPLETED
    assert len(child_sessions) == 2
    assert {child.status for child in child_sessions} == {SessionStatus.COMPLETED}
    result_tool_part = parent_transcript[4].content[0]
    assert result_tool_part.tool_name == "subagent_result"
    assert result_tool_part.structured["retrieval_status"] == "ready"
    assert len(result_tool_part.structured["children"]) == 2
    result_texts = {child["result_text"] for child in result_tool_part.structured["children"]}
    assert result_texts == {"task a done", "task b done"}


def test_interrupting_parent_interrupts_running_background_subagents():
    class BackgroundInterruptProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.child_model_started = asyncio.Event()
            self.parent_second_step_started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            first_text = request.messages[0].content[0].text
            if first_text == "parent task" and len(request.messages) == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_background_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "background review task",
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if first_text == "background review task":
                self.child_model_started.set()
                await asyncio.Event().wait()
                return
            if first_text == "parent task" and request.messages[-1].role == "tool":
                self.parent_second_step_started.set()
                await asyncio.Event().wait()
                return
            raise AssertionError("Unexpected background interrupt provider request.")

    async def run():
        store = InMemorySessionStore()
        provider = BackgroundInterruptProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[
                SubagentTool(
                    app,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.BACKGROUND,
                        )
                    },
                ),
                SubagentResultTool(store),
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

        parent_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="sess_background_parent_interrupt",
                    messages=[Message.text("user", "parent task")],
                ),
            )
        )
        await asyncio.wait_for(provider.child_model_started.wait(), timeout=1)
        await asyncio.wait_for(provider.parent_second_step_started.wait(), timeout=1)
        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_background_parent_interrupt",
                    reason="stop parent and background children",
                )
            )
        ]
        parent_events = await asyncio.wait_for(parent_task, timeout=2)
        child_sessions = await store.list_sessions(
            SessionQuery(parent_session_id="sess_background_parent_interrupt")
        )
        child_events = await store.load_events(child_sessions[0].id)
        return interrupt_events, parent_events, child_sessions, child_events

    interrupt_events, parent_events, child_sessions, child_events = asyncio.run(run())

    assert interrupt_events[-1].type == EventType.SESSION_INTERRUPTED
    assert parent_events[-1].type == EventType.SESSION_INTERRUPTED
    assert len(child_sessions) == 1
    assert child_sessions[0].status == SessionStatus.INTERRUPTED
    child_interrupted = [
        event for event in child_events if event.type == EventType.SESSION_INTERRUPTED
    ]
    assert len(child_interrupted) == 1
    assert child_interrupted[0].payload["metadata"]["source"] == (
        "background_subagent_parent_interrupt"
    )


def test_subagent_tool_background_reports_start_failure_as_tool_error():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_background_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "Review the changes.",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("parent handled missing child"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[
            SubagentTool(
                app,
                agents={
                    "reviewer": SubagentSpec(
                        agent_name="missing_reviewer",
                        mode=SubagentExecutionMode.BACKGROUND,
                    )
                },
            )
        ],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_subagent_background_missing_child",
                messages=[Message.text("user", "Review auth.")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    parent_transcript = asyncio.run(store.load_transcript("sess_subagent_background_missing_child"))
    tool_result = parent_transcript[2].content[0]
    assert tool_result.is_error is True
    assert tool_result.structured["mode"] == "background"
    assert tool_result.structured["status"] == "start_failed"
    child_sessions = asyncio.run(
        store.list_sessions(
            SessionQuery(parent_session_id="sess_subagent_background_missing_child")
        )
    )
    assert child_sessions == []


def test_subagent_tool_returns_child_failure_as_tool_error():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "Review the changes.",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.error("review provider unavailable")],
            [
                ModelStreamEvent.text_delta("parent recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[
            SubagentTool(
                app,
                agents={
                    "reviewer": SubagentSpec(
                        agent_name="reviewer",
                        description="Review delegated work.",
                    )
                },
            )
        ],
    )
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_subagent_parent_failure",
                causal_budget_id="job_subagent_failure",
                messages=[Message.text("user", "Implement and review auth.")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    child_sessions = asyncio.run(
        store.list_sessions(
            SessionQuery(
                parent_session_id="sess_subagent_parent_failure",
            )
        )
    )
    assert len(child_sessions) == 1
    assert child_sessions[0].status == SessionStatus.FAILED
    parent_transcript = asyncio.run(store.load_transcript("sess_subagent_parent_failure"))
    tool_result = parent_transcript[2].content[0]
    assert tool_result.is_error is True
    assert "review provider unavailable" in tool_result.content
    assert tool_result.structured["child_session_id"] == child_sessions[0].id
    assert tool_result.structured["status"] == "session.failed"
    assert provider.requests[2].messages[-1].role == "tool"


def test_subagent_tool_caps_child_result_text():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "Review the changes.",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("abcdef"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent received capped result"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[
            SubagentTool(
                app,
                agents={
                    "reviewer": SubagentSpec(
                        agent_name="reviewer",
                        result_max_chars=4,
                    )
                },
            )
        ],
    )
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_subagent_capped",
                messages=[Message.text("user", "Review auth.")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    parent_transcript = asyncio.run(store.load_transcript("sess_subagent_capped"))
    tool_result = parent_transcript[2].content[0]
    assert tool_result.content == "abcd"
    assert tool_result.structured["result_truncated"] is True
    assert tool_result.structured["result_max_chars"] == 4


def test_subagent_tool_interrupts_child_session_when_parent_is_interrupted():
    class BlockingChildProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.child_model_started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            request_index = len(self.requests)
            if request_index == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "Review the changes.",
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if request_index == 2:
                self.child_model_started.set()
                await asyncio.Event().wait()
                return
            raise AssertionError(f"Unexpected request {request_index}")

    async def run():
        store = InMemorySessionStore()
        provider = BlockingChildProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[
                SubagentTool(
                    app,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            description="Review delegated work.",
                        )
                    },
                )
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

        parent_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="sess_subagent_parent_interrupt",
                    messages=[Message.text("user", "Implement and review auth.")],
                ),
            )
        )
        await asyncio.wait_for(provider.child_model_started.wait(), timeout=1)

        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_subagent_parent_interrupt",
                    reason="stop delegated work",
                )
            )
        ]
        parent_events = await asyncio.wait_for(parent_task, timeout=1)
        child_sessions = await store.list_sessions(
            SessionQuery(parent_session_id="sess_subagent_parent_interrupt")
        )
        child_events = await store.load_events(child_sessions[0].id)
        return interrupt_events, parent_events, child_sessions, child_events

    interrupt_events, parent_events, child_sessions, child_events = asyncio.run(run())

    assert interrupt_events[-1].type == EventType.SESSION_INTERRUPTED
    assert parent_events[-1].type == EventType.SESSION_INTERRUPTED
    assert len(child_sessions) == 1
    assert child_sessions[0].status == SessionStatus.INTERRUPTED
    child_interrupted_events = [
        event for event in child_events if event.type == EventType.SESSION_INTERRUPTED
    ]
    assert len(child_interrupted_events) == 1
    assert child_interrupted_events[0].payload["metadata"] == {"source": "subagent_tool"}


def test_subagent_tool_interrupts_child_session_during_startup_window():
    class DelayedChildStartupStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.child_running = asyncio.Event()
            self.release_child_startup = asyncio.Event()

        async def transition_status(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
        ) -> Session:
            session = await super().transition_status(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
            )
            if (
                session.parent_session_id == "sess_subagent_parent_startup_interrupt"
                and to_status == SessionStatus.RUNNING
            ):
                self.child_running.set()
                await self.release_child_startup.wait()
            return session

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform,
        ) -> Session:
            session = await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
            )
            if (
                session.parent_session_id == "sess_subagent_parent_startup_interrupt"
                and to_status == SessionStatus.INTERRUPTING
            ):
                self.release_child_startup.set()
            return session

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "Review the changes.",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("should not finish"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )

    async def run():
        store = DelayedChildStartupStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[
                SubagentTool(
                    app,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            description="Review delegated work.",
                        )
                    },
                )
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

        parent_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="sess_subagent_parent_startup_interrupt",
                    messages=[Message.text("user", "Implement and review auth.")],
                ),
            )
        )
        await asyncio.wait_for(store.child_running.wait(), timeout=1)

        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_subagent_parent_startup_interrupt",
                    reason="stop while child starts",
                )
            )
        ]
        parent_events = await asyncio.wait_for(parent_task, timeout=1)
        child_sessions = await store.list_sessions(
            SessionQuery(parent_session_id="sess_subagent_parent_startup_interrupt")
        )
        child_events = await store.load_events(child_sessions[0].id)
        return interrupt_events, parent_events, child_sessions, child_events

    interrupt_events, parent_events, child_sessions, child_events = asyncio.run(run())

    assert interrupt_events[-1].type == EventType.SESSION_INTERRUPTED
    assert parent_events[-1].type == EventType.SESSION_INTERRUPTED
    assert len(child_sessions) == 1
    assert child_sessions[0].status == SessionStatus.INTERRUPTED
    assert [
        event.type for event in child_events if event.type == EventType.SESSION_INTERRUPTED
    ] == [EventType.SESSION_INTERRUPTED]


def test_subagent_tool_child_cleanup_failure_does_not_mask_parent_interruption():
    class BlockingChildProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.child_model_started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            request_index = len(self.requests)
            if request_index == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "Review the changes.",
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if request_index == 2:
                self.child_model_started.set()
                await asyncio.Event().wait()
                return
            raise AssertionError(f"Unexpected request {request_index}")

    class FailingInterruptRuntime:
        def __init__(self, app: CayuApp) -> None:
            self.app = app

        def run(self, request: RunRequest) -> AsyncIterator[Event]:
            return self.app.run(request)

        async def interrupt_session(
            self,
            request: InterruptSessionRequest,
        ) -> AsyncIterator[Event]:
            raise RuntimeError("child interrupt cleanup unavailable")
            yield  # pragma: no cover

    async def run():
        store = InMemorySessionStore()
        provider = BlockingChildProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        subagent_runtime = FailingInterruptRuntime(app)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[
                SubagentTool(
                    subagent_runtime,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            description="Review delegated work.",
                        )
                    },
                )
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))

        parent_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="sess_subagent_parent_cleanup_failure",
                    messages=[Message.text("user", "Implement and review auth.")],
                ),
            )
        )
        await asyncio.wait_for(provider.child_model_started.wait(), timeout=1)

        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_subagent_parent_cleanup_failure",
                    reason="stop delegated work",
                )
            )
        ]
        parent_events = await asyncio.wait_for(parent_task, timeout=1)
        parent_transcript = await store.load_transcript("sess_subagent_parent_cleanup_failure")
        parent_session_events = await store.load_events("sess_subagent_parent_cleanup_failure")
        return interrupt_events, parent_events, parent_transcript, parent_session_events

    interrupt_events, parent_events, parent_transcript, parent_session_events = asyncio.run(run())

    assert interrupt_events[-1].type == EventType.SESSION_INTERRUPTED
    assert parent_events[-1].type == EventType.SESSION_INTERRUPTED
    tool_result = parent_transcript[2].content[0]
    assert tool_result.is_error is True
    assert tool_result.artifacts == [
        {
            "type": "cayu.subagent_cleanup_error.v1",
            "child_session_id": tool_result.artifacts[0]["child_session_id"],
            "error": "child interrupt cleanup unavailable",
            "error_type": "RuntimeError",
        }
    ]
    failed_tool_events = [
        event for event in parent_session_events if event.type == EventType.TOOL_CALL_FAILED
    ]
    assert failed_tool_events
    assert failed_tool_events[0].payload["result"]["artifacts"] == tool_result.artifacts


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


def test_cayu_app_runtime_hook_can_fork_and_dispatch_followup_work():
    class RecordingDispatcher(Dispatcher):
        def __init__(self) -> None:
            self.requests: list[DispatchRequest] = []

        async def submit(self, runtime, request: DispatchRequest) -> DispatchHandle:
            self.requests.append(request)
            return DispatchHandle(
                dispatch_id=request.dispatch_id,
                session_id=request.session_id,
                task_id=request.task_id,
                backend="recording",
                status=DispatchStatus.SUBMITTED,
                metadata={"queued": True},
            )

    class FollowupHook(RuntimeHook):
        def __init__(self) -> None:
            self.session_status: SessionStatus | None = None
            self.task_id: str | None = None
            self.handle: DispatchHandle | None = None
            self.actions: list[dict] = []

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            source_session = context.session
            if source_session.metadata.get("purpose") == "knowledge_extraction":
                return
            self.session_status = source_session.status
            child_session_id = f"{source_session.id}_knowledge"
            await context.fork_session(
                ForkSessionRequest(
                    source_session_id=source_session.id,
                    session_id=child_session_id,
                    metadata={"purpose": "knowledge_extraction"},
                )
            )
            task = await context.create_task(
                TaskCreate(
                    type="knowledge_extraction",
                    session_id=child_session_id,
                    assigned_agent_name=source_session.agent_name,
                    input={"source_session_id": source_session.id},
                )
            )
            self.task_id = task.id
            self.handle = await context.dispatch(
                DispatchRequest(
                    session_id=child_session_id,
                    dispatch_id="dispatch_knowledge_1",
                    task_id=task.id,
                    messages=[Message.text("user", "Extract implementation knowledge.")],
                )
            )
            self.actions = context.actions

    store = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    dispatcher = RecordingDispatcher()
    hook = FollowupHook()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("main task done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        runtime_hooks=[hook],
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="builder", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="builder",
                session_id="sess_hook_source",
                messages=[Message.text("user", "Build the feature.")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
    ]
    assert hook.session_status == SessionStatus.COMPLETED
    assert hook.handle == DispatchHandle(
        dispatch_id="dispatch_knowledge_1",
        session_id="sess_hook_source_knowledge",
        task_id=hook.task_id,
        backend="recording",
        status=DispatchStatus.SUBMITTED,
        metadata={"queued": True},
    )
    assert [request.session_id for request in dispatcher.requests] == ["sess_hook_source_knowledge"]
    assert [action["type"] for action in hook.actions] == [
        "fork_session",
        "create_task",
        "dispatch",
    ]

    child = asyncio.run(store.load("sess_hook_source_knowledge"))
    assert child is not None
    assert child.parent_session_id == "sess_hook_source"
    assert child.status == SessionStatus.COMPLETED
    assert child.metadata == {"purpose": "knowledge_extraction"}
    task = asyncio.run(tasks.load_task(hook.task_id))
    assert task is not None
    assert task.status == TaskStatus.PENDING
    assert task.session_id == "sess_hook_source_knowledge"
    assert events[-1].payload["hook_name"] == "FollowupHook"
    assert events[-1].payload["phase"] == "after_session_completed"
    assert [action["type"] for action in events[-1].payload["actions"]] == [
        "fork_session",
        "create_task",
        "dispatch",
    ]


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


def test_cayu_app_after_tool_call_hook_observes_tool_result_and_emits_events():
    class ToolObservationHook(RuntimeHook):
        def __init__(self) -> None:
            self.tool_name: str | None = None
            self.tool_call_id: str | None = None
            self.arguments: dict | None = None
            self.result: ToolResult | None = None
            self.task_id: str | None = None
            self.tool_event_type: EventType | str | None = None

        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            self.tool_name = context.tool_name
            self.tool_call_id = context.tool_call_id
            self.arguments = context.arguments
            self.result = context.result
            self.task_id = context.task_id
            self.tool_event_type = context.tool_event.type
            self.arguments["text"] = "mutated"
            if self.result.structured is not None:
                self.result.structured["echoed"] = "mutated"
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
    assert hook.result.structured == {"agent": "assistant", "echoed": "mutated"}
    assert hook.task_id == "task_tool_hook"
    assert hook.tool_event_type == EventType.TOOL_CALL_COMPLETED
    assert [
        (event.payload["phase"], event.payload["hook_name"], event.payload["scope"])
        for event in events
        if event.type == EventType.HOOK_STARTED
    ] == [("after_tool_call", "ToolObservationHook", "agent")]
    assert events[-1].type == EventType.SESSION_COMPLETED

    stored_events = asyncio.run(store.load_events("sess_tool_hook"))
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
        EventType.TOOL_CALL_COMPLETED,
        EventType.HOOK_STARTED,
        "custom.tool.observed",
        EventType.HOOK_COMPLETED,
    ]
    hook_completed = next(
        event for event in stored_events if event.type == EventType.HOOK_COMPLETED
    )
    assert hook_completed.payload["tool_name"] == "echo"
    assert hook_completed.payload["tool_call_id"] == "call_echo"
    assert hook_completed.payload["phase"] == "after_tool_call"

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
    assert hook_failed.payload["tool_call_id"] == "call_echo"
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
    assert events[-1].payload["actions"] == [
        {
            "type": "emit_custom_event",
            "payload": {
                "event_id": stored_events[-2].id,
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
        "Hook-emitted custom events must use the custom. namespace."
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


def test_cayu_app_retries_retryable_model_error_before_tool_side_effects():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_failed_attempt",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.error("OpenAI API request failed with HTTP 429: rate limit"),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_successful_attempt",
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
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_retry_before_tool",
                messages=[Message.text("user", "use the tool")],
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_retry_before_tool"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.MODEL_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 3
    assert len(tool.calls) == 1
    assert events[3].payload["reason"] == "http_status"
    assert events[3].payload["status_code"] == 429
    assert events[3].payload["attempt"] == 1
    assert events[3].payload["next_attempt"] == 2
    assert events[6].payload["tool_call_id"] == "call_successful_attempt"
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert transcript[1].content[0].type == "tool_call"
    assert transcript[1].content[0].tool_call_id == "call_successful_attempt"


def test_cayu_app_does_not_retry_without_retry_policy():
    provider = FakeProvider(
        [
            ModelStreamEvent.error("OpenAI API request failed with HTTP 429: rate limit"),
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
                session_id="sess_retry_disabled",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.SESSION_FAILED,
    ]
    assert len(provider.requests) == 1
    assert EventType.MODEL_RETRY not in [event.type for event in events]


def test_cayu_app_does_not_retry_non_retryable_model_error():
    provider = FakeProvider(
        [
            ModelStreamEvent.error("OpenAI API request failed with HTTP 400: bad request"),
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
                session_id="sess_retry_non_retryable",
                messages=[Message.text("user", "hi")],
                retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.SESSION_FAILED,
    ]
    assert len(provider.requests) == 1
    assert EventType.MODEL_RETRY not in [event.type for event in events]


def test_cayu_app_retries_provider_exception_and_keeps_transcript_clean():
    class TimeoutThenSuccessProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise TimeoutError("stream idle timeout")
            yield ModelStreamEvent.text_delta("ok")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    store = InMemorySessionStore()
    provider = TimeoutThenSuccessProvider()
    app = CayuApp(
        session_store=store,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_retry_exception",
                messages=[Message.text("user", "hi")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_retry_exception"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.MODEL_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2
    assert events[3].payload["reason"] == "timeout"
    assert [message.role for message in transcript] == ["user", "assistant"]
    assert transcript[1].content[0].text == "ok"


def test_cayu_app_tags_failed_attempt_stream_events_and_keeps_transcript_clean():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("partial answer"),
                ModelStreamEvent.error("OpenAI API request failed with HTTP 500: unavailable"),
            ],
            [
                ModelStreamEvent.text_delta("final answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_retry_stream_attempt_metadata",
                messages=[Message.text("user", "hi")],
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_retry_stream_attempt_metadata"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_ERROR,
        EventType.MODEL_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[2].payload == {
        "delta": "partial answer",
        "step": 1,
        "attempt": 1,
        "max_attempts": 2,
    }
    assert events[3].payload["attempt"] == 1
    assert events[3].payload["max_attempts"] == 2
    assert events[6].payload == {
        "delta": "final answer",
        "step": 1,
        "attempt": 2,
        "max_attempts": 2,
    }
    assert events[7].payload["attempt"] == 2
    assert events[7].payload["max_attempts"] == 2
    assert [message.role for message in transcript] == ["user", "assistant"]
    assert transcript[1].content[0].text == "final answer"


def test_cayu_app_emits_model_error_for_final_failed_exception_attempt():
    class AlwaysTimeoutProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            raise TimeoutError("stream idle timeout")
            yield

    provider = AlwaysTimeoutProvider()
    app = CayuApp(
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_retry_final_exception_error",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.MODEL_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.SESSION_FAILED,
    ]
    assert len(provider.requests) == 2
    assert events[2].payload == {
        "error": "stream idle timeout",
        "error_type": "TimeoutError",
        "step": 1,
        "attempt": 1,
        "max_attempts": 2,
    }
    assert events[5].payload == {
        "error": "stream idle timeout",
        "error_type": "TimeoutError",
        "step": 1,
        "attempt": 2,
        "max_attempts": 2,
    }


def test_cayu_app_does_not_emit_model_error_for_non_retryable_contract_failure():
    provider = FakeProvider(
        [
            ModelStreamEvent(
                type=ModelStreamEventType.TOOL_CALL,
                payload={"name": "echo", "arguments": "not-an-object"},
            )
        ]
    )
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
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
                session_id="sess_retry_non_retryable_contract_failure",
                messages=[Message.text("user", "bad call")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.SESSION_FAILED,
    ]


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
    assert events[-1].payload["interruption_type"] == "tool_approval_required"
    assert events[-1].payload["approval"]["approval_id"] == approval["approval_id"]
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


def test_cayu_app_preserves_structured_output_across_tool_approval():
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
                ModelStreamEvent.tool_call(
                    id="call_final",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "approved"}},
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
    structured_output = StructuredOutputSpec(
        name="approval_answer",
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_approval_structured_output",
                messages=[Message.text("user", "use the tool")],
                structured_output=structured_output,
            ),
        )
    )
    approval = interrupt_events[4].payload["approval"]
    approval_id = approval["approval_id"]
    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_approval_structured_output"))

    assert checkpoint is not None
    assert checkpoint["pending_tool_approval"]["structured_output"]["name"] == ("approval_answer")

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_tool_approval_structured_output",
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
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[7].payload["output"] == {"answer": "approved"}
    assert provider.requests[1].options["structured_output"]["name"] == "approval_answer"


def test_cayu_app_rejects_conflicting_structured_output_on_tool_approval():
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
                ModelStreamEvent.text_delta('{"answer":"approved"}'),
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
                session_id="sess_tool_approval_structured_output_conflict",
                messages=[Message.text("user", "use the tool")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                ),
            ),
        )
    )
    approval_id = interrupt_events[4].payload["approval"]["approval_id"]

    with pytest.raises(ValueError, match="does not match the pending run contract"):
        asyncio.run(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id="sess_tool_approval_structured_output_conflict",
                    approval_id=approval_id,
                    decision=ToolApprovalDecision.APPROVE,
                    structured_output=StructuredOutputSpec(
                        json_schema={
                            "type": "object",
                            "properties": {"different": {"type": "string"}},
                            "required": ["different"],
                        },
                    ),
                ),
            )
        )

    session = asyncio.run(store.load("sess_tool_approval_structured_output_conflict"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert tool.calls == []


def test_cayu_app_budget_limit_stops_approval_before_tool_side_effects():
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
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 100,
                            "total_tokens": 1100,
                        },
                    }
                ),
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
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_approval_cost_limit",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = interrupt_events[4].payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_tool_approval_cost_limit",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
                budget_limits=(fake_budget_limit("0.002"),),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.SESSION_CHECKPOINTED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[1].payload["limit"] == "estimated_cost"
    assert events[2].payload["reason"] == "limit_reached"
    assert tool.calls == []
    assert len(provider.requests) == 1

    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_approval_cost_limit"))
    assert checkpoint == {}
    transcript = asyncio.run(store.load_transcript("sess_tool_approval_cost_limit"))
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]
    session = asyncio.run(store.load("sess_tool_approval_cost_limit"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_approval_limit_counts_only_executable_pending_tools():
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
        tool_policy=DenyEchoRequireSideEffectApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_limit_executable_only",
                messages=[Message.text("user", "use both tools")],
            ),
        )
    )
    approval = interrupt_events[4].payload["approval"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_limit_executable_only",
                approval_id=approval["approval_id"],
                decision=ToolApprovalDecision.APPROVE,
                limits=RunLimits(max_tool_calls=1),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert EventType.SESSION_LIMIT_REACHED not in [event.type for event in events]
    assert side_effect.calls == [{"value": "second"}]
    assert asyncio.run(store.load_checkpoint("sess_approval_limit_executable_only")) == {}

    tool_result_message = provider.requests[1].messages[-1]
    assert tool_result_message.role == "tool"
    assert [part.tool_call_id for part in tool_result_message.content] == [
        "call_1",
        "call_2",
    ]
    assert tool_result_message.content[0].is_error is True
    assert tool_result_message.content[1].content == "recorded"


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


def test_cayu_app_after_tool_call_hook_observes_approval_denial_result():
    class ApprovalDenialHook(RuntimeHook):
        def __init__(self) -> None:
            self.tool_event_type: EventType | str | None = None
            self.result: ToolResult | None = None

        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            self.tool_event_type = context.tool_event.type
            self.result = context.result

    store = InMemorySessionStore()
    hook = ApprovalDenialHook()
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
        tools=[SideEffectTool()],
        tool_policy=RequireApprovalPolicy(),
        runtime_hooks=[hook],
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_approval_deny_hook",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = interrupt_events[4].payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_tool_approval_deny_hook",
                approval_id=approval_id,
                decision=ToolApprovalDecision.DENY,
                reason="not safe",
            ),
        )
    )

    assert hook.tool_event_type == EventType.TOOL_CALL_APPROVAL_DENIED
    assert hook.result is not None
    assert hook.result.content == "Tool call denied by approval: not safe"
    assert hook.result.is_error is True
    assert [
        event.type
        for event in events
        if event.type
        in {
            EventType.TOOL_CALL_APPROVAL_DENIED,
            EventType.HOOK_STARTED,
            EventType.HOOK_COMPLETED,
        }
    ] == [
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
    ]


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
    assert events[-1].payload["interruption_type"] == "tool_approval_required"
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


def test_cayu_app_approval_limit_replays_recorded_tool_outcomes_before_stopping():
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
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 100,
                            "total_tokens": 1100,
                        },
                    }
                ),
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
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_recorded_outcome_limit",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = interrupt_events[4].payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_recorded_outcome_limit",
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
                session_id="sess_approval_recorded_outcome_limit",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
                budget_limits=(fake_budget_limit("0.001"),),
            ),
        )
    )

    assert [event.type for event in retry_events] == [
        EventType.SESSION_RESUMED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_CHECKPOINTED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert retry_events[1].payload["limit"] == "estimated_cost"
    assert tool.calls == [{"value": "secret"}]
    assert len(provider.requests) == 1
    assert asyncio.run(store.load_checkpoint("sess_approval_recorded_outcome_limit")) == {}

    transcript = asyncio.run(store.load_transcript("sess_approval_recorded_outcome_limit"))
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]
    assert transcript[-1].content[0].content == "recorded"


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
    assert retry_events[-1].payload["interruption_type"] == "tool_approval_required"
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


def test_runtime_adds_usage_metrics_to_model_completed_events():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed(
                {
                    "model": "fake-model-version",
                    "usage": {
                        "input_tokens": 12,
                        "input_tokens_details": {"cached_tokens": 5},
                        "output_tokens": 3,
                    },
                }
            ),
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
                session_id="usage_runtime",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["usage"] == {
        "input_tokens": 12,
        "input_tokens_details": {"cached_tokens": 5},
        "output_tokens": 3,
    }
    assert completed.payload["usage_metrics"] == {
        "provider_name": "fake",
        "model": "fake-model-version",
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
        "reasoning_output_tokens": 0,
        "cache": {
            "read_tokens": 0,
            "write_tokens": 0,
            "cached_input_tokens": 5,
            "uncached_input_tokens": 7,
        },
    }


def test_runtime_keeps_raw_model_completed_usage_when_it_cannot_normalize_usage_metrics():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed(
                {
                    "model": "fake-model-version",
                    "usage": {
                        "provider_specific_counter": 123,
                        "provider_specific_cache_mode": "hit",
                    },
                }
            ),
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
                session_id="usage_runtime_raw_only",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["usage"] == {
        "provider_specific_counter": 123,
        "provider_specific_cache_mode": "hit",
    }
    assert "usage_metrics" not in completed.payload


def test_cayu_app_get_session_usage_summarizes_durable_events():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="upper",
                    arguments={"text": "hello"},
                ),
                ModelStreamEvent.completed(
                    {
                        "usage": {
                            "input_tokens": 20,
                            "output_tokens": 4,
                            "input_tokens_details": {"cached_tokens": 8},
                        }
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UpperTool()],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_summary",
                messages=[Message.text("user", "uppercase hello")],
                max_steps=3,
            ),
        )
    )
    summary = asyncio.run(app.get_session_usage("usage_summary"))

    assert summary.model_steps == 2
    assert summary.tool_calls == 1
    assert summary.provider_names == ["fake"]
    assert summary.models == ["fake-model"]
    assert summary.usage.input_tokens == 30
    assert summary.usage.output_tokens == 6
    assert summary.usage.total_tokens == 36
    assert summary.usage.cache.cached_input_tokens == 8
    assert summary.usage.cache.uncached_input_tokens == 22


def test_cayu_app_get_session_usage_queries_only_usage_relevant_events():
    class TrackingStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.load_events_called = False
            self.event_queries: list[EventQuery] = []

        async def load_events(self, session_id: str) -> list[Event]:
            self.load_events_called = True
            return await super().load_events(session_id)

        async def query_events(self, query: EventQuery | None = None):
            if query is not None:
                self.event_queries.append(query)
            return await super().query_events(query)

    store = TrackingStore()
    app = CayuApp(session_store=store)

    async def run() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="usage_query",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            "usage_query",
            [
                Event(type=EventType.SESSION_STARTED, session_id="usage_query"),
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="usage_query",
                    payload={"usage": {"input_tokens": 10, "output_tokens": 2}},
                ),
                Event(type=EventType.MODEL_TEXT_DELTA, session_id="usage_query"),
                Event(type=EventType.TOOL_CALL_STARTED, session_id="usage_query"),
                Event(type=EventType.TOOL_CALL_COMPLETED, session_id="usage_query"),
            ],
        )

        summary = await app.get_session_usage("usage_query")

        assert summary.model_steps == 1
        assert summary.tool_calls == 1
        assert summary.usage.input_tokens == 10
        assert summary.usage.output_tokens == 2

    asyncio.run(run())

    assert store.load_events_called is False
    assert [str(query.event_type) for query in store.event_queries] == [
        "model.completed",
        "tool.call.started",
    ]


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


def test_strip_old_file_attachments_preserves_latest_attachment_result_only():
    old_attachment = file_attachment(
        artifact_id="art_old",
        kind="image",
        filename="old.png",
        content_type="image/png",
        size_bytes=3,
    )
    current_attachment = file_attachment(
        artifact_id="art_current",
        kind="image",
        filename="current.png",
        content_type="image/png",
        size_bytes=7,
    )
    messages = [
        Message.text("user", "old"),
        Message.tool_call(
            tool_call_id="call_old",
            tool_name="read_file",
            arguments={"artifact_id": "art_old"},
        ),
        Message.tool_result(
            tool_call_id="call_old",
            tool_name="read_file",
            content="Attached image artifact art_old: old.png.",
            artifacts=[old_attachment],
        ),
        Message.text("assistant", "old answer"),
        Message.text("user", "current"),
        Message.tool_call(
            tool_call_id="call_current",
            tool_name="read_file",
            arguments={"artifact_id": "art_current"},
        ),
        Message.tool_result(
            tool_call_id="call_current",
            tool_name="read_file",
            content="Attached image artifact art_current: current.png.",
            artifacts=[current_attachment],
        ),
    ]

    projected = strip_old_file_attachments(messages, max_attachment_results=1)

    old_result = projected[2].content[0]
    current_result = projected[6].content[0]
    assert old_result.artifacts == []
    assert "omitted from this provider request" in old_result.content
    assert old_result.structured == {
        "cayu_file_attachments_stripped": [
            {
                "artifact_id": "art_old",
                "filename": "old.png",
                "content_type": "image/png",
                "size_bytes": 3,
                "kind": "image",
            }
        ]
    }
    assert current_result.artifacts == [current_attachment]
    assert messages[2].content[0].artifacts == [old_attachment]

    strip_all = strip_old_file_attachments(messages, max_attachment_results=0)
    assert strip_all[2].content[0].artifacts == []
    assert strip_all[6].content[0].artifacts == []


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


def test_cayu_app_passes_agent_provider_options_to_model_request():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            provider_options={
                "openai": {
                    "prompt_cache_key": "tenant-a-agent",
                    "prompt_cache_retention": "24h",
                }
            },
        )
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_options",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert provider.requests[0].options["openai"] == {
        "prompt_cache_key": "tenant-a-agent",
        "prompt_cache_retention": "24h",
    }
    assert provider.requests[0].options["agent_metadata"] == {}
    assert provider.requests[0].options["step"] == 1


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
    assert events[4].payload["completion"] == {
        "finish_reason": "stop",
        "raw_finish_reason": "stop",
        "status": None,
    }
    assert events[4].payload["step_classification"] == {
        "type": "invalid",
        "reason": "assistant produced no tool calls and no user-visible content",
    }
    assert len(provider.requests) == 1
    assert len(provider.requests[0].messages) == 1
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_records_model_step_classification_for_length_finish():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("partial"),
            ModelStreamEvent.completed(
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                }
            ),
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
                session_id="sess_length_finish",
                messages=[Message.text("user", "hi")],
            ),
        )
    )
    model_completed = events[3]

    assert model_completed.type == EventType.MODEL_COMPLETED
    assert model_completed.payload["completion"] == {
        "finish_reason": "length",
        "raw_finish_reason": "max_output_tokens",
        "status": "incomplete",
    }
    assert model_completed.payload["step_classification"]["type"] == "length"


def test_cayu_app_requires_structured_output_final_tool_call():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta('{"answer":"ok"}'),
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
                session_id="sess_structured_output_requires_tool",
                messages=[Message.text("user", "answer with json")],
                structured_output=StructuredOutputSpec(
                    name="answer",
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=0,
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_requires_tool"))
    session = asyncio.run(store.load("sess_structured_output_requires_tool"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.SESSION_FAILED,
    ]
    assert events[4].payload["valid"] is False
    assert events[4].payload["errors"][0]["message"] == (
        f"Final structured output must be submitted with the `{STRUCTURED_OUTPUT_TOOL_NAME}` tool."
    )
    assert events[-1].payload["error"] == (
        "Structured output validation failed after 1 attempt(s)."
    )
    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert provider.requests[0].options["structured_output"]["name"] == "answer"
    assert provider.requests[0].options["structured_output"]["schema"]["required"] == ["answer"]
    assert [message.role for message in transcript] == ["user", "assistant"]
    assert transcript[-1].content[0].text == '{"answer":"ok"}'


def test_cayu_app_accepts_structured_output_final_tool_call():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_final",
                name=STRUCTURED_OUTPUT_TOOL_NAME,
                arguments={"output": {"answer": "ok"}},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
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
                session_id="sess_structured_output_tool_valid",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    name="answer",
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_tool_valid"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[3].payload["output"] == {"answer": "ok"}
    assert provider.requests[0].tools == [
        {
            "name": STRUCTURED_OUTPUT_TOOL_NAME,
            "description": (
                "Submit the final structured output for this run. Use this only when the "
                "final answer is ready. The value must be provided in the `output` field."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "output": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    }
                },
                "required": ["output"],
                "additionalProperties": False,
            },
        }
    ]
    system_messages = [
        message for message in provider.requests[0].messages if message.role == "system"
    ]
    assert len(system_messages) == 1
    assert STRUCTURED_OUTPUT_TOOL_NAME in system_messages[0].content[0].text
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]
    assert transcript[-1].content[0].tool_name == STRUCTURED_OUTPUT_TOOL_NAME
    assert transcript[-1].content[0].content == "Structured output accepted."
    assert transcript[-1].content[0].structured == {"output": {"answer": "ok"}}
    assert transcript[-1].content[0].is_error is False


def test_cayu_app_retries_invalid_structured_output_final_tool_call():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_final_invalid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"wrong": "value"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_final_valid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "fixed"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
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
                session_id="sess_structured_output_tool_retry",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=1,
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_tool_retry"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[3].payload["errors"][0]["path"] == "$"
    assert events[4].payload["attempt"] == 1
    assert events[7].payload["output"] == {"answer": "fixed"}
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    invalid_result = transcript[2].content[0]
    assert invalid_result.tool_name == STRUCTURED_OUTPUT_TOOL_NAME
    assert invalid_result.is_error is True
    assert "Structured output rejected" in invalid_result.content
    assert STRUCTURED_OUTPUT_TOOL_NAME in invalid_result.content
    assert "plain text" in invalid_result.content
    assert provider.requests[1].messages[-1].role == "tool"
    assert provider.requests[1].messages[-1].content[0].is_error is True


def test_cayu_app_uses_custom_repair_prompt_for_invalid_structured_output_tool_call():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_final_invalid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"wrong": "value"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_final_valid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "fixed"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
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
                session_id="sess_structured_output_tool_custom_repair",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=1,
                    repair_prompt="Call the finalizer tool again with the corrected object.",
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_tool_custom_repair"))

    invalid_result = transcript[2].content[0]
    assert invalid_result.is_error is True
    assert "Structured output rejected" in invalid_result.content
    assert "Call the finalizer tool again with the corrected object." in invalid_result.content


def test_cayu_app_rejects_mixed_structured_output_tool_round_without_side_effects():
    store = InMemorySessionStore()
    side_effect_tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_side_effect",
                    name="side_effect",
                    arguments={"value": "do not run"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_final_mixed",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "too early"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_final_valid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "fixed"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[side_effect_tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_structured_output_tool_mixed",
                messages=[Message.text("user", "use tools and answer")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    max_retries=1,
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_tool_mixed"))

    assert side_effect_tool.calls == []
    assert EventType.TOOL_CALL_STARTED not in [event.type for event in events]
    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[3].payload["errors"][0]["message"] == (
        "Call the structured-output tool by itself, not in the same tool round as other tools."
    )
    mixed_tool_results = transcript[2].content
    assert [part.tool_name for part in mixed_tool_results] == [
        "side_effect",
        STRUCTURED_OUTPUT_TOOL_NAME,
    ]
    assert all(part.is_error for part in mixed_tool_results)


def test_cayu_app_does_not_count_structured_output_tool_against_tool_call_limit():
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
                ModelStreamEvent.tool_call(
                    id="call_final",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "done"}},
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
                session_id="sess_structured_output_tool_limit",
                messages=[Message.text("user", "use one tool and answer")],
                limits=RunLimits(max_tool_calls=1),
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                ),
            ),
        )
    )

    assert EventType.SESSION_LIMIT_REACHED not in [event.type for event in events]
    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[7].payload["output"] == {"answer": "done"}


def test_cayu_app_validates_native_structured_output_final_text():
    store = InMemorySessionStore()
    provider = NativeStructuredOutputFakeProvider(
        [
            ModelStreamEvent.text_delta('{"answer":"ok"}'),
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
                session_id="sess_structured_output_native_valid",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    name="answer",
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    strategy="native",
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_native_valid"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[4].payload["output"] == {"answer": "ok"}
    assert provider.requests[0].options["structured_output"]["strategy"] == "native"
    assert provider.requests[0].tools == []
    assert [message.role for message in provider.requests[0].messages] == ["user"]
    assert [message.role for message in transcript] == ["user", "assistant"]


def test_cayu_app_retries_invalid_native_structured_output_final_text():
    store = InMemorySessionStore()
    provider = NativeStructuredOutputFakeProvider(
        [
            [
                ModelStreamEvent.text_delta('{"wrong":"value"}'),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta('{"answer":"fixed"}'),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
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
                session_id="sess_structured_output_native_retry",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=1,
                    strategy="native",
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_native_retry"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[4].payload["errors"][0]["path"] == "$"
    assert events[9].payload["output"] == {"answer": "fixed"}
    repair_message = transcript[2].content[0].text
    assert "Return only valid JSON" in repair_message
    assert STRUCTURED_OUTPUT_TOOL_NAME not in repair_message


def test_cayu_app_rejects_native_structured_output_for_unsupported_provider():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta('{"answer":"ok"}'),
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
                session_id="sess_structured_output_native_unsupported",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    strategy="native",
                ),
            ),
        )
    )
    session = asyncio.run(store.load("sess_structured_output_native_unsupported"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == (
        "Native structured output is not supported by provider: fake"
    )
    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert provider.requests == []


def test_cayu_app_retries_structured_output_with_durable_repair_prompt():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("not json"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_final_fixed",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "fixed"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
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
                session_id="sess_structured_output_retry",
                messages=[Message.text("user", "answer with json")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=1,
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_retry"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[4].payload["errors"][0]["message"] == (
        f"Final structured output must be submitted with the `{STRUCTURED_OUTPUT_TOOL_NAME}` tool."
    )
    assert events[5].payload["attempt"] == 1
    assert events[8].payload["attempt"] == 2
    assert len(provider.requests) == 2
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    repair_message = transcript[2].content[0].text
    assert STRUCTURED_OUTPUT_TOOL_NAME in repair_message
    assert "plain text" in repair_message
    assert "Validation errors:" in repair_message
    assert provider.requests[1].messages[-1].role == "user"
    assert provider.requests[1].messages[-1].content[0].text == repair_message


def test_cayu_app_fails_structured_output_after_retries_exhausted():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("not json"),
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
                session_id="sess_structured_output_failed",
                messages=[Message.text("user", "answer with json")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    max_retries=0,
                ),
            ),
        )
    )
    session = asyncio.run(store.load("sess_structured_output_failed"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.SESSION_FAILED,
    ]
    assert events[-1].payload["error"] == (
        "Structured output validation failed after 1 attempt(s)."
    )
    assert session is not None
    assert session.status == SessionStatus.FAILED


def test_cayu_app_does_not_write_structured_output_repair_without_remaining_step():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("not json"),
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
                session_id="sess_structured_output_no_step_for_repair",
                messages=[Message.text("user", "answer with json")],
                max_steps=1,
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    max_retries=1,
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_no_step_for_repair"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.SESSION_FAILED,
    ]
    assert EventType.STRUCTURED_OUTPUT_RETRY not in [event.type for event in events]
    assert events[-1].payload["error"] == (
        "Structured output validation failed after 1 attempt(s): "
        "maximum model steps reached before repair."
    )
    assert [message.role for message in transcript] == ["user", "assistant"]
    assert transcript[-1].content[0].text == "not json"


def test_cayu_app_validates_structured_output_only_after_tool_round_finishes():
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
                ModelStreamEvent.tool_call(
                    id="call_final",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "from tool"}},
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
                session_id="sess_structured_output_after_tool",
                messages=[Message.text("user", "use tool then answer with json")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    }
                ),
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
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2
    assert events[7].payload["output"] == {"answer": "from tool"}


def test_cayu_app_validates_native_structured_output_only_after_tool_round_finishes():
    store = InMemorySessionStore()
    provider = NativeStructuredOutputFakeProvider(
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
                ModelStreamEvent.text_delta('{"answer":"from tool"}'),
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
                session_id="sess_native_structured_output_after_tool",
                messages=[Message.text("user", "use tool then answer with json")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    strategy="native",
                ),
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
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2
    assert provider.requests[0].options["structured_output"]["strategy"] == "native"
    assert provider.requests[1].options["structured_output"]["strategy"] == "native"
    assert events[8].payload["output"] == {"answer": "from tool"}


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

    class ArtifactStoreLike:
        id = "artifacts"

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

    with pytest.raises(TypeError, match="artifact_store"):
        Environment(
            EnvironmentSpec(name="artifact_store_like"),
            artifact_store=ArtifactStoreLike(),  # type: ignore[arg-type]
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


def test_interrupt_session_marks_pending_session_interrupted_and_emits_event():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]))
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_direct",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_interrupt_direct",
                    reason="operator requested stop",
                    metadata={"actor": "operator"},
                )
            )
        ]
        return events, await store.load("sess_interrupt_direct")

    events, session = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert [event.type for event in events] == [EventType.SESSION_INTERRUPTED]
    assert events[0].payload == {
        "reason": "operator requested stop",
        "metadata": {"actor": "operator"},
        "interruption_type": "operator_requested",
    }


def test_interrupt_session_race_returns_existing_interrupt_event_without_duplicate():
    class PausingInterruptStore(InMemorySessionStore):
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
            checkpoint_transform,
        ) -> Session:
            session = await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
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
        stored_events = await store.load_events("sess_interrupt_race_idempotent")
        return first_events, second_events, stored_events

    first_events, second_events, stored_events = asyncio.run(run())

    assert [event.type for event in first_events] == [EventType.SESSION_INTERRUPTED]
    assert [event.type for event in second_events] == [EventType.SESSION_INTERRUPTED]
    assert first_events[0].id == second_events[0].id
    assert first_events[0].payload["reason"] == "first request"
    assert [event for event in stored_events if event.type == EventType.SESSION_INTERRUPTED] == [
        first_events[0]
    ]


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
            checkpoint_transform,
        ) -> Session:
            session = await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
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
        stored_events = await store.load_events("sess_run_interrupt_race")
        return run_events, interrupt_events, stored_events

    run_events, interrupt_events, stored_events = asyncio.run(run())

    stored_interrupt_events = [
        event for event in stored_events if event.type == EventType.SESSION_INTERRUPTED
    ]
    assert len(stored_interrupt_events) == 1
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    assert run_events[-1].id == interrupt_events[0].id == stored_interrupt_events[0].id
    assert stored_interrupt_events[0].payload["reason"] == "external interrupt"


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
        checkpoint = await store.load_checkpoint("sess_cross_app_interrupt")
        return run_events, interrupt_events, checkpoint

    run_events, interrupt_events, checkpoint = asyncio.run(run())

    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    assert interrupt_events == [run_events[-1]]
    assert run_events[-1].payload == {
        "reason": "operator stop from api",
        "metadata": {"actor": "operator"},
        "interruption_type": "operator_requested",
    }
    assert checkpoint == {}


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
        checkpoint = await store.load_checkpoint("sess_interrupt_direct_stream_closed")
        return first_event, checkpoint

    first_event, checkpoint = asyncio.run(run())

    assert first_event.type == EventType.SESSION_INTERRUPTED
    assert first_event.payload["reason"] == "operator stop"
    assert checkpoint == {}


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
        checkpoint = await store.load_checkpoint("sess_interrupt_active_stream_closed")
        return run_event, interrupt_events, checkpoint

    run_event, interrupt_events, checkpoint = asyncio.run(run())

    assert run_event.type == EventType.SESSION_INTERRUPTED
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert run_event.id == interrupt_events[0].id
    assert run_event.payload["reason"] == "operator stop"
    assert checkpoint == {}


def test_interrupt_session_persists_payload_atomically_with_interrupting_status():
    class AtomicInterruptStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.checked_inside_transition = False

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform,
        ):
            session = await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
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
        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform,
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
    monkeypatch.setattr(runtime_app_module, "_ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS", 100)
    monkeypatch.setattr(runtime_app_module, "_ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S", 0.01)

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
            if app._is_session_interruption_request_active("sess_cancel_interrupt_request"):
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return app._is_session_interruption_request_active("sess_cancel_interrupt_request")

    assert asyncio.run(run()) is False


def test_interrupt_session_returns_terminal_event_when_provider_delays_cancellation(monkeypatch):
    monkeypatch.setattr(runtime_app_module, "_ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS", 2)
    monkeypatch.setattr(runtime_app_module, "_ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S", 0)

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
    assert EventType.SESSION_INTERRUPTED not in [event.type for event in events_before_release]
    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    assert run_events[-1].id == interrupt_events[0].id
    assert [event.type for event in events_after_release].count(EventType.SESSION_INTERRUPTED) == 1
    assert EventType.MODEL_COMPLETED not in [event.type for event in events_after_release]


def test_interrupt_session_does_not_finalize_unowned_running_session(monkeypatch):
    monkeypatch.setattr(runtime_app_module, "_ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS", 2)
    monkeypatch.setattr(runtime_app_module, "_ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S", 0)

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
    monkeypatch.setattr(runtime_app_module, "_ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS", 2)
    monkeypatch.setattr(runtime_app_module, "_ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S", 0)

    class LosingTransitionStore(InMemorySessionStore):
        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform,
        ) -> Session:
            if session_id == "sess_interrupt_transition_loser_finalizing":
                await super().transition_status_and_checkpoint(
                    session_id,
                    from_statuses=from_statuses,
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=checkpoint_transform,
                )
                raise ValueError("lost transition")
            return await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
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
    assert interrupt_events[0].payload == {
        "reason": "operator stop",
        "metadata": {},
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
    assert len(failed_tool_events) == 1
    assert failed_tool_events[0].payload["tool_call_id"] == "call_1"
    assert failed_tool_events[0].payload["result"]["is_error"] is True
    assert failed_tool_events[0].payload["result"]["structured"] == {
        "interrupted": True,
        "tool_call_id": "call_1",
        "tool_name": "blocking_tool",
    }
    stored_event_types = [event.type for event in stored_events]
    assert stored_event_types.index(EventType.TOOL_CALL_FAILED) < stored_event_types.index(
        EventType.SESSION_INTERRUPTED
    )
    validate_context_messages(transcript)
    assert transcript[-1].role == "tool"
    assert transcript[-1].content[0].tool_call_id == "call_1"
    assert transcript[-1].content[0].is_error is True


def test_cancelled_runner_cleanup_diagnostics_are_preserved_in_tool_result():
    cleanup_artifact = {
        "type": "cayu.runner_cleanup.v1",
        "adapter": "e2b",
        "action": "kill_sandbox",
        "status": "timeout",
        "timeout_s": 0.01,
    }

    class CleanupDiagnosticTool(Tool):
        spec = ToolSpec(
            name="cleanup_diagnostic_tool",
            description="Raise runner cancellation diagnostics.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.started: asyncio.Event | None = None
            self.never_complete: asyncio.Event | None = None

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            if self.started is None or self.never_complete is None:
                raise AssertionError("CleanupDiagnosticTool test events were not initialized.")
            self.started.set()
            try:
                await self.never_complete.wait()
            except asyncio.CancelledError as exc:
                raise RunnerCancelledError(artifacts=[cleanup_artifact]) from exc
            return ToolResult(content="unexpected")

    tool = CleanupDiagnosticTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="cleanup_diagnostic_tool",
                arguments={},
            ),
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
        tool.never_complete = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interrupt_tool_cleanup_diagnostics",
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await tool.started.wait()
        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_interrupt_tool_cleanup_diagnostics",
                    reason="operator stop",
                )
            )
        ]
        run_events = await run_task
        stored_events = await app.session_store.load_events(
            "sess_interrupt_tool_cleanup_diagnostics"
        )
        transcript = await app.session_store.load_transcript(
            "sess_interrupt_tool_cleanup_diagnostics"
        )
        return run_events, interrupt_events, stored_events, transcript

    run_events, interrupt_events, stored_events, transcript = asyncio.run(run())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert run_events[-1].id == interrupt_events[0].id
    failed_tool_events = [
        event for event in stored_events if event.type == EventType.TOOL_CALL_FAILED
    ]
    assert len(failed_tool_events) == 1
    assert failed_tool_events[0].payload["result"]["artifacts"] == [cleanup_artifact]
    validate_context_messages(transcript)
    assert transcript[-1].role == "tool"
    assert transcript[-1].content[0].artifacts == [cleanup_artifact]


def test_cancelled_runner_cleanup_diagnostics_are_attached_only_to_active_tool():
    cleanup_artifact = {
        "type": "cayu.runner_cleanup.v1",
        "adapter": "e2b",
        "action": "kill_sandbox",
        "status": "timeout",
        "timeout_s": 0.01,
    }

    class CleanupDiagnosticTool(Tool):
        spec = ToolSpec(
            name="cleanup_diagnostic_tool",
            description="Raise runner cancellation diagnostics.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.started: asyncio.Event | None = None
            self.never_complete: asyncio.Event | None = None

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            if self.started is None or self.never_complete is None:
                raise AssertionError("CleanupDiagnosticTool test events were not initialized.")
            self.started.set()
            try:
                await self.never_complete.wait()
            except asyncio.CancelledError as exc:
                raise RunnerCancelledError(artifacts=[cleanup_artifact]) from exc
            return ToolResult(content="unexpected")

    tool = CleanupDiagnosticTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_active",
                name="cleanup_diagnostic_tool",
                arguments={},
            ),
            ModelStreamEvent.tool_call(
                id="call_not_started",
                name="echo",
                arguments={"text": "never runs"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool, EchoTool()],
    )

    async def run():
        tool.started = asyncio.Event()
        tool.never_complete = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interrupt_tool_cleanup_diagnostics_round",
                    messages=[Message.text("user", "use tools")],
                ),
            )
        )
        await tool.started.wait()
        _ = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_interrupt_tool_cleanup_diagnostics_round",
                    reason="operator stop",
                )
            )
        ]
        run_events = await run_task
        transcript = await app.session_store.load_transcript(
            "sess_interrupt_tool_cleanup_diagnostics_round"
        )
        stored_events = await app.session_store.load_events(
            "sess_interrupt_tool_cleanup_diagnostics_round"
        )
        return run_events, transcript, stored_events

    run_events, transcript, stored_events = asyncio.run(run())

    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    failed_tool_events = [
        event for event in stored_events if event.type == EventType.TOOL_CALL_FAILED
    ]
    assert [event.payload["tool_call_id"] for event in failed_tool_events] == [
        "call_active",
        "call_not_started",
    ]
    assert failed_tool_events[0].payload["result"]["artifacts"] == [cleanup_artifact]
    assert failed_tool_events[1].payload["result"]["artifacts"] == []
    validate_context_messages(transcript)
    assert transcript[-1].role == "tool"
    result_parts = transcript[-1].content
    assert [part.tool_call_id for part in result_parts] == [
        "call_active",
        "call_not_started",
    ]
    assert result_parts[0].artifacts == [cleanup_artifact]
    assert result_parts[1].artifacts == []


def test_interrupt_session_suppresses_late_tool_events_while_finalizing(monkeypatch):
    monkeypatch.setattr(runtime_app_module, "_ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS", 2)
    monkeypatch.setattr(runtime_app_module, "_ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S", 0)

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
    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    assert run_events[-1].id == interrupt_events[0].id
    event_types_after_release = [event.type for event in events_after_release]
    assert event_types_after_release.count(EventType.SESSION_INTERRUPTED) == 1
    assert EventType.TOOL_CALL_COMPLETED not in event_types_after_release
    assert event_types_after_release.count(EventType.TOOL_CALL_FAILED) == 1
    assert event_types_after_release[-1] == EventType.SESSION_INTERRUPTED
    validate_context_messages(transcript)
    assert transcript[-1].role == "tool"
    assert transcript[-1].content[0].tool_call_id == "call_1"
    assert transcript[-1].content[0].content == "Tool call interrupted before completion."
    assert transcript[-1].content[0].is_error is True


def test_repeated_interrupt_waits_for_active_interruption_terminal_event():
    class BlockingSink(EventSink):
        def __init__(self) -> None:
            self.failed_seen: asyncio.Event | None = None
            self.release_failed_event: asyncio.Event | None = None

        async def emit(self, event: Event) -> None:
            if event.type == EventType.TOOL_CALL_FAILED:
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
        await sink.failed_seen.wait()
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
    assert first_events[0].payload == {
        "reason": "first interrupt",
        "metadata": {},
        "interruption_type": "operator_requested",
    }
    stored_event_types = [event.type for event in stored_events]
    assert stored_event_types.index(EventType.TOOL_CALL_FAILED) < stored_event_types.index(
        EventType.SESSION_INTERRUPTED
    )
    assert stored_event_types.count(EventType.SESSION_INTERRUPTED) == 1


def test_concurrent_interrupt_transition_loser_waits_for_terminal_event():
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
            checkpoint_transform,
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
            return await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
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
    assert stored_interrupt_events == [run_events[-1]]


def test_interrupt_session_preserves_completed_tool_results_in_interrupted_round():
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

    blocking_tool = BlockingTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_echo",
                name="echo",
                arguments={"text": "first"},
            ),
            ModelStreamEvent.tool_call(
                id="call_block",
                name="blocking_tool",
                arguments={},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool(), blocking_tool],
    )

    async def run():
        blocking_tool.started = asyncio.Event()
        blocking_tool.cancelled = asyncio.Event()
        blocking_tool.never_complete = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interrupt_partial_tool_round",
                    messages=[Message.text("user", "use tools")],
                ),
            )
        )
        await blocking_tool.started.wait()
        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_interrupt_partial_tool_round",
                    reason="operator stop",
                )
            )
        ]
        await asyncio.wait_for(blocking_tool.cancelled.wait(), timeout=1)
        run_events = await run_task
        stored_events = await app.session_store.load_events("sess_interrupt_partial_tool_round")
        transcript = await app.session_store.load_transcript("sess_interrupt_partial_tool_round")
        return run_events, interrupt_events, stored_events, transcript

    run_events, interrupt_events, stored_events, transcript = asyncio.run(run())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    assert run_events[-1].id == interrupt_events[0].id
    assert [event.type for event in stored_events].count(EventType.TOOL_CALL_COMPLETED) == 1
    assert [event.type for event in stored_events].count(EventType.TOOL_CALL_FAILED) == 1
    validate_context_messages(transcript)
    result_parts = transcript[-1].content
    assert [part.tool_call_id for part in result_parts] == ["call_echo", "call_block"]
    assert result_parts[0].content == "first"
    assert result_parts[0].is_error is False
    assert result_parts[1].content == "Tool call interrupted before completion."
    assert result_parts[1].is_error is True


def test_interrupt_session_preserves_tool_result_when_interrupted_after_tool_returns():
    store = InMemorySessionStore()

    class InterruptingAfterReturnTool(Tool):
        spec = ToolSpec(
            name="interrupting_after_return_tool",
            description="Interrupt session immediately before returning a real result.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            await store.update_status(ctx.session_id, SessionStatus.INTERRUPTED)
            return ToolResult(
                content="real completed result",
                structured={"completed": True},
            )

    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_finished",
                name="interrupting_after_return_tool",
                arguments={},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[InterruptingAfterReturnTool()],
    )

    async def run():
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_after_tool_return",
                messages=[Message.text("user", "use tool")],
            ),
        )
        transcript = await store.load_transcript("sess_interrupt_after_tool_return")
        stored_events = await store.load_events("sess_interrupt_after_tool_return")
        return events, transcript, stored_events

    events, transcript, stored_events = asyncio.run(run())

    assert events[-1].type == EventType.SESSION_INTERRUPTED
    assert [event.type for event in stored_events].count(EventType.TOOL_CALL_COMPLETED) == 1
    assert [event.type for event in stored_events].count(EventType.TOOL_CALL_FAILED) == 0
    tool_event = next(
        event for event in stored_events if event.type == EventType.TOOL_CALL_COMPLETED
    )
    assert tool_event.payload["result"]["content"] == "real completed result"
    validate_context_messages(transcript)
    assert transcript[-1].role == "tool"
    assert len(transcript[-1].content) == 1
    assert transcript[-1].content[0].tool_call_id == "call_finished"
    assert transcript[-1].content[0].content == "real completed result"
    assert transcript[-1].content[0].structured == {"completed": True}
    assert transcript[-1].content[0].is_error is False


def test_interrupt_session_closes_tool_round_when_interrupted_after_assistant_tool_call_append():
    class InterruptingAfterAssistantToolCallStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.interrupt_after_next_assistant_tool_call_append = False

        async def append_transcript_messages(
            self,
            session_id: str,
            messages: list[Message],
        ) -> None:
            await super().append_transcript_messages(session_id, messages)
            if self.interrupt_after_next_assistant_tool_call_append and any(
                any(type(part) is ToolCallPart for part in message.content) for message in messages
            ):
                self.interrupt_after_next_assistant_tool_call_append = False
                await self.update_status(session_id, SessionStatus.INTERRUPTED)

    store = InterruptingAfterAssistantToolCallStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_echo",
                name="echo",
                arguments={"text": "should not execute"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    async def run():
        store.interrupt_after_next_assistant_tool_call_append = True
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_after_assistant_tool_call_append",
                messages=[Message.text("user", "use tool")],
            ),
        )
        transcript = await store.load_transcript("sess_interrupt_after_assistant_tool_call_append")
        stored_events = await store.load_events("sess_interrupt_after_assistant_tool_call_append")
        return events, transcript, stored_events

    events, transcript, stored_events = asyncio.run(run())

    assert events[-1].type == EventType.SESSION_INTERRUPTED
    assert [event.type for event in stored_events].count(EventType.TOOL_CALL_COMPLETED) == 0
    assert [event.type for event in stored_events].count(EventType.TOOL_CALL_FAILED) == 1
    validate_context_messages(transcript)
    assert transcript[-1].role == "tool"
    assert len(transcript[-1].content) == 1
    assert transcript[-1].content[0].tool_call_id == "call_echo"
    assert transcript[-1].content[0].content == "Tool call interrupted before completion."
    assert transcript[-1].content[0].structured == {
        "interrupted": True,
        "tool_call_id": "call_echo",
        "tool_name": "echo",
    }
    assert transcript[-1].content[0].is_error is True


def test_interrupt_session_does_not_leave_pending_approval_when_interrupted_after_policy_plan():
    store = InMemorySessionStore()

    class InterruptingApprovalPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            await store.update_status(request.session.id, SessionStatus.INTERRUPTED)
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason=f"Approval required for {request.tool_name}.",
            )

    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_echo",
                name="echo",
                arguments={"text": "should not execute"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        tool_policy=InterruptingApprovalPolicy(),
    )

    async def run():
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_after_policy_plan",
                messages=[Message.text("user", "use tool")],
            ),
        )
        transcript = await store.load_transcript("sess_interrupt_after_policy_plan")
        stored_events = await store.load_events("sess_interrupt_after_policy_plan")
        checkpoint = await store.load_checkpoint("sess_interrupt_after_policy_plan")
        return events, transcript, stored_events, checkpoint

    events, transcript, stored_events, checkpoint = asyncio.run(run())

    assert events[-1].type == EventType.SESSION_INTERRUPTED
    assert EventType.TOOL_CALL_APPROVAL_REQUESTED not in [event.type for event in stored_events]
    assert [event.type for event in stored_events].count(EventType.TOOL_CALL_FAILED) == 1
    assert checkpoint is None
    validate_context_messages(transcript)
    assert transcript[-1].role == "tool"
    assert transcript[-1].content[0].tool_call_id == "call_echo"
    assert transcript[-1].content[0].content == "Tool call interrupted before completion."
    assert transcript[-1].content[0].is_error is True


def test_interrupt_session_preserves_tool_results_when_interrupted_before_append():
    class InterruptingTranscriptStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.interrupt_on_next_tool_result_append = False

        async def append_transcript_messages(
            self,
            session_id: str,
            messages: list[Message],
        ) -> None:
            if self.interrupt_on_next_tool_result_append and any(
                message.role == "tool" for message in messages
            ):
                self.interrupt_on_next_tool_result_append = False
                await self.update_status(session_id, SessionStatus.INTERRUPTED)
                raise asyncio.CancelledError
            await super().append_transcript_messages(session_id, messages)

    store = InterruptingTranscriptStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_echo",
                name="echo",
                arguments={"text": "finished"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    async def run():
        store.interrupt_on_next_tool_result_append = True
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_after_tool_before_append",
                messages=[Message.text("user", "use tool")],
            ),
        )
        transcript = await store.load_transcript("sess_interrupt_after_tool_before_append")
        stored_events = await store.load_events("sess_interrupt_after_tool_before_append")
        return events, transcript, stored_events

    events, transcript, stored_events = asyncio.run(run())

    assert events[-1].type == EventType.SESSION_INTERRUPTED
    assert [event.type for event in stored_events].count(EventType.SESSION_INTERRUPTED) == 1
    assert [event.type for event in stored_events].count(EventType.TOOL_CALL_COMPLETED) == 1
    assert [event.type for event in stored_events].count(EventType.TOOL_CALL_FAILED) == 0
    validate_context_messages(transcript)
    assert transcript[-1].role == "tool"
    assert len(transcript[-1].content) == 1
    assert transcript[-1].content[0].tool_call_id == "call_echo"
    assert transcript[-1].content[0].content == "finished"
    assert transcript[-1].content[0].is_error is False
