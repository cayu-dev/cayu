from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

import pytest
from pydantic import ValidationError

import cayu.runtime.app as runtime_app_module
from cayu.artifacts import (
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    LocalArtifactStore,
    file_attachment,
)
from cayu.core import AgentSpec, Event, EventType, Message, TextPart, ToolCallPart, ToolResultPart
from cayu.core.messages import FilePart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.environments import (
    BoundWorkspace,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    WorkspaceBinding,
    WorkspaceInstructionsConfig,
    WorkspaceSnapshot,
)
from cayu.providers import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelContextOverflowError,
    ModelContextPressureProfile,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    NativeStructuredOutputSchemaInvalid,
    OpenAIProvider,
)
from cayu.proxies import CredentialProxy, PassthroughProxy, ProxyAuthorizationResult
from cayu.runners import RunnerCancelledError
from cayu.runtime import (
    TAINT_LABELS_METADATA_KEY,
    TOOL_POLICY_REAUTHORIZATION_METADATA_KEY,
    AfterToolCallDecision,
    AllowAllToolPolicy,
    AllowlistRule,
    BeforeStopContext,
    BeforeStopDecision,
    BeforeToolCallDecision,
    BeforeToolCallHookContext,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    BudgetWindow,
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextCompactor,
    ContextCountingConfig,
    ContextCountingMode,
    ContextPolicy,
    ContextPressureEstimate,
    ContextPressureOverhead,
    ContextRequest,
    ContextUsageState,
    DefaultContextPolicy,
    Dispatcher,
    DispatchHandle,
    DispatchRequest,
    DispatchStatus,
    EventQuery,
    EventSink,
    ForkSessionRequest,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionsRecoveryRequest,
    InMemoryBudgetLedger,
    InMemoryEventSink,
    InMemorySessionStore,
    InMemoryTaskStore,
    InterruptSessionRequest,
    KnowledgeInjectionPolicy,
    LoopPolicy,
    MessageWindowContextPolicy,
    ModelCompactor,
    ModelPricing,
    NativeStructuredOutputUnsupported,
    ObservedDeltaContextEstimator,
    ParameterConstrainedToolPolicy,
    PendingToolApproval,
    PricingCatalog,
    RecentTurnsContextPolicy,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RetryPolicy,
    RunLimits,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    Session,
    SessionIdentity,
    SessionListResult,
    SessionQuery,
    SessionStatus,
    StaticToolPolicy,
    StructuredOutputSpec,
    TaintAwareToolPolicy,
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
    ToolRoundRecoveryRequest,
    TranscriptDigestCompactor,
    UsageTriggeredContextPolicy,
    UserInputResponse,
    default_compaction_prompt,
    strip_old_file_attachments,
    trim_context_messages,
    trim_context_turns,
)
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime.app import _require_native_structured_output_support
from cayu.runtime.context import (
    ContextBuildResult,
    RuntimeManagedContextPolicy,
    validate_context_messages,
)
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.storage import InMemoryKnowledgeStore, KnowledgeEntry
from cayu.tools import (
    SubagentExecutionMode,
    SubagentResultTool,
    SubagentSpec,
    SubagentTool,
)
from cayu.tools.user_input import UserInputTool
from cayu.vaults import ResolvedSecret, SecretRef, StaticVault
from cayu.workspaces import LocalWorkspace, Workspace, WorkspaceListResult, WorkspaceReadResult


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


class CountingProvider(FakeProvider):
    def __init__(
        self,
        events: list[ModelStreamEvent] | list[list[ModelStreamEvent]],
        *,
        count_result: InputTokenCountResult | None = None,
        count_error: Exception | None = None,
    ) -> None:
        super().__init__(events)
        self.count_result = count_result
        self.count_error = count_error
        self.count_requests: list[ModelRequest] = []

    async def count_input_tokens(
        self,
        request: ModelRequest,
    ) -> InputTokenCountResult | None:
        self.count_requests.append(request)
        if self.count_error is not None:
            raise self.count_error
        return self.count_result


class ContextOverflowProvider(ModelProvider):
    name = "overflow"

    def __init__(
        self,
        *,
        overflow_requests: int = 1,
        success_events: list[ModelStreamEvent] | None = None,
    ) -> None:
        self.overflow_requests = overflow_requests
        self.success_events = success_events or [
            ModelStreamEvent.text_delta("recovered"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) <= self.overflow_requests:
            raise ModelContextOverflowError(
                "context too large",
                provider=self.name,
                status_code=400,
                error_code="context_length_exceeded",
            )
        for event in self.success_events:
            yield event


class EventFlattenedOverflowProvider(ModelProvider):
    """Contract-violating provider: flattens overflow into an error event.

    `stream()` must raise `ModelContextOverflowError`, but a provider that
    wraps every failure with `ModelStreamEvent.error(..., cause=exc)` still
    carries the typed overflow identity in the payload. The runtime must route
    that into context-overflow recovery instead of generic retries.
    """

    name = "overflow-event"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.error(
                "context too large",
                cause=ModelContextOverflowError(
                    "context too large",
                    provider=self.name,
                    status_code=400,
                    error_code="context_length_exceeded",
                ),
            )
            return
        yield ModelStreamEvent.text_delta("recovered")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class OtherProvider(FakeProvider):
    name = "other"


class NativeStructuredOutputFakeProvider(FakeProvider):
    supports_native_structured_output = True


class SchemaRejectingProvider(NativeStructuredOutputFakeProvider):
    def preflight_native_structured_output_schema(self, json_schema: dict[str, Any]) -> None:
        raise NativeStructuredOutputSchemaInvalid("$: rejected by provider preflight.")


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

    mutation_blocked = False

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        async for event in super().stream(request):
            if len(self.requests) == 1:
                # Messages are frozen; the runtime history cannot be corrupted
                # even by a misbehaving provider.
                try:
                    request.messages[0].content[0].text = "mutated by provider"
                except ValidationError:
                    self.mutation_blocked = True
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
        if not self.failed_close_once and any(message.role == "tool" for message in messages):
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


class FailingOrdinaryToolResultCloseStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.failed_tool_round_close_once = False

    async def append_transcript_messages_and_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint: dict,
    ) -> None:
        if not self.failed_tool_round_close_once and any(
            message.role == "tool" for message in messages
        ):
            self.failed_tool_round_close_once = True
            raise RuntimeError("ordinary tool round close unavailable")
        await super().append_transcript_messages_and_checkpoint(
            session_id,
            messages,
            checkpoint,
        )


class FailingAfterPendingToolRoundCheckpointStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.failed_pending_tool_round_once = False

    async def append_transcript_messages_and_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint: dict,
    ) -> None:
        await super().append_transcript_messages_and_checkpoint(
            session_id,
            messages,
            checkpoint,
        )
        if (
            not self.failed_pending_tool_round_once
            and "pending_tool_round" in checkpoint
            and any(message.role == "assistant" for message in messages)
        ):
            self.failed_pending_tool_round_once = True
            raise RuntimeError("pending tool round checkpoint persisted before crash")


class BarrierSideEffectTool(Tool):
    spec = ToolSpec(
        name="side_effect",
        description="Record execution.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []
        self._entered = 0
        self._both_entered = asyncio.Event()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        self._entered += 1
        if self._entered >= 2:
            self._both_entered.set()
        # Hold both calls until each has started so a terminal-event crash leaves
        # BOTH started-but-unresolved (deterministic multi-call recovery shape).
        await self._both_entered.wait()
        return ToolResult(content="recorded")


class FailingAllTerminalToolEventStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.failing = True

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        if self.failing and any(
            event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            for event in events
        ):
            raise RuntimeError("terminal tool event unavailable")
        await super().append_events(session_id, events)


class FailingPostPersistEventsLoadStore(FailingTerminalToolEventStore):
    """Crash the round like the base store, then fail the first event load that
    happens after a manual-recovery terminal event was persisted (the
    remaining-unknown recheck inside recover_tool_round)."""

    def __init__(self) -> None:
        super().__init__()
        self.arm_post_persist_load_failure = False

    async def load_events(self, session_id: str) -> list[Event]:
        events = await super().load_events(session_id)
        if self.arm_post_persist_load_failure and any(
            event.payload.get("manual_recovery") is True for event in events
        ):
            self.arm_post_persist_load_failure = False
            raise RuntimeError("events unavailable after manual recovery persisted")
        return events


class FailingSecondTerminalToolEventStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.completed_tool_events = 0

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        for event in events:
            if event.type == EventType.TOOL_CALL_COMPLETED:
                self.completed_tool_events += 1
                if self.completed_tool_events == 2:
                    raise RuntimeError("second terminal tool event unavailable")
        await super().append_events(session_id, events)


class RecordingListSessionsStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.session_queries: list[SessionQuery] = []

    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        copied_query = SessionQuery() if query is None else query.model_copy(deep=True)
        self.session_queries.append(copied_query)
        return await super().list_sessions(query)


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

    async def delete(self, path: str) -> None:
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


class WorkspaceIdTool(Tool):
    spec = ToolSpec(
        name="workspace_id",
        description="Return the active workspace id.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=ctx.workspace_id or "none",
            structured={"workspace_id": ctx.workspace_id},
        )


class RecordingWorkspaceBinding(WorkspaceBinding):
    def __init__(
        self,
        *,
        bound_workspace: Workspace | None = None,
        snapshot: WorkspaceSnapshot | None = None,
        final_snapshot: WorkspaceSnapshot | None = None,
        fail_bind: bool = False,
        fail_finalize: bool = False,
    ) -> None:
        self.bound_workspace = bound_workspace
        self.snapshot = snapshot
        self.final_snapshot = final_snapshot
        self.fail_bind = fail_bind
        self.fail_finalize = fail_finalize
        self.bind_calls: list[dict] = []
        self.finalize_calls: list[dict] = []

    async def bind(
        self,
        workspace: Workspace | None,
        runner: Any,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        self.bind_calls.append(
            {
                "workspace": workspace,
                "runner": runner,
                "session_id": session_id,
                "agent_name": agent_name,
                "environment_name": environment_name,
                "metadata": metadata,
            }
        )
        if self.fail_bind:
            raise RuntimeError("bind failed")
        return BoundWorkspace(
            workspace=self.bound_workspace if self.bound_workspace is not None else workspace,
            source_workspace=workspace,
            runner=runner,
            path="/bound",
            metadata={"binding": "recording"},
            snapshot=self.snapshot,
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        self.finalize_calls.append(
            {
                "bound": bound,
                "outcome": outcome,
                "metadata": metadata,
            }
        )
        if self.fail_finalize:
            raise RuntimeError("finalize failed")
        return self.final_snapshot


class RecordingEnvironmentFactory(EnvironmentFactory):
    def __init__(
        self,
        environment: Environment,
        *,
        fail_create: bool = False,
        metadata: dict[str, Any] | None = None,
        reconnect_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.environment = environment
        self.fail_create = fail_create
        self.metadata = metadata or {}
        self.reconnect_metadata = reconnect_metadata or {}
        self.requests: list[EnvironmentFactoryRequest] = []

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)
        if self.fail_create:
            raise RuntimeError("factory failed")
        return EnvironmentFactoryResult(
            environment=self.environment,
            metadata=self.metadata,
            reconnect_metadata=self.reconnect_metadata,
        )


class RecordingSessionFailedHook(RuntimeHook):
    def __init__(self) -> None:
        self.contexts: list[RuntimeHookContext] = []

    async def after_session_failed(self, context: RuntimeHookContext) -> None:
        self.contexts.append(context)


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
        self.contexts: list[ToolContext] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        self.contexts.append(ctx)
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


def test_context_counting_is_off_by_default() -> None:
    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        count_result=InputTokenCountResult(
            input_tokens=12,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        ),
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_context_counting_default_off",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.count_requests == []
    assert EventType.CONTEXT_COUNTED not in {event.type for event in events}
    assert EventType.CONTEXT_COUNT_RECONCILED not in {event.type for event in events}


def test_cayu_app_emits_turn_completed_with_aggregated_usage() -> None:
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "model": "fake-model",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "total_tokens": 14,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "model": "fake-model",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 3,
                            "total_tokens": 15,
                        },
                    }
                ),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_turn_completed",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    event_types = [event.type for event in events]
    turn_index = event_types.index(EventType.TURN_COMPLETED)
    assert turn_index < event_types.index(EventType.SESSION_COMPLETED)
    turn = events[turn_index]
    assert turn.payload["status"] == "completed"
    assert turn.payload["step_count"] == 2
    assert turn.payload["tool_call_count"] == 1
    assert turn.payload["token_usage"]["input_tokens"] == 22
    assert turn.payload["token_usage"]["output_tokens"] == 7
    assert turn.payload["token_usage"]["total_tokens"] == 29
    assert turn.payload["models"] == ["fake-model"]
    assert turn.payload["provider_names"] == ["fake"]
    assert isinstance(turn.payload["duration_ms"], int)


def test_cayu_app_turn_completed_on_resume_excludes_prior_turn_usage() -> None:
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "model": "fake-model",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "total_tokens": 14,
                        },
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "model": "fake-model",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 3,
                            "total_tokens": 15,
                        },
                    }
                ),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run() -> tuple[list[Event], list[Event]]:
        first_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_turn_completed_resume",
                messages=[Message.text("user", "first")],
            ),
        )
        resumed_events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_turn_completed_resume",
                messages=[Message.text("user", "second")],
            ),
        )
        return first_events, resumed_events

    first_events, resumed_events = asyncio.run(run())

    first_turn = next(event for event in first_events if event.type == EventType.TURN_COMPLETED)
    resumed_turn = next(event for event in resumed_events if event.type == EventType.TURN_COMPLETED)

    assert first_turn.payload["step_count"] == 1
    assert first_turn.payload["tool_call_count"] == 0
    assert first_turn.payload["token_usage"]["input_tokens"] == 10
    assert first_turn.payload["token_usage"]["output_tokens"] == 4
    assert first_turn.payload["token_usage"]["total_tokens"] == 14

    assert resumed_turn.payload["step_count"] == 1
    assert resumed_turn.payload["tool_call_count"] == 0
    assert resumed_turn.payload["token_usage"]["input_tokens"] == 12
    assert resumed_turn.payload["token_usage"]["output_tokens"] == 3
    assert resumed_turn.payload["token_usage"]["total_tokens"] == 15


def test_context_pressure_estimate_reconciles_against_actual_input_usage() -> None:
    user_text = "secret pressure text should not leak into pressure event payload"
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "model": "fake-model",
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "total_tokens": 23,
                    },
                }
            ),
        ]
    )
    app = CayuApp(
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_context_pressure_reconcile",
                messages=[Message.text("user", user_text)],
            ),
        )
    )

    event_types = [event.type for event in events]
    estimated_index = event_types.index(EventType.CONTEXT_PRESSURE_ESTIMATED)
    started_index = event_types.index(EventType.MODEL_STARTED)
    completed_index = event_types.index(EventType.MODEL_COMPLETED)
    reconciled_index = event_types.index(EventType.CONTEXT_PRESSURE_RECONCILED)
    assert estimated_index < started_index < completed_index < reconciled_index

    estimated = events[estimated_index]
    assert estimated.payload["provider"] == "fake"
    assert estimated.payload["model"] == "fake-model"
    assert estimated.payload["messages"] == {"count": 1, "roles": ["user"]}
    assert estimated.payload["tools"] == {"count": 0}
    assert estimated.payload["estimate"]["method"] == "local_full_request_estimate"
    assert estimated.payload["estimate"]["estimated_context_input_tokens"] > 0
    assert (
        estimated.payload["estimate"]["estimated_context_window_tokens"]
        == estimated.payload["estimate"]["estimated_context_input_tokens"]
    )
    assert user_text not in json.dumps(estimated.payload, sort_keys=True)

    reconciled = events[reconciled_index]
    assert reconciled.payload["observation_id"] == estimated.payload["observation_id"]
    pre_call_estimate = reconciled.payload["pre_call_estimate"]
    expected_delta = 20 - pre_call_estimate["estimated_context_input_tokens"]
    assert reconciled.payload["actual_input_tokens"] == 20
    assert reconciled.payload["delta_tokens"] == expected_delta
    assert reconciled.payload["relative_error"] == expected_delta / 20
    assert reconciled.payload["reconciled"] is True


def test_context_counting_observe_emits_count_and_reconciliation_events() -> None:
    user_text = "secret phrase should not be in context count event payload"
    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "model": "fake-model",
                    "usage": {
                        "input_tokens": 15,
                        "output_tokens": 2,
                        "total_tokens": 17,
                    },
                }
            ),
        ],
        count_result=InputTokenCountResult(
            input_tokens=12,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
            components={"messages": 10, "tools": 2},
        ),
    )
    app = CayuApp(
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_context_counting_observe",
                messages=[Message.text("user", user_text)],
            ),
        )
    )

    event_types = [event.type for event in events]
    counted_index = event_types.index(EventType.CONTEXT_COUNTED)
    started_index = event_types.index(EventType.MODEL_STARTED)
    completed_index = event_types.index(EventType.MODEL_COMPLETED)
    reconciled_index = event_types.index(EventType.CONTEXT_COUNT_RECONCILED)
    assert counted_index < started_index < completed_index < reconciled_index
    assert len(provider.count_requests) == 1

    counted = events[counted_index]
    assert counted.payload["provider"] == "fake"
    assert counted.payload["model"] == "fake-model"
    assert isinstance(counted.payload["observation_id"], str)
    assert counted.payload["observation_id"]
    assert counted.payload["messages"] == {"count": 1, "roles": ["user"]}
    assert counted.payload["tools"] == {"count": 0}
    assert counted.payload["count"] == {
        "input_tokens": 12,
        "method": "official",
        "confidence": "high",
        "components": {"messages": 10, "tools": 2},
        "metadata": {},
    }
    assert user_text not in json.dumps(counted.payload, sort_keys=True)

    reconciled = events[reconciled_index]
    assert reconciled.payload["observation_id"] == counted.payload["observation_id"]
    assert reconciled.payload["pre_call_count"]["input_tokens"] == 12
    assert reconciled.payload["actual_input_tokens"] == 15
    assert reconciled.payload["delta_tokens"] == 3
    assert reconciled.payload["relative_error"] == 0.2
    assert reconciled.payload["reconciled"] is True


def test_context_counting_uses_defensive_request_copy_for_provider_counter() -> None:
    class MutatingCountingProvider(CountingProvider):
        async def count_input_tokens(
            self,
            request: ModelRequest,
        ) -> InputTokenCountResult | None:
            result = await super().count_input_tokens(request)
            part = request.messages[0].content[0]
            assert isinstance(part, TextPart)
            # Message parts are frozen; a misbehaving counter cannot corrupt
            # the shared transcript at all.
            with pytest.raises(ValidationError):
                part.text = "mutated by counter"  # type: ignore[misc]
            request.tools.append({"name": "mutated_tool"})
            request.options["temperature"] = 1
            return result

    provider = MutatingCountingProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        count_result=InputTokenCountResult(
            input_tokens=12,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        ),
    )
    app = CayuApp(
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_context_counting_copy",
                messages=[Message.text("user", "original prompt")],
            ),
        )
    )

    count_part = provider.count_requests[0].messages[0].content[0]
    stream_part = provider.requests[0].messages[0].content[0]
    assert isinstance(count_part, TextPart)
    assert isinstance(stream_part, TextPart)
    assert count_part.text == "original prompt"
    assert stream_part.text == "original prompt"
    assert provider.count_requests[0].tools == [{"name": "mutated_tool"}]
    assert provider.requests[0].tools == []
    assert provider.count_requests[0].options["temperature"] == 1
    assert "temperature" not in provider.requests[0].options


def test_context_counting_failure_is_observable_and_does_not_block_model_call() -> None:
    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        count_error=RuntimeError("counter endpoint unavailable"),
    )
    app = CayuApp(
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_context_counting_failure",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    event_types = [event.type for event in events]
    assert EventType.CONTEXT_COUNT_FAILED in event_types
    assert EventType.MODEL_COMPLETED in event_types
    assert EventType.CONTEXT_COUNTED not in event_types
    assert EventType.CONTEXT_COUNT_RECONCILED not in event_types

    failed = next(event for event in events if event.type == EventType.CONTEXT_COUNT_FAILED)
    assert failed.payload["error"] == "counter endpoint unavailable"
    assert failed.payload["error_type"] == "RuntimeError"


def test_cayu_app_passes_environment_knowledge_store_to_tools() -> None:
    class KnowledgeStoreCheckTool(Tool):
        spec = ToolSpec(
            name="check_knowledge_store",
            description="Check whether the tool context has a knowledge store.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            entry = await ctx.knowledge_store.get_entry("entry_1")
            return ToolResult(
                content="knowledge store available",
                structured={
                    "has_knowledge_store": ctx.knowledge_store is not None,
                    "entry_text": entry.text if entry is not None else None,
                },
            )

    knowledge_store = InMemoryKnowledgeStore(
        [KnowledgeEntry(id="entry_1", text="Runtime knowledge is available.")]
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_knowledge",
                    name="check_knowledge_store",
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
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            knowledge_store=knowledge_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[KnowledgeStoreCheckTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_knowledge_store_context",
                messages=[Message.text("user", "check knowledge")],
            ),
        )
    )

    tool_event = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    result_payload = tool_event.payload["result"]
    assert result_payload["structured"] == {
        "has_knowledge_store": True,
        "entry_text": "Runtime knowledge is available.",
    }


def test_cayu_app_knowledge_injection_adds_model_context_without_rewriting_transcript() -> None:
    store = InMemorySessionStore()
    knowledge_store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id="git_policy",
                text=(
                    "For GitHub push from a remote sandbox, use a brokered Git HTTP "
                    "proxy so credentials stay outside the sandbox."
                ),
                namespace="project:cayu",
                labels={"project": "cayu"},
                kind="procedure",
                title="Remote sandbox Git credential boundary",
            ),
            KnowledgeEntry(
                id="email_policy",
                text="SendGrid requests should use the email credential proxy.",
                namespace="project:cayu",
                labels={"project": "cayu"},
                kind="procedure",
            ),
        ]
    )
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("use the git proxy"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            knowledge_store=knowledge_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=KnowledgeInjectionPolicy(
            namespace="project:cayu",
            labels={"project": "cayu"},
            max_hits=1,
            max_bytes=800,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_knowledge_injection",
                messages=[Message.text("user", "How should I push to GitHub from a sandbox?")],
            ),
        )
    )

    assert [event.type for event in events[:5]] == [
        EventType.SESSION_STARTED,
        EventType.KNOWLEDGE_SEARCH_STARTED,
        EventType.KNOWLEDGE_SEARCH_COMPLETED,
        EventType.KNOWLEDGE_INJECTED,
        EventType.MODEL_STARTED,
    ]
    provider_context = provider.requests[0].messages
    assert [message.role for message in provider_context] == ["user", "assistant", "tool"]
    assert provider_context[0].content[0].text == "How should I push to GitHub from a sandbox?"
    injected_call = provider_context[1].content[0]
    assert injected_call.tool_name == "cayu_knowledge_retrieval"
    assert injected_call.tool_call_id == "cayu_knowledge_step_1"
    injected_result = provider_context[2].content[0]
    assert injected_result.tool_call_id == injected_call.tool_call_id
    injected_text = injected_result.content
    assert "entry_id='git_policy'" in injected_text
    assert "brokered Git HTTP proxy" in injected_text
    assert "untrusted reference data" in injected_text
    assert "<untrusted_knowledge>" in injected_text
    assert "</untrusted_knowledge>" in injected_text

    injected_event = next(event for event in events if event.type == EventType.KNOWLEDGE_INJECTED)
    assert injected_event.payload["hit_count"] == 1
    assert injected_event.payload["tool_call_id"] == "cayu_knowledge_step_1"
    source = injected_event.payload["sources"][0]
    assert source["entry_id"] == "git_policy"
    assert source["namespace"] == "project:cayu"
    assert source["kind"] == "procedure"
    assert source["title"] == "Remote sandbox Git credential boundary"
    assert source["chunk_id"] == "git_policy:0"
    assert source["chunk_index"] == 0
    assert source["score"] > 0
    assert source["score_kind"] == "inmemory_keyword"
    assert source["reason"] == "chunk text match"
    assert "brokered Git HTTP proxy" not in str(injected_event.payload)
    search_started = next(
        event for event in events if event.type == EventType.KNOWLEDGE_SEARCH_STARTED
    )
    assert search_started.payload["query"] == {
        "namespace": "project:cayu",
        "labels": {"project": "cayu"},
        "kinds": None,
        "visibilities": None,
        "aspects": [],
        "impact_targets": [],
        "source_type": None,
        "source_id": None,
        "mode": "auto",
        "include_expired": False,
        "limit": 1,
        "max_bytes": 800,
    }

    transcript = asyncio.run(store.load_transcript("sess_knowledge_injection"))
    assert [message.content[0].text for message in transcript] == [
        "How should I push to GitHub from a sandbox?",
        "use the git proxy",
    ]


def test_cayu_app_knowledge_injection_can_be_disabled() -> None:
    knowledge_store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id="git_policy",
                text="GitHub push should use a brokered proxy.",
            )
        ]
    )
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("plain answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            knowledge_store=knowledge_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=KnowledgeInjectionPolicy(enabled=False),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_knowledge_injection_disabled",
                messages=[Message.text("user", "GitHub push?")],
            ),
        )
    )

    assert not any(str(event.type).startswith("knowledge.") for event in events)
    assert [message.content[0].text for message in provider.requests[0].messages] == [
        "GitHub push?"
    ]


def test_cayu_app_knowledge_injection_caps_inserted_context_bytes() -> None:
    knowledge_store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id="large_policy",
                text="alpha " + ("large knowledge body " * 80),
            )
        ]
    )
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("bounded"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            knowledge_store=knowledge_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=KnowledgeInjectionPolicy(max_hits=1, max_bytes=220),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_knowledge_injection_cap",
                messages=[Message.text("user", "alpha")],
            ),
        )
    )

    injected_text = provider.requests[0].messages[2].content[0].content
    assert len(injected_text.encode("utf-8")) <= 220
    assert "[knowledge context truncated]" in injected_text
    injected_event = next(event for event in events if event.type == EventType.KNOWLEDGE_INJECTED)
    assert injected_event.payload["injected_bytes"] <= 220


def test_cayu_app_knowledge_injection_search_failure_can_opt_into_fail_open() -> None:
    class FailingKnowledgeStore(InMemoryKnowledgeStore):
        async def search(self, query):
            raise RuntimeError("knowledge search unavailable")

    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("answered without injected knowledge"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            knowledge_store=FailingKnowledgeStore(
                [KnowledgeEntry(id="git_policy", text="GitHub push uses a proxy.")]
            ),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=KnowledgeInjectionPolicy(fail_open=True),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_knowledge_injection_fail_open",
                messages=[Message.text("user", "GitHub push?")],
            ),
        )
    )

    assert [event.type for event in events[:4]] == [
        EventType.SESSION_STARTED,
        EventType.KNOWLEDGE_SEARCH_STARTED,
        EventType.KNOWLEDGE_SEARCH_FAILED,
        EventType.MODEL_STARTED,
    ]
    failed_event = next(
        event for event in events if event.type == EventType.KNOWLEDGE_SEARCH_FAILED
    )
    assert failed_event.payload["error"] == "knowledge search unavailable"
    assert [message.content[0].text for message in provider.requests[0].messages] == [
        "GitHub push?"
    ]


def test_cayu_app_knowledge_injection_fails_closed_by_default_and_emits_failure_first() -> None:
    class FailingKnowledgeStore(InMemoryKnowledgeStore):
        async def search(self, query):
            raise RuntimeError("knowledge search unavailable")

    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            knowledge_store=FailingKnowledgeStore(
                [KnowledgeEntry(id="git_policy", text="GitHub push uses a proxy.")]
            ),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        # No `fail_open` argument: search failures must fail closed by default.
        context_policy=KnowledgeInjectionPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_knowledge_injection_fail_closed",
                messages=[Message.text("user", "GitHub push?")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.KNOWLEDGE_SEARCH_STARTED,
        EventType.KNOWLEDGE_SEARCH_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    failed_event = next(
        event for event in events if event.type == EventType.KNOWLEDGE_SEARCH_FAILED
    )
    assert failed_event.payload["error"] == "knowledge search unavailable"
    assert events[-1].payload == {
        "error": "knowledge search unavailable",
        "error_type": "RuntimeError",
    }
    assert provider.requests == []


def test_knowledge_injection_fail_closed_preserves_completed_compaction_checkpoint() -> None:
    class FailingKnowledgeStore(InMemoryKnowledgeStore):
        async def search(self, query):
            raise RuntimeError("knowledge search unavailable")

    store = InMemorySessionStore()
    compactor = RecordingCompactor()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            knowledge_store=FailingKnowledgeStore(
                [KnowledgeEntry(id="current_policy", text="Current deployment policy.")]
            ),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=KnowledgeInjectionPolicy(
            CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
                compact_after_messages=2,
            ),
            fail_open=False,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_knowledge_fail_closed_after_compaction",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current deployment"),
                ],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.KNOWLEDGE_SEARCH_STARTED,
        EventType.KNOWLEDGE_SEARCH_FAILED,
        EventType.SESSION_CHECKPOINTED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[5].payload == {
        "checkpoint": "context_compaction",
        "compacted_transcript_cursor": 2,
        "previous_compacted_transcript_cursor": 0,
        "newly_compacted_message_count": 2,
        "recent_message_count": 1,
    }
    checkpoint = asyncio.run(store.load_checkpoint("sess_knowledge_fail_closed_after_compaction"))
    assert checkpoint == {
        "context_compaction": {
            "version": 1,
            "summary": "old|old answer",
            "compacted_transcript_cursor": 2,
            "metadata": {"request_count": 1},
        }
    }
    assert provider.requests == []


def test_knowledge_injection_policy_composes_with_checkpoint_compaction() -> None:
    store = InMemorySessionStore()
    compactor = RecordingCompactor()
    knowledge_store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id="current_policy",
                text="Current deployment requests should check the rollout policy.",
            )
        ]
    )
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            knowledge_store=knowledge_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=KnowledgeInjectionPolicy(
            CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
                compact_after_messages=2,
            ),
            max_hits=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_knowledge_injection_compaction",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current deployment"),
                ],
            ),
        )
    )

    assert [event.type for event in events[:7]] == [
        EventType.SESSION_STARTED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.KNOWLEDGE_SEARCH_STARTED,
        EventType.KNOWLEDGE_SEARCH_COMPLETED,
        EventType.KNOWLEDGE_INJECTED,
        EventType.SESSION_CHECKPOINTED,
    ]
    provider_context = provider.requests[0].messages
    assert [message.role for message in provider_context] == ["user", "user", "assistant", "tool"]
    assert (
        provider_context[0].content[0].text == "Previous session context summary:\nold|old answer"
    )
    assert provider_context[1].content[0].text == "current deployment"
    assert provider_context[2].content[0].tool_name == "cayu_knowledge_retrieval"
    assert "entry_id='current_policy'" in provider_context[3].content[0].content

    checkpoint = asyncio.run(store.load_checkpoint("sess_knowledge_injection_compaction"))
    assert checkpoint == {
        "context_compaction": {
            "version": 1,
            "summary": "old|old answer",
            "compacted_transcript_cursor": 2,
            "metadata": {"request_count": 1},
        }
    }


def test_cayu_app_redacts_tool_results_before_events_transcript_and_context() -> None:
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret_value = "sk-runtime-secret-value"

    class LeakyTool(Tool):
        spec = ToolSpec(
            name="leak",
            description="Return a result containing a known secret.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(
                content=f"connected with {secret_value}",
                structured={
                    "token": secret_value,
                    "nested": ["safe", f"prefix-{secret_value}-suffix"],
                },
                artifacts=[
                    {
                        "type": "debug",
                        "metadata": {"authorization": f"Bearer {secret_value}"},
                    }
                ],
            )

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_leak", name="leak", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[LeakyTool()])

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_result_redaction",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    tool_event = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    result_payload = tool_event.payload["result"]
    assert secret_value not in str(result_payload)
    assert REDACTED_SECRET in result_payload["content"]
    assert result_payload["structured"]["token"] == REDACTED_SECRET
    assert result_payload["structured"]["nested"][1] == f"prefix-{REDACTED_SECRET}-suffix"
    assert result_payload["artifacts"][0]["metadata"]["authorization"] == (
        f"Bearer {REDACTED_SECRET}"
    )

    transcript = asyncio.run(store.load_transcript("sess_tool_result_redaction"))
    tool_message = next(message for message in transcript if message.role == "tool")
    tool_part = tool_message.content[0]
    assert isinstance(tool_part, ToolResultPart)
    assert secret_value not in str(tool_part.model_dump(mode="json"))
    assert tool_part.content == f"connected with {REDACTED_SECRET}"
    assert tool_part.structured is not None
    assert tool_part.structured["token"] == REDACTED_SECRET
    assert tool_part.artifacts[0]["metadata"]["authorization"] == f"Bearer {REDACTED_SECRET}"

    second_request = provider.requests[1]
    assert secret_value not in str(
        [message.model_dump(mode="json") for message in second_request.messages]
    )


def test_cayu_app_redacts_proxy_resolved_secrets_from_tool_results() -> None:
    from cayu.vaults import REDACTED_SECRET

    secret_value = "sk-proxy-resolved-secret"

    class ProxyLeakingTool(Tool):
        spec = ToolSpec(
            name="proxy_leak",
            description="Resolve a proxy secret and accidentally return it.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.proxy is not None
            resolved = await ctx.proxy.resolve(
                SecretRef(name="api_key"),
                scope={"session_id": ctx.session_id},
            )
            raw_secret = resolved.value.get_secret_value()
            return ToolResult(
                content=f"connected with {raw_secret}",
                structured={
                    "token": raw_secret,
                    "nested": ["safe", f"prefix-{raw_secret}-suffix"],
                },
                artifacts=[
                    {
                        "type": "debug",
                        "metadata": {"authorization": f"Bearer {raw_secret}"},
                    }
                ],
            )

    store = InMemorySessionStore()
    vault = StaticVault({"api_key": secret_value})
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_proxy_leak", name="proxy_leak", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=vault,
            proxy=PassthroughProxy(vault),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ProxyLeakingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_secret_redaction",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    tool_event = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    result_payload = tool_event.payload["result"]
    assert secret_value not in str(result_payload)
    assert result_payload["content"] == f"connected with {REDACTED_SECRET}"
    assert result_payload["structured"]["token"] == REDACTED_SECRET
    assert result_payload["structured"]["nested"][1] == f"prefix-{REDACTED_SECRET}-suffix"
    assert result_payload["artifacts"][0]["metadata"]["authorization"] == (
        f"Bearer {REDACTED_SECRET}"
    )

    transcript = asyncio.run(store.load_transcript("sess_proxy_secret_redaction"))
    tool_message = next(message for message in transcript if message.role == "tool")
    tool_part = tool_message.content[0]
    assert isinstance(tool_part, ToolResultPart)
    assert secret_value not in str(tool_part.model_dump(mode="json"))
    assert tool_part.content == f"connected with {REDACTED_SECRET}"
    assert tool_part.structured is not None
    assert tool_part.structured["token"] == REDACTED_SECRET
    assert tool_part.artifacts[0]["metadata"]["authorization"] == f"Bearer {REDACTED_SECRET}"

    second_request = provider.requests[1]
    assert secret_value not in str(
        [message.model_dump(mode="json") for message in second_request.messages]
    )


def test_cayu_app_emits_redacted_proxy_authorization_events() -> None:
    from cayu.vaults import REDACTED_SECRET

    secret_value = "sk-authorized-proxy-secret"

    class AuditingProxy(CredentialProxy):
        def __init__(self, vault: StaticVault) -> None:
            self._passthrough = PassthroughProxy(vault)

        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict[str, Any] | None = None,
        ) -> ResolvedSecret:
            return await self._passthrough.resolve(ref, scope=scope)

        async def authorize_request(
            self,
            *,
            destination: str,
            credential: SecretRef | None = None,
            action: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> ProxyAuthorizationResult:
            if action == "deny_email":
                return ProxyAuthorizationResult(
                    allowed=False,
                    reason="destination denied",
                    metadata={"policy": "blocked", "debug": secret_value},
                )
            return ProxyAuthorizationResult(
                allowed=True,
                metadata={"policy": "allowlist", "debug": secret_value},
            )

    class ProxyAuthorizingTool(Tool):
        spec = ToolSpec(
            name="proxy_auth",
            description="Resolve and authorize proxy use.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.proxy is not None
            await ctx.proxy.resolve(SecretRef(name="sendgrid_api_key"))
            allowed = await ctx.proxy.authorize_request(
                destination="https://api.sendgrid.com/v3/mail/send",
                credential=SecretRef(name="sendgrid_api_key"),
                action="send_email",
                metadata={"template": "invoice_reminder"},
            )
            denied = await ctx.proxy.authorize_request(
                destination="https://api.sendgrid.com/v3/mail/send",
                credential=SecretRef(name="sendgrid_api_key"),
                action="deny_email",
                metadata={"template": "invoice_reminder"},
            )
            return ToolResult(
                content="checked",
                structured={
                    "allowed": allowed.allowed,
                    "denied": denied.allowed,
                },
            )

    store = InMemorySessionStore()
    vault = StaticVault({"sendgrid_api_key": secret_value})
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_proxy_auth", name="proxy_auth", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=vault,
            proxy=AuditingProxy(vault),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ProxyAuthorizingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_authorization_events",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    proxy_events = [event for event in events if event.type == EventType.CREDENTIAL_PROXY_CHECKED]
    assert len(proxy_events) == 2
    assert [event.type for event in events].index(EventType.CREDENTIAL_PROXY_CHECKED) < [
        event.type for event in events
    ].index(EventType.TOOL_CALL_COMPLETED)

    allowed_event = proxy_events[0]
    assert allowed_event.tool_name == "proxy_auth"
    assert allowed_event.payload["tool_call_id"] == "call_proxy_auth"
    assert allowed_event.payload["destination"] == "https://api.sendgrid.com/v3/mail/send"
    assert allowed_event.payload["credential"] == "sendgrid_api_key"
    assert allowed_event.payload["action"] == "send_email"
    assert allowed_event.payload["metadata"] == {"template": "invoice_reminder"}
    assert allowed_event.payload["allowed"] is True
    assert allowed_event.payload["reason"] is None
    assert allowed_event.payload["result_metadata"] == {
        "policy": "allowlist",
        "debug": REDACTED_SECRET,
    }

    denied_event = proxy_events[1]
    assert denied_event.payload["allowed"] is False
    assert denied_event.payload["reason"] == "destination denied"
    assert denied_event.payload["result_metadata"] == {
        "policy": "blocked",
        "debug": REDACTED_SECRET,
    }
    assert secret_value not in str([event.model_dump(mode="json") for event in proxy_events])


def test_cayu_app_records_original_proxy_authorization_metadata() -> None:
    class MutatingProxy(CredentialProxy):
        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict[str, Any] | None = None,
        ) -> ResolvedSecret:
            raise AssertionError("resolve should not be called")

        async def authorize_request(
            self,
            *,
            destination: str,
            credential: SecretRef | None = None,
            action: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> ProxyAuthorizationResult:
            if metadata is None:
                raise AssertionError("metadata should be copied before proxy delegation")
            metadata["template"] = "mutated"
            metadata["new_field"] = "proxy-added"
            if credential is not None:
                credential.metadata["scope"] = "mutated"
            return ProxyAuthorizationResult(allowed=True)

    class ProxyAuthorizingTool(Tool):
        spec = ToolSpec(
            name="proxy_auth",
            description="Authorize proxy use.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.proxy is not None
            await ctx.proxy.authorize_request(
                destination="https://api.sendgrid.com/v3/mail/send",
                credential=SecretRef(
                    name="sendgrid_api_key",
                    metadata={"scope": "email"},
                ),
                action="send_email",
                metadata={"template": "invoice_reminder"},
            )
            return ToolResult(content="checked")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_proxy_auth", name="proxy_auth", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            proxy=MutatingProxy(),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ProxyAuthorizingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_authorization_metadata_copy",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    proxy_events = [event for event in events if event.type == EventType.CREDENTIAL_PROXY_CHECKED]
    assert len(proxy_events) == 1
    assert proxy_events[0].payload["metadata"] == {"template": "invoice_reminder"}
    assert proxy_events[0].payload["credential"] == "sendgrid_api_key"


def test_cayu_app_redacts_proxy_secret_after_tool_mutates_resolved_secret() -> None:
    from pydantic import SecretStr

    from cayu.vaults import REDACTED_SECRET

    secret_value = "sk-mutated-proxy-secret"

    class MutatingSecretTool(Tool):
        spec = ToolSpec(
            name="mutate_secret",
            description="Mutate returned proxy secret and leak original value.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.proxy is not None
            secret = await ctx.proxy.resolve(SecretRef(name="api_key"))
            raw_value = secret.value.get_secret_value()
            secret.value = SecretStr("changed-after-resolve")
            secret.metadata["leak"] = "mutated"
            return ToolResult(
                content=f"using {raw_value}",
                structured={"token": raw_value},
                artifacts=[
                    {
                        "type": "test.secret",
                        "value": raw_value,
                    }
                ],
            )

    vault = StaticVault({"api_key": secret_value})
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_mutate_secret",
                    name="mutate_secret",
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
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=vault,
            proxy=PassthroughProxy(vault),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[MutatingSecretTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_resolved_secret_copy",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    tool_events = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    assert len(tool_events) == 1
    result = tool_events[0].payload["result"]
    assert result["content"] == f"using {REDACTED_SECRET}"
    assert result["structured"] == {"token": REDACTED_SECRET}
    assert result["artifacts"] == [{"type": "test.secret", "value": REDACTED_SECRET}]
    assert secret_value not in str([event.model_dump(mode="json") for event in events])


def test_cayu_app_copies_proxy_resolve_inputs_before_delegation() -> None:
    from pydantic import SecretStr

    class MutatingResolveProxy(CredentialProxy):
        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict[str, Any] | None = None,
        ) -> ResolvedSecret:
            ref.metadata["scope"] = "proxy-mutated"
            if scope is not None:
                scope["tenant"] = "proxy-mutated"
            return ResolvedSecret(
                name=ref.name,
                value=SecretStr("resolved-secret"),
                metadata={"source": "mutating-proxy"},
            )

        async def authorize_request(
            self,
            *,
            destination: str,
            credential: SecretRef | None = None,
            action: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> ProxyAuthorizationResult:
            raise AssertionError("authorize_request should not be called")

    class ProxyResolvingTool(Tool):
        spec = ToolSpec(
            name="resolve_proxy",
            description="Resolve proxy secret with mutable inputs.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.proxy is not None
            ref = SecretRef(name="api_key", metadata={"scope": "email"})
            scope = {"tenant": "acme"}
            await ctx.proxy.resolve(ref, scope=scope)
            return ToolResult(
                content="checked",
                structured={
                    "ref_metadata": ref.metadata,
                    "scope": scope,
                },
            )

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_resolve_proxy",
                    name="resolve_proxy",
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
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            proxy=MutatingResolveProxy(),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ProxyResolvingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_resolve_input_copy",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    tool_events = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    assert len(tool_events) == 1
    assert tool_events[0].payload["result"]["structured"] == {
        "ref_metadata": {"scope": "email"},
        "scope": {"tenant": "acme"},
    }


def test_cayu_app_rejects_invalid_proxy_resolve_scope_before_delegation() -> None:
    class FailingProxy(CredentialProxy):
        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict[str, Any] | None = None,
        ) -> ResolvedSecret:
            raise AssertionError("invalid scope should fail before proxy delegation")

        async def authorize_request(
            self,
            *,
            destination: str,
            credential: SecretRef | None = None,
            action: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> ProxyAuthorizationResult:
            raise AssertionError("authorize_request should not be called")

    class InvalidScopeTool(Tool):
        spec = ToolSpec(
            name="invalid_scope",
            description="Resolve proxy secret with invalid scope.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.proxy is not None
            await ctx.proxy.resolve(
                SecretRef(name="api_key"),
                scope=["not", "an", "object"],  # type: ignore[arg-type]
            )
            return ToolResult(content="unexpected")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_invalid_scope",
                    name="invalid_scope",
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
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            proxy=FailingProxy(),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[InvalidScopeTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_invalid_scope",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    tool_events = [event for event in events if event.type == EventType.TOOL_CALL_FAILED]
    assert len(tool_events) == 1
    assert "`scope` must be a JSON object" in tool_events[0].payload["result"]["content"]


def test_cayu_app_rejects_invalid_proxy_authorization_metadata_before_delegation() -> None:
    class FailingProxy(CredentialProxy):
        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict[str, Any] | None = None,
        ) -> ResolvedSecret:
            raise AssertionError("resolve should not be called")

        async def authorize_request(
            self,
            *,
            destination: str,
            credential: SecretRef | None = None,
            action: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> ProxyAuthorizationResult:
            raise AssertionError("invalid metadata should fail before proxy delegation")

    class InvalidMetadataTool(Tool):
        spec = ToolSpec(
            name="invalid_metadata",
            description="Authorize proxy use with invalid metadata.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.proxy is not None
            await ctx.proxy.authorize_request(
                destination="https://api.sendgrid.com/v3/mail/send",
                metadata=["not", "an", "object"],  # type: ignore[arg-type]
            )
            return ToolResult(content="unexpected")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_invalid_metadata",
                    name="invalid_metadata",
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
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            proxy=FailingProxy(),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[InvalidMetadataTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_invalid_metadata",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    tool_events = [event for event in events if event.type == EventType.TOOL_CALL_FAILED]
    assert len(tool_events) == 1
    assert "`metadata` must be a JSON object" in tool_events[0].payload["result"]["content"]
    assert EventType.CREDENTIAL_PROXY_CHECKED not in [event.type for event in events]


def test_cayu_app_redacts_blocked_tool_result_event_payload() -> None:
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret_value = "policy-secret-value"

    class SecretPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason=f"blocked because {secret_value}",
                metadata={"token": secret_value},
            )

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_echo",
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
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        tool_policy=SecretPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_blocked_tool_result_redaction",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    blocked_event = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert secret_value not in str(blocked_event.payload)
    assert blocked_event.payload["reason"] == f"blocked because {REDACTED_SECRET}"
    assert blocked_event.payload["metadata"]["token"] == REDACTED_SECRET
    assert blocked_event.payload["result"]["content"] == f"blocked because {REDACTED_SECRET}"
    assert blocked_event.payload["result"]["structured"]["metadata"]["token"] == REDACTED_SECRET

    transcript = asyncio.run(store.load_transcript("sess_blocked_tool_result_redaction"))
    tool_message = next(message for message in transcript if message.role == "tool")
    assert isinstance(tool_message.content[0], ToolResultPart)
    assert secret_value not in str(tool_message.model_dump(mode="json"))


def fake_budget_limit(
    max_estimated_cost: str,
    *,
    scope: Literal["session", "run"] = "session",
    allow_unpriced: bool = False,
    window: BudgetWindow | str | None = None,
    action: Literal["interrupt", "notify"] = "interrupt",
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
        action=action,
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


async def collect_user_input_events(
    app: CayuApp,
    response: UserInputResponse,
) -> list[Event]:
    return [event async for event in app.resolve_user_input(response)]


async def collect_tool_approval_recovery_events(
    app: CayuApp,
    request: ToolApprovalRecoveryRequest,
) -> list[Event]:
    return [event async for event in app.recover_tool_approval(request)]


async def collect_tool_round_recovery_events(
    app: CayuApp,
    request: ToolRoundRecoveryRequest,
) -> list[Event]:
    return [event async for event in app.recover_tool_round(request)]


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

    with pytest.raises(TypeError, match="secret_redactor"):
        CayuApp(secret_redactor="not a redactor")  # type: ignore[arg-type]

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


def test_cayu_app_environment_factory_creates_environment_for_session(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        workspace = LocalWorkspace(workspace_root, workspace_id="factory-workspace")
        factory = RecordingEnvironmentFactory(
            Environment(EnvironmentSpec(name="dynamic"), workspace=workspace),
            metadata={"sandbox_id": "sbx_123"},
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="workspace_id",
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
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WorkspaceIdTool()],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory",
                labels={"project": "alpha"},
                metadata={"owner": "test"},
                messages=[Message.text("user", "run tool")],
            ),
        )
        session = await store.load("sess_factory")
        return events, session, factory, workspace

    events, session, factory, workspace = asyncio.run(run())

    assert session is not None
    assert session.environment_name == "dynamic"
    assert [event.type for event in events[:3]] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.SESSION_STARTED,
    ]
    assert events[0].payload == {
        "factory_type": "RecordingEnvironmentFactory",
        "requested_environment_name": "dynamic",
        "parent_session_id": None,
        "causal_budget_id": "sess_factory",
        "labels": {"project": "alpha"},
    }
    assert events[1].payload == {
        "factory_type": "RecordingEnvironmentFactory",
        "requested_environment_name": "dynamic",
        "parent_session_id": None,
        "causal_budget_id": "sess_factory",
        "labels": {"project": "alpha"},
        "environment_name": "dynamic",
        "result_metadata": {"sandbox_id": "sbx_123"},
        "reconnect_metadata": {},
    }
    assert len(factory.requests) == 1
    assert factory.requests[0].session_id == "sess_factory"
    assert factory.requests[0].agent_name == "assistant"
    assert factory.requests[0].environment_name == "dynamic"
    assert factory.requests[0].labels == {"project": "alpha"}
    assert factory.requests[0].metadata == {"owner": "test"}

    tool_events = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    assert tool_events[0].payload["result"]["structured"] == {"workspace_id": workspace.id}


def test_cayu_app_environment_factory_failure_fails_session_before_start_event(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
            ),
            fail_create=True,
        )
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("unreached"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_fail",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_factory_fail")
        return events, session, factory, provider

    events, session, factory, provider = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.SESSION_FAILED,
    ]
    assert events[1].payload["error"] == "factory failed"
    assert events[1].payload["error_type"] == "RuntimeError"
    assert events[2].payload["error"] == "factory failed"
    assert events[2].payload["error_type"] == "RuntimeError"
    assert len(factory.requests) == 1
    assert provider.requests == []


def test_cayu_app_environment_factory_failure_runs_failed_session_hooks(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
            ),
            fail_create=True,
        )
        hook = RecordingSessionFailedHook()
        app = CayuApp(
            session_store=store,
            runtime_hooks=[hook],
            enable_logging=False,
        )
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_fail_hook",
                messages=[Message.text("user", "run")],
            ),
        )
        return events, hook

    events, hook = asyncio.run(run())

    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.SESSION_FAILED,
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
    ]
    assert len(hook.contexts) == 1
    assert hook.contexts[0].session.status == SessionStatus.FAILED
    assert hook.contexts[0].terminal_event.type == EventType.SESSION_FAILED
    assert hook.contexts[0].terminal_event.payload["error"] == "factory failed"


def test_cayu_app_environment_factory_workspace_instruction_failure_fails_session(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        (workspace_root / "AGENTS.md").write_text("too long")
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
                workspace_instructions=WorkspaceInstructionsConfig(
                    paths=("AGENTS.md",),
                    max_bytes=3,
                ),
            )
        )
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("unreached"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_instruction_fail",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_factory_instruction_fail")
        return events, session, provider

    events, session, provider = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert "exceeds 3 bytes" in events[2].payload["error"]
    assert provider.requests == []


def test_cayu_app_environment_factory_workspace_instruction_failure_runs_hooks(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        (workspace_root / "AGENTS.md").write_text("too long")
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
                workspace_instructions=WorkspaceInstructionsConfig(
                    paths=("AGENTS.md",),
                    max_bytes=3,
                ),
            )
        )
        hook = RecordingSessionFailedHook()
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("unreached"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(
            session_store=store,
            runtime_hooks=[hook],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_instruction_fail_hook",
                messages=[Message.text("user", "run")],
            ),
        )
        return events, hook, provider

    events, hook, provider = asyncio.run(run())

    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
    ]
    assert len(hook.contexts) == 1
    assert hook.contexts[0].session.status == SessionStatus.FAILED
    assert hook.contexts[0].terminal_event.type == EventType.SESSION_FAILED
    assert "exceeds 3 bytes" in hook.contexts[0].terminal_event.payload["error"]
    assert provider.requests == []


def test_cayu_app_environment_factory_result_name_must_match_registration(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="different"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
            )
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_name_mismatch",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_factory_name_mismatch")
        return events, session

    events, session = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.SESSION_FAILED,
    ]
    assert "different environment name" in events[1].payload["error"]


def test_cayu_app_environment_factory_output_composes_with_workspace_binding(tmp_path):
    async def run():
        store = InMemorySessionStore()
        configured_root = tmp_path / "configured"
        configured_root.mkdir()
        bound_root = tmp_path / "bound"
        bound_root.mkdir()
        configured_workspace = LocalWorkspace(configured_root, workspace_id="configured")
        bound_workspace = LocalWorkspace(bound_root, workspace_id="bound")
        binding = RecordingWorkspaceBinding(bound_workspace=bound_workspace)
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=configured_workspace,
                binding=binding,
            )
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="workspace_id",
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
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WorkspaceIdTool()],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_binding",
                messages=[Message.text("user", "run")],
            ),
        )
        return events, binding, configured_workspace, bound_workspace

    events, binding, configured_workspace, bound_workspace = asyncio.run(run())

    assert [event.type for event in events[:5]] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.ENVIRONMENT_BINDING_STARTED,
        EventType.ENVIRONMENT_BINDING_COMPLETED,
        EventType.SESSION_STARTED,
    ]
    assert binding.bind_calls[0]["workspace"] is configured_workspace
    assert binding.bind_calls[0]["environment_name"] == "dynamic"
    tool_events = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    assert tool_events[0].payload["result"]["structured"] == {"workspace_id": bound_workspace.id}


def test_cayu_app_resume_uses_stored_factory_backed_environment(tmp_path):
    async def run():
        store = InMemorySessionStore()
        dynamic_root = tmp_path / "dynamic"
        dynamic_root.mkdir()
        static_root = tmp_path / "static"
        static_root.mkdir()
        dynamic_workspace = LocalWorkspace(dynamic_root, workspace_id="dynamic-workspace")
        static_workspace = LocalWorkspace(static_root, workspace_id="static-workspace")
        factory = RecordingEnvironmentFactory(
            Environment(EnvironmentSpec(name="dynamic"), workspace=dynamic_workspace)
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="workspace_id",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="call_2",
                        name="workspace_id",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done again"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_environment(
            Environment(EnvironmentSpec(name="static"), workspace=static_workspace),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WorkspaceIdTool()],
        )

        run_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_resume",
                environment_name="dynamic",
                messages=[Message.text("user", "run tool")],
            ),
        )
        resume_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="sess_factory_resume",
                    messages=[Message.text("user", "run again")],
                )
            )
        ]
        return run_events, resume_events, factory, dynamic_workspace

    run_events, resume_events, factory, dynamic_workspace = asyncio.run(run())

    assert len(factory.requests) == 2
    assert [request.environment_name for request in factory.requests] == [
        "dynamic",
        "dynamic",
    ]
    assert run_events[0].type == EventType.ENVIRONMENT_FACTORY_STARTED
    assert resume_events[0].type == EventType.ENVIRONMENT_FACTORY_STARTED
    tool_events = [
        event
        for event in [*run_events, *resume_events]
        if event.type == EventType.TOOL_CALL_COMPLETED
    ]
    assert [event.payload["result"]["structured"]["workspace_id"] for event in tool_events] == [
        dynamic_workspace.id,
        dynamic_workspace.id,
    ]


def test_cayu_app_environment_factory_reconnect_metadata_round_trips_on_resume(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "dynamic"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="dynamic-workspace"),
            ),
            reconnect_metadata={"sandbox_id": "sbx_1"},
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("done again"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        run_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_reconnect",
                messages=[Message.text("user", "run")],
            ),
        )
        factory.reconnect_metadata = {"sandbox_id": "sbx_1", "generation": 2}
        resume_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="sess_factory_reconnect",
                    messages=[Message.text("user", "resume")],
                )
            )
        ]
        checkpoint = await store.load_checkpoint("sess_factory_reconnect")
        return run_events, resume_events, factory, checkpoint

    run_events, resume_events, factory, checkpoint = asyncio.run(run())

    completed_events = [
        event
        for event in [*run_events, *resume_events]
        if event.type == EventType.ENVIRONMENT_FACTORY_COMPLETED
    ]
    assert [request.reconnect_metadata for request in factory.requests] == [
        {},
        {"sandbox_id": "sbx_1"},
    ]
    assert [event.payload["reconnect_metadata"] for event in completed_events] == [
        {"sandbox_id": "sbx_1"},
        {"sandbox_id": "sbx_1", "generation": 2},
    ]
    assert checkpoint is not None
    assert checkpoint["environment_factory_reconnect"] == {
        "dynamic": {"sandbox_id": "sbx_1", "generation": 2}
    }


def test_cayu_app_environment_factory_reconnect_metadata_survives_context_checkpoint(
    tmp_path,
):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "dynamic"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="dynamic-workspace"),
            ),
            reconnect_metadata={"sandbox_id": "sbx_compaction"},
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("done again"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="You are careful.",
            ),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )

        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_reconnect_compaction",
                messages=[
                    Message.text("user", "old one"),
                    Message.text("assistant", "old answer one"),
                    Message.text("user", "current"),
                ],
            ),
        )
        first_checkpoint = await store.load_checkpoint("sess_factory_reconnect_compaction")
        _ = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="sess_factory_reconnect_compaction",
                    messages=[Message.text("user", "resume")],
                )
            )
        ]
        return factory, first_checkpoint

    factory, first_checkpoint = asyncio.run(run())

    assert first_checkpoint is not None
    assert first_checkpoint["environment_factory_reconnect"] == {
        "dynamic": {"sandbox_id": "sbx_compaction"}
    }
    assert "context_compaction" in first_checkpoint
    assert [request.reconnect_metadata for request in factory.requests] == [
        {},
        {"sandbox_id": "sbx_compaction"},
    ]


def test_cayu_app_binds_environment_for_session_tools_and_finalize(tmp_path):
    async def run():
        store = InMemorySessionStore()
        configured_root = tmp_path / "configured"
        configured_root.mkdir()
        bound_root = tmp_path / "bound"
        bound_root.mkdir()
        configured_workspace = LocalWorkspace(configured_root, workspace_id="configured")
        bound_workspace = LocalWorkspace(bound_root, workspace_id="bound")
        binding = RecordingWorkspaceBinding(bound_workspace=bound_workspace)
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="workspace_id",
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
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=configured_workspace,
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WorkspaceIdTool()],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding",
                messages=[Message.text("user", "run tool")],
            ),
        )
        return events, binding, configured_workspace, bound_workspace

    events, binding, configured_workspace, bound_workspace = asyncio.run(run())

    assert [event.type for event in events[:3]] == [
        EventType.ENVIRONMENT_BINDING_STARTED,
        EventType.ENVIRONMENT_BINDING_COMPLETED,
        EventType.SESSION_STARTED,
    ]
    assert events[0].payload == {
        "binding_type": "RecordingWorkspaceBinding",
        "configured_workspace_id": configured_workspace.id,
        "has_configured_runner": False,
    }
    assert events[1].payload == {
        "binding_type": "RecordingWorkspaceBinding",
        "configured_workspace_id": configured_workspace.id,
        "has_configured_runner": False,
        "source_workspace_id": configured_workspace.id,
        "bound_workspace_id": bound_workspace.id,
        "bound_path": "/bound",
        "bound_metadata": {"binding": "recording"},
        "bound_snapshot": None,
        "has_bound_runner": False,
    }
    assert [event.type for event in events[-3:]] == [
        EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
        EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[-3].payload["configured_workspace_id"] == configured_workspace.id
    assert events[-3].payload["source_workspace_id"] == configured_workspace.id
    assert events[-3].payload["bound_workspace_id"] == bound_workspace.id
    assert events[-3].payload["outcome"] == "completed"
    assert events[-3].payload["bound_snapshot"] is None
    assert events[-2].payload["outcome"] == "completed"
    assert events[-2].payload["final_snapshot"] is None
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(binding.bind_calls) == 1
    assert binding.bind_calls[0]["workspace"] is configured_workspace
    assert binding.bind_calls[0]["session_id"] == "sess_binding"
    assert binding.bind_calls[0]["agent_name"] == "assistant"
    assert binding.bind_calls[0]["environment_name"] == "local"
    assert len(binding.finalize_calls) == 1
    assert binding.finalize_calls[0]["outcome"] == "completed"
    assert binding.finalize_calls[0]["bound"].workspace is bound_workspace

    tool_events = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    assert tool_events[0].payload["result"]["structured"] == {"workspace_id": bound_workspace.id}


def test_cayu_app_binding_events_include_workspace_snapshots(tmp_path):
    async def run():
        store = InMemorySessionStore()
        configured_root = tmp_path / "configured"
        configured_root.mkdir()
        bound_root = tmp_path / "bound"
        bound_root.mkdir()
        configured_workspace = LocalWorkspace(configured_root, workspace_id="configured")
        bound_workspace = LocalWorkspace(bound_root, workspace_id="bound")
        bind_snapshot = WorkspaceSnapshot(
            snapshot_id="snap_bind",
            workspace_id=bound_workspace.id,
            version="commit-a",
            source="git",
            metadata={"branch": "main"},
        )
        final_snapshot = WorkspaceSnapshot(
            snapshot_id="snap_final",
            workspace_id=bound_workspace.id,
            version="commit-b",
            source="git",
            metadata={"dirty": False},
        )
        binding = RecordingWorkspaceBinding(
            bound_workspace=bound_workspace,
            snapshot=bind_snapshot,
            final_snapshot=final_snapshot,
        )
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=configured_workspace,
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding_snapshot",
                messages=[Message.text("user", "run")],
            ),
        )
        return events, binding, bind_snapshot, final_snapshot

    events, binding, bind_snapshot, final_snapshot = asyncio.run(run())

    binding_completed = next(
        event for event in events if event.type == EventType.ENVIRONMENT_BINDING_COMPLETED
    )
    finalize_started = next(
        event for event in events if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED
    )
    finalize_completed = next(
        event for event in events if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
    )
    assert binding_completed.payload["bound_snapshot"] == {
        "snapshot_id": bind_snapshot.snapshot_id,
        "workspace_id": bind_snapshot.workspace_id,
        "version": bind_snapshot.version,
        "source": bind_snapshot.source,
        "metadata": bind_snapshot.metadata,
    }
    assert finalize_started.payload["bound_snapshot"] == binding_completed.payload["bound_snapshot"]
    assert finalize_completed.payload["final_snapshot"] == {
        "snapshot_id": final_snapshot.snapshot_id,
        "workspace_id": final_snapshot.workspace_id,
        "version": final_snapshot.version,
        "source": final_snapshot.source,
        "metadata": final_snapshot.metadata,
    }
    assert binding.finalize_calls[0]["bound"].snapshot == bind_snapshot


def test_cayu_app_binding_failure_fails_session_before_start_event():
    async def run():
        store = InMemorySessionStore()
        binding = RecordingWorkspaceBinding(fail_bind=True)
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("unreached"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), binding=binding),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding_fail",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_binding_fail")
        return events, session, binding, provider

    events, session, binding, provider = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_BINDING_STARTED,
        EventType.ENVIRONMENT_BINDING_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[1].payload["error"] == "bind failed"
    assert events[1].payload["error_type"] == "RuntimeError"
    assert events[3].payload["error"] == "bind failed"
    assert events[3].payload["error_type"] == "RuntimeError"
    assert len(binding.bind_calls) == 1
    assert binding.finalize_calls == []
    assert provider.requests == []


def test_cayu_app_binding_finalize_failure_is_reported_on_terminal_event():
    async def run():
        store = InMemorySessionStore()
        binding = RecordingWorkspaceBinding(fail_finalize=True)
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), binding=binding),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding_finalize_fail",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_binding_finalize_fail")
        return events, session, binding

    events, session, binding = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert [event.type for event in events[-3:]] == [
        EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
        EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[-2].payload["error"] == "finalize failed"
    assert events[-2].payload["error_type"] == "RuntimeError"
    assert events[-2].payload["outcome"] == "completed"
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(binding.finalize_calls) == 1
    assert events[-1].payload["binding_finalize_error"] == {
        "error": "finalize failed",
        "error_type": "RuntimeError",
        "outcome": "completed",
    }


def test_cayu_app_invalid_binding_finalize_snapshot_is_reported_on_terminal_event():
    async def run():
        store = InMemorySessionStore()
        binding = RecordingWorkspaceBinding()
        binding.final_snapshot = object()  # type: ignore[assignment]
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), binding=binding),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding_invalid_finalize_snapshot",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_binding_invalid_finalize_snapshot")
        return events, session

    events, session = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert events[-2].type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
    assert events[-2].payload["error"] == (
        "Workspace snapshot copies require a WorkspaceSnapshot or None."
    )
    assert events[-2].payload["error_type"] == "TypeError"
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert events[-1].payload["binding_finalize_error"] == {
        "error": "Workspace snapshot copies require a WorkspaceSnapshot or None.",
        "error_type": "TypeError",
        "outcome": "completed",
    }


def test_cayu_app_binds_environment_for_approved_tool_continuation(tmp_path):
    async def run():
        store = InMemorySessionStore()
        configured_root = tmp_path / "configured"
        configured_root.mkdir()
        bound_root = tmp_path / "bound"
        bound_root.mkdir()
        configured_workspace = LocalWorkspace(configured_root, workspace_id="configured")
        bound_workspace = LocalWorkspace(bound_root, workspace_id="bound")
        binding = RecordingWorkspaceBinding(bound_workspace=bound_workspace)
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="workspace_id",
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
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=configured_workspace,
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WorkspaceIdTool()],
            tool_policy=RequireApprovalPolicy(),
        )

        first_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding_approval",
                messages=[Message.text("user", "run tool")],
            ),
        )
        approval_event = next(
            event for event in first_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]
        approved_events = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="sess_binding_approval",
                    approval_id=approval_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        return first_events, approved_events, binding, bound_workspace

    first_events, approved_events, binding, bound_workspace = asyncio.run(run())

    assert first_events[0].type == EventType.ENVIRONMENT_BINDING_STARTED
    assert first_events[1].type == EventType.ENVIRONMENT_BINDING_COMPLETED
    assert first_events[-3].type == EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED
    assert first_events[-2].type == EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
    assert first_events[-1].type == EventType.SESSION_INTERRUPTED
    assert approved_events[0].type == EventType.SESSION_RESUMED
    assert approved_events[1].type == EventType.ENVIRONMENT_BINDING_STARTED
    assert approved_events[2].type == EventType.ENVIRONMENT_BINDING_COMPLETED
    assert approved_events[-3].type == EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED
    assert approved_events[-2].type == EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
    assert approved_events[-1].type == EventType.SESSION_COMPLETED
    assert [call["outcome"] for call in binding.finalize_calls] == [
        "interrupted",
        "completed",
    ]
    assert len(binding.bind_calls) == 2
    tool_events = [
        event for event in approved_events if event.type == EventType.TOOL_CALL_COMPLETED
    ]
    assert tool_events[0].payload["result"]["structured"] == {"workspace_id": bound_workspace.id}


def test_cayu_app_validates_tool_approval_retry_before_binding(tmp_path):
    async def run():
        store = InMemorySessionStore()
        configured_root = tmp_path / "configured"
        configured_root.mkdir()
        binding = RecordingWorkspaceBinding()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="workspace_id",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(configured_root, workspace_id="configured"),
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WorkspaceIdTool()],
            tool_policy=RequireApprovalPolicy(),
        )

        first_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding_invalid_approval_retry",
                messages=[Message.text("user", "run tool")],
            ),
        )
        approval_event = next(
            event for event in first_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]
        bind_calls_after_interrupt = len(binding.bind_calls)
        finalize_calls_after_interrupt = len(binding.finalize_calls)

        await store.append_event(
            "sess_binding_invalid_approval_retry",
            Event(
                type=EventType.TOOL_CALL_APPROVAL_DENIED,
                session_id="sess_binding_invalid_approval_retry",
                agent_name="assistant",
                environment_name="local",
                tool_name="workspace_id",
                payload={
                    "approval_id": approval_id,
                    "tool_call_id": "call_1",
                    "result": ToolResult(content="denied").model_dump(),
                },
            ),
        )

        retry_events = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="sess_binding_invalid_approval_retry",
                    approval_id=approval_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        session = await store.load("sess_binding_invalid_approval_retry")
        return (
            retry_events,
            session,
            binding,
            bind_calls_after_interrupt,
            finalize_calls_after_interrupt,
        )

    (
        retry_events,
        session,
        binding,
        bind_calls_after_interrupt,
        finalize_calls_after_interrupt,
    ) = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert [event.type for event in retry_events] == [EventType.SESSION_INTERRUPTED]
    assert "already denied" in retry_events[0].payload["error"]
    assert len(binding.bind_calls) == bind_calls_after_interrupt
    assert len(binding.finalize_calls) == finalize_calls_after_interrupt


def test_cayu_app_validates_tool_approval_retry_before_factory_resolution(tmp_path):
    async def run():
        store = InMemorySessionStore()
        configured_root = tmp_path / "configured"
        configured_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(configured_root, workspace_id="configured"),
            )
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="workspace_id",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WorkspaceIdTool()],
            tool_policy=RequireApprovalPolicy(),
        )

        first_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_invalid_approval_retry",
                messages=[Message.text("user", "run tool")],
            ),
        )
        approval_event = next(
            event for event in first_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]
        factory_calls_after_interrupt = len(factory.requests)

        await store.append_event(
            "sess_factory_invalid_approval_retry",
            Event(
                type=EventType.TOOL_CALL_APPROVAL_DENIED,
                session_id="sess_factory_invalid_approval_retry",
                agent_name="assistant",
                environment_name="dynamic",
                tool_name="workspace_id",
                payload={
                    "approval_id": approval_id,
                    "tool_call_id": "call_1",
                    "result": ToolResult(content="denied").model_dump(),
                },
            ),
        )

        retry_events = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="sess_factory_invalid_approval_retry",
                    approval_id=approval_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        session = await store.load("sess_factory_invalid_approval_retry")
        return retry_events, session, factory, factory_calls_after_interrupt

    retry_events, session, factory, factory_calls_after_interrupt = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert [event.type for event in retry_events] == [EventType.SESSION_INTERRUPTED]
    assert "already denied" in retry_events[0].payload["error"]
    assert len(factory.requests) == factory_calls_after_interrupt


def test_cayu_app_recovery_resolves_factory_before_resume_events(tmp_path):
    async def run():
        store = FailingTerminalToolEventStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
            ),
            reconnect_metadata={"sandbox_id": "sbx_recovery"},
        )
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
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        tool = SideEffectTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireApprovalPolicy(),
        )

        interrupt_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_recovery",
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event
            for event in interrupt_events
            if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]

        retry_events = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_factory_recovery",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        factory.reconnect_metadata = {"sandbox_id": "sbx_recovery", "generation": 2}
        recovery_events = await collect_tool_approval_recovery_events(
            app,
            ToolApprovalRecoveryRequest(
                session_id="sess_factory_recovery",
                approval_id=approval_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="side effect completed externally",
            ),
        )
        checkpoint = await store.load_checkpoint("sess_factory_recovery")
        return retry_events, recovery_events, factory, checkpoint

    retry_events, recovery_events, factory, checkpoint = asyncio.run(run())

    assert [event.type for event in retry_events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert [event.type for event in recovery_events[:3]] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.SESSION_RESUMED,
    ]
    assert EventType.TOOL_CALL_COMPLETED in [event.type for event in recovery_events]
    assert len(factory.requests) == 3
    assert [request.reconnect_metadata for request in factory.requests] == [
        {},
        {"sandbox_id": "sbx_recovery"},
        {"sandbox_id": "sbx_recovery"},
    ]
    assert checkpoint is not None
    assert checkpoint["environment_factory_reconnect"] == {
        "dynamic": {"sandbox_id": "sbx_recovery", "generation": 2}
    }


def test_cayu_app_recovery_factory_failure_returns_to_interrupted_before_resume(tmp_path):
    async def run():
        store = FailingTerminalToolEventStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
            )
        )
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
        tool = SideEffectTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireApprovalPolicy(),
        )

        interrupt_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_recovery_failure",
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event
            for event in interrupt_events
            if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]

        retry_events = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_factory_recovery_failure",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        factory.fail_create = True
        recovery_events = await collect_tool_approval_recovery_events(
            app,
            ToolApprovalRecoveryRequest(
                session_id="sess_factory_recovery_failure",
                approval_id=approval_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="side effect completed externally",
            ),
        )
        session = await store.load("sess_factory_recovery_failure")
        return retry_events, recovery_events, session

    retry_events, recovery_events, session = asyncio.run(run())

    assert [event.type for event in retry_events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert [event.type for event in recovery_events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert recovery_events[-1].payload["error"] == "factory failed"
    assert (
        recovery_events[-1].payload["approval_id"]
        == recovery_events[-1].payload["approval"]["approval_id"]
    )
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[3].payload["limit"] == "total_tokens"
    assert events[3].payload["actual"] == 11
    assert events[3].payload["maximum"] == 10
    assert events[4].payload["reason"] == "limit_reached"
    assert events[6].payload["interruption_type"] == "limit_reached"
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
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[4].payload["limit"] == "total_tokens"
    assert events[6].payload["interruption_type"] == "limit_reached"

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
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[4].payload["limit"] == "estimated_cost"
    assert events[4].payload["maximum"] == "0.002"
    assert events[4].payload["actual"] == "0.002"
    assert events[4].payload["cost_summary"]["total_cost"] == "0.002"
    assert events[6].payload["interruption_type"] == "limit_reached"

    session = asyncio.run(store.load("sess_cost_limit"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_request_notify_budget_emits_event_without_stopping_session():
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
                session_id="sess_request_notify_budget",
                messages=[Message.text("user", "answer")],
                budget_limits=(fake_budget_limit("0.002", action="notify"),),
            ),
        )
    )
    session = asyncio.run(store.load("sess_request_notify_budget"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[4].payload["action"] == "notify"
    assert events[4].payload["limit_reached"] is True
    assert events[4].payload["maximum"] == "0.002"
    assert events[4].payload["actual"] == "0.002"
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_request_notify_budget_dedupes_during_tool_round():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="upper",
                    arguments={"text": "one"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="upper",
                    arguments={"text": "two"},
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
                ModelStreamEvent.text_delta("done"),
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
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UpperTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_notify_budget_tool_round",
                messages=[Message.text("user", "call tools")],
                budget_limits=(fake_budget_limit("0.002", action="notify"),),
                max_steps=2,
            ),
        )
    )
    session = asyncio.run(store.load("sess_request_notify_budget_tool_round"))
    budget_events = [event for event in events if event.type == EventType.BUDGET_LIMIT_REACHED]

    assert [event.payload["action"] for event in budget_events] == ["notify"]
    assert [event.type for event in events].count(EventType.TOOL_CALL_COMPLETED) == 2
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert tool.calls == [{"step": 1}]
    assert events[7].payload["limit"] == "tool_calls"
    assert events[7].payload["actual"] == 2
    assert events[8].payload["tool_call_id"] == "call_2"
    assert events[8].payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="sess_tool_limit",
        tool_round_id=events[8].payload["tool_round_id"],
        tool_call_id="call_2",
    )

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
    # Mid-round elapsed re-checks between tool calls are sequential-mode behavior.
    app = CayuApp(session_store=store, max_parallel_tool_calls=1)
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert not any(event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED for event in events)
    assert tool.calls == []

    checkpoint = asyncio.run(store.load_checkpoint("sess_elapsed_limit_before_approval"))
    assert checkpoint == {}


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
                limits=RunLimits(max_total_tokens=10, scope="session"),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2


def test_cayu_app_default_scope_resume_ignores_prior_session_usage():
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
                session_id="sess_default_scope",
                messages=[Message.text("user", "first")],
            ),
        )
    )

    # RunLimits defaults to scope="run", so a resume with an unstated scope
    # measures only this invocation's usage — the prior turn's 10 tokens do
    # not count against the cap.
    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_default_scope",
                messages=[Message.text("user", "second")],
                limits=RunLimits(max_total_tokens=10),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2


def test_cayu_app_session_scoped_elapsed_limit_measures_session_lifetime():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("first answer"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                }
            ),
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
                session_id="sess_session_elapsed",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    # Backdate the session so its lifetime already exceeds the cap; the resume
    # itself starts fresh on the invocation clock.
    store._sessions["sess_session_elapsed"].created_at = datetime.now(UTC) - timedelta(seconds=120)

    events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_session_elapsed",
                messages=[Message.text("user", "second")],
                limits=RunLimits(max_elapsed_seconds=60, scope="session"),
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_RESUMED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[1].payload["limit"] == "elapsed_seconds"
    assert len(provider.requests) == 1


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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
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
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert second_events[1].payload["scope"] == "app"
    assert second_events[1].payload["actual"] == "1"
    assert second_events[2].payload["limit_reached"] is True
    assert second_session is not None
    assert second_session.status == SessionStatus.INTERRUPTED
    assert len(provider.requests) == 1


def test_cayu_app_notify_budget_emits_event_without_stopping_session():
    store = InMemorySessionStore()
    provider = FakeProvider(
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
                    action="notify",
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_app_budget_notify",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_app_budget_notify"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[6].payload["action"] == "notify"
    assert events[6].payload["limit_reached"] is True
    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert len(provider.requests) == 1


def test_cayu_app_policy_can_notify_before_stricter_interrupt_budget():
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
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
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
                    action="notify",
                ),
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("2"),
                    pricing=fake_budget_limit("10").pricing,
                    action="interrupt",
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
                session_id="sess_budget_notify_then_interrupt_first",
                messages=[Message.text("user", "first")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_budget_notify_then_interrupt_second",
                messages=[Message.text("user", "second")],
            ),
        )
    )
    first_session = asyncio.run(store.load("sess_budget_notify_then_interrupt_first"))
    second_session = asyncio.run(store.load("sess_budget_notify_then_interrupt_second"))

    assert EventType.BUDGET_LIMIT_REACHED in [event.type for event in first_events]
    first_notify_events = [
        event
        for event in first_events
        if event.type == EventType.BUDGET_LIMIT_REACHED and event.payload["action"] == "notify"
    ]
    assert len(first_notify_events) == 1
    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_CHECKED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert second_events[8].payload["action"] == "interrupt"
    assert first_session is not None
    assert first_session.status == SessionStatus.COMPLETED
    assert second_session is not None
    assert second_session.status == SessionStatus.INTERRUPTED
    assert len(provider.requests) == 2


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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    failed = next(
        event for event in second_events if event.type == EventType.BUDGET_RESERVATION_FAILED
    )
    assert failed.payload["accepted"] is False
    assert failed.payload["requested"] == "1"
    assert failed.payload["actual"] == "1.75"
    assert len(provider.requests) == 1


def test_cayu_app_request_budget_reservation_reconciles_model_step():
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
    app = CayuApp(session_store=store, budget_ledger=InMemoryBudgetLedger())
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_budget_reservation",
                messages=[Message.text("user", "hello")],
                budget_limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("2"),
                        pricing=fake_budget_limit("10").pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=1_000_000,
                            max_output_tokens=0,
                        ),
                    ),
                ),
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    reserved = next(event for event in events if event.type == EventType.BUDGET_RESERVED)
    reconciled = next(event for event in events if event.type == EventType.BUDGET_RECONCILED)
    assert reserved.payload["requested"] == "1"
    assert reserved.payload["actual"] == "1"
    assert reconciled.payload["actual_amount"] == "0.25"
    assert reconciled.payload["released_amount"] == "0.75"


def test_cayu_app_request_budget_reservation_stops_second_session_sharing_the_budget():
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
    app = CayuApp(session_store=store, budget_ledger=InMemoryBudgetLedger())
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    shared_limit = BudgetLimit(
        scope="agent",
        key="assistant",
        max_estimated_cost=Decimal("1"),
        pricing=fake_budget_limit("10").pricing,
        reservation=BudgetReservation(
            max_input_tokens=1_000_000,
            max_output_tokens=0,
        ),
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_reservation_first",
                messages=[Message.text("user", "first")],
                budget_limits=(shared_limit,),
            ),
        )
    )
    second_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_request_reservation_second",
                messages=[Message.text("user", "second")],
                budget_limits=(shared_limit,),
            ),
        )
    )

    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert second_events[-1].type == EventType.SESSION_INTERRUPTED
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert "no matching pricing" in first_events[2].payload["message"]
    assert first_events[2].payload["unpriced_model_steps"] == 0
    assert [event.type for event in second_events] == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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


def test_cayu_app_rejects_fork_to_agent_with_different_provider():
    class OtherProvider(FakeProvider):
        name = "other"

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("first answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_provider(OtherProvider([]))
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    app.register_agent(AgentSpec(name="other-agent", model="other-model", provider_name="other"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_fork_provider_source",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    with pytest.raises(ValueError, match="different provider"):
        asyncio.run(
            collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id="sess_fork_provider_source",
                    session_id="sess_fork_provider_child",
                    agent_name="other-agent",
                ),
            )
        )

    assert asyncio.run(store.load("sess_fork_provider_child")) is None


def test_session_query_validates_label_selector_requirements():
    query = SessionQuery(
        label_selectors=[
            {"key": "project", "operator": "in", "values": ["ap_q2", "research"]},
            {"key": "workflow", "operator": "exists"},
        ]
    )

    assert [selector.key for selector in query.label_selectors] == ["project", "workflow"]

    with pytest.raises(ValidationError):
        SessionQuery(label_selectors={"key": "project", "operator": "exists"})


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
        EventType.TURN_COMPLETED,
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
    ).sessions
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


class _TaintSourceTool(Tool):
    spec = ToolSpec(
        name="read_web",
        description="Fetch untrusted web content.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="untrusted page content")


class _ProtectedEmailTool(Tool):
    spec = ToolSpec(
        name="send_email",
        description="Send an email to an external address.",
        input_schema={
            "type": "object",
            "properties": {"body": {"type": "string"}},
        },
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content="email sent")


def _taint_aware_policy() -> TaintAwareToolPolicy:
    return TaintAwareToolPolicy(
        taint_sources={"read_web": ["web"]},
        protected_tools={"send_email": ["web"]},
        decision=ToolPolicyDecision.DENY,
    )


def _build_taint_app(
    provider: ModelProvider,
) -> tuple[CayuApp, InMemorySessionStore, _ProtectedEmailTool]:
    """Wire a parent (subagent + taint source) and a reviewer (protected send_email), both under the
    same TaintAwareToolPolicy, sharing one provider whose event batches are consumed in call order."""
    store = InMemorySessionStore()
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
    email_tool = _ProtectedEmailTool()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[subagent_tool, _TaintSourceTool()],
        tool_policy=_taint_aware_policy(),
    )
    app.register_agent(
        AgentSpec(name="reviewer", model="fake-model"),
        tools=[email_tool],
        tool_policy=_taint_aware_policy(),
    )
    return app, store, email_tool


def test_subagent_inherits_parent_taint_and_gates_protected_tool():
    # Call order across the shared parent+child provider is deterministic: parent read_web, parent
    # subagent, child send_email (denied), child final, parent final.
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_read_web", name="read_web", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "exfiltrate the page"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email",
                    name="send_email",
                    arguments={"body": "untrusted page content"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("review done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app, store, email_tool = _build_taint_app(provider)

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_taint_parent",
                messages=[Message.text("user", "Summarize the page and email it.")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED

    child_sessions = asyncio.run(
        store.list_sessions(SessionQuery(parent_session_id="sess_taint_parent"))
    ).sessions
    assert len(child_sessions) == 1
    child = child_sessions[0]

    # The parent's read_web taint is carried across the boundary onto the child session, even
    # though it was acquired from a tool result (never present on the parent's run metadata).
    assert child.metadata.get(TAINT_LABELS_METADATA_KEY) == ["web"]

    # The inherited taint makes the child's protected send_email a taint-aware policy denial: the
    # tool never runs and the tool result carries the "protected" reason (not some unrelated error).
    assert email_tool.calls == []
    child_transcript = asyncio.run(store.load_transcript(child.id))
    assert [message.role for message in child_transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    denied_result = child_transcript[2].content[0]
    assert denied_result.is_error is True
    assert "protected" in denied_result.content.lower()


def test_subagent_without_parent_taint_allows_protected_tool():
    # Deterministic call order: parent subagent, child send_email (allowed), child final, parent final.
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "send a greeting"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email",
                    name="send_email",
                    arguments={"body": "hello"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("review done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app, store, email_tool = _build_taint_app(provider)

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_untainted_parent",
                messages=[Message.text("user", "Just email a greeting.")],
            ),
        )
    )

    child = asyncio.run(
        store.list_sessions(SessionQuery(parent_session_id="sess_untainted_parent"))
    ).sessions[0]

    # No parent taint means no seed on the child session and no gate on send_email.
    assert child.metadata.get(TAINT_LABELS_METADATA_KEY) is None

    child_transcript = asyncio.run(store.load_transcript(child.id))
    assert child_transcript[2].content[0].is_error is False
    assert child_transcript[2].content[0].content == "email sent"
    assert email_tool.calls == [{"body": "hello"}]


def test_subagent_metadata_arg_cannot_forge_child_taint():
    # An injected parent must not be able to forge child taint by putting the reserved taint key in
    # its model-controlled `metadata` argument. The parent is untainted, so if the forged label
    # leaked through, the child's send_email would be denied; the fix strips it so send_email runs.
    # Deterministic call order: parent subagent (with forged metadata), child send_email, child
    # final, parent final.
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={
                        "agent": "reviewer",
                        "task": "send a greeting",
                        "metadata": {TAINT_LABELS_METADATA_KEY: ["web"]},
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email",
                    name="send_email",
                    arguments={"body": "hello"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("review done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app, store, email_tool = _build_taint_app(provider)

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_forged_parent",
                messages=[Message.text("user", "Delegate a greeting.")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    child = asyncio.run(
        store.list_sessions(SessionQuery(parent_session_id="sess_forged_parent"))
    ).sessions[0]

    # The forged reserved key is stripped from the model-supplied metadata, so the child is not
    # tainted and the protected send_email runs instead of being gated by fabricated taint.
    assert child.metadata.get(TAINT_LABELS_METADATA_KEY) is None
    assert email_tool.calls == [{"body": "hello"}]


def test_subagent_inherits_same_round_taint_source():
    # A taint source (read_web) and the subagent emitted in ONE model round: the barriered subagent
    # must inherit the same-round taint, or the child runs its protected tool untainted.
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_read_web", name="read_web", arguments={}),
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "exfiltrate the page"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email",
                    name="send_email",
                    arguments={"body": "untrusted page content"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("review done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app, store, email_tool = _build_taint_app(provider)

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_same_round",
                messages=[Message.text("user", "Summarize the page and email it.")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    child = asyncio.run(
        store.list_sessions(SessionQuery(parent_session_id="sess_same_round"))
    ).sessions[0]
    assert child.metadata.get(TAINT_LABELS_METADATA_KEY) == ["web"]
    assert email_tool.calls == []


def _build_pausing_taint_app(
    provider: ModelProvider,
    *,
    parent_policy: TaintAwareToolPolicy,
    parent_tools: list,
) -> tuple[CayuApp, InMemorySessionStore, _ProtectedEmailTool]:
    """Parent (subagent + given extra tools/policy) delegates to a reviewer whose protected
    send_email is DENY-gated once tainted."""
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    subagent_tool = SubagentTool(
        app,
        agents={
            "reviewer": SubagentSpec(agent_name="reviewer", description="Review delegated work.")
        },
    )
    email_tool = _ProtectedEmailTool()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[subagent_tool, *parent_tools],
        tool_policy=parent_policy,
    )
    app.register_agent(
        AgentSpec(name="reviewer", model="fake-model"),
        tools=[email_tool],
        tool_policy=_taint_aware_policy(),
    )
    return app, store, email_tool


def test_subagent_taint_survives_approval_pause():
    # A tainted session pauses for approval on a protected subagent; after approval the resumed
    # subagent must still create a tainted child so the child's protected send_email is denied.
    parent_policy = TaintAwareToolPolicy(
        taint_sources={"read_web": ["web"]},
        protected_tools={"subagent": ["web"]},
        decision=ToolPolicyDecision.REQUIRE_APPROVAL,
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "email the secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email", name="send_email", arguments={"body": "secret"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("review done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app, store, email_tool = _build_pausing_taint_app(
        provider, parent_policy=parent_policy, parent_tools=[_TaintSourceTool()]
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_appr_taint",
                messages=[Message.text("user", "Delegate.")],
                metadata={TAINT_LABELS_METADATA_KEY: ["web"]},
            ),
        )
    )
    approval_event = next(
        event for event in first_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_appr_taint",
                approval_id=approval_event.payload["approval"]["approval_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    child = asyncio.run(
        store.list_sessions(SessionQuery(parent_session_id="sess_appr_taint"))
    ).sessions[0]
    assert child.metadata.get(TAINT_LABELS_METADATA_KEY) == ["web"]
    assert email_tool.calls == []


def test_subagent_taint_survives_user_input_pause():
    # A tainted session pauses for user input in a round that also delegates to a subagent; on
    # resume the subagent must still create a tainted child so its protected send_email is denied.
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_subagent",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "email the secret"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_ask",
                    name="ask_user",
                    arguments={"question": "Proceed?"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email", name="send_email", arguments={"body": "secret"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("review done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app, store, email_tool = _build_pausing_taint_app(
        provider, parent_policy=_taint_aware_policy(), parent_tools=[UserInputTool()]
    )

    pause_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_input_taint",
                messages=[Message.text("user", "Delegate.")],
                metadata={TAINT_LABELS_METADATA_KEY: ["web"]},
            ),
        )
    )
    input_id = next(
        event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
    ).payload["input_id"]
    asyncio.run(
        collect_user_input_events(
            app,
            UserInputResponse(session_id="sess_input_taint", input_id=input_id, answer="yes"),
        )
    )

    child = asyncio.run(
        store.list_sessions(SessionQuery(parent_session_id="sess_input_taint"))
    ).sessions[0]
    assert child.metadata.get(TAINT_LABELS_METADATA_KEY) == ["web"]
    assert email_tool.calls == []


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

        child_sessions = (
            await store.list_sessions(
                SessionQuery(parent_session_id="sess_subagent_background_parent")
            )
        ).sessions
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
        child_sessions = (
            await store.list_sessions(
                SessionQuery(parent_session_id="sess_subagent_background_result_parent")
            )
        ).sessions
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
        child_sessions = (
            await store.list_sessions(
                SessionQuery(parent_session_id="sess_subagent_background_all_parent")
            )
        ).sessions
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
        child_sessions = (
            await store.list_sessions(
                SessionQuery(parent_session_id="sess_background_parent_interrupt")
            )
        ).sessions
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
    ).sessions
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
    ).sessions
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


def test_collect_subagent_result_returns_last_message_tail_truncated():
    from cayu.tools.subagents import _collect_subagent_result

    def _delta(text: str) -> Event:
        return Event(
            type=EventType.MODEL_TEXT_DELTA,
            session_id="sess_child",
            payload={"delta": text},
        )

    events = [
        Event(type=EventType.MODEL_STARTED, session_id="sess_child"),
        _delta("EARLY draft that should be discarded entirely"),
        Event(type=EventType.MODEL_STARTED, session_id="sess_child"),
        _delta("final "),
        _delta("answer"),
        Event(type=EventType.SESSION_COMPLETED, session_id="sess_child"),
    ]

    async def _stream():
        for event in events:
            yield event

    result = asyncio.run(_collect_subagent_result(_stream(), max_chars=64))

    # The result is the LAST assistant message, not the first deltas of every turn.
    assert result.text == "final answer"
    assert result.text_truncated is False
    assert result.event_count == len(events)
    assert result.terminal is not None
    assert result.terminal.type == EventType.SESSION_COMPLETED


def test_collect_subagent_result_tail_truncates_last_message_only():
    from cayu.tools.subagents import _collect_subagent_result

    events = [
        Event(type=EventType.MODEL_STARTED, session_id="sess_child"),
        Event(
            type=EventType.MODEL_TEXT_DELTA,
            session_id="sess_child",
            payload={"delta": "abcdefgh"},
        ),
        Event(type=EventType.SESSION_COMPLETED, session_id="sess_child"),
    ]

    async def _stream():
        for event in events:
            yield event

    result = asyncio.run(_collect_subagent_result(_stream(), max_chars=4))

    assert result.text == "abcd"
    assert result.text_truncated is True


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
        child_sessions = (
            await store.list_sessions(
                SessionQuery(parent_session_id="sess_subagent_parent_interrupt")
            )
        ).sessions
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
        child_sessions = (
            await store.list_sessions(
                SessionQuery(parent_session_id="sess_subagent_parent_startup_interrupt")
            )
        ).sessions
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
        metadata={"events": 6},
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
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert dispatch_events[0].payload == {
        "agent_name": "assistant",
        "appended_messages": 1,
        "dispatch_id": "dispatch_fork_1",
        "task_id": task.id,
        "parent_session_id": "sess_dispatch_fork_source",
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
        EventType.TURN_COMPLETED,
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


class _RecordingEchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo text, recording each invocation.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content=args["text"], structured={"echoed": args["text"]})


def _tool_call_provider() -> FakeProvider:
    return FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_echo",
                    name="echo",
                    arguments={"text": "original"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )


def test_before_tool_call_hook_modifies_arguments():
    class ArgModifierHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            modified = dict(context.arguments)
            modified["text"] = "modified"
            return BeforeToolCallDecision(action="proceed_modified", modified_arguments=modified)

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[ArgModifierHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_mod",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert tool.calls == [{"text": "modified"}]
    transcript = asyncio.run(store.load_transcript("sess_before_mod"))
    assert transcript[2].content[0].structured == {"echoed": "modified"}
    assert [event.payload["phase"] for event in events if event.type == EventType.HOOK_STARTED] == [
        "before_tool_call"
    ]
    assert events[-1].type == EventType.SESSION_COMPLETED


class _DenyTextPolicy(ToolPolicy):
    """Deny when arguments['text'] == 'forbidden'; allow otherwise."""

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.arguments.get("text") == "forbidden":
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="forbidden text")
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class _RewriteTextHook(RuntimeHook):
    def __init__(self, value: str) -> None:
        self._value = value

    async def before_tool_call(self, context: BeforeToolCallHookContext) -> BeforeToolCallDecision:
        modified = dict(context.arguments)
        modified["text"] = self._value
        return BeforeToolCallDecision(action="proceed_modified", modified_arguments=modified)


def _run_reauth_case(session_id, policy, hooks, extra_hooks=()):
    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=policy,
        runtime_hooks=[*hooks, *extra_hooks],
    )
    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript(session_id))
    return tool, events, transcript


def test_before_tool_call_modified_args_are_reauthorized_and_denied():
    # Policy allows the model's original "original", but a hook rewrites to "forbidden";
    # re-authorization on the effective args must block the call (the tool never runs).
    tool, events, transcript = _run_reauth_case(
        "sess_reauth_deny", _DenyTextPolicy(), [_RewriteTextHook("forbidden")]
    )
    assert tool.calls == []
    blocked = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert blocked.payload["blocked_by"] == "tool_policy_reauthorization"
    assert blocked.payload["decision"] == "deny"
    assert transcript[2].content[0].is_error is True


def test_before_tool_call_modified_args_pass_reauthorization():
    # A hook rewrites to an allowed value; re-authorization permits it and the tool runs with it.
    tool, _events, transcript = _run_reauth_case(
        "sess_reauth_allow", _DenyTextPolicy(), [_RewriteTextHook("allowed")]
    )
    assert tool.calls == [{"text": "allowed"}]
    assert transcript[2].content[0].structured == {"echoed": "allowed"}


def test_reauthorized_effective_args_reach_after_hook_and_event():
    seen: dict[str, object] = {}

    class CaptureArgs(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            seen["args"] = context.arguments

    tool, events, _transcript = _run_reauth_case(
        "sess_effective_args",
        AllowAllToolPolicy(),
        [_RewriteTextHook("modified")],
        extra_hooks=[CaptureArgs()],
    )
    assert tool.calls == [{"text": "modified"}]
    # after_tool_call sees the EFFECTIVE args, and the result event records them for audit.
    assert seen["args"] == {"text": "modified"}
    completed = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert completed.payload["effective_arguments"] == {"text": "modified"}


def test_before_tool_call_modified_args_requiring_approval_are_blocked():
    class ApprovalForModifiedPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            if request.arguments.get("text") == "needs_approval":
                return ToolPolicyResult(
                    decision=ToolPolicyDecision.REQUIRE_APPROVAL, reason="needs approval"
                )
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    # v1 fail-safe: a hook that rewrites args into a REQUIRE_APPROVAL state is blocked, not resumed.
    tool, events, _transcript = _run_reauth_case(
        "sess_reauth_approval",
        ApprovalForModifiedPolicy(),
        [_RewriteTextHook("needs_approval")],
    )
    assert tool.calls == []
    blocked = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert blocked.payload["blocked_by"] == "tool_policy_reauthorization"
    assert blocked.payload["decision"] == "require_approval"


class _RecordMarkerPolicy(ToolPolicy):
    """Allow everything; record the reauthorization marker seen on each authorize() call."""

    def __init__(self) -> None:
        self.markers: list[object] = []

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        self.markers.append(request.metadata.get(TOOL_POLICY_REAUTHORIZATION_METADATA_KEY))
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


def test_reauthorization_of_modified_args_is_marked_for_stateful_policies():
    policy = _RecordMarkerPolicy()
    tool, _events, _transcript = _run_reauth_case(
        "sess_reauth_marker", policy, [_RewriteTextHook("modified")]
    )
    # Two authorizations for a hook-modified call: initial (no marker), then re-auth (marked True)
    # so a stateful policy can re-check the limit without double-counting.
    assert tool.calls == [{"text": "modified"}]
    assert policy.markers == [None, True]


def test_unmodified_call_is_authorized_once_without_reauth_marker():
    policy = _RecordMarkerPolicy()
    tool, _events, _transcript = _run_reauth_case("sess_no_reauth", policy, [])
    # No hook modification → a single authorization, no re-auth, no marker.
    assert tool.calls == [{"text": "original"}]
    assert policy.markers == [None]


def test_after_tool_call_hook_sees_redacted_arguments():
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret_value = "sk-arg-secret"
    seen: dict[str, object] = {}

    class ObservingHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            seen["arguments"] = context.arguments

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_echo", name="echo", arguments={"text": secret_value}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        runtime_hooks=[ObservingHook()],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_arg_redact",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    # The after-hook observes redacted arguments, not just a redacted result.
    assert secret_value not in str(seen["arguments"])
    assert seen["arguments"] == {"text": REDACTED_SECRET}


def test_composed_after_hook_never_sees_prior_hooks_raw_secret_result():
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret_value = "sk-x"
    seen: dict[str, object] = {}

    class InjectSecretHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision:
            return AfterToolCallDecision(
                action="modify", modified_result=ToolResult(content=f"leak {secret_value}")
            )

    class ObserveResultHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            seen["content"] = context.result.content

    store = InMemorySessionStore()
    # App-scope hook injects the secret via modify; agent-scope hook observes the threaded result.
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
        runtime_hooks=[InjectSecretHook()],
    )
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        runtime_hooks=[ObserveResultHook()],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compose_redact",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    # The second after-hook sees the first hook's modification redacted, never the raw secret.
    assert secret_value not in str(seen["content"])
    assert REDACTED_SECRET in str(seen["content"])


class _ModifyTextBeforeHook(RuntimeHook):
    def __init__(self, value: str) -> None:
        self._value = value

    async def before_tool_call(self, context: BeforeToolCallHookContext) -> BeforeToolCallDecision:
        modified = dict(context.arguments)
        modified["text"] = self._value
        return BeforeToolCallDecision(action="proceed_modified", modified_arguments=modified)


class _CaptureAfterArgsHook(RuntimeHook):
    def __init__(self) -> None:
        self.arguments: object = None

    async def after_tool_call(self, context: ToolCallHookContext) -> None:
        self.arguments = context.arguments


def test_effective_args_flow_to_short_circuit_terminal_path():
    class ShortCircuitHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            return BeforeToolCallDecision(
                action="short_circuit", synthetic_result=ToolResult(content="cached")
            )

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    capture = _CaptureAfterArgsHook()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[_ModifyTextBeforeHook("modified-before-short"), ShortCircuitHook(), capture],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_eff_sc",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    # A hook modified args, then a later hook short-circuited: the effective args must reach the
    # after-hook and the result event, even though the tool never ran.
    assert tool.calls == []
    assert capture.arguments == {"text": "modified-before-short"}
    completed = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert completed.payload["effective_arguments"] == {"text": "modified-before-short"}


def test_effective_args_flow_to_block_terminal_path():
    class BlockHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            return BeforeToolCallDecision(action="block", block_reason="blocked")

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    capture = _CaptureAfterArgsHook()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[_ModifyTextBeforeHook("modified-before-block"), BlockHook(), capture],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_eff_block",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert tool.calls == []
    assert capture.arguments == {"text": "modified-before-block"}
    blocked = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert blocked.payload["effective_arguments"] == {"text": "modified-before-block"}


def test_invalid_after_tool_call_decision_fails_even_when_observe_only():
    class BadReturnHook(RuntimeHook):
        async def after_tool_call(self, context):
            return "not a decision"  # invalid: not an AfterToolCallDecision or None

    class DenyPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="no")

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=DenyPolicy(),
        runtime_hooks=[BadReturnHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_bad_observe",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    # Observe-only (denial) path still validates the decision → hook.failed, not hook.completed.
    hook_failed = next(event for event in events if event.type == EventType.HOOK_FAILED)
    assert hook_failed.payload["phase"] == "after_tool_call"
    assert hook_failed.payload["error_type"] == "TypeError"
    transcript = asyncio.run(store.load_transcript("sess_bad_observe"))
    assert transcript[2].content[0].is_error is True


def test_before_tool_call_hook_short_circuits_without_running_tool():
    class ShortCircuitHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            return BeforeToolCallDecision(
                action="short_circuit",
                synthetic_result=ToolResult(content="cached", structured={"cached": True}),
            )

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[ShortCircuitHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_short",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert tool.calls == []
    completed = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert completed.payload["short_circuited_by"] == "before_tool_call_hook"
    transcript = asyncio.run(store.load_transcript("sess_before_short"))
    assert transcript[2].content[0].content == "cached"
    assert transcript[2].content[0].structured == {"cached": True}
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_before_tool_call_hook_blocks_the_call():
    class BlockHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            return BeforeToolCallDecision(action="block", block_reason="not allowed")

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[BlockHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_block",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert tool.calls == []
    blocked = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert blocked.payload["blocked_by"] == "before_tool_call_hook"
    assert blocked.payload["reason"] == "not allowed"
    transcript = asyncio.run(store.load_transcript("sess_before_block"))
    result_part = transcript[2].content[0]
    assert result_part.content == "not allowed"
    assert result_part.is_error is True
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_before_tool_call_hook_failure_proceeds_unmodified():
    class FailingBeforeHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            raise RuntimeError("before hook broke")

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[FailingBeforeHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_fail",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    hook_failed = next(event for event in events if event.type == EventType.HOOK_FAILED)
    assert hook_failed.payload["phase"] == "before_tool_call"
    assert hook_failed.payload["error"] == "before hook broke"
    assert tool.calls == [{"text": "original"}]
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_before_tool_call_invalid_decision_proceeds_unmodified():
    class InvalidDecisionHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            # Missing modified_arguments raises ValidationError inside the hook body.
            return BeforeToolCallDecision(action="proceed_modified")

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[InvalidDecisionHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_before_invalid",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    hook_failed = next(event for event in events if event.type == EventType.HOOK_FAILED)
    assert hook_failed.payload["phase"] == "before_tool_call"
    assert hook_failed.payload["error_type"] == "ValidationError"
    assert tool.calls == [{"text": "original"}]
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_after_tool_call_hook_modifies_result_before_transcript():
    class ResultModifierHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision:
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(content="rewritten", structured={"modified": True}),
            )

    store = InMemorySessionStore()
    provider = _tool_call_provider()
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        runtime_hooks=[ResultModifierHook()],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_after_mod",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    transcript = asyncio.run(store.load_transcript("sess_after_mod"))
    assert transcript[2].content[0].content == "rewritten"
    assert transcript[2].content[0].structured == {"modified": True}
    # The model's next turn sees the modified result.
    assert provider.requests[1].messages[-1].content[0].structured == {"modified": True}
    # The result event is persisted after the hook lifecycle events.
    stored_events = asyncio.run(store.load_events("sess_after_mod"))
    assert [
        event.type
        for event in stored_events
        if event.type
        in {
            EventType.TOOL_CALL_COMPLETED,
            EventType.HOOK_STARTED,
            EventType.HOOK_COMPLETED,
        }
    ] == [
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
        EventType.TOOL_CALL_COMPLETED,
    ]


def test_after_tool_call_hooks_compose_app_then_agent_scope():
    class AppendHook(RuntimeHook):
        def __init__(self, suffix: str) -> None:
            self._suffix = suffix

        async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision:
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(content=context.result.content + self._suffix),
            )

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_echo", name="echo", arguments={"text": "seed"}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, runtime_hooks=[AppendHook("+app")])
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        runtime_hooks=[AppendHook("+agent")],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_after_compose",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    transcript = asyncio.run(store.load_transcript("sess_after_compose"))
    assert transcript[2].content[0].content == "seed+app+agent"


def test_after_tool_call_modification_is_redacted():
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret_value = "sk-injected-by-hook"

    class SecretInjectingHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision:
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(content=f"leaked {secret_value}"),
            )

    store = InMemorySessionStore()
    provider = _tool_call_provider()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        runtime_hooks=[SecretInjectingHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_after_redact",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    # Redaction runs after the hook's modification, so the injected secret is still scrubbed.
    tool_event = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert secret_value not in str(tool_event.payload["result"])
    assert REDACTED_SECRET in tool_event.payload["result"]["content"]
    transcript = asyncio.run(store.load_transcript("sess_after_redact"))
    assert transcript[2].content[0].content == f"leaked {REDACTED_SECRET}"


def test_tool_call_hooks_apply_to_subagent_tool_calls():
    observed: list[tuple[str, str]] = []

    class RecordingAppHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision | None:
            observed.append(("before", context.tool_name))
            return None

        async def after_tool_call(
            self, context: ToolCallHookContext
        ) -> AfterToolCallDecision | None:
            observed.append(("after", context.tool_name))
            return None

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_sub",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "do it"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_child_echo", name="echo", arguments={"text": "child"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("child done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False, runtime_hooks=[RecordingAppHook()])
    subagent_tool = SubagentTool(
        app,
        agents={"reviewer": SubagentSpec(agent_name="reviewer", description="Review.")},
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="parent", model="fake-model"), tools=[subagent_tool])
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"), tools=[EchoTool()])

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                session_id="sess_sub_hooks",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    # App-scope hooks fire on the parent's subagent call AND the child's echo call (propagation).
    assert ("before", "subagent") in observed
    assert ("before", "echo") in observed
    assert ("after", "echo") in observed


def test_after_tool_call_hook_sees_redacted_result():
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret_value = "sk-leaked-to-hook"
    seen: dict[str, object] = {}

    class LeakyEchoTool(Tool):
        spec = ToolSpec(
            name="echo",
            description="Return a secret.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content=f"token {secret_value}", structured={"token": secret_value})

    class ObservingHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            seen["content"] = context.result.content
            seen["structured"] = context.result.structured

    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[LeakyEchoTool()],
        runtime_hooks=[ObservingHook()],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_hook_redact",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    # The hook observes the already-redacted result, never the raw secret.
    assert secret_value not in str(seen["content"])
    assert REDACTED_SECRET in str(seen["content"])
    assert seen["structured"] == {"token": REDACTED_SECRET}


def test_after_tool_call_modify_ignored_on_policy_denied_result():
    class DenyPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="nope")

    class RewriteHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision:
            return AfterToolCallDecision(
                action="modify", modified_result=ToolResult(content="APPROVED")
            )

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=DenyPolicy(),
        runtime_hooks=[RewriteHook()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_deny_modify",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert tool.calls == []
    # The after-hook still fires on the denial, but its modify is ignored (observe-only).
    assert any(event.type == EventType.HOOK_STARTED for event in events)
    assert any(event.type == EventType.TOOL_CALL_BLOCKED for event in events)
    transcript = asyncio.run(store.load_transcript("sess_deny_modify"))
    result_part = transcript[2].content[0]
    assert result_part.content != "APPROVED"
    assert result_part.is_error is True


def test_after_tool_call_modify_applies_on_short_circuit_result():
    class ShortCircuitHook(RuntimeHook):
        async def before_tool_call(
            self, context: BeforeToolCallHookContext
        ) -> BeforeToolCallDecision:
            return BeforeToolCallDecision(
                action="short_circuit", synthetic_result=ToolResult(content="cached")
            )

    class RewriteHook(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision:
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(content=f"rewritten-{context.result.content}"),
            )

    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[ShortCircuitHook(), RewriteHook()],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_sc_modify",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert tool.calls == []
    transcript = asyncio.run(store.load_transcript("sess_sc_modify"))
    assert transcript[2].content[0].content == "rewritten-cached"


def test_before_tool_call_decision_rejects_stray_block_reason():
    with pytest.raises(ValidationError, match="block_reason is only valid with block"):
        BeforeToolCallDecision(action="proceed", block_reason="nope")


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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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


def test_cayu_app_links_claimed_task_to_successful_run():
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
                task_id="task_runtime_claimed_success",
                type="respond",
                assigned_agent_name="assistant",
            )
        )
        claimed = await task_store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None
        assert claimed.worker_id == "worker_a"
        assert claimed.session_id is None

        app = CayuApp(session_store=session_store, task_store=task_store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_task_claimed_success",
                task_id="task_runtime_claimed_success",
                task_worker_id="worker_a",
                messages=[Message.text("user", "hi")],
            ),
        )
        task = await task_store.load_task("task_runtime_claimed_success")
        return events, task

    events, task = asyncio.run(run_task_session())

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TASK_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TASK_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    assert task.session_id == "sess_task_claimed_success"
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert events[1].payload["task_id"] == "task_runtime_claimed_success"


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
        EventType.TURN_COMPLETED,
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
        EventType.MODEL_ATTEMPT_DISCARDED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 3
    assert len(tool.calls) == 1
    assert events[3].payload["reason"] == "http_status"
    assert events[3].payload["status_code"] == 429
    assert events[3].payload["attempt"] == 1
    assert events[3].payload["next_attempt"] == 2
    assert events[4].payload["attempt"] == 1
    assert events[4].payload["next_attempt"] == 2
    assert events[4].payload["status_code"] == 429
    assert events[7].payload["tool_call_id"] == "call_successful_attempt"
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.MODEL_ATTEMPT_DISCARDED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2
    assert events[3].payload["reason"] == "timeout"
    assert events[4].payload["reason"] == "timeout"
    assert events[4].payload["attempt"] == 1
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
        EventType.MODEL_ATTEMPT_DISCARDED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
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
    assert events[4].payload["attempt"] == 1
    assert events[4].payload["next_attempt"] == 2
    assert events[4].payload["status_code"] == 500
    assert events[7].payload == {
        "delta": "final answer",
        "step": 1,
        "attempt": 2,
        "max_attempts": 2,
    }
    assert events[8].payload["attempt"] == 2
    assert events[8].payload["max_attempts"] == 2
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
        EventType.MODEL_ATTEMPT_DISCARDED,
        EventType.MODEL_STARTED,
        EventType.MODEL_ERROR,
        EventType.TURN_COMPLETED,
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
    assert events[4].payload["attempt"] == 1
    assert events[4].payload["reason"] == "timeout"
    assert events[6].payload == {
        "error": "stream idle timeout",
        "error_type": "TimeoutError",
        "step": 1,
        "attempt": 2,
        "max_attempts": 2,
    }


def test_cayu_app_retries_using_typed_provider_status_code_without_http_text():
    # The error message carries no HTTP token; retry classification must come
    # from the typed provider status code preserved on the error-event payload.
    typed_error = ModelProviderError(
        "provider overloaded",
        provider="fake",
        status_code=529,
    )
    provider = FakeProvider(
        [
            [ModelStreamEvent.error("provider overloaded", cause=typed_error)],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
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
                session_id="sess_typed_status_retry",
                messages=[Message.text("user", "hi")],
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            ),
        )
    )

    types = [event.type for event in events]
    assert EventType.MODEL_RETRY in types
    discarded = next(e for e in events if e.type == EventType.MODEL_ATTEMPT_DISCARDED)
    assert discarded.payload["status_code"] == 529
    assert discarded.payload["attempt"] == 1
    assert discarded.payload["next_attempt"] == 2
    retry_event = next(e for e in events if e.type == EventType.MODEL_RETRY)
    assert retry_event.payload["status_code"] == 529
    assert retry_event.payload["reason"] == "http_status"


def test_cayu_app_does_not_retry_when_provider_marks_error_non_retryable():
    # Message text looks like a transient timeout, but the provider's typed
    # verdict (retryable=False) forces a terminal decision.
    typed_error = ModelProviderError(
        "request timed out",
        provider="fake",
        retryable=False,
    )
    provider = FakeProvider(
        [
            [ModelStreamEvent.error("request timed out", cause=typed_error)],
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
                session_id="sess_typed_non_retryable",
                messages=[Message.text("user", "hi")],
                retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
            ),
        )
    )

    types = [event.type for event in events]
    assert EventType.MODEL_RETRY not in types
    assert EventType.MODEL_ATTEMPT_DISCARDED not in types
    assert len(provider.requests) == 1


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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        "idempotency_key": events[3].payload["idempotency_key"],
        "effect": "external",
        "arguments": {"text": "from tool"},
        "tool_round_id": events[3].payload["tool_round_id"],
    }
    assert events[4].payload["tool_round_id"] == events[3].payload["tool_round_id"]
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


def test_cayu_app_recovers_pending_tool_round_from_recorded_terminal_event():
    store = FailingOrdinaryToolResultCloseStore()
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
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_round_recover_recorded",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_round_recover_recorded"))
    assert checkpoint is not None
    assert "pending_tool_round" in checkpoint
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in initial_events)
    assert initial_events[-1].type == EventType.SESSION_FAILED
    assert tool.calls == [{}]

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_tool_round_recover_recorded",
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert tool.calls == [{}]
    assert any(
        event.type == EventType.SESSION_CHECKPOINTED
        and event.payload
        == {
            "checkpoint": "pending_tool_round",
            "tool_round_id": checkpoint["pending_tool_round"]["round_id"],
            "cleared": True,
            "recovered_tool_calls": 1,
        }
        for event in resume_events
    )
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    assert asyncio.run(store.load_checkpoint("sess_tool_round_recover_recorded")) == {}

    transcript = asyncio.run(store.load_transcript("sess_tool_round_recover_recorded"))
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
    assert transcript[2].content[0].tool_call_id == "call_1"
    assert transcript[2].content[0].content == "recorded"
    assert provider.requests[1].messages[-3].role == "assistant"
    assert provider.requests[1].messages[-2].role == "tool"
    assert provider.requests[1].messages[-1].content[0].text == "continue"


def test_cayu_app_recovers_pending_tool_round_before_tool_started():
    store = FailingAfterPendingToolRoundCheckpointStore()
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
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_round_recover_not_started",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_round_recover_not_started"))
    assert checkpoint is not None
    assert "pending_tool_round" in checkpoint
    assert not any(event.type == EventType.TOOL_CALL_STARTED for event in initial_events)
    assert initial_events[-1].type == EventType.SESSION_FAILED
    assert tool.calls == []

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_tool_round_recover_not_started",
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert tool.calls == []
    recovered_failures = [
        event
        for event in resume_events
        if event.type == EventType.TOOL_CALL_FAILED and event.payload.get("recovered") is True
    ]
    assert len(recovered_failures) == 1
    assert recovered_failures[0].payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="sess_tool_round_recover_not_started",
        tool_round_id=checkpoint["pending_tool_round"]["round_id"],
        tool_call_id="call_1",
    )
    assert recovered_failures[0].payload["result"]["structured"] == {
        "recovered": True,
        "recovery_reason": "pending_tool_round_not_started",
        "tool_round_id": checkpoint["pending_tool_round"]["round_id"],
        "tool_call_id": "call_1",
        "tool_name": "side_effect",
        "started": False,
        "executed": False,
        "outcome_unknown": False,
    }
    assert "was not executed" in recovered_failures[0].payload["result"]["content"]
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    assert asyncio.run(store.load_checkpoint("sess_tool_round_recover_not_started")) == {}

    transcript = asyncio.run(store.load_transcript("sess_tool_round_recover_not_started"))
    recovered_result = transcript[2].content[0]
    assert recovered_result.tool_call_id == "call_1"
    assert recovered_result.is_error is True
    assert "was not executed" in recovered_result.content
    assert provider.requests[1].messages[-3].role == "assistant"
    assert provider.requests[1].messages[-2].role == "tool"
    assert provider.requests[1].messages[-1].content[0].text == "continue"


def test_cayu_app_recovers_pending_tool_round_with_unknown_tool_outcome():
    store = FailingTerminalToolEventStore()
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
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_round_recover_unknown",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_round_recover_unknown"))
    assert checkpoint is not None
    assert "pending_tool_round" in checkpoint
    assert any(event.type == EventType.TOOL_CALL_STARTED for event in initial_events)
    assert not any(event.type == EventType.TOOL_CALL_COMPLETED for event in initial_events)
    assert initial_events[-1].type == EventType.SESSION_FAILED
    assert tool.calls == [{}]

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_tool_round_recover_unknown",
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert tool.calls == [{}]
    recovered_failures = [
        event
        for event in resume_events
        if event.type == EventType.TOOL_CALL_FAILED and event.payload.get("recovered") is True
    ]
    assert len(recovered_failures) == 1
    assert recovered_failures[0].payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="sess_tool_round_recover_unknown",
        tool_round_id=checkpoint["pending_tool_round"]["round_id"],
        tool_call_id="call_1",
    )
    assert recovered_failures[0].payload["result"]["structured"] == {
        "recovered": True,
        "recovery_reason": "pending_tool_round_missing_terminal_event",
        "tool_round_id": checkpoint["pending_tool_round"]["round_id"],
        "tool_call_id": "call_1",
        "tool_name": "side_effect",
        "started": True,
        "outcome_unknown": True,
    }
    assert "outcome is unknown" in recovered_failures[0].payload["result"]["content"]
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    assert asyncio.run(store.load_checkpoint("sess_tool_round_recover_unknown")) == {}

    transcript = asyncio.run(store.load_transcript("sess_tool_round_recover_unknown"))
    recovered_result = transcript[2].content[0]
    assert recovered_result.tool_call_id == "call_1"
    assert recovered_result.is_error is True
    assert "outcome is unknown" in recovered_result.content


def test_cayu_app_recovers_pending_tool_round_without_reusing_old_tool_call_id():
    store = FailingSecondTerminalToolEventStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"round": "old"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"round": "current"},
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
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_round_recover_reused_id",
                messages=[Message.text("user", "use the tool twice")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_round_recover_reused_id"))
    assert checkpoint is not None
    assert "pending_tool_round" in checkpoint
    assert [
        event.payload["tool_round_id"]
        for event in initial_events
        if event.type == EventType.TOOL_CALL_STARTED
    ] == [
        initial_events[3].payload["tool_round_id"],
        checkpoint["pending_tool_round"]["round_id"],
    ]
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in initial_events)
    assert initial_events[-1].type == EventType.SESSION_FAILED
    assert tool.calls == [{"round": "old"}, {"round": "current"}]

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_tool_round_recover_reused_id",
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert tool.calls == [{"round": "old"}, {"round": "current"}]
    recovered_failures = [
        event
        for event in resume_events
        if event.type == EventType.TOOL_CALL_FAILED and event.payload.get("recovered") is True
    ]
    assert len(recovered_failures) == 1
    assert (
        recovered_failures[0].payload["tool_round_id"]
        == checkpoint["pending_tool_round"]["round_id"]
    )
    assert recovered_failures[0].payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="sess_tool_round_recover_reused_id",
        tool_round_id=checkpoint["pending_tool_round"]["round_id"],
        tool_call_id="call_1",
    )
    assert recovered_failures[0].payload["result"]["structured"] == {
        "recovered": True,
        "recovery_reason": "pending_tool_round_missing_terminal_event",
        "tool_round_id": checkpoint["pending_tool_round"]["round_id"],
        "tool_call_id": "call_1",
        "tool_name": "side_effect",
        "started": True,
        "outcome_unknown": True,
    }
    assert "outcome is unknown" in recovered_failures[0].payload["result"]["content"]
    assert resume_events[-1].type == EventType.SESSION_COMPLETED

    transcript = asyncio.run(store.load_transcript("sess_tool_round_recover_reused_id"))
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
    assert transcript[2].content[0].content == "recorded"
    assert "outcome is unknown" in transcript[4].content[0].content


def _crashed_tool_round_app(
    session_id: str,
    store: FailingTerminalToolEventStore | None = None,
) -> tuple[CayuApp, FailingTerminalToolEventStore, SideEffectTool, dict]:
    """Run a session whose only tool call starts but records no terminal event.

    Returns the app, store, tool, and the intact pending_tool_round checkpoint;
    the session is FAILED with `call_1` started-but-unresolved.
    """
    store = store if store is not None else FailingTerminalToolEventStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
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
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    assert initial_events[-1].type == EventType.SESSION_FAILED
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None and "pending_tool_round" in checkpoint
    assert tool.calls == [{}]
    return app, store, tool, checkpoint


def test_cayu_app_recover_tool_round_completed_outcome_resumes_without_unknown():
    session_id = "sess_tool_round_manual_completed"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id)
    round_id = checkpoint["pending_tool_round"]["round_id"]

    recovery_events = asyncio.run(
        collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=round_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="side effect verified externally",
                structured={"verified": True},
                reason="operator checked the downstream system",
            ),
        )
    )

    # The tool was executed exactly once (before the crash) — never re-run.
    assert tool.calls == [{}]
    recovered = next(
        event
        for event in recovery_events
        if event.type == EventType.TOOL_CALL_COMPLETED
        and event.payload.get("manual_recovery") is True
    )
    assert recovered.payload["tool_round_id"] == round_id
    assert recovered.payload["tool_call_id"] == "call_1"
    # The manual terminal event carries the same idempotency key as the crashed
    # execution's started event, and the same tool_round_id the auto-close ledger
    # reads — the continuation reuses this exact event instead of synthesizing.
    assert recovered.payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id=session_id,
        tool_round_id=round_id,
        tool_call_id="call_1",
    )
    assert recovered.payload["result"]["content"] == "side effect verified externally"
    assert recovered.payload["result"]["is_error"] is False
    assert recovery_events[-1].type == EventType.SESSION_COMPLETED
    assert not any(
        "outcome_unknown" in str(event.payload.get("result", "")) for event in recovery_events
    )
    assert asyncio.run(store.load_checkpoint(session_id)) == {}

    transcript = asyncio.run(store.load_transcript(session_id))
    recovered_result = transcript[2].content[0]
    assert recovered_result.tool_call_id == "call_1"
    assert recovered_result.is_error is False
    assert recovered_result.content == "side effect verified externally"
    assert recovered_result.structured == {"verified": True}

    # The round is closed and the checkpoint cleared — a second recovery rejects.
    with pytest.raises(RuntimeError, match="no pending tool round"):
        asyncio.run(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=round_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="again",
                ),
            )
        )


def test_cayu_app_recover_tool_round_failed_outcome_resumes():
    session_id = "sess_tool_round_manual_failed"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id)
    round_id = checkpoint["pending_tool_round"]["round_id"]

    recovery_events = asyncio.run(
        collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=round_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.FAILED,
                message="side effect confirmed absent downstream",
            ),
        )
    )

    assert tool.calls == [{}]
    recovered = next(
        event
        for event in recovery_events
        if event.type == EventType.TOOL_CALL_FAILED and event.payload.get("manual_recovery") is True
    )
    assert recovered.payload["result"]["is_error"] is True
    assert recovery_events[-1].type == EventType.SESSION_COMPLETED
    transcript = asyncio.run(store.load_transcript(session_id))
    recovered_result = transcript[2].content[0]
    assert recovered_result.is_error is True
    assert recovered_result.content == "side effect confirmed absent downstream"


def test_cayu_app_recover_tool_round_rejects_invalid_targets():
    session_id = "sess_tool_round_manual_invalid"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id)
    round_id = checkpoint["pending_tool_round"]["round_id"]

    def recover(**overrides):
        request_kwargs = {
            "session_id": session_id,
            "round_id": round_id,
            "tool_call_id": "call_1",
            "outcome": ToolApprovalRecoveryOutcome.COMPLETED,
            "message": "verified",
        }
        request_kwargs.update(overrides)
        return asyncio.run(
            collect_tool_round_recovery_events(app, ToolRoundRecoveryRequest(**request_kwargs))
        )

    with pytest.raises(KeyError, match="Session not found"):
        recover(session_id="sess_missing")
    with pytest.raises(ValueError, match="does not match pending round"):
        recover(round_id="round_other")
    with pytest.raises(ValueError, match="not part of the paused round"):
        recover(tool_call_id="call_other")

    # Every rejection leaves the crashed status untouched and the checkpoint intact.
    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.FAILED
    assert asyncio.run(store.load_checkpoint(session_id)) == checkpoint

    # A plain resume auto-repairs the round first; manual recovery then has no target.
    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(session_id=session_id, messages=[Message.text("user", "continue")]),
        )
    )
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    with pytest.raises(RuntimeError, match="no pending tool round"):
        recover()
    assert tool.calls == [{}]


def test_cayu_app_recover_tool_round_rejects_never_started_call():
    store = FailingAfterPendingToolRoundCheckpointStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_round_manual_not_started",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    assert initial_events[-1].type == EventType.SESSION_FAILED
    checkpoint = asyncio.run(store.load_checkpoint("sess_tool_round_manual_not_started"))
    assert checkpoint is not None and "pending_tool_round" in checkpoint
    assert tool.calls == []

    with pytest.raises(RuntimeError, match="requires a recorded tool.call.started"):
        asyncio.run(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id="sess_tool_round_manual_not_started",
                    round_id=checkpoint["pending_tool_round"]["round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="verified",
                ),
            )
        )
    session = asyncio.run(store.load("sess_tool_round_manual_not_started"))
    assert session is not None and session.status == SessionStatus.FAILED
    assert tool.calls == []


def test_cayu_app_recover_tool_round_rejects_concurrent_recovery():
    session_id = "sess_tool_round_manual_concurrent"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id)
    round_id = checkpoint["pending_tool_round"]["round_id"]

    def recovery_request(message: str) -> ToolRoundRecoveryRequest:
        return ToolRoundRecoveryRequest(
            session_id=session_id,
            round_id=round_id,
            tool_call_id="call_1",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message=message,
        )

    async def run() -> None:
        first = app.recover_tool_round(recovery_request("verified externally"))
        # Drive the first recovery past its claim: the first yielded event proves
        # the status transition and the in-process registration happened.
        assert await anext(first) is not None

        second = app.recover_tool_round(recovery_request("duplicate attempt"))
        with pytest.raises(RuntimeError, match="active work in this process"):
            await anext(second)

        remaining = [event async for event in first]
        assert remaining[-1].type == EventType.SESSION_COMPLETED

    asyncio.run(run())
    # The duplicate attempt emitted nothing: exactly one terminal event exists for
    # the call, and the tool itself only ever ran once (before the crash).
    events = asyncio.run(store.load_events(session_id))
    manual_terminals = [
        event
        for event in events
        if event.type == EventType.TOOL_CALL_COMPLETED
        and event.payload.get("manual_recovery") is True
    ]
    assert len(manual_terminals) == 1
    assert tool.calls == [{}]


def test_cayu_app_recover_tool_round_post_persist_failure_closes_to_interrupted():
    session_id = "sess_tool_round_manual_post_persist"
    store = FailingPostPersistEventsLoadStore()
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id, store=store)
    round_id = checkpoint["pending_tool_round"]["round_id"]

    store.arm_post_persist_load_failure = True
    recovery_events = asyncio.run(
        collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=round_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="verified externally",
            ),
        )
    )

    # The failure hit AFTER the terminal event persisted: instead of rolling back
    # to the crashed status (which can be non-resumable), the session closes to
    # INTERRUPTED with the failure recorded on the terminal event.
    interrupted = recovery_events[-1]
    assert interrupted.type == EventType.SESSION_INTERRUPTED
    assert interrupted.payload["tool_round_id"] == round_id
    assert interrupted.payload["tool_call_id"] == "call_1"
    assert interrupted.payload["error_type"] == "RuntimeError"
    assert "events unavailable" in interrupted.payload["error"]
    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.INTERRUPTED
    checkpoint_after = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint_after is not None and "pending_tool_round" in checkpoint_after

    # Re-targeting the same call is rejected (evidence is durable) ...
    with pytest.raises(RuntimeError, match="Resume the session"):
        asyncio.run(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=round_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="again",
                ),
            )
        )
    # ... and resume(), as the docstring prescribes, finishes the round from the
    # persisted operator outcome without re-running the tool.
    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(session_id=session_id, messages=[Message.text("user", "continue")]),
        )
    )
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == [{}]
    transcript = asyncio.run(store.load_transcript(session_id))
    recovered_result = next(
        part
        for message in transcript
        for part in message.content
        if type(part).__name__ == "ToolResultPart"
    )
    assert recovered_result.content == "verified externally"
    assert recovered_result.is_error is False


def test_cayu_app_recover_tool_round_rejects_native_structured_output_preflight():
    session_id = "sess_tool_round_manual_preflight"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id)

    with pytest.raises(ValueError, match="Native structured output is not supported"):
        asyncio.run(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=checkpoint["pending_tool_round"]["round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="verified",
                    structured_output=StructuredOutputSpec(
                        json_schema={"type": "object", "properties": {}},
                        strategy="native",
                    ),
                ),
            )
        )
    # The preflight fires before any status transition.
    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.FAILED
    assert tool.calls == [{}]


def test_cayu_app_recover_tool_round_multi_call_recovers_iteratively():
    session_id = "sess_tool_round_manual_multi"
    store = FailingAllTerminalToolEventStore()
    tool = BarrierSideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={"n": 1}),
                ModelStreamEvent.tool_call(id="call_2", name="side_effect", arguments={"n": 2}),
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
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use both tools")],
            ),
        )
    )
    assert initial_events[-1].type == EventType.SESSION_FAILED
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None and "pending_tool_round" in checkpoint
    round_id = checkpoint["pending_tool_round"]["round_id"]
    assert len(tool.calls) == 2
    started_ids = {
        event.payload["tool_call_id"]
        for event in asyncio.run(store.load_events(session_id))
        if event.type == EventType.TOOL_CALL_STARTED
    }
    assert started_ids == {"call_1", "call_2"}
    store.failing = False

    first_recovery = asyncio.run(
        collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=round_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="call_1 verified externally",
            ),
        )
    )
    # One call per invocation: call_2 also started without a terminal event, so
    # the session stops INTERRUPTED naming it instead of closing the round.
    assert first_recovery[-1].type == EventType.SESSION_INTERRUPTED
    assert first_recovery[-1].payload["manual_recovery_required"] is True
    assert first_recovery[-1].payload["tool_round_id"] == round_id
    assert first_recovery[-1].payload["tool_call_id"] == "call_2"
    assert first_recovery[-1].payload["tool_name"] == "side_effect"
    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.INTERRUPTED
    checkpoint_after_first = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint_after_first is not None
    assert "pending_tool_round" in checkpoint_after_first

    # The already-recovered call now has a terminal event — re-targeting rejects.
    with pytest.raises(RuntimeError, match="already has a terminal event"):
        asyncio.run(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=round_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="again",
                ),
            )
        )

    second_recovery = asyncio.run(
        collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=round_id,
                tool_call_id="call_2",
                outcome=ToolApprovalRecoveryOutcome.FAILED,
                message="call_2 confirmed failed externally",
            ),
        )
    )
    assert second_recovery[-1].type == EventType.SESSION_COMPLETED
    # Neither tool re-ran during either recovery.
    assert len(tool.calls) == 2
    assert asyncio.run(store.load_checkpoint(session_id)) == {}

    transcript = asyncio.run(store.load_transcript(session_id))
    tool_message = transcript[2]
    results_by_id = {part.tool_call_id: part for part in tool_message.content}
    assert results_by_id["call_1"].content == "call_1 verified externally"
    assert results_by_id["call_1"].is_error is False
    assert results_by_id["call_2"].content == "call_2 confirmed failed externally"
    assert results_by_id["call_2"].is_error is True


def test_cayu_app_recover_tool_round_taints_follow_up_rounds():
    session_id = "sess_tool_round_manual_taint"
    store = FailingTerminalToolEventStore()
    email_tool = _ProtectedEmailTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_read_web", name="read_web", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email",
                    name="send_email",
                    arguments={"body": "page summary"},
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
        tools=[_TaintSourceTool(), email_tool],
        tool_policy=_taint_aware_policy(),
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "read the page, then email it")],
            ),
        )
    )
    assert initial_events[-1].type == EventType.SESSION_FAILED
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None and "pending_tool_round" in checkpoint

    recovery_events = asyncio.run(
        collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=checkpoint["pending_tool_round"]["round_id"],
                tool_call_id="call_read_web",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="untrusted page content",
            ),
        )
    )

    # The manually recovered read_web completion still taints the session: the
    # follow-up round's protected send_email is denied, never executed.
    assert recovery_events[-1].type == EventType.SESSION_COMPLETED
    blocked = [event for event in recovery_events if event.type == EventType.TOOL_CALL_BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].payload["tool_call_id"] == "call_send_email"
    assert email_tool.calls == []


def test_cayu_app_recover_incomplete_session_interrupts_abandoned_running_session():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_abandoned_running",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_abandoned_running", SessionStatus.RUNNING)
        return await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="sess_abandoned_running",
                reason="worker restart",
                metadata={"worker": "w1"},
            )
        )

    result = asyncio.run(setup_and_recover())
    session = asyncio.run(store.load("sess_abandoned_running"))
    events = asyncio.run(store.load_events("sess_abandoned_running"))

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert result.previous_status == SessionStatus.RUNNING
    assert result.status == SessionStatus.INTERRUPTED
    assert result.actions == (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,)
    assert [event.type for event in result.events] == [EventType.SESSION_INTERRUPTED]
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    assert events[-1].payload == {
        "reason": "worker restart",
        "metadata": {"worker": "w1"},
        "interruption_type": "runtime_interrupted",
        "recovered": True,
    }


def test_cayu_app_recover_incomplete_session_skips_terminal_without_registered_agent():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id="sess_recover_terminal_missing_agent",
                messages=[Message.text("user", "done")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_recover_terminal_missing_agent", SessionStatus.COMPLETED)
        return await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_recover_terminal_missing_agent")
        )

    result = asyncio.run(setup_and_recover())

    assert result.previous_status == SessionStatus.COMPLETED
    assert result.status == SessionStatus.COMPLETED
    assert result.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
    assert result.events == ()
    assert result.message == "Session is terminal; recovery skipped."


def test_cayu_app_recover_incomplete_session_finalizes_pending_interrupt():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_stale_interrupting",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_stale_interrupting", SessionStatus.INTERRUPTING)
        await store.checkpoint(
            "sess_stale_interrupting",
            {
                "pending_session_interrupt": {
                    "reason": "deploy",
                    "metadata": {"worker": "old"},
                    "interruption_type": "operator_requested",
                }
            },
        )
        return await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_stale_interrupting")
        )

    result = asyncio.run(setup_and_recover())
    session = asyncio.run(store.load("sess_stale_interrupting"))
    checkpoint = asyncio.run(store.load_checkpoint("sess_stale_interrupting"))

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert checkpoint == {}
    assert result.actions == (IncompleteSessionRecoveryAction.FINALIZED_INTERRUPT,)
    assert [event.type for event in result.events] == [EventType.SESSION_INTERRUPTED]
    assert result.events[0].payload == {
        "reason": "deploy",
        "metadata": {"worker": "old"},
        "interruption_type": "operator_requested",
    }


def test_cayu_app_recover_incomplete_session_is_idempotent_after_interrupt_finalized():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover_twice():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_recover_interrupt_twice",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_recover_interrupt_twice", SessionStatus.INTERRUPTING)
        await store.checkpoint(
            "sess_recover_interrupt_twice",
            {
                "pending_session_interrupt": {
                    "reason": "deploy",
                    "metadata": {},
                    "interruption_type": "operator_requested",
                }
            },
        )
        first = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_recover_interrupt_twice")
        )
        second = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_recover_interrupt_twice")
        )
        events = await store.load_events("sess_recover_interrupt_twice")
        return first, second, events

    first, second, events = asyncio.run(setup_and_recover_twice())

    interrupted_events = [event for event in events if event.type == EventType.SESSION_INTERRUPTED]
    assert len(interrupted_events) == 1
    assert first.actions == (IncompleteSessionRecoveryAction.FINALIZED_INTERRUPT,)
    assert [event.type for event in first.events] == [EventType.SESSION_INTERRUPTED]
    assert second.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
    assert second.events == ()
    assert second.status == SessionStatus.INTERRUPTED


def test_cayu_app_recover_incomplete_session_repairs_tool_round_before_interrupting():
    store = FailingAfterPendingToolRoundCheckpointStore()
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
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recover_incomplete_tool_round",
                messages=[Message.text("user", "use tool")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_recover_incomplete_tool_round"))
    assert checkpoint is not None
    assert "pending_tool_round" in checkpoint
    asyncio.run(store.update_status("sess_recover_incomplete_tool_round", SessionStatus.RUNNING))

    result = asyncio.run(
        app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_recover_incomplete_tool_round")
        )
    )
    session = asyncio.run(store.load("sess_recover_incomplete_tool_round"))
    transcript = asyncio.run(store.load_transcript("sess_recover_incomplete_tool_round"))

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert result.actions == (
        IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
        IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    )
    assert tool.calls == []
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]
    assert transcript[2].content[0].tool_call_id == "call_1"
    assert "was not executed" in transcript[2].content[0].content
    assert asyncio.run(store.load_checkpoint("sess_recover_incomplete_tool_round")) == {}


def test_cayu_app_recover_incomplete_session_uses_recorded_tool_result():
    store = FailingOrdinaryToolResultCloseStore()
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
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recover_incomplete_recorded_tool_result",
                messages=[Message.text("user", "use tool")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("sess_recover_incomplete_recorded_tool_result"))
    assert checkpoint is not None
    assert "pending_tool_round" in checkpoint
    asyncio.run(
        store.update_status(
            "sess_recover_incomplete_recorded_tool_result",
            SessionStatus.RUNNING,
        )
    )

    result = asyncio.run(
        app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="sess_recover_incomplete_recorded_tool_result"
            )
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_recover_incomplete_recorded_tool_result"))

    assert result.actions == (
        IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
        IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    )
    assert tool.calls == [{}]
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]
    assert transcript[2].content[0].tool_call_id == "call_1"
    assert transcript[2].content[0].content == "recorded"
    assert transcript[2].content[0].is_error is False
    assert asyncio.run(store.load_checkpoint("sess_recover_incomplete_recorded_tool_result")) == {}


def test_cayu_app_recover_incomplete_session_preserves_pending_tool_approval():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="side_effect",
                arguments={"value": "requires approval"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
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
                session_id="sess_recover_incomplete_pending_approval",
                messages=[Message.text("user", "use tool")],
            ),
        )
    )
    approval_event = next(
        event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval_id = approval_event.payload["approval"]["approval_id"]
    asyncio.run(
        store.update_status(
            "sess_recover_incomplete_pending_approval",
            SessionStatus.RUNNING,
        )
    )

    result = asyncio.run(
        app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_recover_incomplete_pending_approval")
        )
    )
    session = asyncio.run(store.load("sess_recover_incomplete_pending_approval"))
    checkpoint = asyncio.run(store.load_checkpoint("sess_recover_incomplete_pending_approval"))

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert result.actions == (IncompleteSessionRecoveryAction.PENDING_APPROVAL,)
    assert result.pending_approval_id == approval_id
    assert result.status == SessionStatus.INTERRUPTED
    assert [event.type for event in result.events] == [EventType.SESSION_INTERRUPTED]
    assert checkpoint is not None
    assert checkpoint["pending_tool_approval"]["approval_id"] == approval_id
    assert "pending_session_interrupt" not in checkpoint
    assert tool.calls == []


def test_incomplete_sessions_recovery_request_requires_explicit_statuses():
    with pytest.raises(ValidationError, match="statuses"):
        IncompleteSessionsRecoveryRequest(limit=10)


def test_incomplete_sessions_recovery_request_rejects_empty_or_terminal_statuses():
    with pytest.raises(ValidationError, match="must not be empty"):
        IncompleteSessionsRecoveryRequest(statuses=set(), limit=10)

    with pytest.raises(ValidationError, match="unsupported recovery status"):
        IncompleteSessionsRecoveryRequest(statuses={SessionStatus.COMPLETED}, limit=10)


def test_cayu_app_recover_incomplete_sessions_with_explicit_interrupting_status():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_session(session_id: str, status: SessionStatus) -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", session_id)],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, status)

    async def setup_and_recover():
        await create_session("sess_batch_interrupting", SessionStatus.INTERRUPTING)
        await create_session("sess_batch_running", SessionStatus.RUNNING)
        await create_session("sess_batch_pending", SessionStatus.PENDING)
        await create_session("sess_batch_completed", SessionStatus.COMPLETED)
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.INTERRUPTING},
                limit=10,
            )
        )

    results = asyncio.run(setup_and_recover())

    assert [result.session_id for result in results] == ["sess_batch_interrupting"]
    assert results[0].status == SessionStatus.INTERRUPTED
    assert results[0].actions == (IncompleteSessionRecoveryAction.FINALIZED_INTERRUPT,)


def test_cayu_app_recover_incomplete_sessions_fetches_only_requested_limit():
    store = RecordingListSessionsStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_session(session_id: str) -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", session_id)],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTING)

    async def setup_and_recover():
        await create_session("sess_batch_limit_1")
        await create_session("sess_batch_limit_2")
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.INTERRUPTING},
                limit=1,
            )
        )

    results = asyncio.run(setup_and_recover())

    assert len(results) == 1
    assert len(store.session_queries) == 1
    assert store.session_queries[0].status == SessionStatus.INTERRUPTING
    assert store.session_queries[0].limit == 1
    assert store.session_queries[0].offset == 0


def test_cayu_app_recover_incomplete_sessions_can_include_abandoned_running_and_pending():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_session(session_id: str, status: SessionStatus) -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", session_id)],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, status)

    async def setup_and_recover():
        await create_session("sess_batch_running", SessionStatus.RUNNING)
        await create_session("sess_batch_pending", SessionStatus.PENDING)
        await create_session("sess_batch_completed", SessionStatus.COMPLETED)
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.RUNNING, SessionStatus.PENDING},
                limit=10,
            )
        )

    results = asyncio.run(setup_and_recover())

    assert {result.session_id for result in results} == {
        "sess_batch_running",
        "sess_batch_pending",
    }
    assert {(result.session_id, result.status, result.actions) for result in results} == {
        (
            "sess_batch_running",
            SessionStatus.INTERRUPTED,
            (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,),
        ),
        (
            "sess_batch_pending",
            SessionStatus.INTERRUPTED,
            (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,),
        ),
    }


def test_cayu_app_recover_incomplete_sessions_skips_unregistered_agent_and_continues():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover():
        async def create_session(session_id: str, agent_name: str, status: SessionStatus) -> None:
            await store.create(
                RunRequest(
                    agent_name=agent_name,
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            if status is not SessionStatus.PENDING:
                await store.update_status(session_id, status)

        await create_session("sess_sweep_healthy_a", "assistant", SessionStatus.RUNNING)
        await create_session("sess_sweep_healthy_b", "assistant", SessionStatus.RUNNING)
        await create_session("sess_sweep_ghost", "ghost_agent", SessionStatus.PENDING)
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.RUNNING, SessionStatus.PENDING}
            )
        )

    results = asyncio.run(setup_and_recover())

    by_id = {result.session_id: result for result in results}
    assert by_id["sess_sweep_healthy_a"].actions == (
        IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    )
    assert by_id["sess_sweep_healthy_b"].actions == (
        IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    )
    ghost = by_id["sess_sweep_ghost"]
    assert ghost.actions == (IncompleteSessionRecoveryAction.SKIPPED_UNREGISTERED_AGENT,)
    assert "ghost_agent" in ghost.message
    ghost_session = asyncio.run(store.load("sess_sweep_ghost"))
    assert ghost_session is not None
    assert ghost_session.status == SessionStatus.PENDING


def test_cayu_app_recover_incomplete_sessions_isolates_mid_batch_failure(monkeypatch):
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover():
        for session_id in ("sess_sweep_ok_one", "sess_sweep_broken", "sess_sweep_ok_two"):
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.update_status(session_id, SessionStatus.RUNNING)

        original_load_checkpoint = store.load_checkpoint

        async def broken_load_checkpoint(session_id: str) -> dict[str, Any] | None:
            if session_id == "sess_sweep_broken":
                raise RuntimeError("checkpoint store exploded")
            return await original_load_checkpoint(session_id)

        monkeypatch.setattr(store, "load_checkpoint", broken_load_checkpoint)
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(statuses={SessionStatus.RUNNING})
        )

    results = asyncio.run(setup_and_recover())

    by_id = {result.session_id: result for result in results}
    assert by_id["sess_sweep_ok_one"].actions == (
        IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    )
    assert by_id["sess_sweep_ok_two"].actions == (
        IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    )
    broken = by_id["sess_sweep_broken"]
    assert broken.actions == (IncompleteSessionRecoveryAction.FAILED,)
    assert "checkpoint store exploded" in broken.message


def test_cayu_app_recover_incomplete_sessions_failed_entry_reports_current_status(monkeypatch):
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sweep_mutated",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_sweep_mutated", SessionStatus.RUNNING)

        original_load_checkpoint = store.load_checkpoint
        calls = {"count": 0}

        # The first load_checkpoint call precedes the status transition; the
        # second follows it — failing there leaves the store already mutated
        # (RUNNING -> INTERRUPTING) when the batch handler builds the result.
        async def flaky_load_checkpoint(session_id: str) -> dict[str, Any] | None:
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("checkpoint reload exploded")
            return await original_load_checkpoint(session_id)

        monkeypatch.setattr(store, "load_checkpoint", flaky_load_checkpoint)
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(statuses={SessionStatus.RUNNING})
        )

    results = asyncio.run(setup_and_recover())

    (failed,) = results
    assert failed.actions == (IncompleteSessionRecoveryAction.FAILED,)
    assert "checkpoint reload exploded" in failed.message
    assert failed.previous_status == SessionStatus.RUNNING
    assert failed.status == SessionStatus.INTERRUPTING
    session = asyncio.run(store.load("sess_sweep_mutated"))
    assert session is not None
    assert session.status == failed.status


def test_cayu_app_recover_incomplete_sessions_propagates_cancellation(monkeypatch):
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sweep_cancelled",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_sweep_cancelled", SessionStatus.RUNNING)

        async def cancelled_load_checkpoint(session_id: str) -> dict[str, Any] | None:
            raise asyncio.CancelledError()

        monkeypatch.setattr(store, "load_checkpoint", cancelled_load_checkpoint)
        return await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(statuses={SessionStatus.RUNNING})
        )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(setup_and_recover())


def test_cayu_app_recover_incomplete_session_missing_session_still_raises():
    app = CayuApp(session_store=InMemorySessionStore())

    with pytest.raises(KeyError, match="Session not found"):
        asyncio.run(
            app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="sess_never_created")
            )
        )


def test_cayu_app_recover_incomplete_session_unregistered_agent_returns_typed_skip():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="ghost_agent",
                session_id="sess_single_ghost",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        return await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_single_ghost")
        )

    result = asyncio.run(setup_and_recover())

    assert result.actions == (IncompleteSessionRecoveryAction.SKIPPED_UNREGISTERED_AGENT,)
    assert result.previous_status == SessionStatus.PENDING
    assert result.status == SessionStatus.PENDING
    assert "ghost_agent" in result.message
    session = asyncio.run(store.load("sess_single_ghost"))
    assert session is not None
    assert session.status == SessionStatus.PENDING


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
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert tool.calls == []
    blocked_event = events[4]
    assert blocked_event.tool_name == "side_effect"
    assert blocked_event.payload["tool_round_id"] == events[3].payload["tool_round_id"]
    assert blocked_event.payload == {
        "tool_call_id": "call_1",
        "idempotency_key": blocked_event.payload["idempotency_key"],
        "tool_round_id": blocked_event.payload["tool_round_id"],
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


def test_cayu_app_blocks_tool_call_before_execution_with_parameter_policy():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "external"},
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
        tool_policy=ParameterConstrainedToolPolicy(
            {"side_effect": [AllowlistRule("value", values=["internal"])]}
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_parameter_blocked_tool",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert tool.calls == []
    blocked_event = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert blocked_event.payload["reason"] == "Parameter 'value' value is not allowed."
    assert blocked_event.payload["metadata"] == {
        "policy": "parameter_constrained",
        "tool_name": "side_effect",
        "parameter": "value",
        "rule": "AllowlistRule",
        "rule_index": 0,
    }
    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.content == "Parameter 'value' value is not allowed."
    assert tool_result_part.structured["metadata"]["policy"] == "parameter_constrained"
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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


class ExpiringApprovalPolicy(ToolPolicy):
    def __init__(self, expires_in_seconds: float = 60.0) -> None:
        self.expires_in_seconds = expires_in_seconds

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            reason=f"Approval required for {request.tool_name}.",
            approval_expires_in_seconds=self.expires_in_seconds,
        )


def _reviewer_actor() -> ResolutionActor:
    return ResolutionActor(
        subject="reviewer@example.com",
        tenant="acme",
        source=ResolutionActorSource.REQUEST,
        claims={"role": "admin"},
    )


def _approval_pause_app(
    *,
    store: InMemorySessionStore,
    tool: Tool,
    policy: ToolPolicy,
    clock=None,
) -> tuple[CayuApp, FakeProvider]:
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
                ModelStreamEvent.text_delta("resolved"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, clock=clock)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=policy,
    )
    return app, provider


def test_tool_approval_resolution_events_carry_resolved_by_actor():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    app, _provider = _approval_pause_app(store=store, tool=tool, policy=RequireApprovalPolicy())

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_actor",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    ).payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_actor",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
                reason="looks safe",
                resolved_by=_reviewer_actor(),
            ),
        )
    )

    # `claims` stay on the request and are excluded from event payloads.
    expected_actor = {
        "subject": "reviewer@example.com",
        "tenant": "acme",
        "source": "request",
    }
    resumed = next(event for event in events if event.type == EventType.SESSION_RESUMED)
    assert resumed.payload["resolved_by"] == expected_actor
    assert resumed.payload["expired"] is False
    approved = next(event for event in events if event.type == EventType.TOOL_CALL_APPROVED)
    assert approved.payload["resolved_by"] == expected_actor
    assert tool.calls == [{"value": "secret"}]


def test_tool_approval_denial_events_carry_resolved_by_actor():
    store = InMemorySessionStore()
    tool = SideEffectTool()
    app, _provider = _approval_pause_app(store=store, tool=tool, policy=RequireApprovalPolicy())

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_denial_actor",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    ).payload["approval"]["approval_id"]

    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_denial_actor",
                approval_id=approval_id,
                decision=ToolApprovalDecision.DENY,
                reason="not safe",
                resolved_by=_reviewer_actor(),
            ),
        )
    )

    denied = next(event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_DENIED)
    assert denied.payload["resolved_by"]["subject"] == "reviewer@example.com"
    assert denied.payload["resolved_by"]["source"] == "request"
    assert denied.payload["expired"] is False
    assert not any(event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED for event in events)
    assert tool.calls == []


def test_resolution_actor_rejects_reserved_subject_for_request_sources():
    with pytest.raises(ValidationError, match="reserved for system actors"):
        ResolutionActor(subject="cayu:approval-expiry")
    with pytest.raises(ValidationError, match="reserved for system actors"):
        ResolutionActor(subject="cayu:anything", source=ResolutionActorSource.REQUEST)
    from cayu.runtime.approvals import EXPIRY_RESOLUTION_ACTOR_SUBJECT

    system_actor = ResolutionActor(
        subject=EXPIRY_RESOLUTION_ACTOR_SUBJECT,
        source=ResolutionActorSource.SYSTEM,
    )
    assert system_actor.subject == EXPIRY_RESOLUTION_ACTOR_SUBJECT


def test_expired_tool_approval_resolves_as_deterministic_denial():
    clock = {"now": datetime(2026, 7, 9, 12, 0, tzinfo=UTC)}
    store = InMemorySessionStore()
    tool = SideEffectTool()
    app, _provider = _approval_pause_app(
        store=store,
        tool=tool,
        policy=ExpiringApprovalPolicy(expires_in_seconds=60),
        clock=lambda: clock["now"],
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_expiry",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_event = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    pending = PendingToolApproval.from_event(approval_event)
    assert pending.expires_at == datetime(2026, 7, 9, 12, 1, tzinfo=UTC)

    clock["now"] = datetime(2026, 7, 9, 12, 2, tzinfo=UTC)
    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_expiry",
                approval_id=pending.approval_id,
                decision=ToolApprovalDecision.APPROVE,
                reason="approve it anyway",
                resolved_by=_reviewer_actor(),
            ),
        )
    )

    resumed = next(event for event in events if event.type == EventType.SESSION_RESUMED)
    assert resumed.payload["decision"] == "deny"
    assert resumed.payload["expired"] is True
    assert resumed.payload["resolved_by"]["subject"] == "cayu:approval-expiry"
    assert resumed.payload["resolved_by"]["source"] == "system"

    expired_event = next(
        event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED
    )
    assert expired_event.payload["approval_id"] == pending.approval_id
    assert expired_event.payload["expires_at"] == "2026-07-09T12:01:00+00:00"
    assert expired_event.payload["requested_decision"] == "approve"
    assert expired_event.payload["resolved_by"]["subject"] == "cayu:approval-expiry"
    assert expired_event.payload["triggered_by"]["subject"] == "reviewer@example.com"

    denied = next(event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_DENIED)
    assert denied.payload["expired"] is True
    assert "expired at" in denied.payload["reason"]
    assert denied.payload["resolved_by"]["source"] == "system"

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == []
    session = asyncio.run(store.load("sess_approval_expiry"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_unexpired_tool_approval_resolves_normally():
    clock = {"now": datetime(2026, 7, 9, 12, 0, tzinfo=UTC)}
    store = InMemorySessionStore()
    tool = SideEffectTool()
    app, _provider = _approval_pause_app(
        store=store,
        tool=tool,
        policy=ExpiringApprovalPolicy(expires_in_seconds=60),
        clock=lambda: clock["now"],
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_unexpired",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    ).payload["approval"]["approval_id"]

    clock["now"] = datetime(2026, 7, 9, 12, 0, 30, tzinfo=UTC)
    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_unexpired",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert not any(event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED for event in events)
    assert any(event.type == EventType.TOOL_CALL_APPROVED for event in events)
    assert tool.calls == [{"value": "secret"}]


def test_multi_call_approval_round_uses_minimum_ttl():
    # The round is approved/denied as a whole, so any approval-requiring
    # call's TTL bounds the round: the shortest window wins, not the gating
    # (first) call's.
    class PerToolTtlPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                approval_expires_in_seconds=300 if request.tool_name == "side_effect" else 60,
            )

    clock = {"now": datetime(2026, 7, 9, 12, 0, tzinfo=UTC)}
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "one"},
                ),
                ModelStreamEvent.tool_call(id="call_2", name="echo", arguments={"text": "two"}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, clock=lambda: clock["now"])
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool(), EchoTool()],
        tool_policy=PerToolTtlPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_multi_ttl",
                messages=[Message.text("user", "use both tools")],
            ),
        )
    )
    approval_event = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    pending = PendingToolApproval.from_event(approval_event)
    # The gating call (side_effect, 300s) does not win: echo's 60s bound does.
    assert pending.expires_at == datetime(2026, 7, 9, 12, 1, tzinfo=UTC)


def test_expired_approval_retry_after_recorded_grant_is_not_coerced():
    # Expiry gates the FIRST grant only: a retry of an approval that was
    # granted in-window (crash after tool.call.approved) must honor the grant
    # even when the retry lands after the window — coercing it to a denial
    # would contradict the recorded grant and wedge the session.
    clock = {"now": datetime(2026, 7, 9, 12, 0, tzinfo=UTC)}
    store = InMemorySessionStore()
    tool = SideEffectTool()
    app, _provider = _approval_pause_app(
        store=store,
        tool=tool,
        policy=ExpiringApprovalPolicy(expires_in_seconds=60),
        clock=lambda: clock["now"],
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_expiry_retry",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    ).payload["approval"]["approval_id"]

    # A prior in-window resolve crashed after recording the grant.
    asyncio.run(
        store.append_event(
            "sess_expiry_retry",
            Event(
                type=EventType.TOOL_CALL_APPROVED,
                session_id="sess_expiry_retry",
                agent_name="assistant",
                tool_name="side_effect",
                payload={"approval_id": approval_id, "tool_call_id": "call_1"},
            ),
        )
    )

    clock["now"] = datetime(2026, 7, 9, 12, 5, tzinfo=UTC)
    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_expiry_retry",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert not any(event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED for event in events)
    assert not any(event.type == EventType.TOOL_CALL_APPROVAL_DENIED for event in events)
    assert tool.calls == [{"value": "secret"}]
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_tool_approval_recovery_events_carry_resolved_by_and_expiry_stamp():
    # Recovery reconciles a side effect authorized in-window before a crash;
    # it is never blocked by expiry, but an out-of-window reconciliation is
    # stamped for the audit trail.
    clock = {"now": datetime(2026, 7, 9, 12, 0, tzinfo=UTC)}
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
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, clock=lambda: clock["now"])
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=ExpiringApprovalPolicy(expires_in_seconds=60),
    )

    async def run():
        interrupt_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_recovery_actor",
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_id = next(
            event
            for event in interrupt_events
            if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        ).payload["approval"]["approval_id"]

        # In-window approve crashes when the terminal tool event fails to
        # persist; the stream ends interrupted and recovery is required.
        crashed_events = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_recovery_actor",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        assert crashed_events[-1].type == EventType.SESSION_INTERRUPTED

        # The operator reconciles the external outcome after the window closed.
        clock["now"] = datetime(2026, 7, 9, 12, 5, tzinfo=UTC)
        return await collect_tool_approval_recovery_events(
            app,
            ToolApprovalRecoveryRequest(
                session_id="sess_recovery_actor",
                approval_id=approval_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="side effect completed externally",
                resolved_by=_reviewer_actor(),
            ),
        )

    recovery_events = asyncio.run(run())

    resumed = next(event for event in recovery_events if event.type == EventType.SESSION_RESUMED)
    assert resumed.payload["resolved_by"]["subject"] == "reviewer@example.com"
    assert resumed.payload["expired"] is True
    recovered = next(
        event
        for event in recovery_events
        if event.type == EventType.TOOL_CALL_COMPLETED
        and event.payload.get("manual_recovery") is True
    )
    assert recovered.payload["resolved_by"]["subject"] == "reviewer@example.com"
    assert recovered.payload["resolved_by"]["source"] == "request"
    assert recovered.payload["expired"] is True
    assert not any(event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED for event in recovery_events)
    assert recovery_events[-1].type == EventType.SESSION_COMPLETED


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
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[8].payload["output"] == {"answer": "approved"}
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[1].payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="sess_tool_approval_deny",
        tool_call_id="call_1",
        approval_id=approval_id,
    )
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
    # after_tool_call runs before the result event is persisted, so the denial event lands last.
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
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert recovery_events[1].payload["approval_id"] == approval_id
    assert recovery_events[1].payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="sess_approval_started_without_terminal",
        tool_call_id="call_1",
        approval_id=approval_id,
    )
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


def test_cayu_app_redacts_manual_tool_approval_recovery_result():
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret_value = "manual-recovery-secret"
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
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
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
                session_id="sess_manual_recovery_redaction",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = interrupt_events[4].payload["approval"]["approval_id"]

    _ = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_manual_recovery_redaction",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )
    recovery_events = asyncio.run(
        collect_tool_approval_recovery_events(
            app,
            ToolApprovalRecoveryRequest(
                session_id="sess_manual_recovery_redaction",
                approval_id=approval_id,
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message=f"side effect completed with {secret_value}",
                structured={"token": secret_value},
                artifacts=[{"metadata": {"token": secret_value}}],
                reason=f"operator confirmed {secret_value}",
                metadata={"token": secret_value},
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_manual_recovery_redaction"))

    tool_event = recovery_events[1]
    assert tool_event.type == EventType.TOOL_CALL_COMPLETED
    assert secret_value not in str(tool_event.payload)
    assert tool_event.payload["reason"] == f"operator confirmed {REDACTED_SECRET}"
    assert tool_event.payload["metadata"]["token"] == REDACTED_SECRET
    assert tool_event.payload["result"]["content"] == (
        f"side effect completed with {REDACTED_SECRET}"
    )
    assert tool_event.payload["result"]["structured"]["token"] == REDACTED_SECRET
    assert tool_event.payload["result"]["artifacts"][0]["metadata"]["token"] == REDACTED_SECRET

    tool_message = next(message for message in transcript if message.role == "tool")
    tool_part = tool_message.content[0]
    assert isinstance(tool_part, ToolResultPart)
    assert secret_value not in str(tool_part.model_dump(mode="json"))
    assert tool_part.content == f"side effect completed with {REDACTED_SECRET}"
    assert tool_part.structured == {"token": REDACTED_SECRET}
    assert tool_part.artifacts[0]["metadata"]["token"] == REDACTED_SECRET
    assert provider.requests[1].messages[-1].role == "tool"
    assert secret_value not in str(provider.requests[1].messages[-1].model_dump(mode="json"))


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


def test_cayu_app_wires_environment_proxy_into_tool_context():
    class ProxyProbeTool(Tool):
        spec = ToolSpec(
            name="probe_proxy",
            description="Record proxy identity.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            super().__init__()
            self.proxies: list[object] = []

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            self.proxies.append(ctx.proxy)
            return ToolResult(content="ok")

    vault = StaticVault({"api_key": "sk-secret-123"})
    proxy = PassthroughProxy(vault)
    tool = ProxyProbeTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="probe_proxy",
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
            vault=vault,
            proxy=proxy,
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_context_proxy",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(tool.proxies) == 1
    assert isinstance(tool.proxies[0], CredentialProxy)


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
    assert tool.contexts[0].metadata["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="sess_policy_run_metadata",
        tool_round_id=events[3].payload["tool_round_id"],
        tool_call_id="call_1",
    )
    assert tool.contexts[0].metadata == {
        "tenant": {"id": "tenant_1"},
        "tool_call_id": "call_1",
        "idempotency_key": tool.contexts[0].metadata["idempotency_key"],
        "tool_effect": "external",
    }
    tool.contexts[0].metadata["tenant"]["id"] = "tool-mutated"
    session = asyncio.run(app.session_store.load("sess_policy_run_metadata"))
    assert session is not None
    assert session.metadata == {"tenant": {"id": "tenant_1"}}


def test_cayu_app_taint_policy_requires_approval_from_prior_durable_tool_result():
    store = InMemorySessionStore()
    echo_tool = EchoTool()
    side_effect_tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_read",
                    name="echo",
                    arguments={"text": "external email body"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("stored"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_send",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[echo_tool, side_effect_tool],
        tool_policy=TaintAwareToolPolicy(
            taint_sources={"echo": ["external_email"]},
            protected_tools={"side_effect": ["external_email"]},
        ),
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_taint_prior",
                messages=[Message.text("user", "read external data")],
            ),
        )
    )
    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_taint_prior",
                messages=[Message.text("user", "send outbound action")],
            ),
        )
    )

    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in initial_events)
    approval_event = next(
        event for event in resume_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval = approval_event.payload["approval"]
    assert approval["tool_name"] == "side_effect"
    assert approval["metadata"]["policy"] == "taint_aware"
    assert approval["metadata"]["matched_taint_labels"] == ["external_email"]
    assert side_effect_tool.calls == []


def test_cayu_app_taint_policy_applies_within_same_tool_round():
    echo_tool = EchoTool()
    side_effect_tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_read",
                    name="echo",
                    arguments={"text": "external email body"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_send",
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
        tools=[echo_tool, side_effect_tool],
        tool_policy=TaintAwareToolPolicy(
            taint_sources={"echo": ["external_email"]},
            protected_tools={"side_effect": ["external_email"]},
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_taint_same_round",
                messages=[Message.text("user", "read then send")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.TOOL_CALL_APPROVAL_REQUESTED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    approval_event = next(
        event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval = approval_event.payload["approval"]
    assert approval["tool_name"] == "side_effect"
    assert approval["metadata"]["policy"] == "taint_aware"
    assert approval["metadata"]["matched_taint_labels"] == ["external_email"]
    assert side_effect_tool.calls == []


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
    assert tool.contexts[0].metadata["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="sess_policy_resume_metadata",
        tool_round_id=events[3].payload["tool_round_id"],
        tool_call_id="call_1",
    )
    assert tool.contexts[0].metadata == {
        "resume": {"id": "resume_1"},
        "tool_call_id": "call_1",
        "idempotency_key": tool.contexts[0].metadata["idempotency_key"],
        "tool_effect": "external",
    }
    tool.contexts[0].metadata["resume"]["id"] = "tool-mutated"
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
        EventType.TURN_COMPLETED,
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
            # Messages are frozen; policies cannot corrupt the transcript.
            with pytest.raises(ValidationError):
                request.messages[0].content[0].text = "mutated"  # type: ignore[misc]
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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


def test_model_compactor_ignores_thinking_events():
    # A reasoning-capable compaction provider may emit THINKING events; compaction must
    # ignore them (the summary is built from text) rather than crash on an unknown type.
    provider = FakeProvider(
        [
            ModelStreamEvent.thinking("internal reasoning", provider_state={"signature": "S"}),
            ModelStreamEvent.text_delta("summary "),
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"usage": {"output_tokens": 2}}),
        ]
    )
    compactor = ModelCompactor(provider=provider, model="summary-model")

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert result.summary == "summary done"


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


def test_model_compactor_reports_model_completed_payload_with_usage_metrics():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model-2026-01-01",
                    "usage": {"input_tokens": 40, "output_tokens": 8},
                    "provider_state": {"opaque": True},
                }
            ),
        ]
    )
    compactor = ModelCompactor(provider=provider, model="summary-model")

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert result.model_completed_payloads == [
        {
            "model": "summary-model-2026-01-01",
            "usage": {"input_tokens": 40, "output_tokens": 8},
            "provider_name": "fake",
            "requested_model": "summary-model",
            "purpose": "context_compaction",
            "compactor": "ModelCompactor",
            "usage_metrics": {
                "provider_name": "fake",
                "requested_model": "summary-model",
                "model": "summary-model-2026-01-01",
                "input_tokens": 40,
                "output_tokens": 8,
                "total_tokens": 48,
                "reasoning_output_tokens": 0,
                "cache": {
                    "read_tokens": 0,
                    "write_tokens": 0,
                    "cached_input_tokens": 0,
                    "uncached_input_tokens": 40,
                },
            },
        }
    ]


class FlakyCompactionProvider(ModelProvider):
    name = "flaky"

    def __init__(self, *, failures: int, error: ModelProviderError) -> None:
        self.failures = failures
        self.error = error
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) <= self.failures:
            raise self.error
        yield ModelStreamEvent.text_delta("recovered summary")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def test_model_compactor_retries_transient_provider_errors():
    provider = FlakyCompactionProvider(
        failures=1,
        error=ModelProviderError(
            "provider overloaded",
            provider="flaky",
            status_code=503,
            retryable=True,
        ),
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
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

    assert result.summary == "recovered summary"
    assert len(provider.requests) == 2


def test_model_compactor_does_not_retry_by_default():
    provider = FlakyCompactionProvider(
        failures=1,
        error=ModelProviderError(
            "provider overloaded",
            provider="flaky",
            status_code=503,
            retryable=True,
        ),
    )
    compactor = ModelCompactor(provider=provider, model="summary-model")

    with pytest.raises(ModelProviderError, match="provider overloaded"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )
    assert len(provider.requests) == 1


def test_model_compactor_does_not_retry_terminal_provider_errors():
    provider = FlakyCompactionProvider(
        failures=1,
        error=ModelProviderError(
            "invalid request",
            provider="flaky",
            status_code=400,
            retryable=False,
        ),
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
    )

    with pytest.raises(ModelProviderError, match="invalid request"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )
    assert len(provider.requests) == 1


def test_transcript_digest_compactor_preserves_previous_summary_when_clipping():
    # A large batch of newly compacted messages must not clip the accumulated
    # previous summary out of the front of the combined text.
    compactor = TranscriptDigestCompactor(max_summary_chars=400)
    previous_summary = "decision: ship plan A"

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("user", f"filler {index} " + "x" * 40) for index in range(30)
                ],
                existing_summary=previous_summary,
            )
        )
    )

    assert len(result.summary) <= 400
    assert f"Previous summary:\n{previous_summary}" in result.summary
    assert "Newly compacted transcript:\n[clipped to latest content]" in result.summary
    assert result.summary.endswith("filler 29 " + "x" * 40)


def test_transcript_digest_compactor_budgets_both_oversized_sections():
    compactor = TranscriptDigestCompactor(max_summary_chars=400)
    previous_summary = ("earlier decision " * 40).strip()

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "latest work " * 50)],
                existing_summary=previous_summary,
            )
        )
    )

    assert len(result.summary) <= 400
    previous_section, digest_section = result.summary.split("\n\n")
    assert previous_section.startswith("Previous summary:\n[clipped to latest content]")
    assert previous_section.endswith("earlier decision")
    assert digest_section.startswith("Newly compacted transcript:\n[clipped to latest content]")
    assert len(previous_section) == 199
    assert len(digest_section) == 199


def test_transcript_digest_compactor_keeps_small_previous_summary_unclipped():
    compactor = TranscriptDigestCompactor()
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
    assert result.summary == (
        "Previous summary:\nprevious context\n\n"
        "Newly compacted transcript:\nuser: old request\nassistant: old answer"
    )
    assert compactor.max_summary_chars == 8000


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
        "requested_model": "fake-model",
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


def test_context_policy_receives_previous_actual_context_usage_on_next_call():
    class UsageAwarePolicy(ContextPolicy):
        def __init__(self) -> None:
            self.requests: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.requests.append(request)
            if (
                request.context_usage.last_input_tokens is not None
                and request.context_usage.last_input_tokens >= 50
            ):
                return [request.messages[-1]]
            return request.messages

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 1}}),
            ],
        ]
    )
    policy = UsageAwarePolicy()
    app = CayuApp()
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
                session_id="usage_context_policy",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_context_policy",
                messages=[Message.text("user", "second request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_context_policy",
                messages=[Message.text("user", "third request")],
            ),
        )
    )

    assert len(policy.requests) == 3
    assert policy.requests[0].context_usage.last_input_tokens is None
    assert policy.requests[1].context_usage.last_input_tokens == 100
    assert policy.requests[1].context_usage.last_output_tokens == 4
    assert policy.requests[1].context_usage.last_total_tokens == 104
    assert policy.requests[1].context_usage.last_provider_name == "fake"
    assert policy.requests[1].context_usage.last_model == "fake-model"
    assert policy.requests[2].context_usage.last_input_tokens == 20
    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    assert len(provider.requests[2].messages) > 1
    assert provider.requests[2].messages[-1].content[0].text == "third request"


def test_context_usage_state_uses_latest_completed_event_query():
    class TrackingStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.event_queries: list[EventQuery] = []

        async def query_events(self, query: EventQuery | None = None):
            if query is not None:
                self.event_queries.append(query)
            return await super().query_events(query)

    tracking_store = TrackingStore()
    session_id = "usage_context_policy_latest_event"

    async def run():
        await tracking_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await tracking_store.append_events(
            session_id,
            [
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    payload={
                        "model": "fake-model",
                        "usage": {"input_tokens": index, "output_tokens": 1},
                    },
                )
                for index in range(1, 106)
            ],
        )
        return await runtime_app_module._context_usage_state_for_session(
            session_store=tracking_store,
            session_id=session_id,
        )

    context_usage = asyncio.run(run())

    assert context_usage.last_input_tokens == 105
    assert context_usage.last_output_tokens == 1
    assert context_usage.last_total_tokens == 106
    assert context_usage.last_provider_name is None
    assert context_usage.last_model == "fake-model"
    assert len(tracking_store.event_queries) == 1
    assert tracking_store.event_queries[0].limit == 1
    assert tracking_store.event_queries[0].order_by.value == "sequence_desc"


def test_observed_delta_context_estimator_estimates_only_messages_after_cursor():
    estimator = ObservedDeltaContextEstimator(chars_per_token=4, json_chars_per_token=2)
    usage = ContextUsageState(last_input_tokens=40, last_transcript_cursor=2)
    messages = [
        Message.text("user", "already counted"),
        Message.text("assistant", "already counted answer"),
        Message.text("user", "abcd" * 5),
        Message.tool_result(
            tool_call_id="call_1",
            tool_name="lookup",
            content="xy" * 5,
            structured={"ok": True},
        ),
    ]

    estimate = estimator.estimate(usage=usage, messages=messages)

    assert isinstance(estimate, ContextPressureEstimate)
    assert estimate.method == "observed_plus_estimated_delta"
    assert estimate.confidence == "estimated"
    assert estimate.observed_context_input_tokens == 40
    assert estimate.estimated_delta_input_tokens == 8
    assert estimate.estimated_context_input_tokens == 48
    assert estimate.reserved_output_tokens == 0
    assert estimate.estimated_context_window_tokens == 48
    assert estimate.anchor_transcript_cursor == 2
    assert estimate.current_transcript_cursor == 4
    assert estimate.estimated_message_count == 2


def test_observed_delta_context_estimator_can_estimate_full_request_overhead():
    estimator = ObservedDeltaContextEstimator(chars_per_token=4, json_chars_per_token=2)
    usage = ContextUsageState(last_input_tokens=40, last_transcript_cursor=2)
    messages = [Message.text("user", "abcd" * 5)]
    overhead = ContextPressureOverhead(
        tools=[
            {
                "name": "lookup",
                "description": "x" * 40,
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ],
        structured_output_instruction="return structured output",
        request_options={
            "step": 2,
            "structured_output": {"name": "answer"},
            "openai": {"tool_choice": "auto"},
        },
    )

    estimate = estimator.estimate_full_request(
        usage=usage,
        messages=messages,
        overhead=overhead,
    )

    assert estimate.method == "local_full_request_estimate"
    assert estimate.observed_context_input_tokens == 40
    assert estimate.estimated_message_input_tokens == 5
    assert estimate.estimated_tool_schema_input_tokens > 0
    assert estimate.estimated_structured_output_input_tokens > 0
    assert estimate.estimated_request_options_input_tokens > 0
    assert estimate.estimated_context_input_tokens == (
        estimate.estimated_message_input_tokens
        + estimate.estimated_tool_schema_input_tokens
        + estimate.estimated_structured_output_input_tokens
        + estimate.estimated_request_options_input_tokens
    )
    assert estimate.estimated_context_window_tokens == estimate.estimated_context_input_tokens


def test_observed_delta_context_estimator_counts_only_overhead_delta_after_anchor():
    estimator = ObservedDeltaContextEstimator(chars_per_token=4, json_chars_per_token=2)
    overhead = ContextPressureOverhead(
        tools=[
            {
                "name": "lookup",
                "description": "x" * 80,
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ],
        request_options={"openai": {"tool_choice": "auto"}},
    )
    overhead_tokens = estimator.estimate_full_request(
        usage=ContextUsageState(),
        messages=[Message.text("user", "seed")],
        overhead=overhead,
    ).estimated_request_overhead_input_tokens
    usage = ContextUsageState(
        last_input_tokens=100,
        last_transcript_cursor=1,
        last_context_overhead_input_tokens=overhead_tokens,
    )
    messages = [
        Message.text("user", "already counted"),
        Message.text("user", "abcd" * 5),
    ]

    stable = estimator.estimate_anchored_request(
        usage=usage,
        messages=messages,
        overhead=overhead,
    )
    increased = estimator.estimate_anchored_request(
        usage=usage.model_copy(update={"last_context_overhead_input_tokens": overhead_tokens - 5}),
        messages=messages,
        overhead=overhead,
    )
    decreased = estimator.estimate_anchored_request(
        usage=usage.model_copy(update={"last_context_overhead_input_tokens": overhead_tokens + 5}),
        messages=messages,
        overhead=overhead,
    )

    assert stable.estimated_message_input_tokens == 5
    assert stable.estimated_request_overhead_input_tokens == overhead_tokens
    assert stable.previous_request_overhead_input_tokens == overhead_tokens
    assert stable.estimated_request_overhead_delta_tokens == 0
    assert stable.estimated_delta_input_tokens == 5
    assert stable.estimated_context_input_tokens == 105

    assert increased.estimated_request_overhead_delta_tokens == 5
    assert increased.estimated_context_input_tokens == 110
    assert decreased.estimated_request_overhead_delta_tokens == -5
    assert decreased.estimated_context_input_tokens == 100


def test_observed_delta_context_estimator_uses_provider_specific_image_floor():
    estimator = ObservedDeltaContextEstimator(chars_per_token=5)
    from cayu.core.messages import FilePart

    user_image_message = Message(
        role="user",
        content=[
            TextPart(text="look"),
            FilePart(
                attachment=file_attachment(
                    artifact_id="img_user",
                    kind="image",
                    filename="user.png",
                    content_type="image/png",
                    size_bytes=68,
                )
            ),
        ],
    )
    image_message = Message.tool_result(
        tool_call_id="call_image",
        tool_name="image_tool",
        content="image result",
        artifacts=[
            file_attachment(
                artifact_id="img_1",
                kind="image",
                filename="pixel.png",
                content_type="image/png",
                size_bytes=68,
            )
        ],
    )
    document_message = Message.tool_result(
        tool_call_id="call_document",
        tool_name="document_tool",
        content="document result",
        artifacts=[
            file_attachment(
                artifact_id="doc_1",
                kind="document",
                filename="tiny.pdf",
                content_type="application/pdf",
                size_bytes=120,
            )
        ],
    )

    generic_tokens = estimator.estimate_full_request(
        usage=ContextUsageState(),
        messages=[user_image_message, image_message, document_message],
        overhead=ContextPressureOverhead(image_min_tokens=32),
    )
    anthropic_tokens = estimator.estimate_full_request(
        usage=ContextUsageState(),
        messages=[user_image_message, image_message, document_message],
        overhead=ContextPressureOverhead(
            image_min_tokens=100,
            document_min_tokens=1800,
        ),
    )

    assert generic_tokens.estimated_attachment_input_tokens < 200
    assert anthropic_tokens.estimated_attachment_input_tokens >= 2000
    assert (
        anthropic_tokens.estimated_attachment_input_tokens
        > generic_tokens.estimated_attachment_input_tokens
    )


def test_observed_delta_context_estimator_uses_adaptive_text_density():
    prose = (
        "Remote sandbox Git authentication should use a brokered Git HTTP proxy. "
        "Credentials stay outside the sandbox boundary. " * 100
    )
    dense_json = (
        '{"path":"src/cayu/runtime/context.py","line":123,'
        '"value":"x_y-z.abc/def","ok":true}\n' * 100
    )
    pretty_json = (
        '{\n  "path": "src/cayu/runtime/context.py",\n  "line": 123,\n'
        '  "value": "x_y-z.abc/def",\n  "ok": true\n}\n' * 100
    )
    logs = (
        "2026-07-02T12:34:56Z INFO session=sess_123 step=4 "
        "tool=search_knowledge latency_ms=42 status=ok input_tokens=14947\n" * 100
    )
    csv = (
        "timestamp,session_id,agent,model,input_tokens,output_tokens,status\n"
        "2026-07-02T12:34:56Z,sess_123,assistant,gpt-5.5,14947,7,ok\n" * 100
    )
    code = (
        "def estimate_context_pressure(request):\n"
        "    if request.count_input_tokens is None:\n"
        "        return estimate\n"
        "    return request.count_input_tokens(messages)\n" * 100
    )
    markdown = (
        "# Heading\n\n- `code/path.py`: explanation with punctuation, numbers 12345, "
        'and JSON {"ok": true}.\n' * 100
    )
    estimator = ObservedDeltaContextEstimator(chars_per_token=5)

    prose_tokens = estimator.estimate_message_tokens(Message.text("user", prose))
    dense_json_tokens = estimator.estimate_message_tokens(Message.text("user", dense_json))
    pretty_json_tokens = estimator.estimate_message_tokens(Message.text("user", pretty_json))
    logs_tokens = estimator.estimate_message_tokens(Message.text("user", logs))
    csv_tokens = estimator.estimate_message_tokens(Message.text("user", csv))
    code_tokens = estimator.estimate_message_tokens(Message.text("user", code))
    markdown_tokens = estimator.estimate_message_tokens(Message.text("user", markdown))

    assert prose_tokens == math.ceil(len(prose) / 5)
    assert dense_json_tokens == math.ceil(len(dense_json) / 3)
    assert pretty_json_tokens == math.ceil(len(pretty_json) / 3)
    assert logs_tokens == math.ceil(len(logs) / 2.75)
    assert csv_tokens == math.ceil(len(csv) / 2.5)
    assert code_tokens == math.ceil(len(code) / 5)
    assert markdown_tokens == math.ceil(len(markdown) / 3.75)


def test_observed_delta_context_estimator_uses_tool_schema_density_override():
    estimator = ObservedDeltaContextEstimator(chars_per_token=5, json_chars_per_token=5)
    tools = [
        {
            "name": "heavy",
            "description": "Search many records. " * 200,
            "input_schema": {
                "type": "object",
                "properties": {
                    f"field_{index}": {
                        "type": "string",
                        "description": "Filter value " * 20,
                    }
                    for index in range(20)
                },
            },
        }
    ]

    default_tokens = estimator.estimate_tool_schema_tokens(tools)
    openai_profile_tokens = estimator.estimate_tool_schema_tokens(
        tools,
        chars_per_token=6,
    )

    assert default_tokens > openai_profile_tokens
    assert openai_profile_tokens == math.ceil(
        len(json.dumps(tools[0], ensure_ascii=False, separators=(",", ":"), sort_keys=True)) / 6
    )


def test_context_policy_receives_estimated_context_pressure_on_next_call():
    class PressureAwarePolicy(ContextPolicy):
        def __init__(self) -> None:
            self.requests: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.requests.append(request)
            return request.messages

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 40, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
        ]
    )
    policy = PressureAwarePolicy()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_pressure_estimate",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="context_pressure_estimate",
                messages=[Message.text("user", "abcd" * 4)],
            ),
        )
    )

    first_completed = next(
        event for event in first_events if event.type == EventType.MODEL_COMPLETED
    )
    assert first_completed.payload["transcript_cursor"] == 2
    assert len(policy.requests) == 2
    pressure = policy.requests[1].context_usage.input_pressure
    assert pressure is not None
    assert pressure.observed_context_input_tokens == 40
    assert pressure.estimated_delta_input_tokens == 4
    assert pressure.estimated_context_input_tokens == 44
    assert pressure.estimated_context_window_tokens == 44
    assert pressure.anchor_transcript_cursor == 2
    assert pressure.current_transcript_cursor == 3
    assert pressure.estimated_message_count == 1


def test_context_policy_input_pressure_uses_provider_profile_image_floor():
    class ImageFloorProvider(FakeProvider):
        @property
        def context_pressure_profile(self) -> ModelContextPressureProfile:
            return ModelContextPressureProfile(
                image_min_tokens=100,
                document_min_tokens=1800,
            )

    class PressureAwarePolicy(ContextPolicy):
        def __init__(self) -> None:
            self.requests: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.requests.append(request)
            return request.messages

    provider = ImageFloorProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 40, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 50, "output_tokens": 2}}),
            ],
        ]
    )
    policy = PressureAwarePolicy()
    app = CayuApp()
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
                session_id="context_pressure_provider_profile",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="context_pressure_provider_profile",
                messages=[
                    Message.tool_result(
                        tool_call_id="call_image",
                        tool_name="image_tool",
                        content="image result",
                        artifacts=[
                            file_attachment(
                                artifact_id="img_1",
                                kind="image",
                                filename="pixel.png",
                                content_type="image/png",
                                size_bytes=68,
                            )
                        ],
                    ),
                    Message.tool_result(
                        tool_call_id="call_document",
                        tool_name="document_tool",
                        content="document result",
                        artifacts=[
                            file_attachment(
                                artifact_id="doc_1",
                                kind="document",
                                filename="tiny.pdf",
                                content_type="application/pdf",
                                size_bytes=120,
                            )
                        ],
                    ),
                ],
            ),
        )
    )

    assert len(policy.requests) == 2
    pressure = policy.requests[1].context_usage.input_pressure
    assert pressure is not None
    assert pressure.estimated_attachment_input_tokens >= 1900


def test_usage_triggered_context_policy_can_trigger_on_estimated_context_pressure():
    class LargeSchemaTool(Tool):
        spec = ToolSpec(
            name="large_schema",
            description="x" * 200,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "y" * 200,
                    }
                },
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 40, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=45,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[LargeSchemaTool()],
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_estimated_pressure",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_estimated_pressure",
                messages=[Message.text("user", "abcd" * 5)],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "abcd" * 5
    checkpoint_events = [
        event for event in first_events if event.type == EventType.SESSION_CHECKPOINTED
    ]
    assert len(checkpoint_events) == 1
    payload = checkpoint_events[0].payload
    assert payload["checkpoint"] == "usage_triggered_context"
    assert payload["min_input_tokens"] is None
    assert payload["trigger_estimated_context_tokens"] == 45
    assert payload["reserved_output_tokens"] == 0
    assert payload["min_total_tokens"] is None
    assert payload["last_input_tokens"] is None
    assert payload["last_total_tokens"] is None
    assert payload["last_transcript_cursor"] is None
    assert payload["estimated_context_input_tokens"] >= 45
    assert payload["estimated_context_window_tokens"] >= 45
    assert payload["estimated_delta_input_tokens"] == payload["estimated_context_input_tokens"]
    assert not [event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED]


def test_usage_triggered_context_policy_does_not_double_count_stable_overhead():
    class LargeSchemaTool(Tool):
        spec = ToolSpec(
            name="large_schema",
            description="x" * 1000,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "y" * 1000,
                    }
                },
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 1000, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 1010, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=1250,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[LargeSchemaTool()],
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_stable_overhead",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_stable_overhead",
                messages=[Message.text("user", "abcd" * 5)],
            ),
        )
    )

    first_completed = next(
        event for event in first_events if event.type == EventType.MODEL_COMPLETED
    )
    assert (
        first_completed.payload["context_pressure"]["estimated_request_overhead_input_tokens"] > 300
    )
    assert len(provider.requests[1].messages) == 3
    assert not [event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED]


def test_usage_triggered_context_policy_full_estimates_same_length_projection():
    class ReplacingPolicy(ContextPolicy):
        async def build(self, request: ContextRequest) -> list[Message]:
            return [Message.text("user", "x" * 1000) for _ in request.messages]

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 1010, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=ReplacingPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=450,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_same_length_projection",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_same_length_projection",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    assert not [event for event in first_events if event.type == EventType.SESSION_CHECKPOINTED]
    assert len(provider.requests[0].messages) == 1
    assert provider.requests[0].messages[0].content[0].text == "x" * 1000
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    checkpoint_event = next(
        event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED
    )
    assert checkpoint_event.payload["estimated_context_input_tokens"] >= 300


def test_usage_triggered_context_policy_estimates_base_policy_model_facing_context():
    class InjectingPolicy(ContextPolicy):
        async def build(self, request: ContextRequest) -> list[Message]:
            return [
                *request.messages,
                Message.text("user", "injected context " + ("x" * 400)),
            ]

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 40, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=InjectingPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=80,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_estimated_policy_context",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_estimated_policy_context",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    checkpoint_event = next(
        event for event in first_events if event.type == EventType.SESSION_CHECKPOINTED
    )
    assert checkpoint_event.payload["estimated_context_input_tokens"] >= 80
    assert not [event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED]


def test_usage_triggered_context_policy_includes_reserved_output_in_estimated_trigger():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=94,
        reserved_output_tokens=90,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_reserved_output",
                messages=[Message.text("user", "abcd" * 5)],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    checkpoint_event = next(
        event for event in first_events if event.type == EventType.SESSION_CHECKPOINTED
    )
    assert checkpoint_event.payload["trigger_estimated_context_tokens"] == 94
    assert checkpoint_event.payload["reserved_output_tokens"] == 90
    assert checkpoint_event.payload["estimated_context_input_tokens"] < 100
    assert checkpoint_event.payload["estimated_context_window_tokens"] >= 94


def test_usage_triggered_context_policy_provider_count_can_prevent_estimated_false_positive():
    class LargeSchemaTool(Tool):
        spec = ToolSpec(
            name="large_schema",
            description="x" * 400,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "y" * 400,
                    }
                },
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="")

    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
        ],
        count_result=InputTokenCountResult(
            input_tokens=10,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        ),
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=45,
        verify_estimate_with_provider_count=True,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[LargeSchemaTool()],
        context_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_provider_count_false_positive",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    assert len(provider.count_requests) == 1
    assert provider.count_requests[0].tools
    assert not [event for event in events if event.type == EventType.SESSION_CHECKPOINTED]


def test_usage_triggered_context_policy_provider_count_is_not_called_when_far_from_threshold():
    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
        ],
        count_result=InputTokenCountResult(
            input_tokens=10,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        ),
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=10_000,
        verify_estimate_with_provider_count=True,
    )
    app = CayuApp()
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
                session_id="usage_triggered_provider_count_far",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    assert provider.count_requests == []


def test_usage_triggered_context_policy_large_delta_can_request_provider_count():
    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
        ],
        count_result=InputTokenCountResult(
            input_tokens=12_000,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        ),
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=10_000,
        verify_estimate_with_provider_count=True,
        provider_count_min_delta_tokens=20,
    )
    app = CayuApp()
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
                session_id="usage_triggered_provider_count_large_delta",
                messages=[Message.text("user", "x" * 200)],
            ),
        )
    )

    assert len(provider.count_requests) == 1
    checkpoint_event = next(
        event for event in events if event.type == EventType.SESSION_CHECKPOINTED
    )
    assert checkpoint_event.payload["provider_count_input_tokens"] == 12_000
    assert checkpoint_event.payload["provider_count_context_window_tokens"] == 12_000


def test_usage_triggered_context_policy_stays_triggered_after_previous_actual_usage():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 1}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        min_input_tokens=50,
    )
    app = CayuApp()
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
                session_id="usage_triggered_policy",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy",
                messages=[Message.text("user", "second request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy",
                messages=[Message.text("user", "third request")],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    assert len(provider.requests[2].messages) == 1
    assert provider.requests[2].messages[0].content[0].text == "third request"
    checkpoint_events = [
        event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED
    ]
    assert len(checkpoint_events) == 1
    assert checkpoint_events[0].payload == {
        "checkpoint": "usage_triggered_context",
        "min_input_tokens": 50,
        "min_total_tokens": None,
        "last_input_tokens": 100,
        "last_total_tokens": 104,
    }
    checkpoint = asyncio.run(app.session_store.load_checkpoint("usage_triggered_policy"))
    assert checkpoint == {
        "usage_triggered_context": {
            "version": 1,
            "min_input_tokens": 50,
            "min_total_tokens": None,
            "last_input_tokens": 100,
            "last_total_tokens": 104,
        }
    }


def test_usage_triggered_context_policy_preserves_marker_with_checkpoint_policy():
    class ReplacingCheckpointPolicy(RuntimeManagedContextPolicy):
        def __init__(self) -> None:
            self.calls = 0

        async def build_with_checkpoint(
            self,
            request: ContextRequest,
            *,
            checkpoint: dict[str, Any] | None,
        ) -> ContextBuildResult:
            self.calls += 1
            return ContextBuildResult(
                messages=[request.messages[-1]],
                checkpoint={"triggered_policy": {"calls": self.calls}},
                checkpoint_event_payload={
                    "checkpoint": "triggered_policy",
                    "calls": self.calls,
                },
            )

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 1}}),
            ],
        ]
    )
    triggered_policy = ReplacingCheckpointPolicy()
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=triggered_policy,
        min_input_tokens=50,
    )
    app = CayuApp()
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
                session_id="usage_triggered_policy_checkpoint_merge",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy_checkpoint_merge",
                messages=[Message.text("user", "second request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy_checkpoint_merge",
                messages=[Message.text("user", "third request")],
            ),
        )
    )

    assert triggered_policy.calls == 2
    assert len(provider.requests[1].messages) == 1
    assert len(provider.requests[2].messages) == 1
    checkpoint = asyncio.run(
        app.session_store.load_checkpoint("usage_triggered_policy_checkpoint_merge")
    )
    assert checkpoint == {
        "triggered_policy": {"calls": 2},
        "usage_triggered_context": {
            "version": 1,
            "min_input_tokens": 50,
            "min_total_tokens": None,
            "last_input_tokens": 100,
            "last_total_tokens": 104,
        },
    }


def test_usage_triggered_context_policy_can_be_last_call_only():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 1}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        min_input_tokens=50,
        sticky=False,
    )
    app = CayuApp()
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
                session_id="usage_triggered_policy_last_call_only",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy_last_call_only",
                messages=[Message.text("user", "second request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy_last_call_only",
                messages=[Message.text("user", "third request")],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    assert len(provider.requests[2].messages) > 1
    assert provider.requests[2].messages[-1].content[0].text == "third request"
    checkpoint = asyncio.run(
        app.session_store.load_checkpoint("usage_triggered_policy_last_call_only")
    )
    assert checkpoint is None


def test_usage_triggered_context_policy_supports_total_token_threshold():
    policy = UsageTriggeredContextPolicy(
        base_policy=MessageWindowContextPolicy(max_messages=3),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        min_total_tokens=100,
    )
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current"),
    ]

    async def build_context() -> list[Message]:
        return await policy.build(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
                context_usage=ContextUsageState(last_total_tokens=100),
            )
        )

    context = asyncio.run(build_context())

    assert [message.role for message in context] == ["system", "user"]
    assert [message.content[0].text for message in context] == ["You are careful.", "current"]


def test_usage_triggered_context_policy_requires_a_threshold():
    with pytest.raises(ValueError, match="At least one usage threshold"):
        UsageTriggeredContextPolicy(
            triggered_policy=MessageWindowContextPolicy(max_messages=1),
        )


def test_usage_triggered_context_policy_rejects_runtime_managed_estimated_base_policy():
    with pytest.raises(ValueError, match="side-effect-free base_policy"):
        UsageTriggeredContextPolicy(
            base_policy=CheckpointCompactionContextPolicy(),
            triggered_policy=MessageWindowContextPolicy(max_messages=1),
            trigger_estimated_context_tokens=100,
        )


def test_usage_triggered_context_policy_requires_bool_sticky():
    with pytest.raises(TypeError, match="sticky must be a bool"):
        UsageTriggeredContextPolicy(
            triggered_policy=MessageWindowContextPolicy(max_messages=1),
            min_input_tokens=1,
            sticky=1,
        )


def test_context_overflow_policy_rebuilds_context_and_retries_once():
    provider = ContextOverflowProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_recovery",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    assert [request.messages[0].content[0].text for request in provider.requests] == [
        "old request",
        "new request",
    ]
    assert len(provider.requests[0].messages) == 2
    assert len(provider.requests[1].messages) == 1
    assert [
        event.type
        for event in events
        if event.type
        in {
            EventType.CONTEXT_OVERFLOW_DETECTED,
            EventType.CONTEXT_OVERFLOW_RECOVERING,
            EventType.CONTEXT_OVERFLOW_FAILED,
            EventType.SESSION_COMPLETED,
        }
    ] == [
        EventType.CONTEXT_OVERFLOW_DETECTED,
        EventType.CONTEXT_OVERFLOW_RECOVERING,
        EventType.SESSION_COMPLETED,
    ]
    detected = next(event for event in events if event.type == EventType.CONTEXT_OVERFLOW_DETECTED)
    assert detected.payload == {
        "step": 1,
        "phase": "initial",
        "error": "context too large",
        "error_type": "ModelContextOverflowError",
        "provider": "overflow",
        "original_message_count": 2,
        "status_code": 400,
        "provider_error_code": "context_length_exceeded",
    }
    recovering = next(
        event for event in events if event.type == EventType.CONTEXT_OVERFLOW_RECOVERING
    )
    assert recovering.payload == {
        "step": 1,
        "original_message_count": 2,
        "recovery_message_count": 1,
        "policy": "RecentTurnsContextPolicy",
    }
    transcript = asyncio.run(app.session_store.load_transcript("context_overflow_recovery"))
    assert [message.content[0].text for message in transcript if message.role == "user"] == [
        "old request",
        "new request",
    ]


def test_context_overflow_recovery_from_error_stream_event():
    provider = EventFlattenedOverflowProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_event_recovery",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    assert len(provider.requests) == 2
    assert len(provider.requests[1].messages) == 1
    # The typed overflow is rehydrated before the generic MODEL_ERROR/retry
    # path, so recovery runs exactly like the raised-exception contract.
    assert EventType.MODEL_ERROR not in [event.type for event in events]
    assert [
        event.type
        for event in events
        if event.type
        in {
            EventType.CONTEXT_OVERFLOW_DETECTED,
            EventType.CONTEXT_OVERFLOW_RECOVERING,
            EventType.CONTEXT_OVERFLOW_FAILED,
            EventType.SESSION_COMPLETED,
        }
    ] == [
        EventType.CONTEXT_OVERFLOW_DETECTED,
        EventType.CONTEXT_OVERFLOW_RECOVERING,
        EventType.SESSION_COMPLETED,
    ]
    detected = next(event for event in events if event.type == EventType.CONTEXT_OVERFLOW_DETECTED)
    assert detected.payload == {
        "step": 1,
        "phase": "initial",
        "error": "context too large",
        "error_type": "ModelContextOverflowError",
        "provider": "overflow-event",
        "original_message_count": 2,
        "status_code": 400,
        "provider_error_code": "context_length_exceeded",
    }


def test_context_overflow_error_stream_event_without_policy_fails_typed():
    provider = EventFlattenedOverflowProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_event_no_policy",
                messages=[Message.text("user", "too much")],
            ),
        )
    )

    assert len(provider.requests) == 1
    failed = events[-1]
    assert failed.type == EventType.SESSION_FAILED
    assert failed.payload["error"] == "context too large"
    assert failed.payload["error_type"] == "ModelContextOverflowError"


def test_context_overflow_policy_can_checkpoint_compaction_before_retry():
    provider = ContextOverflowProvider()
    compactor = RecordingCompactor()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=1,
            compact_after_messages=1,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_compaction_recovery",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    assert len(compactor.requests) == 1
    assert [message.content[0].text for message in compactor.requests[0].messages] == [
        "old request"
    ]
    assert len(provider.requests) == 2
    assert len(provider.requests[1].messages) == 2
    assert (
        provider.requests[1]
        .messages[0]
        .content[0]
        .text.startswith("Previous session context summary:")
    )
    assert provider.requests[1].messages[1].content[0].text == "new request"
    assert EventType.SESSION_CHECKPOINTED in [event.type for event in events]
    checkpoint = asyncio.run(
        app.session_store.load_checkpoint("context_overflow_compaction_recovery")
    )
    assert checkpoint == {
        "context_compaction": {
            "version": 1,
            "summary": "old request",
            "compacted_transcript_cursor": 1,
            "metadata": {"request_count": 1},
        }
    }


def test_context_overflow_without_policy_fails_without_retry():
    provider = ContextOverflowProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_no_policy",
                messages=[Message.text("user", "too much")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert EventType.CONTEXT_OVERFLOW_DETECTED not in [event.type for event in events]
    failed = events[-1]
    assert failed.type == EventType.SESSION_FAILED
    assert failed.payload["error"] == "context too large"
    assert failed.payload["error_type"] == "ModelContextOverflowError"


def test_context_overflow_policy_fails_cleanly_when_recovery_overflows():
    provider = ContextOverflowProvider(overflow_requests=2)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_overflow_policy=MessageWindowContextPolicy(max_messages=1),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_overflow_recovery_failed",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )

    assert len(provider.requests) == 2
    assert [
        event.type
        for event in events
        if event.type
        in {
            EventType.CONTEXT_OVERFLOW_DETECTED,
            EventType.CONTEXT_OVERFLOW_RECOVERING,
            EventType.CONTEXT_OVERFLOW_FAILED,
        }
    ] == [
        EventType.CONTEXT_OVERFLOW_DETECTED,
        EventType.CONTEXT_OVERFLOW_RECOVERING,
        EventType.CONTEXT_OVERFLOW_FAILED,
    ]
    failed = next(event for event in events if event.type == EventType.CONTEXT_OVERFLOW_FAILED)
    assert failed.payload == {
        "step": 1,
        "phase": "recovery",
        "error": "context too large",
        "error_type": "ModelContextOverflowError",
        "provider": "overflow",
        "original_message_count": 2,
        "status_code": 400,
        "provider_error_code": "context_length_exceeded",
        "recovery_message_count": 1,
    }
    assert events[-1].type == EventType.SESSION_FAILED


def test_register_agent_rejects_invalid_context_overflow_policy():
    app = CayuApp()
    with pytest.raises(TypeError, match="context_overflow_policy must be a ContextPolicy"):
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_overflow_policy=object(),  # type: ignore[arg-type]
        )


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
    # One query carrying all usage-bearing types: per-type queries sharing a
    # watermark can skip events appended between them (issue #101).
    assert len(store.event_queries) == 1
    assert store.event_queries[0].event_type is None
    assert [str(event_type) for event_type in store.event_queries[0].event_types] == [
        "model.completed",
        "tool.call.started",
    ]


def test_cayu_app_query_all_event_records_preserves_filters():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)

    async def run() -> None:
        for session_id in ("query_all_a", "query_all_b"):
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
        await store.append_event(
            "query_all_a",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="query_all_a",
                timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            ),
        )
        await store.append_event(
            "query_all_b",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="query_all_b",
                timestamp=datetime(2026, 1, 1, 13, 0, tzinfo=UTC),
            ),
        )

        records = await app._query_all_event_records(
            EventQuery(
                session_ids=("query_all_b",),
                event_type=EventType.MODEL_COMPLETED,
                since=datetime(2026, 1, 1, 12, 30, tzinfo=UTC),
                until=datetime(2026, 1, 1, 13, 30, tzinfo=UTC),
                limit=1,
            )
        )
        assert [record.event.session_id for record in records] == ["query_all_b"]

        # The paginator rebuilds the query field-by-field; a dropped
        # `event_types` would silently widen usage reads (the delta leaks in).
        await store.append_event(
            "query_all_b",
            Event(
                type=EventType.MODEL_TEXT_DELTA,
                session_id="query_all_b",
                payload={"delta": "noise"},
            ),
        )
        multi_type_records = await app._query_all_event_records(
            EventQuery(
                session_ids=("query_all_b",),
                event_types=(EventType.MODEL_COMPLETED, EventType.TOOL_CALL_STARTED),
                limit=1,
            )
        )
        assert [record.event.type for record in multi_type_records] == [EventType.MODEL_COMPLETED]

    asyncio.run(run())


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
        EventType.TURN_COMPLETED,
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
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
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
        "finish_reason": "stop",
        "model": "summary-model",
        "provider_name": "fake",
        "requested_model": "summary-model",
        "purpose": "context_compaction",
        "compactor": "ModelCompactor",
    }
    assert events[3].payload == {
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
    assert "model summary" not in str(events[3].payload)

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


def test_cayu_app_counts_compaction_model_spend_in_session_usage():
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 40, "output_tokens": 8},
                }
            ),
        ]
    )
    runtime_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
        ]
    )
    app = CayuApp()
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
                session_id="sess_compaction_usage",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    compaction_completed = [
        event
        for event in events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert len(compaction_completed) == 1
    assert compaction_completed[0].payload["usage_metrics"]["input_tokens"] == 40
    assert compaction_completed[0].payload["usage_metrics"]["model"] == "summary-model"
    assert compaction_completed[0].payload["usage_metrics"]["requested_model"] == "summary-model"

    summary = asyncio.run(app.get_session_usage("sess_compaction_usage"))
    assert summary.model_steps == 2
    assert summary.provider_names == ["fake"]
    assert summary.models == ["summary-model", "fake-model"]
    assert summary.usage.input_tokens == 50
    assert summary.usage.output_tokens == 10
    assert summary.usage.total_tokens == 60


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
        EventType.TURN_COMPLETED,
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
    assert events[4].payload == {
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
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[3].payload == {
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
    assert events[5].payload == {
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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


def test_file_attachment_refs_include_user_file_parts():
    from cayu.core.messages import FilePart
    from cayu.runtime.app import _file_attachment_refs

    user_attachment = file_attachment(
        artifact_id="art_user",
        kind="image",
        filename="photo.png",
        content_type="image/png",
        size_bytes=5,
    )
    tool_attachment = file_attachment(
        artifact_id="art_tool",
        kind="document",
        filename="report.pdf",
        content_type="application/pdf",
        size_bytes=9,
    )
    messages = [
        Message(
            role="user",
            content=[TextPart(text="look"), FilePart(attachment=user_attachment)],
        ),
        Message.tool_call(tool_call_id="call_1", tool_name="read_file"),
        Message.tool_result(
            tool_call_id="call_1",
            tool_name="read_file",
            content="attached",
            artifacts=[tool_attachment],
        ),
    ]

    refs, prompt_ids, tool_ids = _file_attachment_refs(messages)

    assert [ref.artifact_id for ref in refs] == ["art_user", "art_tool"]
    assert prompt_ids == {"art_user"}
    assert tool_ids == {"art_tool"}

    conflicting = file_attachment(
        artifact_id="art_user",
        kind="image",
        filename="different.png",
        content_type="image/png",
        size_bytes=7,
    )
    with pytest.raises(RuntimeError, match="Conflicting file attachment"):
        _file_attachment_refs(
            [
                *messages,
                Message(role="user", content=[FilePart(attachment=conflicting)]),
            ]
        )


def _app_with_artifact_store(tmp_path, **app_kwargs) -> tuple[CayuApp, LocalArtifactStore]:
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    app = CayuApp(enable_logging=False, **app_kwargs)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=store),
        default=True,
    )
    return app, store


def _valid_png_bytes() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _valid_jpeg_bytes() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _valid_pdf_bytes() -> bytes:
    import io

    import pypdf

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_attach_file_saves_reference_and_persists_bytes(tmp_path):
    app, store = _app_with_artifact_store(tmp_path)
    png = _valid_png_bytes()

    part = asyncio.run(
        app.attach_file(png, filename="pic.png", kind="image", session_id="sess_attach")
    )

    assert isinstance(part, FilePart)
    attachment = part.attachment
    assert attachment["kind"] == "image"
    assert attachment["filename"] == "pic.png"
    assert attachment["content_type"] == "image/png"
    assert attachment["size_bytes"] == len(png)

    read = asyncio.run(store.read_bytes(attachment["artifact_id"]))
    assert read.content == png


def test_attach_file_infers_content_type_from_filename(tmp_path):
    app, _ = _app_with_artifact_store(tmp_path)

    part = asyncio.run(
        app.attach_file(
            _valid_pdf_bytes(), filename="invoice.pdf", kind="document", session_id="sess_attach"
        )
    )

    assert part.attachment["content_type"] == "application/pdf"


def test_attach_file_rejects_bytes_over_limit(tmp_path):
    app, _ = _app_with_artifact_store(tmp_path, max_file_attachment_bytes=4)

    with pytest.raises(ValueError, match="prompt attachment byte limit"):
        asyncio.run(app.attach_file(b"hello", filename="pic.png", kind="image"))


def test_attach_file_rejects_kind_content_type_mismatch(tmp_path):
    app, _ = _app_with_artifact_store(tmp_path)

    with pytest.raises(ValueError):
        asyncio.run(app.attach_file(_valid_pdf_bytes(), filename="invoice.pdf", kind="image"))


def test_attach_file_rejects_invalid_image_bytes(tmp_path):
    app, _ = _app_with_artifact_store(tmp_path)

    with pytest.raises(ValueError, match="not a valid image"):
        asyncio.run(
            app.attach_file(b"not an image", filename="pic.png", kind="image", session_id="s")
        )


def test_attach_file_rejects_image_bytes_content_type_mismatch(tmp_path):
    # JPEG bytes declared as image/png (via the .png filename) must be rejected before storing.
    app, _ = _app_with_artifact_store(tmp_path)

    with pytest.raises(ValueError, match="image/jpeg but the declared content type is image/png"):
        asyncio.run(
            app.attach_file(_valid_jpeg_bytes(), filename="wrong.png", kind="image", session_id="s")
        )

    # The correctly-labeled JPEG (via .jpg) is accepted.
    part = asyncio.run(
        app.attach_file(_valid_jpeg_bytes(), filename="ok.jpg", kind="image", session_id="s")
    )
    assert part.attachment["content_type"] == "image/jpeg"


def test_attach_file_requires_artifact_store(tmp_path):
    app = CayuApp(enable_logging=False)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)

    with pytest.raises(RuntimeError, match="statically-registered artifact store"):
        asyncio.run(app.attach_file(_valid_png_bytes(), filename="pic.png", kind="image"))


def _user_image_part(artifact_id: str, filename: str) -> FilePart:
    return FilePart(
        attachment=file_attachment(
            artifact_id=artifact_id,
            kind="image",
            filename=filename,
            content_type="image/png",
            size_bytes=5,
        )
    )


def test_strip_old_prompt_files_keeps_latest_attach_turn():
    # Turn 1 attaches a file; turn 2 attaches another. The earlier prompt file is projected to a
    # text note; the latest attach turn keeps its FilePart. Durable transcript stays intact.
    messages = [
        Message(role="user", content=[TextPart(text="turn1"), _user_image_part("art_a", "a.png")]),
        Message.text("assistant", "ok"),
        Message(role="user", content=[TextPart(text="turn2"), _user_image_part("art_c", "c.png")]),
    ]

    projected = strip_old_file_attachments(messages)

    assert [type(part).__name__ for part in projected[0].content] == ["TextPart", "TextPart"]
    note = projected[0].content[1].text
    assert "omitted from this provider request" in note
    assert "artifact_id=art_a" in note
    assert [type(part).__name__ for part in projected[2].content] == ["TextPart", "FilePart"]
    # Durable transcript input is untouched (projection only).
    assert type(messages[0].content[1]) is FilePart


def test_strip_old_prompt_files_keeps_multi_file_prompt_intact_on_its_turn():
    messages = [
        Message(
            role="user",
            content=[
                TextPart(text="describe both"),
                _user_image_part("art_a", "a.png"),
                _user_image_part("art_b", "b.png"),
            ],
        )
    ]

    projected = strip_old_file_attachments(messages)

    kept = [type(part).__name__ for part in projected[0].content]
    assert kept.count("FilePart") == 2


def test_strip_old_prompt_files_projects_after_a_later_text_turn():
    # Once the attach turn has been answered and a new (text-only) user turn begins, the earlier
    # prompt file is projected to a note so its bytes are not re-sent on every later request.
    messages = [
        Message(role="user", content=[TextPart(text="turn1"), _user_image_part("art_a", "a.png")]),
        Message.text("assistant", "ok"),
        Message.text("user", "plain follow-up"),
    ]

    projected = strip_old_file_attachments(messages)

    assert not any(type(part) is FilePart for part in projected[0].content)
    note = next(
        part.text
        for part in projected[0].content
        if type(part) is TextPart and "omitted" in part.text
    )
    assert "art_a" in note


def test_strip_old_prompt_files_keeps_file_live_through_its_tool_loop():
    # A prompt file must survive its OWN run's tool loop: an assistant/tool response after the file
    # with no NEWER user turn is still the current attach turn, so the file stays provider-resolvable.
    messages = [
        Message(
            role="user", content=[TextPart(text="summarize"), _user_image_part("art_a", "a.png")]
        ),
        Message.tool_call(tool_call_id="call_1", tool_name="read_file"),
        Message.tool_result(tool_call_id="call_1", tool_name="read_file", content="page text"),
    ]

    projected = strip_old_file_attachments(messages)

    assert any(type(part) is FilePart for part in projected[0].content)


def test_strip_old_file_attachments_strips_old_tool_and_prompt_files_together():
    # Both stripping mechanisms fire in one call: an over-budget tool-result attachment AND an
    # earlier prompt file. Each keeps only its latest occurrence; the two are independent.
    old_tool = file_attachment(
        artifact_id="tool_old",
        kind="image",
        filename="told.png",
        content_type="image/png",
        size_bytes=3,
    )
    new_tool = file_attachment(
        artifact_id="tool_new",
        kind="image",
        filename="tnew.png",
        content_type="image/png",
        size_bytes=7,
    )
    messages = [
        Message(
            role="user", content=[TextPart(text="turn1"), _user_image_part("file_old", "old.png")]
        ),
        Message.tool_call(tool_call_id="call_old", tool_name="read_file"),
        Message.tool_result(
            tool_call_id="call_old", tool_name="read_file", content="old", artifacts=[old_tool]
        ),
        Message(
            role="user", content=[TextPart(text="turn2"), _user_image_part("file_new", "new.png")]
        ),
        Message.tool_call(tool_call_id="call_new", tool_name="read_file"),
        Message.tool_result(
            tool_call_id="call_new", tool_name="read_file", content="new", artifacts=[new_tool]
        ),
    ]

    projected = strip_old_file_attachments(messages, max_attachment_results=1)

    # Old prompt file -> note; latest prompt file kept.
    assert not any(type(part) is FilePart for part in projected[0].content)
    assert any(type(part) is FilePart for part in projected[3].content)
    # Old tool-result attachment stripped; latest tool-result attachment kept.
    assert projected[2].content[0].artifacts == []
    assert projected[5].content[0].artifacts == [new_tool]


def test_strip_old_prompt_files_keeps_all_files_from_the_same_turn():
    # Two file-bearing user messages in ONE turn (no assistant response between them) must both
    # reach the model; only a prior turn's files (separated by an assistant response) are stripped.
    same_turn = strip_old_file_attachments(
        [
            Message(role="user", content=[_user_image_part("art_a", "a.png")]),
            Message(role="user", content=[_user_image_part("art_b", "b.png")]),
        ]
    )
    assert any(type(part) is FilePart for part in same_turn[0].content)
    assert any(type(part) is FilePart for part in same_turn[1].content)

    across_turns = strip_old_file_attachments(
        [
            Message(role="user", content=[_user_image_part("art_a", "a.png")]),
            Message.text("assistant", "ok"),
            Message(role="user", content=[_user_image_part("art_b", "b.png")]),
        ]
    )
    assert not any(type(part) is FilePart for part in across_turns[0].content)
    assert any(type(part) is FilePart for part in across_turns[2].content)


def test_attach_file_rejects_inference_for_encoded_suffix(tmp_path):
    app, _ = _app_with_artifact_store(tmp_path)

    with pytest.raises(ValueError, match="encoding"):
        asyncio.run(app.attach_file(b"gzipped", filename="invoice.pdf.gz", kind="document"))


def test_noteify_unresolvable_prompt_files_replaces_only_targeted_parts():
    messages = [
        Message(
            role="user",
            content=[
                TextPart(text="look"),
                _user_image_part("art_bad", "bad.png"),
                _user_image_part("art_ok", "ok.png"),
            ],
        )
    ]

    projected = runtime_app_module.noteify_unresolvable_prompt_files(messages, {"art_bad"})

    kept = projected[0].content
    assert [type(part).__name__ for part in kept] == ["TextPart", "FilePart", "TextPart"]
    assert kept[1].attachment["artifact_id"] == "art_ok"
    assert "could not be resolved" in kept[2].text
    assert "art_bad" in kept[2].text


def test_resolved_file_attachments_fail_open_only_for_prompt_files(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    app = CayuApp(enable_logging=False)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=store),
        default=True,
    )
    registered_env = app._environments["local"]
    session = Session(
        id="sess_resolve",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        causal_budget_id="sess_resolve",
    )

    def missing(artifact_id):
        return file_attachment(
            artifact_id=artifact_id,
            kind="image",
            filename="ghost.png",
            content_type="image/png",
            size_bytes=5,
        )

    async def resolve(messages):
        return await runtime_app_module._resolved_file_attachments(
            messages=messages,
            session=session,
            registered_environment=registered_env,
            max_file_attachment_bytes=8_000_000,
            max_total_file_attachment_bytes=32_000_000,
            max_file_attachments_per_request=20,
        )

    # Prompt file with a missing artifact: fails open into the unresolvable set (no raise).
    prompt_only = [Message(role="user", content=[FilePart(attachment=missing("art_ghost"))])]
    resolved, unresolvable = asyncio.run(resolve(prompt_only))
    assert resolved == {}
    assert unresolvable == {"art_ghost"}

    # A malformed artifact id (not an 'art_' id) raises ValueError on read but still fails open.
    bad_id = [Message(role="user", content=[FilePart(attachment=missing("nope"))])]
    resolved, unresolvable = asyncio.run(resolve(bad_id))
    assert unresolvable == {"nope"}

    # Tool-result attachment with a missing artifact stays fail-closed.
    tool_only = [
        Message.tool_call(tool_call_id="c1", tool_name="read_file"),
        Message.tool_result(
            tool_call_id="c1",
            tool_name="read_file",
            content="x",
            artifacts=[missing("art_toolghost")],
        ),
    ]
    with pytest.raises(FileNotFoundError):
        asyncio.run(resolve(tool_only))

    # An id referenced by BOTH a prompt file and a tool result stays fail-closed (the tool-result
    # provider path would brick on a missing resolved entry).
    shared = [
        Message(role="user", content=[FilePart(attachment=missing("art_shared"))]),
        Message.tool_call(tool_call_id="c2", tool_name="read_file"),
        Message.tool_result(
            tool_call_id="c2",
            tool_name="read_file",
            content="x",
            artifacts=[missing("art_shared")],
        ),
    ]
    with pytest.raises(FileNotFoundError):
        asyncio.run(resolve(shared))


def test_run_fails_open_on_unresolvable_prompt_file(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("ok"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=store),
        default=True,
    )

    async def flow() -> FilePart:
        # Attach under one session but run under another -> the live prompt file is unresolvable.
        part = await app.attach_file(
            _valid_png_bytes(), filename="pic.png", kind="image", session_id="sess_other"
        )
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_run",
                messages=[Message(role="user", content=[TextPart(text="describe"), part])],
            )
        ):
            pass
        return part

    part = asyncio.run(flow())
    request = provider.requests[-1]

    # The run reached the provider (did not brick) with the file projected to a note, no bytes.
    assert part.attachment["artifact_id"] not in request.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
    user_message = next(message for message in request.messages if message.role == "user")
    assert not any(type(part_) is FilePart for part_ in user_message.content)
    assert any(
        type(part_) is TextPart and "could not be resolved" in part_.text
        for part_ in user_message.content
    )


def test_multi_turn_resume_ships_prompt_file_only_on_its_attach_turn(tmp_path):
    # End-to-end run -> resume through the real runtime: turn 1 attaches a file, turn 2 resumes and
    # attaches another. The turn-2 model request must carry the new file's bytes only; the old prompt
    # file must be projected to a text note and never re-resolved to bytes.
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("turn1"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("turn2"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=store),
        default=True,
    )

    async def flow() -> tuple[FilePart, FilePart]:
        sid = "sess_multiturn_strip"
        old_part = await app.attach_file(
            _valid_png_bytes(), filename="old.png", kind="image", session_id=sid
        )
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=sid,
                messages=[Message(role="user", content=[TextPart(text="turn1"), old_part])],
            )
        ):
            pass
        new_part = await app.attach_file(
            _valid_png_bytes(), filename="new.png", kind="image", session_id=sid
        )
        async for _ in app.resume(
            ResumeRequest(
                session_id=sid,
                messages=[Message(role="user", content=[TextPart(text="turn2"), new_part])],
            )
        ):
            pass
        return old_part, new_part

    old_part, new_part = asyncio.run(flow())
    old_id = old_part.attachment["artifact_id"]
    new_id = new_part.attachment["artifact_id"]

    turn2_request = provider.requests[-1]

    # Only the latest attach turn's file is resolved to bytes for the provider.
    resolved = turn2_request.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
    assert set(resolved) == {new_id}

    user_messages = [message for message in turn2_request.messages if message.role == "user"]
    old_turn, new_turn = user_messages[0], user_messages[1]
    # Old prompt file projected to a text note; new prompt file kept as a FilePart.
    assert not any(type(part) is FilePart for part in old_turn.content)
    assert any(type(part) is FilePart for part in new_turn.content)
    old_note = " ".join(part.text for part in old_turn.content if type(part) is TextPart)
    assert "omitted from this provider request" in old_note
    assert old_id in old_note


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


def test_cayu_app_renders_explicit_workspace_instructions_with_agent_prompt():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace_instructions="Run tests with uv run pytest.",
        ),
        default=True,
    )
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
                session_id="sess_workspace_instructions",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert [message.role for message in provider.requests[0].messages] == [
        "system",
        "user",
    ]
    system_text = provider.requests[0].messages[0].content[0].text
    assert "[Agent instructions]\nYou are careful." in system_text
    assert "[Workspace instructions]" in system_text
    assert "Source: explicit" in system_text
    assert "Run tests with uv run pytest." in system_text
    assert provider.requests[0].messages[1].content[0].text == "hi"
    transcript = asyncio.run(app.session_store.load_transcript("sess_workspace_instructions"))
    assert [message.role for message in transcript] == ["system", "user", "assistant"]
    assert transcript[0].content[0].text == system_text


def test_cayu_app_loads_workspace_instructions_from_configured_file(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("Use pnpm and do not edit generated files.\n")
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="repo"),
            workspace=LocalWorkspace(tmp_path, workspace_id="repo"),
            workspace_instructions=WorkspaceInstructionsConfig(paths=("CLAUDE.md",)),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_workspace_instruction_file",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert [message.role for message in provider.requests[0].messages] == [
        "system",
        "user",
    ]
    system_text = provider.requests[0].messages[0].content[0].text
    assert "Source: CLAUDE.md" in system_text
    assert "Use pnpm and do not edit generated files." in system_text


def test_cayu_app_merges_workspace_instruction_files(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Use uv for Python commands.\n")
    (tmp_path / ".cayu").mkdir()
    (tmp_path / ".cayu" / "AGENTS.md").write_text("Run focused tests before broad tests.\n")
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="repo"),
            workspace=LocalWorkspace(tmp_path, workspace_id="repo"),
            workspace_instructions=WorkspaceInstructionsConfig(mode="merge"),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_merged_workspace_instruction_files",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    system_text = provider.requests[0].messages[0].content[0].text
    assert "Source: AGENTS.md, .cayu/AGENTS.md" in system_text
    assert "Source: AGENTS.md\nUse uv for Python commands." in system_text
    assert "Source: .cayu/AGENTS.md\nRun focused tests before broad tests." in system_text


def test_workspace_instruction_file_config_rejects_paths_outside_workspace():
    with pytest.raises(ValidationError, match="must stay inside the workspace"):
        WorkspaceInstructionsConfig(paths=("../AGENTS.md",))


def test_cayu_app_rejects_oversized_workspace_instructions_before_session_create(tmp_path):
    (tmp_path / "AGENTS.md").write_text("too long")
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="repo"),
            workspace=LocalWorkspace(tmp_path, workspace_id="repo"),
            workspace_instructions=WorkspaceInstructionsConfig(
                paths=("AGENTS.md",),
                max_bytes=3,
            ),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(ValueError, match="exceeds 3 bytes"):
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_oversized_workspace_instructions",
                    messages=[Message.text("user", "hi")],
                ),
            )
        )

    session = asyncio.run(app.session_store.load("sess_oversized_workspace_instructions"))
    assert session is None
    assert provider.requests == []


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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
    assert provider.mutation_blocked
    assert provider.requests[0].messages[0].content[0].text == "original"
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.TURN_COMPLETED,
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
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[5].payload["valid"] is False
    assert events[5].payload["errors"][0]["message"] == (
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
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[4].payload["output"] == {"answer": "ok"}
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


def test_cayu_app_redacts_structured_output_tool_result_before_transcript():
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret_value = "sk-live-structured-output-secret"
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_final",
                name=STRUCTURED_OUTPUT_TOOL_NAME,
                arguments={"output": {"answer": secret_value}},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_structured_output_tool_redacted",
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
    transcript = asyncio.run(store.load_transcript("sess_structured_output_tool_redacted"))

    assert events[3].type == EventType.STRUCTURED_OUTPUT_VALIDATING
    assert events[4].type == EventType.STRUCTURED_OUTPUT_VALIDATED
    assert events[4].payload["output"] == {"answer": REDACTED_SECRET}
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]
    tool_part = transcript[-1].content[0]
    assert isinstance(tool_part, ToolResultPart)
    assert tool_part.structured == {"output": {"answer": REDACTED_SECRET}}
    assert secret_value not in repr(tool_part)


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
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[4].payload["errors"][0]["path"] == "$"
    assert events[5].payload["attempt"] == 1
    assert events[9].payload["output"] == {"answer": "fixed"}
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


def test_cayu_app_redacts_structured_output_tool_validation_errors():
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret_value = "structured-output-error-secret"
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_final_invalid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": secret_value}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_final_valid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "ok"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_structured_output_tool_error_redaction",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"enum": ["ok"]}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=1,
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_structured_output_tool_error_redaction"))

    failed_event = events[4]
    retry_event = events[5]
    assert failed_event.type == EventType.STRUCTURED_OUTPUT_FAILED
    assert retry_event.type == EventType.STRUCTURED_OUTPUT_RETRY
    assert secret_value not in str(failed_event.payload)
    assert secret_value not in str(retry_event.payload)
    assert REDACTED_SECRET in failed_event.payload["errors"][0]["message"]

    invalid_result = transcript[2].content[0]
    assert isinstance(invalid_result, ToolResultPart)
    assert invalid_result.tool_name == STRUCTURED_OUTPUT_TOOL_NAME
    assert invalid_result.is_error is True
    assert secret_value not in str(invalid_result.model_dump(mode="json"))
    assert REDACTED_SECRET in invalid_result.content
    assert REDACTED_SECRET in invalid_result.structured["structured_output_errors"][0]["message"]
    assert secret_value not in str(provider.requests[1].messages[-1].model_dump(mode="json"))


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
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[4].payload["errors"][0]["message"] == (
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
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[8].payload["output"] == {"answer": "done"}


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
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[5].payload["output"] == {"answer": "ok"}
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
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[5].payload["errors"][0]["path"] == "$"
    assert events[11].payload["output"] == {"answer": "fixed"}
    repair_message = transcript[2].content[0].text
    assert "Return only valid JSON" in repair_message
    assert STRUCTURED_OUTPUT_TOOL_NAME not in repair_message


def test_cayu_app_redacts_native_structured_output_repair_prompt_errors():
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret_value = "native-structured-output-error-secret"
    store = InMemorySessionStore()
    provider = NativeStructuredOutputFakeProvider(
        [
            [
                ModelStreamEvent.text_delta(f'{{"answer":"{secret_value}"}}'),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta('{"answer":"ok"}'),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_native_structured_output_error_redaction",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"enum": ["ok"]}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=1,
                    strategy="native",
                ),
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_native_structured_output_error_redaction"))

    failed_event = events[5]
    retry_event = events[6]
    assert failed_event.type == EventType.STRUCTURED_OUTPUT_FAILED
    assert retry_event.type == EventType.STRUCTURED_OUTPUT_RETRY
    assert secret_value not in str(failed_event.payload)
    assert secret_value not in str(retry_event.payload)
    assert REDACTED_SECRET in failed_event.payload["errors"][0]["message"]

    repair_message = transcript[2].content[0].text
    assert "Return only valid JSON" in repair_message
    assert secret_value not in repair_message
    assert REDACTED_SECRET in repair_message
    assert secret_value not in str(provider.requests[1].messages[-1].model_dump(mode="json"))


def _native_answer_spec() -> StructuredOutputSpec:
    return StructuredOutputSpec(
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
        strategy="native",
    )


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

    with pytest.raises(NativeStructuredOutputUnsupported, match="provider: fake") as excinfo:
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_structured_output_native_unsupported",
                    messages=[Message.text("user", "answer with structured output")],
                    structured_output=_native_answer_spec(),
                ),
            )
        )

    # Still a ValueError, so existing handlers (server 4xx mapping) keep working.
    assert isinstance(excinfo.value, ValueError)
    # Preflight contract: no session persisted, no model request made.
    assert asyncio.run(store.load("sess_structured_output_native_unsupported")) is None
    assert provider.requests == []


def test_cayu_app_rejects_native_structured_output_on_model_routed_provider():
    # Model-pattern routing can select a non-native provider purely by model
    # name, even when the default provider WOULD support native mode.
    class NativeDefaultProvider(NativeStructuredOutputFakeProvider):
        name = "native-default"

    store = InMemorySessionStore()
    routed = FakeProvider(
        [
            ModelStreamEvent.text_delta('{"answer":"ok"}'),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(NativeDefaultProvider([]), default=True)
    app.register_provider(routed, model_patterns=["fake-*"])
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(NativeStructuredOutputUnsupported, match="provider: fake"):
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_structured_output_native_routed",
                    messages=[Message.text("user", "answer with structured output")],
                    structured_output=_native_answer_spec(),
                ),
            )
        )

    assert asyncio.run(store.load("sess_structured_output_native_routed")) is None
    assert routed.requests == []


def test_cayu_app_resume_rejects_native_structured_output_for_unsupported_provider():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
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
                session_id="sess_native_resume",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    with pytest.raises(NativeStructuredOutputUnsupported, match="provider: fake"):
        asyncio.run(
            collect_resume_events(
                app,
                ResumeRequest(
                    session_id="sess_native_resume",
                    messages=[Message.text("user", "again")],
                    structured_output=_native_answer_spec(),
                ),
            )
        )

    # Checked before the status transition: the session stays resumable.
    session = asyncio.run(store.load("sess_native_resume"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_rejects_native_structured_output_on_tool_approval():
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
                session_id="sess_native_tool_approval",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    ).payload["approval"]["approval_id"]

    # The paused run had no spec, so the resolver's NATIVE spec would be
    # adopted — and must be rejected before the status transition.
    with pytest.raises(NativeStructuredOutputUnsupported, match="provider: fake"):
        asyncio.run(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id="sess_native_tool_approval",
                    approval_id=approval_id,
                    decision=ToolApprovalDecision.APPROVE,
                    structured_output=_native_answer_spec(),
                ),
            )
        )

    session = asyncio.run(store.load("sess_native_tool_approval"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_rejects_native_structured_output_on_tool_approval_recovery():
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
                session_id="sess_native_approval_recovery",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    ).payload["approval"]["approval_id"]

    with pytest.raises(NativeStructuredOutputUnsupported, match="provider: fake"):
        asyncio.run(
            collect_tool_approval_recovery_events(
                app,
                ToolApprovalRecoveryRequest(
                    session_id="sess_native_approval_recovery",
                    approval_id=approval_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="side effect completed externally",
                    structured_output=_native_answer_spec(),
                ),
            )
        )

    session = asyncio.run(store.load("sess_native_approval_recovery"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_require_native_structured_output_support_helper():
    app = CayuApp(session_store=InMemorySessionStore())
    app.register_provider(FakeProvider([]), default=True)
    registered_provider = app._get_registered_provider("fake")

    # No spec and TOOL specs pass; only NATIVE on a non-native provider raises.
    _require_native_structured_output_support(None, registered_provider=registered_provider)
    _require_native_structured_output_support(
        StructuredOutputSpec(json_schema={"type": "object"}),
        registered_provider=registered_provider,
    )
    with pytest.raises(NativeStructuredOutputUnsupported):
        _require_native_structured_output_support(
            _native_answer_spec(), registered_provider=registered_provider
        )


def test_cayu_app_rejects_invalid_native_structured_output_schema_before_run():
    store = InMemorySessionStore()
    provider = SchemaRejectingProvider([])
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(
        NativeStructuredOutputSchemaInvalid, match="rejected by provider preflight"
    ) as excinfo:
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_native_schema_invalid",
                    messages=[Message.text("user", "answer with structured output")],
                    structured_output=_native_answer_spec(),
                ),
            )
        )

    # Still a ValueError, so existing handlers (server 4xx mapping) keep working.
    assert isinstance(excinfo.value, ValueError)
    # Preflight contract: no session persisted, no model request made.
    assert asyncio.run(store.load("sess_native_schema_invalid")) is None
    assert provider.requests == []


def test_cayu_app_rejects_openai_native_schema_before_first_model_call():
    # The issue-#151 acceptance path end-to-end: a schema OpenAI strict mode
    # always refuses (no `additionalProperties: false`) fails before any
    # session is created and before the transport is touched.
    class UnreachableTransport:
        def stream_response_events(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            raise AssertionError("preflight must reject before any model request")

        async def create_response(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("preflight must reject before any model request")

    store = InMemorySessionStore()
    provider = OpenAIProvider(api_key="test-key", transport=UnreachableTransport())
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test"))

    with pytest.raises(
        NativeStructuredOutputSchemaInvalid,
        match=r"\$: .*additionalProperties: false",
    ):
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_openai_native_schema_invalid",
                    messages=[Message.text("user", "answer with structured output")],
                    structured_output=_native_answer_spec(),
                ),
            )
        )

    assert asyncio.run(store.load("sess_openai_native_schema_invalid")) is None


def test_cayu_app_rejects_invalid_native_structured_output_schema_on_tool_approval():
    # Composition: the resolver adopts the effective spec from the pending
    # checkpoint, and the provider schema preflight must run on THAT spec
    # before the status transition — not just on the run() request path.
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = SchemaRejectingProvider(
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
                session_id="sess_native_schema_tool_approval",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    ).payload["approval"]["approval_id"]

    # The paused run had no spec, so the resolver's NATIVE spec would be
    # adopted — and its schema must be rejected before the status transition.
    with pytest.raises(NativeStructuredOutputSchemaInvalid, match="rejected by provider preflight"):
        asyncio.run(
            collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id="sess_native_schema_tool_approval",
                    approval_id=approval_id,
                    decision=ToolApprovalDecision.APPROVE,
                    structured_output=_native_answer_spec(),
                ),
            )
        )

    session = asyncio.run(store.load("sess_native_schema_tool_approval"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


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
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert events[5].payload["errors"][0]["message"] == (
        f"Final structured output must be submitted with the `{STRUCTURED_OUTPUT_TOOL_NAME}` tool."
    )
    assert events[6].payload["attempt"] == 1
    assert events[10].payload["attempt"] == 2
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
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.TURN_COMPLETED,
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
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.TURN_COMPLETED,
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
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2
    assert events[8].payload["output"] == {"answer": "from tool"}


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
                structured_output=_native_answer_spec(),
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
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2
    assert provider.requests[0].options["structured_output"]["strategy"] == "native"
    assert provider.requests[1].options["structured_output"]["strategy"] == "native"
    assert events[9].payload["output"] == {"answer": "from tool"}


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


def test_cayu_app_preserves_labels_when_environment_sets_name():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    session_store = InMemorySessionStore()
    app = CayuApp(session_store=session_store)
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_environment_labels",
                labels={"owner": "org_123", "project": "ap_q2"},
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    session = asyncio.run(session_store.load("sess_environment_labels"))
    assert session is not None
    assert session.environment_name == "local"
    assert session.labels == {"owner": "org_123", "project": "ap_q2"}


def test_cayu_app_rejects_invalid_environment_lookup_name():
    app = CayuApp()
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)

    with pytest.raises(ValueError, match="environment.name"):
        app.get_environment("")

    with pytest.raises(ValueError, match="environment.name"):
        app.get_environment(" ")


def test_cayu_app_registers_and_selects_default_environment_factory(tmp_path):
    workspace_root = tmp_path / "factory"
    workspace_root.mkdir()
    factory = RecordingEnvironmentFactory(
        Environment(
            EnvironmentSpec(name="dynamic", metadata={"kind": "dynamic"}),
            workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
        )
    )
    app = CayuApp()
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic", metadata={"kind": "registration"}),
        factory,
        default=True,
    )

    assert app.get_environment_factory() is factory
    assert app.get_environment_factory("dynamic") is factory
    with pytest.raises(RuntimeError, match="factory-backed"):
        app.get_environment()


def test_cayu_app_rejects_invalid_environment_factory_registration_inputs(tmp_path):
    class FactoryLike:
        pass

    class EnvironmentSpecSubclass(EnvironmentSpec):
        pass

    workspace_root = tmp_path / "factory"
    workspace_root.mkdir()
    factory = RecordingEnvironmentFactory(
        Environment(
            EnvironmentSpec(name="dynamic"),
            workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
        )
    )
    app = CayuApp()

    with pytest.raises(TypeError, match="EnvironmentSpec"):
        app.register_environment_factory(
            EnvironmentSpecSubclass(name="dynamic"),
            factory,
        )

    with pytest.raises(TypeError, match="EnvironmentFactory"):
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            FactoryLike(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="bool"):
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default="true",  # type: ignore[arg-type]
        )


def test_cayu_app_rejects_duplicate_environment_and_factory_names(tmp_path):
    workspace_root = tmp_path / "factory"
    workspace_root.mkdir()
    factory = RecordingEnvironmentFactory(
        Environment(
            EnvironmentSpec(name="dynamic"),
            workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
        )
    )
    app = CayuApp()
    app.register_environment(Environment(EnvironmentSpec(name="local")))

    with pytest.raises(ValueError, match="Environment already registered"):
        app.register_environment_factory(EnvironmentSpec(name="local"), factory)

    app.register_environment_factory(EnvironmentSpec(name="dynamic"), factory)
    with pytest.raises(ValueError, match="Environment already registered"):
        app.register_environment(Environment(EnvironmentSpec(name="dynamic")))


def test_cayu_app_rejects_environment_factory_lookup_for_static_environment():
    app = CayuApp()
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)

    with pytest.raises(RuntimeError, match="not factory-backed"):
        app.get_environment_factory()


def test_cayu_app_rejects_invalid_environment_factory_lookup_name():
    app = CayuApp()

    with pytest.raises(ValueError, match="environment.name"):
        app.get_environment_factory("")

    with pytest.raises(ValueError, match="environment.name"):
        app.get_environment_factory(" ")


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

    # Environment subclasses are advertised extension points and are accepted.
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
    assert [event.type for event in run_events[-2:]] == [
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert run_events[-2].payload["status"] == "interrupted"
    assert run_events[-1].id == interrupt_events[0].id == stored_interrupt_events[0].id
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
    assert event_types_after_release[-2] == EventType.TURN_COMPLETED
    assert event_types_after_release[-1] == EventType.SESSION_INTERRUPTED


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
    assert failed_tool_events[0].payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="sess_interrupt_tool_call",
        tool_round_id=failed_tool_events[0].payload["tool_round_id"],
        tool_call_id="call_1",
    )
    assert failed_tool_events[0].payload["result"]["is_error"] is True
    assert failed_tool_events[0].payload["result"]["structured"] == {
        "interrupted": True,
        "tool_call_id": "call_1",
        "tool_name": "blocking_tool",
        "tool_round_id": failed_tool_events[0].payload["tool_round_id"],
    }
    stored_event_types = [event.type for event in stored_events]
    assert stored_event_types.index(EventType.TOOL_CALL_FAILED) < stored_event_types.index(
        EventType.SESSION_INTERRUPTED
    )
    validate_context_messages(transcript)
    assert transcript[-1].role == "tool"
    assert transcript[-1].content[0].tool_call_id == "call_1"
    assert transcript[-1].content[0].is_error is True


def test_interrupt_session_preserves_proxy_authorization_events() -> None:
    class ProxyBlockingTool(Tool):
        spec = ToolSpec(
            name="proxy_blocking_tool",
            description="Authorize proxy use and block until cancelled.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.authorized: asyncio.Event | None = None
            self.cancelled: asyncio.Event | None = None
            self.never_complete: asyncio.Event | None = None

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            if self.authorized is None or self.cancelled is None or self.never_complete is None:
                raise AssertionError("ProxyBlockingTool test events were not initialized.")
            assert ctx.proxy is not None
            await ctx.proxy.authorize_request(
                destination="https://api.sendgrid.com/v3/mail/send",
                credential=SecretRef(name="sendgrid_api_key"),
                action="send_email",
                metadata={"template": "invoice_reminder"},
            )
            self.authorized.set()
            try:
                await self.never_complete.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return ToolResult(content="unexpected")

    tool = ProxyBlockingTool()
    vault = StaticVault({"sendgrid_api_key": "sk-proxy-interrupt-secret"})
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_proxy_blocking",
                name="proxy_blocking_tool",
                arguments={},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=vault,
            proxy=PassthroughProxy(vault),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    async def run():
        tool.authorized = asyncio.Event()
        tool.cancelled = asyncio.Event()
        tool.never_complete = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interrupt_proxy_authorization",
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await tool.authorized.wait()
        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_interrupt_proxy_authorization",
                    reason="operator stop",
                )
            )
        ]
        await asyncio.wait_for(tool.cancelled.wait(), timeout=1)
        run_events = await run_task
        stored_events = await app.session_store.load_events("sess_interrupt_proxy_authorization")
        return run_events, interrupt_events, stored_events

    run_events, interrupt_events, stored_events = asyncio.run(run())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    proxy_events = [
        event for event in stored_events if event.type == EventType.CREDENTIAL_PROXY_CHECKED
    ]
    assert len(proxy_events) == 1
    assert proxy_events[0].payload["tool_call_id"] == "call_proxy_blocking"
    assert proxy_events[0].payload["destination"] == "https://api.sendgrid.com/v3/mail/send"
    assert proxy_events[0].payload["credential"] == "sendgrid_api_key"
    assert proxy_events[0].payload["action"] == "send_email"
    assert proxy_events[0].payload["allowed"] is True
    stored_event_types = [event.type for event in stored_events]
    assert stored_event_types.index(EventType.CREDENTIAL_PROXY_CHECKED) < stored_event_types.index(
        EventType.TOOL_CALL_FAILED
    )
    assert stored_event_types.index(EventType.TOOL_CALL_FAILED) < stored_event_types.index(
        EventType.SESSION_INTERRUPTED
    )


def test_generic_cancellation_does_not_emit_proxy_authorization_events() -> None:
    class ProxyBlockingTool(Tool):
        spec = ToolSpec(
            name="proxy_blocking_tool",
            description="Authorize proxy use and block until cancelled.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.authorized: asyncio.Event | None = None
            self.cancelled: asyncio.Event | None = None
            self.never_complete: asyncio.Event | None = None

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            if self.authorized is None or self.cancelled is None or self.never_complete is None:
                raise AssertionError("ProxyBlockingTool test events were not initialized.")
            assert ctx.proxy is not None
            await ctx.proxy.authorize_request(
                destination="https://api.sendgrid.com/v3/mail/send",
                credential=SecretRef(name="sendgrid_api_key"),
                action="send_email",
            )
            self.authorized.set()
            try:
                await self.never_complete.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return ToolResult(content="unexpected")

    tool = ProxyBlockingTool()
    vault = StaticVault({"sendgrid_api_key": "sk-proxy-cancel-secret"})
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_proxy_blocking",
                name="proxy_blocking_tool",
                arguments={},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=vault,
            proxy=PassthroughProxy(vault),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    async def run():
        tool.authorized = asyncio.Event()
        tool.cancelled = asyncio.Event()
        tool.never_complete = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_generic_cancel_proxy_authorization",
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await tool.authorized.wait()
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        await asyncio.wait_for(tool.cancelled.wait(), timeout=1)
        return await app.session_store.load_events("sess_generic_cancel_proxy_authorization")

    stored_events = asyncio.run(run())

    assert EventType.CREDENTIAL_PROXY_CHECKED not in [event.type for event in stored_events]
    assert EventType.SESSION_INTERRUPTED not in [event.type for event in stored_events]


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


def test_cancelled_runner_cleanup_diagnostics_are_redacted_in_tool_result():
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret_value = "cleanup-secret-token"
    cleanup_artifact = {
        "type": "cayu.runner_cleanup.v1",
        "adapter": "e2b",
        "status": "timeout",
        "stderr": f"failed with token {secret_value}",
        "metadata": {"token": secret_value},
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
    app = CayuApp(secret_redactor=SecretRedactor(secret_value))
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
                    session_id="sess_interrupt_tool_cleanup_redaction",
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await tool.started.wait()
        _ = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_interrupt_tool_cleanup_redaction",
                    reason="operator stop",
                )
            )
        ]
        run_events = await run_task
        stored_events = await app.session_store.load_events("sess_interrupt_tool_cleanup_redaction")
        transcript = await app.session_store.load_transcript(
            "sess_interrupt_tool_cleanup_redaction"
        )
        return run_events, stored_events, transcript

    run_events, stored_events, transcript = asyncio.run(run())

    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    failed_tool_event = next(
        event for event in stored_events if event.type == EventType.TOOL_CALL_FAILED
    )
    assert secret_value not in str(failed_tool_event.payload)
    artifact = failed_tool_event.payload["result"]["artifacts"][0]
    assert artifact["stderr"] == f"failed with token {REDACTED_SECRET}"
    assert artifact["metadata"]["token"] == REDACTED_SECRET

    validate_context_messages(transcript)
    assert transcript[-1].role == "tool"
    tool_part = transcript[-1].content[0]
    assert isinstance(tool_part, ToolResultPart)
    assert secret_value not in str(tool_part.model_dump(mode="json"))
    assert tool_part.artifacts[0]["metadata"]["token"] == REDACTED_SECRET


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
    # A "not yet started" trailing tool call requires sequential execution.
    app = CayuApp(max_parallel_tool_calls=1)
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
    assert event_types_after_release.count(EventType.TOOL_CALL_FAILED) == 1
    assert event_types_after_release[-2] == EventType.TURN_COMPLETED
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
            await self._interrupt_if_assistant_tool_call_appended(session_id, messages)

        async def append_transcript_messages_and_checkpoint(
            self,
            session_id: str,
            messages: list[Message],
            checkpoint: dict,
        ) -> None:
            await super().append_transcript_messages_and_checkpoint(
                session_id,
                messages,
                checkpoint,
            )
            await self._interrupt_if_assistant_tool_call_appended(session_id, messages)

        async def _interrupt_if_assistant_tool_call_appended(
            self,
            session_id: str,
            messages: list[Message],
        ) -> None:
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
        "tool_round_id": transcript[-1].content[0].structured["tool_round_id"],
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
    assert checkpoint == {}
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
            await self._maybe_interrupt_before_tool_result_append(session_id, messages)
            await super().append_transcript_messages(session_id, messages)

        async def append_transcript_messages_and_checkpoint(
            self,
            session_id: str,
            messages: list[Message],
            checkpoint: dict,
        ) -> None:
            await self._maybe_interrupt_before_tool_result_append(session_id, messages)
            await super().append_transcript_messages_and_checkpoint(
                session_id,
                messages,
                checkpoint,
            )

        async def _maybe_interrupt_before_tool_result_append(
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


def test_tool_call_times_out_and_session_continues():
    class SlowTool(Tool):
        spec = ToolSpec(
            name="slow_tool",
            description="Sleep past the configured tool timeout.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            await asyncio.sleep(30)
            return ToolResult(content="unexpected")

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_slow", name="slow_tool", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, tool_timeout_seconds=0.05)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SlowTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_timeout",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    failed_events = [event for event in events if event.type == EventType.TOOL_CALL_FAILED]
    assert len(failed_events) == 1
    assert failed_events[0].payload["result"]["content"] == (
        "Tool call timed out after 0.05 seconds."
    )
    assert failed_events[0].payload["result"]["is_error"] is True
    assert events[-1].type == EventType.SESSION_COMPLETED

    transcript = asyncio.run(store.load_transcript("sess_tool_timeout"))
    tool_message = next(message for message in transcript if message.role == "tool")
    assert tool_message.content[0].content == "Tool call timed out after 0.05 seconds."
    assert tool_message.content[0].is_error is True


def test_run_tool_does_not_mislabel_tool_raised_timeout_error():
    from cayu.runtime._tool_execution import run_tool

    class TimeoutRaisingTool(Tool):
        spec = ToolSpec(
            name="timeout_raiser",
            description="Raise TimeoutError from inside the tool.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            raise TimeoutError("upstream service timed out")

    result = asyncio.run(
        run_tool(
            tool=TimeoutRaisingTool(),
            ctx=ToolContext(session_id="sess_tool_raised_timeout"),
            arguments={},
            timeout_seconds=5,
        )
    )

    assert result.is_error is True
    assert "upstream service timed out" in result.content
    assert "timed out after" not in result.content


def test_parallel_tool_round_executes_calls_concurrently_by_default():
    class RendezvousTool(Tool):
        spec = ToolSpec(
            name="rendezvous",
            description="Wait until every call in the round has started.",
            input_schema={
                "type": "object",
                "properties": {"slot": {"type": "string"}},
                "required": ["slot"],
            },
        )

        def __init__(self, expected_arrivals: int) -> None:
            self.expected_arrivals = expected_arrivals
            self.arrivals = 0
            self.all_arrived = asyncio.Event()

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            self.arrivals += 1
            if self.arrivals >= self.expected_arrivals:
                self.all_arrived.set()
            # Deadlocks under sequential execution: each call waits for the
            # other call in the same round to start.
            await asyncio.wait_for(self.all_arrived.wait(), timeout=2)
            return ToolResult(content=args["slot"])

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_a",
                    name="rendezvous",
                    arguments={"slot": "a"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_b",
                    name="rendezvous",
                    arguments={"slot": "b"},
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
        tools=[RendezvousTool(expected_arrivals=2)],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_parallel_round",
                messages=[Message.text("user", "fan out")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    tool_events = [
        (event.type, event.payload["tool_call_id"])
        for event in events
        if event.type in {EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_COMPLETED}
    ]
    # Buffered parallel execution preserves the model's tool-call order.
    assert tool_events == [
        (EventType.TOOL_CALL_STARTED, "call_a"),
        (EventType.TOOL_CALL_COMPLETED, "call_a"),
        (EventType.TOOL_CALL_STARTED, "call_b"),
        (EventType.TOOL_CALL_COMPLETED, "call_b"),
    ]

    transcript = asyncio.run(store.load_transcript("sess_parallel_round"))
    tool_message = next(message for message in transcript if message.role == "tool")
    assert [part.tool_call_id for part in tool_message.content] == ["call_a", "call_b"]
    assert [part.content for part in tool_message.content] == ["a", "b"]


def test_parallel_tool_round_composes_with_tool_call_limit_and_usage_counting():
    # Composition pin for issue #101's real-world trigger: a genuinely
    # concurrent tool round under RunLimits. The whole round is gated against
    # limits BEFORE execution (mid-round re-evaluation is skipped), and the
    # usage summary must count every tool.call.started the parallel round
    # appended — the undercount the watermark race used to cause.
    class RendezvousTool(Tool):
        spec = ToolSpec(
            name="rendezvous",
            description="Wait until every call in the round has started.",
            input_schema={
                "type": "object",
                "properties": {"slot": {"type": "string"}},
                "required": ["slot"],
            },
        )

        def __init__(self, expected_arrivals: int) -> None:
            self.expected_arrivals = expected_arrivals
            self.arrivals = 0
            self.all_arrived = asyncio.Event()

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            self.arrivals += 1
            if self.arrivals >= self.expected_arrivals:
                self.all_arrived.set()
            # Deadlocks under sequential execution: each call waits for the
            # other call in the same round to start.
            await asyncio.wait_for(self.all_arrived.wait(), timeout=2)
            return ToolResult(content=args["slot"])

    store = InMemorySessionStore()
    tool = RendezvousTool(expected_arrivals=2)
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_a", name="rendezvous", arguments={"slot": "a"}),
                ModelStreamEvent.tool_call(id="call_b", name="rendezvous", arguments={"slot": "b"}),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                    }
                ),
            ],
            [
                ModelStreamEvent.tool_call(id="call_c", name="rendezvous", arguments={"slot": "c"}),
                ModelStreamEvent.tool_call(id="call_d", name="rendezvous", arguments={"slot": "d"}),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 20, "output_tokens": 4, "total_tokens": 24},
                    }
                ),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_parallel_limit",
                messages=[Message.text("user", "fan out twice")],
                limits=RunLimits(max_tool_calls=3),
            ),
        )
    )

    # Round 1 (2 parallel calls) fits the limit and genuinely ran concurrently.
    assert tool.arrivals == 2
    started_ids = [
        event.payload["tool_call_id"]
        for event in events
        if event.type == EventType.TOOL_CALL_STARTED
    ]
    assert started_ids == ["call_a", "call_b"]

    # Round 2 (2 more calls) would exceed max_tool_calls=3; the whole round is
    # gated before execution, so neither call starts.
    limit_event = next(event for event in events if event.type == EventType.SESSION_LIMIT_REACHED)
    assert limit_event.payload["limit"] == "tool_calls"
    assert limit_event.payload["actual"] == 4
    failed_ids = {
        event.payload["tool_call_id"]
        for event in events
        if event.type == EventType.TOOL_CALL_FAILED
    }
    assert failed_ids == {"call_c", "call_d"}
    assert events[-1].type == EventType.SESSION_INTERRUPTED

    # The usage summary counts every started call of the parallel round and
    # both model steps — the undercount the watermark race used to cause.
    summary = asyncio.run(app.get_session_usage("sess_parallel_limit"))
    assert summary.tool_calls == 2
    assert summary.model_steps == 2
    assert summary.usage.total_tokens == 36


def test_cayu_app_usage_summary_spans_tool_approval_pause_and_resume():
    # Composition pin: usage accounting rebuilds from the full event tail on
    # every entrance, so totals must span both segments of a paused run.
    store = InMemorySessionStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "pause here"},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 20, "output_tokens": 4, "total_tokens": 24},
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
        tool_policy=RequireApprovalPolicy(),
    )

    interrupt_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_usage_across_pause",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    approval_id = next(
        event for event in interrupt_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    ).payload["approval"]["approval_id"]

    resume_events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_usage_across_pause",
                approval_id=approval_id,
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )
    assert resume_events[-1].type == EventType.SESSION_COMPLETED

    summary = asyncio.run(app.get_session_usage("sess_usage_across_pause"))
    assert summary.model_steps == 2
    assert summary.tool_calls == 1
    assert summary.usage.total_tokens == 36


def test_parallel_tool_round_concurrency_is_capped_by_semaphore():
    class ConcurrencyProbeTool(Tool):
        spec = ToolSpec(
            name="probe",
            description="Record how many calls run concurrently.",
            input_schema={
                "type": "object",
                "properties": {"step": {"type": "integer"}},
                "required": ["step"],
            },
        )

        def __init__(self) -> None:
            self.current = 0
            self.max_concurrent = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            self.current += 1
            self.max_concurrent = max(self.max_concurrent, self.current)
            await asyncio.sleep(0.02)
            self.current -= 1
            return ToolResult(content=str(args["step"]))

    probe = ConcurrencyProbeTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="probe", arguments={"step": 1}),
                ModelStreamEvent.tool_call(id="call_2", name="probe", arguments={"step": 2}),
                ModelStreamEvent.tool_call(id="call_3", name="probe", arguments={"step": 3}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(max_parallel_tool_calls=2)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[probe],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_parallel_capped",
                messages=[Message.text("user", "fan out")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert probe.max_concurrent == 2
    completed_ids = [
        event.payload["tool_call_id"]
        for event in events
        if event.type == EventType.TOOL_CALL_COMPLETED
    ]
    assert completed_ids == ["call_1", "call_2", "call_3"]


def test_parallel_tool_call_timeouts_do_not_serialize_the_round():
    class SleepyTool(Tool):
        spec = ToolSpec(
            name="sleepy",
            description="Sleep forever; the per-call timeout stops it.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            await asyncio.sleep(30)
            return ToolResult(content="unexpected")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="sleepy", arguments={}),
                ModelStreamEvent.tool_call(id="call_2", name="sleepy", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(tool_timeout_seconds=0.05)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SleepyTool()],
    )

    started = time.monotonic()
    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_parallel_timeouts",
                messages=[Message.text("user", "fan out")],
            ),
        )
    )
    elapsed = time.monotonic() - started

    assert events[-1].type == EventType.SESSION_COMPLETED
    failed_events = [event for event in events if event.type == EventType.TOOL_CALL_FAILED]
    assert [event.payload["tool_call_id"] for event in failed_events] == ["call_1", "call_2"]
    for event in failed_events:
        assert event.payload["result"]["content"] == "Tool call timed out after 0.05 seconds."
    # Both timeouts elapse concurrently; well under two sequential timeouts
    # plus scheduling slack.
    assert elapsed < 1.0


@pytest.mark.parametrize(
    ("kwargs", "error_type", "error"),
    [
        (
            {"tool_timeout_seconds": 0},
            ValueError,
            "tool_timeout_seconds must be greater than zero.",
        ),
        (
            {"tool_timeout_seconds": "5"},
            TypeError,
            "tool_timeout_seconds must be a number or None.",
        ),
        (
            {"max_parallel_tool_calls": 0},
            ValueError,
            "max_parallel_tool_calls must be greater than zero.",
        ),
        (
            {"max_parallel_tool_calls": 2.5},
            TypeError,
            "max_parallel_tool_calls must be an integer.",
        ),
    ],
)
def test_cayu_app_validates_tool_execution_settings(kwargs, error_type, error):
    with pytest.raises(error_type, match=error):
        CayuApp(**kwargs)


# --------------------------------------------------------------------------- #
# Pending tool approvals persist the original run's config (limits, budgets,
# max_steps, retry policy) so resolving does not restart with fresh defaults.
# --------------------------------------------------------------------------- #
def test_pending_tool_approval_run_config_round_trips_json_checkpoint():
    from cayu.runtime.approvals import (
        PendingToolApproval,
        PendingToolCallApproval,
        copy_pending_tool_approval,
    )

    pending = PendingToolApproval(
        approval_id="appr_1",
        tool_call_id="call_1",
        tool_name="side_effect",
        agent_name="assistant",
        tool_calls=[PendingToolCallApproval(tool_call_id="call_1", tool_name="side_effect")],
        max_steps=7,
        limits=RunLimits(max_tool_calls=3, scope="session"),
        budget_limits=(fake_budget_limit("2.50"),),
        retry_policy=RetryPolicy(max_attempts=3),
        expires_at=datetime(2026, 7, 9, 12, 30, tzinfo=UTC),
    )

    # Same round-trip as the durable checkpoint: model_dump(mode="json") in,
    # PendingToolApproval(**payload) out.
    restored = PendingToolApproval(**pending.model_dump(mode="json"))
    assert restored.max_steps == 7
    assert restored.limits == RunLimits(max_tool_calls=3, scope="session")
    assert restored.budget_limits == (fake_budget_limit("2.50"),)
    assert restored.retry_policy == RetryPolicy(max_attempts=3)
    assert restored.expires_at == pending.expires_at

    copied = copy_pending_tool_approval(pending)
    assert copied.max_steps == 7
    assert copied.limits == pending.limits
    assert copied.budget_limits == pending.budget_limits
    assert copied.retry_policy == pending.retry_policy
    assert copied.expires_at == pending.expires_at


def test_pending_tool_approval_loads_legacy_checkpoint_without_run_config():
    from cayu.runtime.approvals import PendingToolApproval, PendingToolCallApproval

    legacy = PendingToolApproval(
        approval_id="appr_legacy",
        tool_call_id="call_1",
        tool_name="side_effect",
        agent_name="assistant",
        tool_calls=[PendingToolCallApproval(tool_call_id="call_1", tool_name="side_effect")],
    )
    payload = legacy.model_dump(mode="json")
    for key in ("max_steps", "limits", "budget_limits", "retry_policy", "expires_at"):
        payload.pop(key)

    restored = PendingToolApproval(**payload)
    assert restored.max_steps is None
    assert restored.limits is None
    assert restored.budget_limits is None
    assert restored.retry_policy is None
    assert restored.expires_at is None


def test_effective_approval_run_config_prefers_override_then_pending_then_default():
    from cayu.runtime.app import (
        _DEFAULT_APPROVAL_MAX_STEPS,
        _effective_approval_budget_limits,
        _effective_approval_max_steps,
        _effective_approval_retry_policy,
        _effective_approval_run_limits,
    )
    from cayu.runtime.approvals import PendingToolApproval, PendingToolCallApproval

    def pending(**kwargs) -> PendingToolApproval:
        return PendingToolApproval(
            approval_id="appr_1",
            tool_call_id="call_1",
            tool_name="side_effect",
            agent_name="assistant",
            tool_calls=[PendingToolCallApproval(tool_call_id="call_1", tool_name="side_effect")],
            **kwargs,
        )

    persisted = pending(
        max_steps=9,
        limits=RunLimits(max_tool_calls=2, scope="session"),
        budget_limits=(fake_budget_limit("1.00"),),
        retry_policy=RetryPolicy(max_attempts=4),
    )
    legacy = pending()

    # Explicit overrides on the approval request win.
    assert _effective_approval_max_steps(max_steps=3, pending_approval=persisted) == 3
    assert _effective_approval_run_limits(
        limits=RunLimits(max_tool_calls=5),
        pending_approval=persisted,
    ) == RunLimits(max_tool_calls=5)
    assert (
        _effective_approval_budget_limits(
            budget_limits=(),
            pending_approval=persisted,
        )
        == ()
    )
    override_policy = RetryPolicy(max_attempts=2)
    assert (
        _effective_approval_retry_policy(
            retry_policy=override_policy,
            pending_approval=persisted,
        )
        is override_policy
    )

    # No override -> the original run's persisted config is restored.
    assert _effective_approval_max_steps(max_steps=None, pending_approval=persisted) == 9
    assert _effective_approval_run_limits(
        limits=None,
        pending_approval=persisted,
    ) == RunLimits(max_tool_calls=2, scope="session")
    assert _effective_approval_budget_limits(
        budget_limits=None,
        pending_approval=persisted,
    ) == (fake_budget_limit("1.00"),)
    assert _effective_approval_retry_policy(
        retry_policy=None,
        pending_approval=persisted,
    ) == RetryPolicy(max_attempts=4)

    # Legacy approvals without persisted config fall back to the old defaults.
    assert (
        _effective_approval_max_steps(max_steps=None, pending_approval=legacy)
        == _DEFAULT_APPROVAL_MAX_STEPS
    )
    assert _effective_approval_run_limits(limits=None, pending_approval=legacy) == RunLimits()
    assert _effective_approval_budget_limits(budget_limits=None, pending_approval=legacy) == ()
    assert _effective_approval_retry_policy(retry_policy=None, pending_approval=legacy) is None


def _run_config_approval_app() -> tuple[InMemorySessionStore, SideEffectTool, CayuApp]:
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
                    id="call_2",
                    name="echo",
                    arguments={"text": "after approval"},
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
        tools=[tool, EchoTool()],
        tool_policy=SideEffectApprovalPolicy(),
    )
    return store, tool, app


def test_cayu_app_resolving_tool_approval_restores_original_run_limits():
    store, tool, app = _run_config_approval_app()
    session_id = "sess_approval_restores_run_config"

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
                max_steps=7,
                limits=RunLimits(max_tool_calls=1, scope="session"),
                retry_policy=RetryPolicy(max_attempts=3),
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None
    pending = checkpoint["pending_tool_approval"]
    assert pending["max_steps"] == 7
    assert pending["limits"]["max_tool_calls"] == 1
    assert pending["limits"]["scope"] == "session"
    assert pending["retry_policy"]["max_attempts"] == 3
    assert pending["budget_limits"] == []

    # Approving without overrides resumes under the ORIGINAL run's limits: the
    # approved tool call consumes the session's single allowed tool call, so
    # the follow-up echo round trips the limit instead of executing.
    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=pending["approval_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )
    assert tool.calls == [{"value": "secret"}]
    limit_events = [event for event in events if event.type == EventType.SESSION_LIMIT_REACHED]
    assert len(limit_events) == 1
    assert limit_events[0].payload["limit"] == "tool_calls"
    session = asyncio.run(store.load(session_id))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_cayu_app_tool_approval_explicit_limits_override_persisted_run_config():
    store, tool, app = _run_config_approval_app()
    session_id = "sess_approval_override_run_config"

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
                limits=RunLimits(max_tool_calls=1, scope="session"),
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None
    pending = checkpoint["pending_tool_approval"]

    # An explicit limits override on the approval request wins over the
    # persisted run config, so the session runs to completion.
    events = asyncio.run(
        collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=pending["approval_id"],
                decision=ToolApprovalDecision.APPROVE,
                limits=RunLimits(),
            ),
        )
    )
    assert tool.calls == [{"value": "secret"}]
    assert events[-1].type == EventType.SESSION_COMPLETED
