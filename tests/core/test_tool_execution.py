from __future__ import annotations

import asyncio
import json
import threading
import warnings
from collections.abc import AsyncIterator
from datetime import UTC, datetime, tzinfo
from typing import Any, cast

import pytest
from tests.core._event_projection_support import private_events_for_public_events

from cayu.core import AgentSpec, Event, EventType, Message, ToolCallPart, ToolResultPart
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    AfterToolCallDecision,
    BeforeToolCallDecision,
    BeforeToolCallHookContext,
    CayuApp,
    InMemorySessionStore,
    InterruptSessionRequest,
    PendingActionQuery,
    RunLimits,
    RunRequest,
    RuntimeHook,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolCallHookContext,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.runtime import _tool_execution as tool_execution
from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgePublicationConflict,
    KnowledgePublicationReceipt,
)
from cayu.tools.commands import ExecCommandTool
from cayu.tools.files import (
    DeleteFileTool,
    EditFileTool,
    ListArtifactsTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from cayu.tools.knowledge import (
    ListKnowledgeTool,
    ReadKnowledgeTool,
    RememberKnowledgeTool,
    SearchKnowledgeTool,
)
from cayu.tools.subagents import SubagentResultTool, SubagentTool
from cayu.vaults import SecretRedactor


class _TestKnowledgeStore(InMemoryKnowledgeStore):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("access_scope", KnowledgeAccessScope.privileged())
        super().__init__(*args, **kwargs)


_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "tag": {"type": "string"},
        "delay": {"type": "number"},
    },
    "required": ["tag"],
}


class _ScriptedProvider(ModelProvider):
    """Emits one round of the given tool calls, then finishes on the next step."""

    name = "fake"

    def __init__(self, tool_calls: list[tuple[str, str, dict]]) -> None:
        self._tool_calls = tool_calls
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            for call_id, name, arguments in self._tool_calls:
                yield ModelStreamEvent.tool_call(id=call_id, name=name, arguments=arguments)
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _Recorder:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.order: list[str] = []
        self.completed: list[str] = []
        self.context_idempotency_keys: list[str | None] = []
        self.metadata_idempotency_keys: list[str | None] = []
        self.metadata_tool_effects: list[str | None] = []


class _StalledKnowledgePublicationStore(_TestKnowledgeStore):
    def __init__(self) -> None:
        super().__init__()
        self.dispatched = asyncio.Event()
        self.release = asyncio.Event()
        self.settled = asyncio.Event()
        self.operation_id: str | None = None
        self.publish_calls = 0

    async def publish_entry_revision(
        self,
        entry,
        chunks,
        *,
        operation_id,
        expected_revision=None,
        activation_authority=None,
    ):
        self.publish_calls += 1
        self.operation_id = operation_id
        self.dispatched.set()
        await self.release.wait()
        receipt = await super().publish_entry_revision(
            entry,
            chunks,
            operation_id=operation_id,
            expected_revision=expected_revision,
            activation_authority=activation_authority,
        )
        self.settled.set()
        return receipt


class _CancellationResistantKnowledgeReadStore(_TestKnowledgeStore):
    def __init__(self, *, phase: str) -> None:
        super().__init__()
        self.phase = phase
        self.read_started = threading.Event()
        self.release = threading.Event()
        self.read_finished = threading.Event()
        self.publish_calls = 0

    async def _stall(self) -> None:
        await asyncio.to_thread(self._blocking_read)

    def _blocking_read(self) -> None:
        self.read_started.set()
        try:
            if not self.release.wait(timeout=5):
                raise RuntimeError("test knowledge read barrier timed out")
        finally:
            self.read_finished.set()

    async def load_entry_publication_receipt(self, operation_id):
        if self.phase == "receipt":
            await self._stall()
        return await super().load_entry_publication_receipt(operation_id)

    async def get_entry(self, entry_id):
        if self.phase == "entry":
            await self._stall()
        return await super().get_entry(entry_id)

    async def publish_entry_revision(
        self,
        entry,
        chunks,
        *,
        operation_id,
        expected_revision=None,
        activation_authority=None,
    ):
        self.publish_calls += 1
        return await super().publish_entry_revision(
            entry,
            chunks,
            operation_id=operation_id,
            expected_revision=expected_revision,
            activation_authority=activation_authority,
        )


class _RecordingTool(Tool):
    """Tool that records concurrency and ordering via a shared recorder."""

    def __init__(
        self,
        recorder: _Recorder,
        *,
        name: str = "recording_tool",
        parallel_safe: bool = True,
        effect: ToolEffect = ToolEffect.NONE,
    ) -> None:
        super().__init__(
            ToolSpec(
                name=name,
                description="records execution",
                input_schema=_TOOL_SCHEMA,
                parallel_safe=parallel_safe,
                effect=effect,
            )
        )
        self._recorder = recorder

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        rec = self._recorder
        tag = args["tag"]
        rec.active += 1
        rec.max_active = max(rec.max_active, rec.active)
        rec.order.append(f"start:{tag}")
        rec.context_idempotency_keys.append(ctx.idempotency_key)
        rec.metadata_idempotency_keys.append(ctx.metadata.get("idempotency_key"))
        rec.metadata_tool_effects.append(ctx.metadata.get("tool_effect"))
        try:
            await asyncio.sleep(args.get("delay", 0.05))
        finally:
            rec.active -= 1
        rec.order.append(f"end:{tag}")
        rec.completed.append(tag)
        return ToolResult(content=tag)


class _CapturePolicy(ToolPolicy):
    def __init__(self) -> None:
        self.requests: list[ToolPolicyRequest] = []

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        self.requests.append(request)
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class _FakeSubagentRuntime:
    async def run(self, request: RunRequest) -> AsyncIterator[Event]:
        if False:
            yield Event(type=EventType.SESSION_STARTED, session_id=request.session_id)

    async def interrupt_session(self, request: InterruptSessionRequest) -> AsyncIterator[Event]:
        if False:
            yield Event(type=EventType.SESSION_INTERRUPTED, session_id=request.session_id)


async def _collect(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


async def _collect_interrupt(
    app: CayuApp,
    request: InterruptSessionRequest,
) -> list[Event]:
    return [event async for event in app.interrupt_session(request)]


def _build(
    *,
    tools: list[Tool],
    tool_calls: list[tuple[str, str, dict]],
    max_parallel_tool_calls: int = 4,
    tool_policy: ToolPolicy | None = None,
) -> CayuApp:
    app = CayuApp(
        session_store=InMemorySessionStore(),
        enable_logging=False,
        max_parallel_tool_calls=max_parallel_tool_calls,
    )
    app.register_provider(_ScriptedProvider(tool_calls), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=tools,
        tool_policy=tool_policy,
    )
    return app


def test_builtin_mutating_tools_are_not_parallel_safe() -> None:
    # Cayu's side-effecting built-ins must opt out of concurrent execution so the
    # default-on parallel engine never runs concurrent writes/commands/knowledge mutations.
    assert ExecCommandTool.spec.parallel_safe is False
    assert EditFileTool.spec.parallel_safe is False
    assert DeleteFileTool.spec.parallel_safe is False
    assert WriteFileTool.spec.parallel_safe is False
    assert RememberKnowledgeTool.spec.parallel_safe is False
    assert ExecCommandTool.spec.workspace_mutation is True
    assert EditFileTool.spec.workspace_mutation is True
    assert DeleteFileTool.spec.workspace_mutation is True
    assert WriteFileTool.spec.workspace_mutation is True
    assert RememberKnowledgeTool.spec.workspace_mutation is False
    assert ExecCommandTool.spec.effect is ToolEffect.EXTERNAL
    assert EditFileTool.spec.effect is ToolEffect.EXTERNAL
    assert DeleteFileTool.spec.effect is ToolEffect.EXTERNAL
    assert WriteFileTool.spec.effect is ToolEffect.EXTERNAL
    assert RememberKnowledgeTool.spec.effect is ToolEffect.EXTERNAL
    assert ReadFileTool.spec.effect is ToolEffect.EXTERNAL
    assert ListFilesTool.spec.effect is ToolEffect.NONE
    assert ListArtifactsTool.spec.effect is ToolEffect.NONE
    assert SearchKnowledgeTool.spec.effect is ToolEffect.NONE
    assert ListKnowledgeTool.spec.effect is ToolEffect.NONE
    assert ReadKnowledgeTool.spec.effect is ToolEffect.NONE
    assert SubagentTool(_FakeSubagentRuntime(), agents={"helper": "helper"}).spec.effect is (
        ToolEffect.EXTERNAL
    )
    assert SubagentResultTool(InMemorySessionStore()).spec.effect is ToolEffect.NONE


def test_remember_knowledge_ambiguous_failure_event_is_bounded_and_content_free() -> None:
    knowledge_canary = "private knowledge content must not enter failure evidence"
    exception_canary = "private store diagnostic must not enter failure evidence"

    class AmbiguousKnowledgeStore(_TestKnowledgeStore):
        async def publish_entry_revision(
            self,
            entry,
            chunks,
            *,
            operation_id,
            expected_revision=None,
            activation_authority=None,
        ):
            del activation_authority
            raise RuntimeError(f"{exception_canary}: {knowledge_canary}")

    async def run():
        session_store = InMemorySessionStore()
        knowledge_store = AmbiguousKnowledgeStore()
        app = CayuApp(session_store=session_store, enable_logging=False)
        app.register_provider(
            _ScriptedProvider(
                [("call_remember", "remember_knowledge", {"text": knowledge_canary})]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="knowledge-test"),
                knowledge_store=knowledge_store,
            ),
            default=True,
        )
        tool = RememberKnowledgeTool(spec=RememberKnowledgeTool.spec.model_copy(deep=True))
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )
        events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_remember_failure_event",
                messages=[Message.text("user", "remember this")],
            ),
        )
        private_events = await private_events_for_public_events(session_store, events)
        transcript = await session_store.load_transcript("s_remember_failure_event")
        return events, private_events, transcript

    events, private_events, transcript = asyncio.run(run())

    public_failure = next(event for event in events if event.type == EventType.TOOL_CALL_FAILED)
    private_failure = next(
        event for event in private_events if event.type == EventType.TOOL_CALL_FAILED
    )
    for failure in (public_failure, private_failure):
        result = failure.payload["result"]
        encoded = json.dumps(result, sort_keys=True)
        event_payload = json.dumps(failure.payload, sort_keys=True)
        assert result["is_error"] is True
        assert result["content"] == (
            "Failed to store knowledge safely. No unowned cleanup was attempted."
        )
        assert result["structured"]["error"] == "knowledge_write_failed"
        assert result["structured"]["outcome"] == "ambiguous_publication"
        assert result["structured"]["cleanup"] == "not_attempted_unowned"
        assert failure.payload["arguments_state"] == "unavailable"
        assert "arguments" not in failure.payload
        assert len(encoded.encode("utf-8")) <= 1_024
        assert knowledge_canary not in event_payload
        assert exception_canary not in event_payload

    tool_call = next(
        part for message in transcript for part in message.content if isinstance(part, ToolCallPart)
    )
    assert tool_call.arguments == {}
    tool_result = next(
        part
        for message in transcript
        for part in message.content
        if isinstance(part, ToolResultPart)
    )
    transcript_payload = json.dumps(tool_result.model_dump(mode="json"), sort_keys=True)
    assert knowledge_canary not in transcript_payload
    assert exception_canary not in transcript_payload


def test_tool_timeout_consumes_only_its_owned_cancellation_request() -> None:
    class SlowTool(Tool):
        spec = ToolSpec(
            name="slow_tool",
            description="Wait beyond the tool deadline.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def run() -> tuple[tool_execution.ToolExecutionOutcome, int]:
        outcome = await tool_execution.run_tool(
            tool=SlowTool(),
            effect=ToolEffect.NONE,
            ctx=ToolContext(session_id="tool-timeout-cancellation-owner"),
            arguments={},
            redactor=SecretRedactor,
            timeout_seconds=0.01,
        )
        current_task = asyncio.current_task()
        assert current_task is not None
        return outcome, current_task.cancelling()

    outcome, cancellation_requests = asyncio.run(run())

    assert outcome.result.is_error is True
    assert outcome.result.content == "Tool call timed out after 0.01 seconds."
    assert cancellation_requests == 0


def test_tool_timeout_isolated_owner_preserves_later_caller_cancellation() -> None:
    class CancellationResettingTool(Tool):
        spec = ToolSpec(
            name="cancellation_resetting_tool",
            description="Consume the child deadline before caller cancellation.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.deadline_consumed = asyncio.Event()

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                child_task = asyncio.current_task()
                assert child_task is not None
                child_task.uncancel()
                self.deadline_consumed.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def run() -> tuple[asyncio.CancelledError, int]:
        tool = CancellationResettingTool()
        execution = asyncio.create_task(
            tool_execution.run_tool(
                tool=tool,
                effect=ToolEffect.NONE,
                ctx=ToolContext(session_id="isolated-tool-timeout-cancellation"),
                arguments={},
                redactor=SecretRedactor,
                timeout_seconds=0.01,
            )
        )
        await asyncio.wait_for(tool.deadline_consumed.wait(), timeout=1)
        execution.cancel("caller cancellation after tool deadline")
        with pytest.raises(
            asyncio.CancelledError,
            match="caller cancellation after tool deadline",
        ) as raised:
            await execution
        assert execution.cancelled() is True
        return raised.value, execution.cancelling()

    cancellation, cancellation_requests = asyncio.run(run())

    assert cancellation.args == ("caller cancellation after tool deadline",)
    assert cancellation_requests == 1


def test_remember_knowledge_timeout_returns_while_owned_publication_finishes() -> None:
    async def run():
        session_store = InMemorySessionStore()
        knowledge_store = _StalledKnowledgePublicationStore()
        tool = RememberKnowledgeTool()
        app = CayuApp(
            session_store=session_store,
            enable_logging=False,
            tool_timeout_seconds=0.05,
        )
        app.register_provider(
            _ScriptedProvider(
                [("call_remember", "remember_knowledge", {"text": "Retained knowledge."})]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="knowledge-test"),
                knowledge_store=knowledge_store,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )
        events = await asyncio.wait_for(
            _collect(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="s_remember_stalled_publication",
                    messages=[Message.text("user", "remember this")],
                ),
            ),
            timeout=1,
        )
        assert knowledge_store.dispatched.is_set()
        assert knowledge_store.settled.is_set() is False
        assert knowledge_store.publish_calls == 1
        assert len(tool._publication_owner) == 1
        knowledge_store.release.set()
        await asyncio.wait_for(knowledge_store.settled.wait(), timeout=2)

        async def wait_for_owner_release() -> None:
            while tool._publication_owner:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_owner_release(), timeout=2)
        operation_id = knowledge_store.operation_id
        assert operation_id is not None
        receipt = await knowledge_store.load_entry_publication_receipt(operation_id)
        return events, receipt

    events, receipt = asyncio.run(run())

    timed_out = next(event for event in events if event.type is EventType.TOOL_CALL_FAILED)
    assert timed_out.payload["terminal_outcome"] == "tool_execution_timeout"
    assert timed_out.payload["result"]["content"] == "Tool call timed out after 0.05 seconds."
    assert timed_out.payload["arguments_state"] == "unavailable"
    assert receipt is not None


def test_remember_knowledge_operator_interrupt_returns_while_publication_finishes() -> None:
    async def run():
        session_store = InMemorySessionStore()
        knowledge_store = _StalledKnowledgePublicationStore()
        tool = RememberKnowledgeTool()
        app = CayuApp(session_store=session_store, enable_logging=False)
        app.register_provider(
            _ScriptedProvider(
                [("call_remember", "remember_knowledge", {"text": "Retained knowledge."})]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="knowledge-test"),
                knowledge_store=knowledge_store,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )
        run_task = asyncio.create_task(
            _collect(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="s_remember_stalled_publication_interrupt",
                    messages=[Message.text("user", "remember this")],
                ),
            )
        )
        await asyncio.wait_for(knowledge_store.dispatched.wait(), timeout=1)
        interrupt_events = await asyncio.wait_for(
            _collect_interrupt(
                app,
                InterruptSessionRequest(
                    session_id="s_remember_stalled_publication_interrupt",
                    reason="operator interrupt",
                ),
            ),
            timeout=1,
        )
        run_events = await asyncio.wait_for(run_task, timeout=1)
        assert knowledge_store.settled.is_set() is False
        assert knowledge_store.publish_calls == 1
        assert len(tool._publication_owner) == 1
        knowledge_store.release.set()
        await asyncio.wait_for(knowledge_store.settled.wait(), timeout=2)

        async def wait_for_owner_release() -> None:
            while tool._publication_owner:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_owner_release(), timeout=2)
        operation_id = knowledge_store.operation_id
        assert operation_id is not None
        receipt = await knowledge_store.load_entry_publication_receipt(operation_id)
        return run_events, interrupt_events, receipt

    run_events, interrupt_events, receipt = asyncio.run(run())

    assert any(event.type is EventType.SESSION_INTERRUPTED for event in run_events)
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert receipt is not None


def test_app_shutdown_seals_and_bounds_registered_knowledge_publications() -> None:
    class ShutdownStalledStore(_StalledKnowledgePublicationStore):
        def __init__(self) -> None:
            super().__init__()
            self.stopped = asyncio.Event()
            self.receipt_reads = 0

        async def load_entry_publication_receipt(self, operation_id):
            self.receipt_reads += 1
            return await super().load_entry_publication_receipt(operation_id)

        async def publish_entry_revision(
            self,
            entry,
            chunks,
            *,
            operation_id,
            expected_revision=None,
            activation_authority=None,
        ):
            try:
                return await super().publish_entry_revision(
                    entry,
                    chunks,
                    operation_id=operation_id,
                    expected_revision=expected_revision,
                    activation_authority=activation_authority,
                )
            finally:
                self.stopped.set()

    async def run():
        store = ShutdownStalledStore()
        tool = RememberKnowledgeTool()
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )
        context = ToolContext(
            session_id="shutdown-publication",
            idempotency_key="shutdown-publication-operation",
            knowledge_store=store,
        )
        invocation = asyncio.create_task(tool.run(context, {"text": "Retained knowledge."}))
        await asyncio.wait_for(store.dispatched.wait(), timeout=1)
        invocation.cancel("caller left")
        with pytest.raises(asyncio.CancelledError, match="caller left"):
            await invocation

        first_close = asyncio.create_task(app.drain_knowledge_publications(timeout_s=0.01))
        await asyncio.sleep(0)
        second_close = asyncio.create_task(app.drain_knowledge_publications(timeout_s=1))
        first_result, second_result = await asyncio.gather(first_close, second_close)
        await asyncio.wait_for(store.stopped.wait(), timeout=1)
        while tool._publication_owner:
            await asyncio.sleep(0)

        receipt_reads_before_rejection = store.receipt_reads
        rejected = await tool.run(
            context.model_copy(update={"idempotency_key": "post-shutdown-operation"}),
            {"text": "New knowledge after shutdown."},
        )
        late_tool = RememberKnowledgeTool()
        app.register_agent(
            AgentSpec(name="late-agent", model="fake-model"),
            tools=[late_tool],
        )
        late_rejected = await late_tool.run(
            context.model_copy(update={"idempotency_key": "late-registration-operation"}),
            {"text": "Late registered knowledge."},
        )
        return (
            first_result,
            second_result,
            rejected,
            late_rejected,
            store.publish_calls,
            receipt_reads_before_rejection,
            store.receipt_reads,
        )

    (
        first_result,
        second_result,
        rejected,
        late_rejected,
        publish_calls,
        receipt_reads_before_rejection,
        receipt_reads,
    ) = asyncio.run(run())

    assert first_result is False
    assert second_result is False
    assert rejected.is_error is True
    assert rejected.structured["outcome"] == "publication_owner_closed"
    assert late_rejected.is_error is True
    assert late_rejected.structured["outcome"] == "publication_owner_closed"
    assert publish_calls == 1
    assert receipt_reads == receipt_reads_before_rejection


def test_failed_registration_after_shutdown_does_not_seal_an_unregistered_tool() -> None:
    app = CayuApp(enable_logging=False)
    app.seal_knowledge_publications()
    tool = RememberKnowledgeTool()

    with pytest.raises(ValueError, match="Duplicate tool registered for agent"):
        app.register_agent(
            AgentSpec(name="rejected-agent", model="fake-model"),
            tools=[tool, tool],
        )

    assert tool._publication_owner.sealed is False


@pytest.mark.parametrize("read_phase", ["receipt", "entry"])
def test_remember_knowledge_timeout_abandons_cancellation_resistant_read(
    read_phase: str,
) -> None:
    async def run():
        session_store = InMemorySessionStore()
        knowledge_store = _CancellationResistantKnowledgeReadStore(phase=read_phase)
        tool = RememberKnowledgeTool()
        app = CayuApp(
            session_store=session_store,
            enable_logging=False,
            tool_timeout_seconds=0.05,
        )
        app.register_provider(
            _ScriptedProvider(
                [("call_remember", "remember_knowledge", {"text": "Read timeout knowledge."})]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="knowledge-test"),
                knowledge_store=knowledge_store,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )
        events = await asyncio.wait_for(
            _collect(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=f"s_remember_stalled_{read_phase}_timeout",
                    messages=[Message.text("user", "remember this")],
                ),
            ),
            timeout=1,
        )
        assert knowledge_store.read_started.is_set()
        assert knowledge_store.read_finished.is_set() is False
        assert knowledge_store.publish_calls == 0
        assert len(tool._read_operations) == 1
        knowledge_store.release.set()

        async def wait_for_read_finish() -> None:
            while not knowledge_store.read_finished.is_set():
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_read_finish(), timeout=1)

        async def wait_for_read_release() -> None:
            while tool._read_operations:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_read_release(), timeout=1)
        return events

    events = asyncio.run(run())

    timed_out = next(event for event in events if event.type is EventType.TOOL_CALL_FAILED)
    assert timed_out.payload["terminal_outcome"] == "tool_execution_timeout"
    assert timed_out.payload["arguments_state"] == "unavailable"


@pytest.mark.parametrize("read_phase", ["receipt", "entry"])
def test_remember_knowledge_operator_interrupt_abandons_cancellation_resistant_read(
    read_phase: str,
) -> None:
    async def run():
        session_store = InMemorySessionStore()
        knowledge_store = _CancellationResistantKnowledgeReadStore(phase=read_phase)
        tool = RememberKnowledgeTool()
        app = CayuApp(session_store=session_store, enable_logging=False)
        app.register_provider(
            _ScriptedProvider(
                [("call_remember", "remember_knowledge", {"text": "Read interrupt knowledge."})]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="knowledge-test"),
                knowledge_store=knowledge_store,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )
        session_id = f"s_remember_stalled_{read_phase}_interrupt"
        run_task = asyncio.create_task(
            _collect(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "remember this")],
                ),
            )
        )

        async def wait_for_read_start() -> None:
            while not knowledge_store.read_started.is_set():
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_read_start(), timeout=1)
        interrupt_events = await asyncio.wait_for(
            _collect_interrupt(
                app,
                InterruptSessionRequest(session_id=session_id, reason="operator interrupt"),
            ),
            timeout=1,
        )
        run_events = await asyncio.wait_for(run_task, timeout=1)
        assert knowledge_store.read_finished.is_set() is False
        assert knowledge_store.publish_calls == 0
        assert len(tool._read_operations) == 1
        knowledge_store.release.set()

        async def wait_for_read_finish() -> None:
            while not knowledge_store.read_finished.is_set():
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_read_finish(), timeout=1)

        async def wait_for_read_release() -> None:
            while tool._read_operations:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_read_release(), timeout=1)
        return run_events, interrupt_events

    run_events, interrupt_events = asyncio.run(run())

    assert any(event.type is EventType.SESSION_INTERRUPTED for event in run_events)
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]


@pytest.mark.parametrize("failure_kind", ["hostile_subclass", "mutated_reason"])
def test_remember_knowledge_detaches_hostile_conflict_classification_at_runtime(
    failure_kind: str,
    caplog,
    capsys,
) -> None:
    canary = f"private conflict diagnostic from {failure_kind}"

    class HostileConflict(KnowledgePublicationConflict):
        def __getattribute__(self, name):
            if name == "reason":
                raise RuntimeError(canary)
            return super().__getattribute__(name)

    class HostileReason:
        def __hash__(self):
            raise RuntimeError(canary)

        def __eq__(self, other):
            del other
            raise RuntimeError(canary)

        def __repr__(self):
            raise RuntimeError(canary)

        def __str__(self):
            raise RuntimeError(canary)

    if failure_kind == "hostile_subclass":
        failure = HostileConflict("entry_occupied")
    else:
        failure = KnowledgePublicationConflict("entry_occupied")
        cast("Any", failure).reason = HostileReason()

    class HostileConflictStore(_TestKnowledgeStore):
        async def publish_entry_revision(
            self,
            entry,
            chunks,
            *,
            operation_id,
            expected_revision=None,
            activation_authority=None,
        ):
            del entry, chunks, operation_id, activation_authority
            raise failure

    async def run():
        session_store = InMemorySessionStore()
        app = CayuApp(session_store=session_store, enable_logging=False)
        app.register_provider(
            _ScriptedProvider(
                [("call_remember", "remember_knowledge", {"text": "Safe public knowledge."})]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="knowledge-test"),
                knowledge_store=HostileConflictStore(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RememberKnowledgeTool()],
        )
        events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"s_hostile_conflict_{failure_kind}",
                messages=[Message.text("user", "remember this")],
            ),
        )
        private_events = await private_events_for_public_events(session_store, events)
        transcript = await session_store.load_transcript(f"s_hostile_conflict_{failure_kind}")
        return events, private_events, transcript

    events, private_events, transcript = asyncio.run(run())
    failure_event = next(event for event in events if event.type == EventType.TOOL_CALL_FAILED)
    assert failure_event.payload["result"]["structured"]["outcome"] == "ambiguous_publication"
    rendered = repr(
        {
            "public": [event.model_dump(mode="json") for event in events],
            "private": [event.model_dump(mode="json") for event in private_events],
            "transcript": [message.model_dump(mode="json") for message in transcript],
        }
    )
    captured = capsys.readouterr()
    assert canary not in rendered
    assert canary not in caplog.text
    assert canary not in captured.out
    assert canary not in captured.err


def test_remember_knowledge_omits_unconfirmed_receipt_id_from_failure_evidence(
    caplog,
    capsys,
) -> None:
    canary = "private_receipt_entry_id_canary"

    class ConflictingReceiptStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.receipts: dict[str, KnowledgePublicationReceipt] = {}
            self.publish_calls = 0

        async def load_entry_publication_receipt(self, operation_id):
            receipt = self.receipts.get(operation_id)
            if receipt is None:
                committed_at = datetime.now(UTC)
                receipt = KnowledgePublicationReceipt(
                    operation_id=operation_id,
                    entry_id=canary,
                    entry_revision=1,
                    expected_revision=None,
                    request_sha256="0" * 64,
                    entry_created_at=committed_at,
                    entry_updated_at=committed_at,
                    committed_at=committed_at,
                )
                self.receipts[operation_id] = receipt
            return receipt

        async def publish_entry_revision(
            self,
            entry,
            chunks,
            *,
            operation_id,
            expected_revision=None,
            activation_authority=None,
        ):
            del entry, chunks, operation_id, activation_authority
            self.publish_calls += 1
            raise AssertionError("A prior receipt must prevent publication dispatch.")

    async def run():
        session_store = InMemorySessionStore()
        knowledge_store = ConflictingReceiptStore()
        app = CayuApp(session_store=session_store, enable_logging=False)
        app.register_provider(
            _ScriptedProvider(
                [("call_remember", "remember_knowledge", {"text": "Safe public knowledge."})]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="knowledge-test"),
                knowledge_store=knowledge_store,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RememberKnowledgeTool()],
        )
        events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_conflicting_receipt_id",
                messages=[Message.text("user", "remember this")],
            ),
        )
        private_events = await private_events_for_public_events(session_store, events)
        transcript = await session_store.load_transcript("s_conflicting_receipt_id")
        return events, private_events, transcript, knowledge_store.publish_calls

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        events, private_events, transcript, publish_calls = asyncio.run(run())

    public_failure = next(event for event in events if event.type == EventType.TOOL_CALL_FAILED)
    private_failure = next(
        event for event in private_events if event.type == EventType.TOOL_CALL_FAILED
    )
    for failure in (public_failure, private_failure):
        assert failure.payload["result"]["structured"] == {
            "error": "knowledge_write_failed",
            "outcome": "activation_receipt_missing",
            "cleanup": "not_attempted_unowned",
        }
    rendered = repr(
        {
            "public": [event.model_dump(mode="json") for event in events],
            "private": [event.model_dump(mode="json") for event in private_events],
            "transcript": [message.model_dump(mode="json") for message in transcript],
        }
    )
    captured = capsys.readouterr()
    assert publish_calls == 0
    assert canary not in rendered
    assert canary not in repr([str(warning.message) for warning in caught_warnings])
    assert canary not in caplog.text
    assert canary not in captured.out
    assert canary not in captured.err


def test_remember_knowledge_success_withholds_arguments_from_events_and_transcript(
    caplog,
    capsys,
) -> None:
    canaries = {
        "text": "private_knowledge_text_canary",
        "title": "private_knowledge_title_canary",
        "kind": "private_knowledge_kind_canary",
        "aspect": "private_knowledge_aspect_canary",
    }
    arguments = {
        "text": canaries["text"],
        "title": canaries["title"],
        "kind": canaries["kind"],
        "aspects": [canaries["aspect"]],
    }

    async def run():
        session_store = InMemorySessionStore()
        knowledge_store = _TestKnowledgeStore()
        app = CayuApp(session_store=session_store, enable_logging=False)
        provider = _ScriptedProvider(
            [
                ("call_remember_first", "remember_knowledge", arguments),
                ("call_remember_again", "remember_knowledge", arguments),
            ]
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="knowledge-test"),
                knowledge_store=knowledge_store,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RememberKnowledgeTool()],
        )
        events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_remember_success_projection",
                messages=[Message.text("user", "remember this")],
            ),
        )
        private_events = await private_events_for_public_events(session_store, events)
        transcript = await session_store.load_transcript("s_remember_success_projection")
        completed = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
        entry_id = completed[0].payload["result"]["structured"]["entry"]["entry_id"]
        stored_entry = await knowledge_store.get_entry(entry_id)
        return events, private_events, transcript, stored_entry, provider.requests

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        events, private_events, transcript, stored_entry, provider_requests = asyncio.run(run())

    assert stored_entry is not None
    assert stored_entry.text == canaries["text"]
    assert stored_entry.title == canaries["title"]
    assert stored_entry.kind == canaries["kind"]
    assert stored_entry.aspects == [canaries["aspect"]]

    public_completed = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
    private_completed = [
        event for event in private_events if event.type is EventType.TOOL_CALL_COMPLETED
    ]
    assert len(public_completed) == 2
    assert len(private_completed) == 2
    for completed in (*public_completed, *private_completed):
        assert completed.payload["arguments_state"] == "unavailable"
        assert "arguments" not in completed.payload
        assert "effective_arguments" not in completed.payload
        structured = completed.payload["result"]["structured"]
        assert structured["entry"] == {
            "entry_id": stored_entry.id,
            "revision": stored_entry.revision,
            "status": "pending",
        }
    assert [event.payload["result"]["structured"]["written"] for event in public_completed] == [
        True,
        False,
    ]
    assert [
        event.payload["result"]["structured"]["already_known"] for event in public_completed
    ] == [False, True]

    rendered = repr(
        {
            "public": [event.model_dump(mode="json") for event in events],
            "private": [event.model_dump(mode="json") for event in private_events],
            "transcript": [message.model_dump(mode="json") for message in transcript],
            "provider_messages": [
                message.model_dump(mode="json") for message in provider_requests[1].messages
            ],
        }
    )
    captured = capsys.readouterr()
    for canary in canaries.values():
        assert canary not in rendered
        assert canary not in repr([str(warning.message) for warning in caught_warnings])
        assert canary not in caplog.text
        assert canary not in captured.out
        assert canary not in captured.err


@pytest.mark.parametrize(
    ("boundary", "expected_event_type"),
    [
        ("policy_denial", EventType.TOOL_CALL_BLOCKED),
        ("hook_block", EventType.TOOL_CALL_BLOCKED),
        ("hook_short_circuit", EventType.TOOL_CALL_COMPLETED),
        ("hook_after_modify", EventType.TOOL_CALL_COMPLETED),
        ("policy_reauthorization", EventType.TOOL_CALL_BLOCKED),
    ],
)
def test_private_tool_arguments_cannot_reenter_terminal_output_through_extensions(
    boundary: str,
    expected_event_type: EventType,
    caplog,
    capsys,
) -> None:
    canaries = {
        "text": f"private_{boundary}_knowledge_text",
        "decision": f"private_{boundary}_extension_output",
    }

    class ArgumentEchoPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            if boundary == "policy_denial":
                return ToolPolicyResult(
                    decision=ToolPolicyDecision.DENY,
                    reason=canaries["decision"],
                    metadata={"private": canaries["decision"]},
                )
            if (
                boundary == "policy_reauthorization"
                and request.arguments.get("title") == canaries["decision"]
            ):
                return ToolPolicyResult(
                    decision=ToolPolicyDecision.DENY,
                    reason=canaries["decision"],
                    metadata={"private": canaries["decision"]},
                )
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    class ArgumentEchoHook(RuntimeHook):
        captured: str | None = None

        async def before_tool_call(
            self,
            context: BeforeToolCallHookContext,
        ) -> BeforeToolCallDecision | None:
            assert context.arguments["text"] == canaries["text"]
            self.captured = context.arguments["text"]
            if boundary == "hook_block":
                return BeforeToolCallDecision(
                    action="block",
                    block_reason=canaries["decision"],
                )
            if boundary == "hook_short_circuit":
                return BeforeToolCallDecision(
                    action="short_circuit",
                    synthetic_result=ToolResult(
                        content=canaries["decision"],
                        structured={"private": canaries["decision"]},
                        artifacts=[{"private": canaries["decision"]}],
                    ),
                )
            if boundary == "policy_reauthorization":
                return BeforeToolCallDecision(
                    action="proceed_modified",
                    modified_arguments={
                        **context.arguments,
                        "title": canaries["decision"],
                    },
                )
            return None

        async def after_tool_call(
            self,
            context: ToolCallHookContext,
        ) -> AfterToolCallDecision | None:
            if boundary != "hook_after_modify":
                return None
            assert context.arguments == {}
            assert self.captured == canaries["text"]
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(
                    content=self.captured,
                    structured={"private": self.captured},
                ),
            )

    async def run():
        session_store = InMemorySessionStore()
        provider = _ScriptedProvider(
            [
                (
                    "call_remember",
                    "remember_knowledge",
                    {"text": canaries["text"]},
                )
            ]
        )
        app = CayuApp(
            session_store=session_store,
            enable_logging=False,
            runtime_hooks=[ArgumentEchoHook()],
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="knowledge-test"),
                knowledge_store=_TestKnowledgeStore(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RememberKnowledgeTool()],
            tool_policy=ArgumentEchoPolicy(),
        )
        session_id = f"s_private_argument_{boundary}"
        events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "remember this")],
            ),
        )
        private_events = await private_events_for_public_events(session_store, events)
        transcript = await session_store.load_transcript(session_id)
        return events, private_events, transcript, provider.requests

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        events, private_events, transcript, provider_requests = asyncio.run(run())

    terminal = next(event for event in events if event.type is expected_event_type)
    assert terminal.payload["arguments_state"] == "unavailable"
    assert "arguments" not in terminal.payload
    assert "effective_arguments" not in terminal.payload
    rendered = repr(
        {
            "public": [event.model_dump(mode="json") for event in events],
            "private": [event.model_dump(mode="json") for event in private_events],
            "transcript": [message.model_dump(mode="json") for message in transcript],
            "followup_requests": [
                request.model_dump(mode="json") for request in provider_requests[1:]
            ],
        }
    )
    captured = capsys.readouterr()
    for canary in canaries.values():
        assert canary not in rendered
        assert canary not in repr([str(warning.message) for warning in caught_warnings])
        assert canary not in caplog.text
        assert canary not in captured.out
        assert canary not in captured.err


def test_private_tool_policy_output_stays_quarantined_across_approval_resume(
    caplog,
    capsys,
) -> None:
    knowledge_canary = "private_approval_knowledge_text"
    policy_canary = "private_approval_policy_output"

    class ApprovalPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            if request.tool_name == "remember_knowledge":
                assert request.arguments["text"] == knowledge_canary
                return ToolPolicyResult(
                    decision=ToolPolicyDecision.DENY,
                    reason=policy_canary,
                    metadata={"private": policy_canary},
                )
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason="Approve the sibling recording tool.",
            )

    async def run():
        session_store = InMemorySessionStore()
        provider = _ScriptedProvider(
            [
                (
                    "call_remember",
                    "remember_knowledge",
                    {"text": knowledge_canary},
                ),
                ("call_gate", "recording_tool", {"tag": "approved"}),
            ]
        )
        recorder = _Recorder()
        app = CayuApp(session_store=session_store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="knowledge-test"),
                knowledge_store=_TestKnowledgeStore(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RememberKnowledgeTool(), _RecordingTool(recorder)],
            tool_policy=ApprovalPolicy(),
        )
        session_id = "s_private_argument_approval_resume"
        initial = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "remember this")],
            ),
        )
        requested = next(
            event for event in initial if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval = requested.payload["approval"]
        checkpoint = await session_store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["pending_tool_approval"]["publish_arguments"] is False
        pending_actions = await session_store.query_pending_actions(
            PendingActionQuery(session_id=session_id)
        )
        resumed = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=session_id,
                    approval_id=approval["approval_id"],
                    tool_round_id=approval["tool_round_id"],
                    tool_call_id=approval["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        events = [*initial, *resumed]
        private_events = await private_events_for_public_events(session_store, events)
        transcript = await session_store.load_transcript(session_id)
        return (
            events,
            private_events,
            transcript,
            provider.requests,
            pending_actions,
            recorder.completed,
        )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        (
            events,
            private_events,
            transcript,
            provider_requests,
            pending_actions,
            completed_tools,
        ) = asyncio.run(run())

    requested = next(
        event for event in events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    approval = requested.payload["approval"]
    assert approval["arguments_state"] == "quarantined"
    assert "arguments" not in approval
    assert "reason" not in approval
    assert "metadata" not in approval
    for call in approval["tool_calls"]:
        assert call["arguments_state"] == "quarantined"
        assert "arguments" not in call
        assert "reason" not in call
        assert "metadata" not in call
    assert len(pending_actions.actions) == 1
    assert pending_actions.actions[0].arguments is None
    assert pending_actions.actions[0].detail == "Approval required"
    assert completed_tools == ["approved"]
    recovered_denial = next(
        event
        for event in events
        if event.type is EventType.TOOL_CALL_BLOCKED and event.tool_name == "remember_knowledge"
    )
    assert recovered_denial.payload["arguments_state"] == "unavailable"
    rendered = repr(
        {
            "public": [event.model_dump(mode="json") for event in events],
            "private": [event.model_dump(mode="json") for event in private_events],
            "transcript": [message.model_dump(mode="json") for message in transcript],
            "followup_requests": [
                request.model_dump(mode="json") for request in provider_requests[1:]
            ],
        }
    )
    captured = capsys.readouterr()
    for canary in (knowledge_canary, policy_canary):
        assert canary not in rendered
        assert canary not in repr([str(warning.message) for warning in caught_warnings])
        assert canary not in caplog.text
        assert canary not in captured.out
        assert canary not in captured.err


@pytest.mark.parametrize(
    ("output_phase", "expected_event_type", "expected_outcome"),
    [
        ("entry_read", EventType.TOOL_CALL_COMPLETED, None),
        ("receipt_read", EventType.TOOL_CALL_FAILED, "receipt_read_failed"),
        ("publication", EventType.TOOL_CALL_COMPLETED, None),
    ],
)
def test_remember_knowledge_contains_store_output_cancellation_during_validation(
    output_phase: str,
    expected_event_type: EventType,
    expected_outcome: str | None,
    caplog,
    capsys,
) -> None:
    canary = f"private cancellation from {output_phase} output validation"

    class CancellingTimezone(tzinfo):
        def utcoffset(self, value):
            del value
            raise asyncio.CancelledError(canary)

        def dst(self, value):
            del value
            return None

    def hostile_datetime() -> datetime:
        return datetime(2026, 8, 13, 12, 0, tzinfo=CancellingTimezone())

    class ForgedOutputStore(_TestKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.publish_calls = 0

        async def get_entry(self, entry_id):
            if output_phase != "entry_read":
                return await super().get_entry(entry_id)
            return KnowledgeEntry(id=entry_id, text="Untrusted store material.").model_copy(
                update={"created_at": hostile_datetime()}
            )

        async def load_entry_publication_receipt(self, operation_id):
            receipt = await super().load_entry_publication_receipt(operation_id)
            if output_phase != "receipt_read" or receipt is not None:
                return receipt
            committed_at = datetime.now(UTC)
            return KnowledgePublicationReceipt(
                operation_id=operation_id,
                entry_id="forged-receipt-entry",
                entry_revision=1,
                expected_revision=None,
                request_sha256="0" * 64,
                entry_created_at=committed_at,
                entry_updated_at=committed_at,
                committed_at=committed_at,
            ).model_copy(update={"committed_at": hostile_datetime()})

        async def publish_entry_revision(
            self,
            entry,
            chunks,
            *,
            operation_id,
            expected_revision=None,
            activation_authority=None,
        ):
            self.publish_calls += 1
            receipt = await super().publish_entry_revision(
                entry,
                chunks,
                operation_id=operation_id,
                expected_revision=expected_revision,
                activation_authority=activation_authority,
            )
            if output_phase != "publication":
                return receipt
            return receipt.model_copy(update={"committed_at": hostile_datetime()})

    async def run():
        session_store = InMemorySessionStore()
        knowledge_store = ForgedOutputStore()
        app = CayuApp(session_store=session_store, enable_logging=False)
        app.register_provider(
            _ScriptedProvider(
                [("call_remember", "remember_knowledge", {"text": "Safe public knowledge."})]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="knowledge-test"),
                knowledge_store=knowledge_store,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RememberKnowledgeTool()],
        )
        events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"s_forged_{output_phase}_output",
                messages=[Message.text("user", "remember this")],
            ),
        )
        private_events = await private_events_for_public_events(session_store, events)
        transcript = await session_store.load_transcript(f"s_forged_{output_phase}_output")
        return events, private_events, transcript, knowledge_store.publish_calls

    async def run_and_observe_task():
        task = asyncio.create_task(run())
        result = await task
        return result, task.cancelling(), task.cancelled()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        (
            (events, private_events, transcript, publish_calls),
            cancelling_count,
            task_cancelled,
        ) = asyncio.run(run_and_observe_task())

    terminal = next(event for event in events if event.type == expected_event_type)
    if expected_outcome is None:
        assert terminal.payload["result"]["is_error"] is False
    else:
        assert terminal.payload["result"]["structured"]["outcome"] == expected_outcome
    expected_publish_calls = 0 if output_phase == "receipt_read" else 1
    assert publish_calls == expected_publish_calls
    assert cancelling_count == 0
    assert task_cancelled is False
    rendered = repr(
        {
            "public": [event.model_dump(mode="json") for event in events],
            "private": [event.model_dump(mode="json") for event in private_events],
            "transcript": [message.model_dump(mode="json") for message in transcript],
        }
    )
    captured = capsys.readouterr()
    assert canary not in rendered
    assert canary not in repr([str(warning.message) for warning in caught_warnings])
    assert canary not in caplog.text
    assert canary not in captured.out
    assert canary not in captured.err


def test_tool_idempotency_key_preserves_component_boundaries() -> None:
    first = tool_execution.tool_idempotency_key(
        session_id="session\x00round",
        tool_call_id="call",
        tool_round_id="approval",
    )
    second = tool_execution.tool_idempotency_key(
        session_id="session",
        tool_call_id="call",
        tool_round_id="round\x00approval",
    )

    assert first != second


def test_parallel_safe_tools_run_concurrently() -> None:
    recorder = _Recorder()
    app = _build(
        tools=[_RecordingTool(recorder, name="safe_tool")],
        tool_calls=[
            ("a", "safe_tool", {"tag": "a"}),
            ("b", "safe_tool", {"tag": "b"}),
        ],
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_concurrent",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    # The two parallel-safe calls overlapped in time.
    assert recorder.max_active == 2


def test_tool_context_receives_stable_idempotency_key_from_round_identity() -> None:
    recorder = _Recorder()
    app = _build(
        tools=[_RecordingTool(recorder, name="identity_tool")],
        tool_calls=[("call_1", "identity_tool", {"tag": "a"})],
    )

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_identity",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    private_events = asyncio.run(private_events_for_public_events(app.session_store, events))

    started = next(event for event in private_events if event.type == EventType.TOOL_CALL_STARTED)
    completed = next(
        event for event in private_events if event.type == EventType.TOOL_CALL_COMPLETED
    )

    key = started.payload["idempotency_key"]
    assert key.startswith("cayu-tool:v1:")
    assert len(key) == len("cayu-tool:v1:") + 64
    assert recorder.context_idempotency_keys == [key]
    assert recorder.metadata_idempotency_keys == [key]

    expected_key = tool_execution.tool_idempotency_key(
        session_id="s_identity",
        tool_round_id=started.payload["tool_round_id"],
        tool_call_id="call_1",
    )
    assert key == expected_key
    assert completed.payload["tool_round_id"] == started.payload["tool_round_id"]
    assert completed.payload["idempotency_key"] == key
    assert completed.payload["arguments_state"] == "finalized"
    assert completed.payload["arguments"] == {"tag": "a"}


def test_tool_effect_reaches_policy_started_event_and_tool_context_metadata() -> None:
    recorder = _Recorder()
    policy = _CapturePolicy()
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(_ScriptedProvider([("call_1", "idem_tool", {"tag": "a"})]), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[
            _RecordingTool(
                recorder,
                name="idem_tool",
                effect=ToolEffect.IDEMPOTENT,
            )
        ],
        tool_policy=policy,
    )

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_effect",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    started = next(event for event in events if event.type == EventType.TOOL_CALL_STARTED)
    assert started.payload["effect"] == "idempotent"
    assert [request.tool_effect for request in policy.requests] == [ToolEffect.IDEMPOTENT]
    assert recorder.metadata_tool_effects == ["idempotent"]


def test_parallel_safe_false_does_not_overlap_and_runs_after_the_batch() -> None:
    recorder = _Recorder()
    app = _build(
        tools=[
            _RecordingTool(recorder, name="safe_tool"),
            _RecordingTool(recorder, name="serial_tool", parallel_safe=False),
        ],
        tool_calls=[
            ("a", "safe_tool", {"tag": "a"}),
            ("b", "safe_tool", {"tag": "b"}),
            ("c", "serial_tool", {"tag": "c"}),
        ],
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_unsafe",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    # The two parallel-safe calls overlapped; the serial tool ran alone, after both finished.
    assert recorder.max_active == 2
    assert recorder.order.index("start:c") > recorder.order.index("end:a")
    assert recorder.order.index("start:c") > recorder.order.index("end:b")


def test_mixed_round_tool_results_keep_model_order() -> None:
    # Model emits [safe a, unsafe b, safe c]; execution runs the safe batch first then the unsafe
    # tool, but the tool_result parts must line up with the assistant tool-call order.
    recorder = _Recorder()
    app = _build(
        tools=[
            _RecordingTool(recorder, name="safe_tool"),
            _RecordingTool(recorder, name="serial_tool", parallel_safe=False),
        ],
        tool_calls=[
            ("a", "safe_tool", {"tag": "a"}),
            ("b", "serial_tool", {"tag": "b"}),
            ("c", "safe_tool", {"tag": "c"}),
        ],
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_order",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    transcript = asyncio.run(app.session_store.load_transcript("s_order"))
    tool_message = next(message for message in transcript if message.role == "tool")
    result_ids = [
        part.tool_call_id for part in tool_message.content if isinstance(part, ToolResultPart)
    ]
    assert result_ids == ["a", "b", "c"]


def test_parallel_safe_false_is_an_ordering_barrier_in_model_position() -> None:
    # [safe A, safe B, unsafe C, safe D]: A/B run concurrently, then the unsafe barrier C alone,
    # then D — preserving model order (NOT A/B/D before C).
    recorder = _Recorder()
    app = _build(
        tools=[
            _RecordingTool(recorder, name="safe_tool"),
            _RecordingTool(recorder, name="serial_tool", parallel_safe=False),
        ],
        tool_calls=[
            ("a", "safe_tool", {"tag": "a"}),
            ("b", "safe_tool", {"tag": "b"}),
            ("c", "serial_tool", {"tag": "c"}),
            ("d", "safe_tool", {"tag": "d"}),
        ],
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_barrier",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert recorder.max_active == 2  # A and B overlapped; the barrier never overlaps
    order = recorder.order
    # C runs only after both A and B finished, and D only after C finished.
    assert order.index("start:c") > order.index("end:a")
    assert order.index("start:c") > order.index("end:b")
    assert order.index("start:d") > order.index("end:c")
    parts = _tool_result_parts_ordered(app, "s_barrier")
    assert parts == ["a", "b", "c", "d"]


def test_parallel_safe_false_barrier_runs_before_later_safe_reads() -> None:
    # Regression (read-after-write): [unsafe C, safe A, safe B] must run C first (barrier at
    # position 0), so the reads never execute before the write.
    recorder = _Recorder()
    app = _build(
        tools=[
            _RecordingTool(recorder, name="safe_tool"),
            _RecordingTool(recorder, name="serial_tool", parallel_safe=False),
        ],
        tool_calls=[
            ("c", "serial_tool", {"tag": "c"}),
            ("a", "safe_tool", {"tag": "a"}),
            ("b", "safe_tool", {"tag": "b"}),
        ],
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_raw",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    order = recorder.order
    assert order.index("end:c") < order.index("start:a")
    assert order.index("end:c") < order.index("start:b")


def test_parallel_safe_uses_registered_declaration_copy() -> None:
    recorder = _Recorder()
    serial_tool = _RecordingTool(
        recorder,
        name="serial_tool",
        parallel_safe=False,
        effect=ToolEffect.IDEMPOTENT,
    )
    policy = _CapturePolicy()
    app = _build(
        tools=[
            serial_tool,
            _RecordingTool(recorder, name="safe_tool"),
        ],
        tool_calls=[
            ("c", "serial_tool", {"tag": "c"}),
            ("a", "safe_tool", {"tag": "a"}),
        ],
        tool_policy=policy,
    )

    serial_tool.spec = ToolSpec(
        name="serial_tool",
        description="mutated after registration",
        input_schema=_TOOL_SCHEMA,
        parallel_safe=True,
        effect=ToolEffect.NONE,
    )

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_registered_parallel_safe_copy",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    order = recorder.order
    assert order.index("end:c") < order.index("start:a")
    started = next(
        event
        for event in events
        if event.type == EventType.TOOL_CALL_STARTED and event.tool_name == "serial_tool"
    )
    assert started.payload["effect"] == "idempotent"
    assert policy.requests[0].tool_effect is ToolEffect.IDEMPOTENT


def _tool_result_parts_ordered(app: CayuApp, session_id: str) -> list[str]:
    transcript = asyncio.run(app.session_store.load_transcript(session_id))
    tool_message = next(m for m in transcript if m.role == "tool")
    return [p.tool_call_id for p in tool_message.content if isinstance(p, ToolResultPart)]


class _CancelTool(Tool):
    spec = ToolSpec(
        name="cancel_tool",
        description="raises CancelledError (simulates a leaked cancel scope)",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        raise asyncio.CancelledError()


def test_parallel_spontaneous_cancel_does_not_brick_the_round() -> None:
    # A parallel tool that raises CancelledError with no session interrupt must not leave the round
    # half-open (a dangling assistant tool-call with no tool_result bricks every later step). Each
    # un-terminated call is completed with a synthesized error result so the session stays runnable.
    recorder = _Recorder()
    app = _build(
        tools=[_CancelTool(), _RecordingTool(recorder, name="safe_tool")],
        tool_calls=[
            ("call_1", "cancel_tool", {}),
            ("call_2", "safe_tool", {"tag": "b"}),
        ],
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_cancel",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED  # not bricked
    transcript = asyncio.run(app.session_store.load_transcript("s_cancel"))
    tool_message = next(m for m in transcript if m.role == "tool")
    results = {p.tool_call_id: p for p in tool_message.content if isinstance(p, ToolResultPart)}
    assert set(results) == {"call_1", "call_2"}  # complete round — every call has a result
    assert results["call_1"].is_error is True  # synthesized abnormal-termination error
    abnormal_event = next(
        event
        for event in asyncio.run(private_events_for_public_events(app.session_store, events))
        if event.type == EventType.TOOL_CALL_FAILED
        and event.payload.get("abnormal_termination") is True
    )
    assert abnormal_event.payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="s_cancel",
        tool_round_id=abnormal_event.payload["tool_round_id"],
        tool_call_id="call_1",
    )
    durable_events = asyncio.run(app.session_store.load_events("s_cancel"))
    assert abnormal_event in durable_events


def test_max_parallel_one_runs_sequentially() -> None:
    # A global cap of 1 (CayuApp(max_parallel_tool_calls=1)) forces one-at-a-time execution even
    # for parallel-safe tools — the app-level off switch for concurrency.
    recorder = _Recorder()
    app = _build(
        tools=[_RecordingTool(recorder, name="safe_tool")],
        tool_calls=[
            ("a", "safe_tool", {"tag": "a"}),
            ("b", "safe_tool", {"tag": "b"}),
        ],
        max_parallel_tool_calls=1,
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_seq",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert recorder.max_active == 1


def test_limit_stop_mid_round_does_not_strand_later_segment_tool_calls(monkeypatch) -> None:
    # A parallel_safe=False barrier splits a round into multiple sequential segments. When a run
    # limit trips in a non-last segment, the limit-close must record a tool_result for EVERY
    # remaining round call (later segments included). Otherwise those assistant tool_calls dangle
    # with no matching tool_result and the session is unresumable. Regression test: the limit-close
    # is scoped to the whole round, not just the tripping segment.
    import cayu.runtime._session_engine as session_engine_module

    clock = {"value": 0.0}
    monkeypatch.setattr(session_engine_module.time, "monotonic", lambda: clock["value"])

    class _ClockAdvancingTool(Tool):
        spec = ToolSpec(
            name="advance_clock",
            description="unsafe barrier that advances the fake clock past the elapsed limit",
            input_schema={"type": "object", "properties": {}},
            parallel_safe=False,
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            clock["value"] = 1.0
            return ToolResult(content="advanced")

    recorder = _Recorder()
    app = _build(
        tools=[_ClockAdvancingTool(), _RecordingTool(recorder, name="reader")],
        tool_calls=[
            (
                "call_1",
                "advance_clock",
                {},
            ),  # segment 1 (barrier): runs, pushes elapsed to the limit
            ("call_2", "advance_clock", {}),  # segment 2 (barrier): limit trips before it runs
            ("call_3", "reader", {"tag": "c"}),  # segment 3: stranded without the whole-round close
        ],
        max_parallel_tool_calls=4,
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_limit_segment",
                messages=[Message.text("user", "go")],
                limits=RunLimits(max_elapsed_seconds=1),
            ),
        )
    )

    assert EventType.SESSION_LIMIT_REACHED in {event.type for event in events}
    assert "c" not in recorder.completed  # the later segment never executed after the limit tripped
    transcript = asyncio.run(app.session_store.load_transcript("s_limit_segment"))
    tool_message = next(m for m in transcript if m.role == "tool")
    result_ids = {p.tool_call_id for p in tool_message.content if isinstance(p, ToolResultPart)}
    assert result_ids == {"call_1", "call_2", "call_3"}  # every round call has a result — no dangle
