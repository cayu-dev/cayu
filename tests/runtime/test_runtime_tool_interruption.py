"""Tool interruption contracts."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from tests.runner_cancellation import cancelled_error_with_artifacts
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _assert_proxy_tool_effect_remains_unknown,
    assert_only_model_step_publication_checkpoint,
    collect_events,
    collect_interrupt_events,
)
from tests.runtime.runtime_tool_support import EchoTool, RequireApprovalPolicy, SideEffectTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.configuration import CayuConfig, ToolExecutionConfig
from cayu.context.base import validate_context_messages
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import Event, EventType
from cayu.messages import Message, MessageRole
from cayu.providers.base import ModelStreamEvent
from cayu.proxies.passthrough import PassthroughProxy
from cayu.runtime._interruption_coordinator import suppress_interruption_cascade
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import InterruptSessionRequest, RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult
from cayu.vaults.base import SecretRef
from cayu.vaults.static import StaticVault


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
        await _assert_proxy_tool_effect_remains_unknown(
            app.session_store, "sess_interrupt_proxy_authorization"
        )
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
    assert EventType.TOOL_CALL_FAILED not in stored_event_types
    assert EventType.TOOL_CALL_COMPLETED not in stored_event_types
    assert stored_event_types.count(EventType.TOOL_EFFECT_OUTCOME_UNKNOWN) == 1


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
        assert run_task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await run_task
        assert run_task.cancelled()
        assert run_task.cancelling() == 1
        await asyncio.wait_for(tool.cancelled.wait(), timeout=1)
        session_id = "sess_generic_cancel_proxy_authorization"
        await _assert_proxy_tool_effect_remains_unknown(app.session_store, session_id)
        return (
            await app.session_store.load_events(session_id),
            await app.session_store.load(session_id),
            await app.session_store.load_transcript(session_id),
            await app.session_store.load_checkpoint(session_id),
        )

    stored_events, session, transcript, checkpoint = asyncio.run(run())

    event_types = [event.type for event in stored_events]
    assert EventType.CREDENTIAL_PROXY_CHECKED not in event_types
    assert EventType.TOOL_CALL_FAILED not in event_types
    assert EventType.TOOL_CALL_COMPLETED not in event_types
    assert event_types.count(EventType.TOOL_EFFECT_OUTCOME_UNKNOWN) == 1
    assert event_types.count(EventType.SESSION_INTERRUPTED) == 1
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    validate_context_messages(transcript)
    assert not any(message.role == "tool" for message in transcript)
    assert checkpoint is not None
    assert "pending_tool_round" in checkpoint
    interrupted = next(
        event for event in stored_events if event.type == EventType.SESSION_INTERRUPTED
    )
    assert interrupted.payload["abandoned"] is True


def test_cancelled_runner_cleanup_diagnostics_are_preserved_without_tool_terminal():
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
                raise cancelled_error_with_artifacts(artifacts=[cleanup_artifact]) from exc
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
    assert not any(
        event.type in {EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_COMPLETED}
        for event in stored_events
    )
    cleanup_events = [
        event for event in stored_events if event.type is EventType.TOOL_EFFECT_CLEANUP_OBSERVED
    ]
    assert len(cleanup_events) == 1
    assert cleanup_events[0].payload["artifacts"] == [cleanup_artifact]
    unknown = [
        event for event in stored_events if event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
    ]
    assert len(unknown) == 1
    assert unknown[0].payload["tool_call_id"] == "call_1"
    validate_context_messages(transcript)
    assert not any(message.role == "tool" for message in transcript)


def test_cancelled_runner_cleanup_diagnostics_are_redacted_in_tool_result():
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

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
                raise cancelled_error_with_artifacts(artifacts=[cleanup_artifact]) from exc
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
    cleanup_event = next(
        event for event in stored_events if event.type == EventType.TOOL_EFFECT_CLEANUP_OBSERVED
    )
    assert secret_value not in str(cleanup_event.payload)
    artifact = cleanup_event.payload["artifacts"][0]
    assert artifact["stderr"] == f"failed with token {REDACTED_SECRET}"
    assert artifact["metadata"]["token"] == REDACTED_SECRET

    validate_context_messages(transcript)
    assert not any(message.role == "tool" for message in transcript)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_cancelled_runner_cleanup_diagnostics_are_attached_only_to_active_tool(
    tmp_path, backend, request
):
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
                raise cancelled_error_with_artifacts(artifacts=[cleanup_artifact]) from exc
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
    store = (
        InMemorySessionStore()
        if backend == "memory"
        else SQLiteSessionStore(tmp_path / "cleanup.db")
    )
    if backend == "sqlite":
        request.addfinalizer(lambda: asyncio.run(store.close()))
    app = CayuApp(
        session_store=store,
        config=CayuConfig(tool_execution=ToolExecutionConfig(max_parallel_tool_calls=1)),
    )
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
    assert failed_tool_events == []
    diagnostics = [
        event for event in stored_events if event.type == EventType.TOOL_EFFECT_CLEANUP_OBSERVED
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0].payload["tool_call_id"] == "call_active"
    assert diagnostics[0].payload["artifacts"] == [cleanup_artifact]
    assert diagnostics[0].payload["truncated"] is False
    unknown = [e for e in stored_events if e.type == EventType.TOOL_EFFECT_OUTCOME_UNKNOWN]
    assert [e.payload["tool_call_id"] for e in unknown] == ["call_active"]
    validate_context_messages(transcript)
    assert not any(message.role == "tool" for message in transcript)
    checkpoint = asyncio.run(
        app.session_store.load_checkpoint("sess_interrupt_tool_cleanup_diagnostics_round")
    )
    assert checkpoint is not None and "pending_tool_round" in checkpoint


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
    # Establish a completed prefix before the second call signals admission.
    app = CayuApp(config=CayuConfig(tool_execution=ToolExecutionConfig(max_parallel_tool_calls=1)))
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
    assert [event.type for event in stored_events].count(EventType.TOOL_CALL_FAILED) == 0
    completed = next(e for e in stored_events if e.type == EventType.TOOL_CALL_COMPLETED)
    assert completed.payload["tool_call_id"] == "call_echo"
    assert completed.payload["result"]["content"] == "first"
    assert completed.payload["result"]["is_error"] is False
    unknown = [e for e in stored_events if e.type == EventType.TOOL_EFFECT_OUTCOME_UNKNOWN]
    assert len(unknown) == 1
    assert unknown[0].payload["tool_call_id"] == "call_block"
    checkpoint = asyncio.run(app.session_store.load_checkpoint("sess_interrupt_partial_tool_round"))
    assert checkpoint is not None and "pending_tool_round" in checkpoint
    validate_context_messages(transcript)
    assert not any(message.role == "tool" for message in transcript)


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


def test_interrupt_session_closes_tool_round_after_assistant_tool_call_is_quarantined():
    class InterruptingAfterAssistantToolCallQuarantineStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.interrupt_after_next_assistant_tool_call_quarantine = False

        async def promote_model_completion_stage(
            self,
            session_id: str,
            *,
            stage_id: str,
            expected_run_epoch: int,
        ):
            result = await super().promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=expected_run_epoch,
            )
            checkpoint = await self.load_checkpoint(session_id)
            pending_round = checkpoint.get("pending_tool_round")
            if (
                self.interrupt_after_next_assistant_tool_call_quarantine
                and type(pending_round) is dict
                and pending_round.get("assistant_message_state") == "quarantined"
            ):
                self.interrupt_after_next_assistant_tool_call_quarantine = False
                await self.update_status(session_id, SessionStatus.INTERRUPTED)
            return result

    store = InterruptingAfterAssistantToolCallQuarantineStore()
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
        store.interrupt_after_next_assistant_tool_call_quarantine = True
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
        "model_step_id": transcript[-1].content[0].model_step_id,
        "model_attempt_id": transcript[-1].content[0].model_attempt_id,
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
    assert_only_model_step_publication_checkpoint(checkpoint)
    validate_context_messages(transcript)
    assert transcript[-1].role == "tool"
    assert transcript[-1].content[0].tool_call_id == "call_echo"
    assert transcript[-1].content[0].content == "Tool call interrupted before completion."
    assert transcript[-1].content[0].is_error is True


def test_remote_interrupt_wins_race_with_policy_plan_publication():
    class BlockingPolicyPublicationStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.policy_publication_started = asyncio.Event()
            self.allow_policy_publication = asyncio.Event()

        async def transform_checkpoint(self, session_id: str, checkpoint_transform) -> None:
            if getattr(checkpoint_transform, "__name__", "") == "publish_policy_plan":
                self.policy_publication_started.set()
                await self.allow_policy_publication.wait()
            await super().transform_checkpoint(session_id, checkpoint_transform)

    async def run() -> tuple[list[Event], list[Event], list[Event], list[Message]]:
        session_id = "sess_remote_interrupt_policy_publication"
        store = BlockingPolicyPublicationStore()
        provider = FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_echo",
                    name="echo",
                    arguments={"text": "must not execute"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        )
        run_app = CayuApp(session_store=store)
        interrupt_app = CayuApp(session_store=store)
        for app in (run_app, interrupt_app):
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[EchoTool()],
            )

        run_task = asyncio.create_task(
            collect_events(
                run_app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await asyncio.wait_for(store.policy_publication_started.wait(), timeout=10)
        with suppress_interruption_cascade():
            interrupt_task = asyncio.create_task(
                collect_interrupt_events(
                    interrupt_app,
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="remote operator stop",
                    ),
                )
            )
        for _ in range(100):
            current = await store.load(session_id)
            if current is not None and current.status is SessionStatus.INTERRUPTING:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("Remote interruption did not claim the session.")
        store.allow_policy_publication.set()
        run_events = await asyncio.wait_for(run_task, timeout=5)
        interrupt_events = await asyncio.wait_for(interrupt_task, timeout=5)
        return (
            run_events,
            interrupt_events,
            await store.load_events(session_id),
            await store.load_transcript(session_id),
        )

    run_events, interrupt_events, stored_events, transcript = asyncio.run(run())

    assert interrupt_events[-1].type is EventType.SESSION_INTERRUPTED
    assert run_events[-1].type is EventType.SESSION_INTERRUPTED
    assert EventType.SESSION_FAILED not in [event.type for event in stored_events]
    assert [event.type for event in stored_events].count(EventType.TOOL_CALL_FAILED) == 1
    validate_context_messages(transcript)
    assert transcript[-1].role == "tool"
    assert transcript[-1].content[0].tool_call_id == "call_echo"
    assert transcript[-1].content[0].content == "Tool call interrupted before completion."


def test_remote_interrupt_wins_race_with_atomic_approval_publication():
    class BlockingApprovalPublicationStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.approval_publication_started = asyncio.Event()
            self.allow_approval_publication = asyncio.Event()

        async def publish_runtime_publication(self, session_id: str, **kwargs):
            if kwargs["request"].kind == "approval-open":
                self.approval_publication_started.set()
                await self.allow_approval_publication.wait()
            return await super().publish_runtime_publication(session_id, **kwargs)

    async def run() -> tuple[list[Event], list[Event], dict[str, Any] | None]:
        session_id = "sess_remote_interrupt_approval_publication"
        store = BlockingApprovalPublicationStore()
        provider = FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_side_effect",
                    name="side_effect",
                    arguments={"value": "must not execute"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        )
        tool = SideEffectTool()
        run_app = CayuApp(session_store=store)
        interrupt_app = CayuApp(session_store=store)
        for app in (run_app, interrupt_app):
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[tool],
                tool_policy=RequireApprovalPolicy(),
            )

        run_task = asyncio.create_task(
            collect_events(
                run_app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await asyncio.wait_for(store.approval_publication_started.wait(), timeout=10)
        with suppress_interruption_cascade():
            interrupt_task = asyncio.create_task(
                collect_interrupt_events(
                    interrupt_app,
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="remote operator stop",
                    ),
                )
            )
        for _ in range(100):
            current = await store.load(session_id)
            if current is not None and current.status is SessionStatus.INTERRUPTING:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("Remote interruption did not claim the session.")
        store.allow_approval_publication.set()
        run_events = await asyncio.wait_for(run_task, timeout=5)
        interrupt_events = await asyncio.wait_for(interrupt_task, timeout=5)
        assert tool.calls == []
        return run_events, interrupt_events, await store.load_checkpoint(session_id)

    run_events, interrupt_events, checkpoint = asyncio.run(run())

    assert interrupt_events[-1].type is EventType.SESSION_INTERRUPTED
    assert run_events[-1].type is EventType.SESSION_INTERRUPTED
    assert EventType.TOOL_CALL_APPROVAL_REQUESTED not in [event.type for event in run_events]
    assert checkpoint is not None
    assert "pending_tool_approval" not in checkpoint
    assert "pending_tool_round" not in checkpoint


def test_local_interrupt_after_atomic_approval_publication_closes_retained_round():
    class BlockingApprovalFanOutStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.approval_event_id: str | None = None
            self.approval_fan_out_started = asyncio.Event()

        async def publish_runtime_publication(self, session_id: str, **kwargs):
            published = await super().publish_runtime_publication(session_id, **kwargs)
            for event in kwargs["request"].events:
                if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED:
                    self.approval_event_id = event.id
            return published

        async def claim_persisted_event_side_effect(self, *, session_id: str, event_id: str):
            if event_id == self.approval_event_id:
                self.approval_fan_out_started.set()
                await asyncio.Event().wait()
            return await super().claim_persisted_event_side_effect(
                session_id=session_id,
                event_id=event_id,
            )

    async def run() -> tuple[
        list[Event],
        list[Event],
        list[Event],
        list[Message],
        dict[str, Any] | None,
        list[dict[str, str]],
    ]:
        session_id = "sess_local_interrupt_after_approval_publication"
        store = BlockingApprovalFanOutStore()
        provider = FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_side_effect",
                    name="side_effect",
                    arguments={"value": "must not execute"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        )
        tool = SideEffectTool()
        app = CayuApp(session_store=store)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireApprovalPolicy(),
        )

        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await asyncio.wait_for(store.approval_fan_out_started.wait(), timeout=10)
        with suppress_interruption_cascade():
            interrupt_task = asyncio.create_task(
                collect_interrupt_events(
                    app,
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="local operator stop",
                    ),
                )
            )
        run_events, interrupt_events = await asyncio.wait_for(
            asyncio.gather(run_task, interrupt_task),
            timeout=5,
        )
        return (
            run_events,
            interrupt_events,
            await store.load_events(session_id),
            await store.load_transcript(session_id),
            await store.load_checkpoint(session_id),
            tool.calls,
        )

    run_events, interrupt_events, stored_events, transcript, checkpoint, tool_calls = asyncio.run(
        run()
    )

    assert run_events[-1].type is EventType.SESSION_INTERRUPTED
    assert interrupt_events[-1].type is EventType.SESSION_INTERRUPTED
    assert EventType.SESSION_FAILED not in [event.type for event in stored_events]
    assert [event.type for event in stored_events].count(EventType.TOOL_CALL_FAILED) == 1
    assert [event.type for event in stored_events].count(
        EventType.TOOL_CALL_APPROVAL_REQUESTED
    ) == 1
    assert tool_calls == []
    assert_only_model_step_publication_checkpoint(checkpoint)
    validate_context_messages(transcript)
    assert transcript[-1].role is MessageRole.TOOL
    assert transcript[-1].content[0].tool_call_id == "call_side_effect"
    assert transcript[-1].content[0].content == "Tool call interrupted before completion."


def test_interrupt_session_preserves_tool_results_when_interrupted_before_append():
    class InterruptingTranscriptStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.interrupt_on_next_tool_round_publication = False
            self.run_task: asyncio.Task[list[Event]] | None = None

        async def publish_runtime_publication(
            self,
            session_id: str,
            *,
            request,
            expected_statuses=None,
            expected_run_epoch=None,
            expected_transcript_cursor=None,
        ):
            if self.interrupt_on_next_tool_round_publication and request.kind == "tool-round":
                self.interrupt_on_next_tool_round_publication = False
                await self.update_status(session_id, SessionStatus.INTERRUPTING)
                if self.run_task is None:
                    raise AssertionError("Run task was not initialized.")
                self.run_task.cancel()
            return await super().publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses=expected_statuses,
                expected_run_epoch=expected_run_epoch,
                expected_transcript_cursor=expected_transcript_cursor,
            )

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
        store.interrupt_on_next_tool_round_publication = True
        store.run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interrupt_after_tool_before_append",
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        events = await store.run_task
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
