"""Every terminal of an edited call carries the marker, the notice and its exact arguments."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_tool_effect_reconciliation_registration import _spec
from tests.runtime.runtime_recovery_support import FailingTerminalToolEventStore

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.approvals.tools import (
    ToolApprovalArgumentsRefused,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.budgets.run_limits import RunLimits
from cayu.environments import Environment, EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import ToolResultPart
from cayu.providers import ModelStreamEvent
from cayu.runtime._tool_execution import (
    APPROVED_ARGUMENTS_NOTICE,
    approved_arguments_notice_content,
    content_carries_approved_arguments_notice,
    receipt_approved_arguments_notice_content,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions.memory import InMemorySessionStore
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyResult
from cayu.vaults import StaticVault
from cayu.vaults.redaction import SecretRedactor

_TERMINALS = {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}


class _SendTool(Tool):
    """Sends a message; ``mode`` selects how the tool body behaves."""

    def __init__(self, *, effect: ToolEffect = ToolEffect.NONE, mode: str = "ok") -> None:
        self.spec = ToolSpec(
            name="send",
            description="Send a message.",
            input_schema={"type": "object", "properties": {"to": {"type": "string"}}},
            effect=effect,
        )
        super().__init__()
        self.mode = mode
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(dict(args))
        if self.mode == "mutate":
            args.pop("to")
        if self.mode == "raise":
            raise RuntimeError("delivery failed")
        return ToolResult(content="sent")


class _ApprovalPolicy(ToolPolicy):
    """Requires approval; re-authorizing an edit can be slow with a short window."""

    def __init__(self, *, reauthorization_delay: float = 0.0, window: float | None = None):
        self.reauthorization_delay = reauthorization_delay
        self.window = window

    async def authorize(self, request):
        if request.metadata.get("cayu:approved_arguments_reauthorization") is True:
            await asyncio.sleep(self.reauthorization_delay)
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                approval_expires_in_seconds=self.window,
            )
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL, approval_expires_in_seconds=3600
        )


def _app(
    tool: Tool, *, store=None, policy=None, dynamic=False, reconcilers=None, secret_redactor=None
) -> CayuApp:
    app = CayuApp(
        session_store=store or InMemorySessionStore(),
        enable_logging=False,
        secret_redactor=secret_redactor,
    )
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="pause", name="send", arguments={"to": "ops"}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    if dynamic:
        app.register_environment(
            Environment(EnvironmentSpec(name="dynamic"), vault=StaticVault({"key": "unused"})),
            default=True,
        )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[tool],
        tool_policy=policy or _ApprovalPolicy(),
        **({} if reconcilers is None else {"tool_effect_reconcilers": reconcilers}),
    )
    return app


async def _drain(stream):
    async with aclosing(stream):
        return [event async for event in stream]


async def _pause(app: CayuApp, session_id: str) -> ToolApprovalRequest:
    events = await _drain(
        app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "send it")],
            )
        )
    )
    paused = next(e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
    return ToolApprovalRequest(
        session_id=session_id,
        approval_id=paused.payload["approval_id"],
        tool_round_id=paused.payload["tool_round_id"],
        tool_call_id=paused.payload["tool_call_id"],
        decision=ToolApprovalDecision.APPROVE,
        arguments={"to": "finance"},
    )


async def _terminal_and_result(app: CayuApp, session_id: str):
    [terminal] = [
        e for e in await app.session_store.load_events(session_id) if e.type in _TERMINALS
    ]
    [result] = [
        part
        for message in await app.session_store.load_transcript(session_id)
        for part in message.content
        if type(part) is ToolResultPart
    ]
    return terminal, result


def test_the_clock_is_read_after_a_slow_reauthorization():
    # The original window is an hour; the edit's own one-second window passes
    # while its 1.2-second re-authorization runs.
    tool = _SendTool()
    app = _app(tool, policy=_ApprovalPolicy(reauthorization_delay=1.2, window=1.0))

    async def scenario():
        request = await _pause(app, "edit-slow-reauthorization")
        with pytest.raises(ToolApprovalArgumentsRefused, match="approval window"):
            await _drain(app.resolve_tool_approval(request))
        checkpoint = await app.session_store.load_checkpoint(request.session_id)
        assert pending_approval_reader.pending_approval_from_checkpoint(checkpoint) is not None

    asyncio.run(scenario())
    assert tool.calls == []


def test_dynamic_scope_manual_recovery_keeps_the_notice():
    store = FailingTerminalToolEventStore()
    tool = _SendTool()
    app = _app(tool, store=store, dynamic=True)

    async def scenario():
        request = await _pause(app, "edit-dynamic-recovery")
        await _drain(app.resolve_tool_approval(request))
        assert store.failed_terminal_once and tool.calls == [{"to": "finance"}]
        events = await _drain(
            app.recover_tool_approval(
                ToolApprovalRecoveryRequest(
                    session_id=request.session_id,
                    approval_id=request.approval_id,
                    tool_round_id=request.tool_round_id,
                    tool_call_id=request.tool_call_id,
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="delivered outside",
                )
            )
        )
        assert events[-1].type is EventType.SESSION_COMPLETED
        return await _terminal_and_result(app, request.session_id)

    terminal, result = asyncio.run(scenario())

    assert terminal.payload["approved_with_changes"] is True
    assert terminal.payload["result"]["content"].startswith(APPROVED_ARGUMENTS_NOTICE)
    assert result.content.startswith(APPROVED_ARGUMENTS_NOTICE)
    assert tool.calls == [{"to": "finance"}]


def test_a_tool_mutating_its_arguments_still_publishes_the_approved_ones():
    tool = _SendTool(mode="mutate")
    app = _app(tool)

    async def scenario():
        request = await _pause(app, "edit-mutating-tool")
        await _drain(app.resolve_tool_approval(request))
        return await _terminal_and_result(app, request.session_id)

    terminal, _result = asyncio.run(scenario())

    assert terminal.payload["arguments"] == {"to": "ops"}
    assert terminal.payload["approved_arguments"] == {"to": "finance"}


@pytest.mark.parametrize("effect", [ToolEffect.IDEMPOTENT, ToolEffect.NONE])
def test_a_failing_edited_call_carries_the_marker_and_notice(effect):
    tool = _SendTool(effect=effect, mode="raise")
    app = _app(tool)

    async def scenario():
        request = await _pause(app, f"edit-failure-{effect.value}")
        await _drain(app.resolve_tool_approval(request))
        return await _terminal_and_result(app, request.session_id)

    terminal, result = asyncio.run(scenario())

    assert terminal.type is EventType.TOOL_CALL_FAILED
    assert terminal.payload["approved_with_changes"] is True
    assert terminal.payload["result"]["content"].startswith(APPROVED_ARGUMENTS_NOTICE)
    assert result.content.startswith(APPROVED_ARGUMENTS_NOTICE)


_RECEIPT_MESSAGE = "Delivery confirmed."


@pytest.mark.parametrize(
    "message",
    [_RECEIPT_MESSAGE, f"{APPROVED_ARGUMENTS_NOTICE}\n\n{_RECEIPT_MESSAGE}"],
    ids=["plain", "already-noticed"],
)
@pytest.mark.parametrize("secret", [None, "person"], ids=["no-secret", "notice-secret"])
def test_receipt_recovery_of_an_edited_call_keeps_the_marker_and_notice(message, secret):
    # A receipt message that already starts with the notice still gets its own
    # notice, so the strict receipt check accepts the settled terminal. The
    # secret "person" overlaps the notice, so the notice is published redacted.
    now = datetime.now(UTC)
    redactor = SecretRedactor() if secret is None else SecretRedactor(secret)

    class _LostAcknowledgementTool(_SendTool):
        def __init__(self) -> None:
            super().__init__(effect=ToolEffect.EXTERNAL)
            self.spec = self.spec.model_copy(
                update={
                    "execution_profile_identity": ExecutionProfileBehaviorIdentity(
                        name="tests:edit-receipt-tool",
                        behavior_version="1",
                        implementation_version="1",
                    )
                }
            )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            self.calls.append(dict(args))
            raise ConnectionError("delivery acknowledgement lost")

    class _Reconciler:
        async def reconcile(self, *, context, receipt):
            return ToolEffectReconciliationResult(
                outcome="completed",
                observation="sent",
                receipt=ToolEffectReceipt(
                    receipt_id="edit-receipt",
                    receipt_schema="deployment",
                    receipt_schema_version=1,
                    tool_call_id=context.tool_call_id,
                    tool_name=context.tool_name,
                    idempotency_key=context.idempotency_key,
                    outcome="completed",
                    message=message,
                    source="reconciler",
                    observed_at=now,
                ),
            )

    tool = _LostAcknowledgementTool()
    app = _app(
        tool,
        secret_redactor=redactor,
        reconcilers={
            "send": ToolEffectReconciliationRegistration(
                reconciler=_Reconciler(),
                spec=_spec(supports_lookup=True, timeout_seconds=60),
            )
        },
    )

    async def scenario():
        request = await _pause(app, "edit-receipt")
        resolved = await _drain(app.resolve_tool_approval(request))
        assert resolved[-1].type is EventType.SESSION_INTERRUPTED, resolved[-1].payload
        target = await app.inspect_tool_effect(
            "edit-receipt", tool_round_id=request.tool_round_id, tool_call_id=request.tool_call_id
        )
        recovered = await _drain(
            app.reconcile_tool_effect(
                ToolEffectReconciliationRequest(**target.model_dump(), lookup=True)
            )
        )
        assert recovered[-1].type is EventType.SESSION_COMPLETED, recovered[-1].payload
        return await _terminal_and_result(app, "edit-receipt")

    terminal, result = asyncio.run(scenario())

    notice = redactor.redact_text(APPROVED_ARGUMENTS_NOTICE)
    published_message = redactor.redact_text(message)
    assert (notice == APPROVED_ARGUMENTS_NOTICE) is (secret is None)
    assert tool.calls == [{"to": "finance"}]
    assert terminal.payload["effect_reconciled"] is True
    assert terminal.payload["approved_with_changes"] is True
    assert terminal.payload["result"]["content"] == f"{notice}\n\n{published_message}"
    assert result.content == f"{notice}\n\n{published_message}"


def test_approving_with_the_proposed_arguments_is_a_plain_approval():
    tool = _SendTool()
    app = _app(tool)

    async def scenario():
        request = await _pause(app, "edit-unchanged")
        await _drain(
            app.resolve_tool_approval(request.model_copy(update={"arguments": {"to": "ops"}}))
        )
        events = await app.session_store.load_events("edit-unchanged")
        [approved] = [e for e in events if e.type is EventType.TOOL_CALL_APPROVED]
        terminal, result = await _terminal_and_result(app, "edit-unchanged")
        return approved, terminal, result

    approved, terminal, result = asyncio.run(scenario())

    assert tool.calls == [{"to": "ops"}]
    assert "arguments_revision" not in approved.payload
    assert "approved_with_changes" not in terminal.payload
    assert "approved_arguments" not in terminal.payload
    assert result.content == "sent"


def test_a_receipt_terminal_is_recognized_with_a_redacted_notice():
    redactor = SecretRedactor("person")
    original = "Delivery confirmed."

    for redacting in (None, redactor):
        content = approved_arguments_notice_content(original, redactor=redacting)
        assert content_carries_approved_arguments_notice(content, original=original)
        for message in (original, content, ""):
            noticed = receipt_approved_arguments_notice_content(message, redactor=redacting)
            assert content_carries_approved_arguments_notice(noticed, original=message)
    assert not content_carries_approved_arguments_notice(original, original=original)
    assert not content_carries_approved_arguments_notice(
        f"Something else.\n\n{original}", original=original
    )
    assert not content_carries_approved_arguments_notice(
        f"{APPROVED_ARGUMENTS_NOTICE}\n\nA different message.", original=original
    )


class _WindowlessApprovalPolicy(ToolPolicy):
    async def authorize(self, request):
        return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)


def test_a_run_limit_skip_of_an_edited_call_keeps_the_marker_and_notice():
    now = [datetime(2026, 10, 18, tzinfo=UTC)]
    tool = _SendTool()
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, clock=lambda: now[0], enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="pause", name="send", arguments={"to": "ops"}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[tool],
        tool_policy=_WindowlessApprovalPolicy(),
    )
    limits = RunLimits(max_elapsed_seconds=10, scope="session")

    async def scenario():
        events = await _drain(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="edit-limit",
                    messages=[Message.text("user", "send it")],
                    limits=limits,
                )
            )
        )
        paused = next(e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
        now[0] = (await store.load("edit-limit")).created_at + timedelta(hours=1)
        await _drain(
            app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="edit-limit",
                    approval_id=paused.payload["approval_id"],
                    tool_round_id=paused.payload["tool_round_id"],
                    tool_call_id=paused.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                    arguments={"to": "finance"},
                )
            )
        )
        return await _terminal_and_result(app, "edit-limit")

    terminal, result = asyncio.run(scenario())

    assert tool.calls == []
    assert terminal.payload["reason"] == "limit_reached"
    assert terminal.payload["approved_with_changes"] is True
    assert terminal.payload["result"]["content"].startswith(APPROVED_ARGUMENTS_NOTICE)
    assert result.content.startswith(APPROVED_ARGUMENTS_NOTICE)
