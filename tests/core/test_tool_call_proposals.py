"""Application code raises an approval-gated tool call without a model turn."""

from __future__ import annotations

import asyncio
import time
from contextlib import aclosing
from pathlib import Path
from uuid import uuid4

import pytest

from cayu import AgentSpec, CayuApp, Message
from cayu.approvals import (
    ToolApprovalArgumentsRefused,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolCallProposalOutcome,
    ToolCallProposalStatus,
    tool_call_proposal_request,
)
from cayu.budgets import default_price_book
from cayu.events import EventType
from cayu.messages import ToolResultPart
from cayu.observability import EventWatcher, serve_event_watchers
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.runtime import ExecutionProfileBehaviorIdentity
from cayu.runtime._tool_call_proposals import (
    TOOL_CALL_PROPOSAL_PROVIDER_NAME,
    ToolCallProposalProvider,
    proposal_message,
)
from cayu.sessions import PendingActionQuery, SessionStatus
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks import Task, TaskCreate, TaskHandlerOutcome, TaskQuery, run_task_worker
from cayu.tools import AlwaysRequireApprovalToolPolicy
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import (
    GuardedToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.vaults.redaction import SecretRedactor

FILED: list[dict] = []


class FileIssue(Tool):
    spec = ToolSpec(
        name="file_issue",
        description="File one issue.",
        input_schema={
            "type": "object",
            "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
            "required": ["title"],
        },
        effect=ToolEffect.EXTERNAL,
        max_terminal_payload_bytes=64 * 1024,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:proposals:file-issue", behavior_version="1", implementation_version="1"
        ),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        FILED.append(dict(args))
        return ToolResult(content=f"Filed {args['title']}.")


class PinnedTitle:
    """Deny any call whose title is not the stored proposal's."""

    execution_profile_identity = ExecutionProfileBehaviorIdentity(
        name="tests:proposals:pinned-title", behavior_version="1", implementation_version="1"
    )

    async def check(self, request: ToolPolicyRequest) -> ToolPolicyResult | None:
        if request.arguments.get("title") != "Stored title":
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY, reason="Arguments differ from the proposal."
            )
        return None


def _build(session_store=None, task_store=None, **kwargs) -> CayuApp:
    # No provider is registered: proposals never call a model.
    app = CayuApp(
        session_store=session_store or InMemorySessionStore(),
        task_store=task_store,
        enable_logging=False,
        **kwargs,
    )
    app.register_agent(
        AgentSpec(name="scout", model="unconfigured-model"),
        tools=[FileIssue()],
        tool_policy=GuardedToolPolicy(
            guards=[PinnedTitle()], then=AlwaysRequireApprovalToolPolicy(tools=["file_issue"])
        ),
    )
    return app


async def _drain(stream) -> list:
    async with aclosing(stream):
        return [event async for event in stream]


def _resolution(outcome: ToolCallProposalOutcome, decision, **fields) -> ToolApprovalRequest:
    assert outcome.approval_id and outcome.tool_round_id and outcome.tool_call_id
    return ToolApprovalRequest(
        session_id=outcome.session_id,
        approval_id=outcome.approval_id,
        tool_round_id=outcome.tool_round_id,
        tool_call_id=outcome.tool_call_id,
        decision=decision,
        **fields,
    )


@pytest.fixture(autouse=True)
def _reset_filed():
    FILED.clear()


def test_approved_proposal_runs_the_tool_once_without_a_provider() -> None:
    async def scenario():
        app = _build()
        pending = await app.propose_tool_call(
            "scout", "file_issue", {"title": "Stored title", "body": "Exact text."}
        )
        assert pending.status is ToolCallProposalStatus.PENDING_APPROVAL
        assert pending.tool_name == "file_issue" and pending.result is None
        assert FILED == []
        # It is an ordinary pending approval: the dashboard reads the same query.
        actions = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id=pending.session_id)
        )
        [action] = actions.actions
        assert action.approval_id is not None
        session = await app.session_store.load(pending.session_id)
        assert session is not None and session.status is SessionStatus.INTERRUPTED
        assert session.provider_name == TOOL_CALL_PROPOSAL_PROVIDER_NAME

        events = await _drain(
            app.resolve_tool_approval(_resolution(pending, ToolApprovalDecision.APPROVE))
        )
        assert events[-1].type is EventType.SESSION_COMPLETED
        settled = ToolCallProposalOutcome.from_events(
            await app.session_store.load_events(pending.session_id)
        )
        cost = await app.get_session_cost(pending.session_id, default_price_book())
        await app.aclose()
        return settled, cost

    settled, cost = asyncio.run(scenario())
    assert FILED == [{"title": "Stored title", "body": "Exact text."}]
    assert settled.status is ToolCallProposalStatus.COMPLETED
    assert settled.result is not None and settled.result.content == "Filed Stored title."
    # Both control steps are priced at zero, with no price book entry needed.
    assert cost.total_cost == 0
    assert cost.model_steps == 2 and cost.unpriced_model_steps == 0


def test_denied_proposal_records_the_denial_and_never_runs() -> None:
    async def scenario():
        app = _build()
        pending = await app.propose_tool_call("scout", "file_issue", {"title": "Stored title"})
        await _drain(
            app.resolve_tool_approval(
                _resolution(pending, ToolApprovalDecision.DENY, reason="Duplicate.")
            )
        )
        events = await app.session_store.load_events(pending.session_id)
        session = await app.session_store.load(pending.session_id)
        await app.aclose()
        return events, session

    events, session = asyncio.run(scenario())
    assert FILED == []
    assert sum(event.type is EventType.TOOL_CALL_APPROVAL_DENIED for event in events) == 1
    outcome = ToolCallProposalOutcome.from_events(events)
    assert outcome.status is ToolCallProposalStatus.DENIED
    assert outcome.reason == "Duplicate."
    assert session is not None and session.status is SessionStatus.COMPLETED


def test_guarded_policy_applies_to_proposals_and_approval_edits() -> None:
    async def scenario():
        app = _build()
        blocked = await app.propose_tool_call("scout", "file_issue", {"title": "Rewritten"})
        pending = await app.propose_tool_call("scout", "file_issue", {"title": "Stored title"})
        # An approver's edit is re-authorized by the same guard and refused.
        with pytest.raises(ToolApprovalArgumentsRefused, match="denied the approved arguments"):
            await _drain(
                app.resolve_tool_approval(
                    _resolution(
                        pending, ToolApprovalDecision.APPROVE, arguments={"title": "Edited"}
                    )
                )
            )
        still = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id=pending.session_id)
        )
        await app.aclose()
        return blocked, still

    blocked, still = asyncio.run(scenario())
    assert blocked.status is ToolCallProposalStatus.BLOCKED
    assert blocked.approval_id is None
    assert blocked.reason == "Arguments differ from the proposal."
    assert len(still.actions) == 1
    assert FILED == []


def test_pending_proposal_resolves_in_a_restarted_process(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite"

    async def scenario():
        first = _build(SQLiteSessionStore(path))
        pending = await first.propose_tool_call(
            "scout", "file_issue", {"title": "Stored title"}, session_id="restart-proposal"
        )
        await first.aclose()

        second = _build(SQLiteSessionStore(path))
        [action] = (
            await second.session_store.query_pending_actions(
                PendingActionQuery(session_id="restart-proposal")
            )
        ).actions
        assert action.approval_id and action.round_id and action.tool_call_id
        events = await _drain(
            second.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="restart-proposal",
                    approval_id=action.approval_id,
                    tool_round_id=action.round_id,
                    tool_call_id=action.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        )
        await second.aclose()
        return pending, events

    pending, events = asyncio.run(scenario())
    assert pending.status is ToolCallProposalStatus.PENDING_APPROVAL
    assert events[-1].type is EventType.SESSION_COMPLETED
    assert FILED == [{"title": "Stored title"}]


def test_attached_task_hands_off_and_completes_after_approval(tmp_path: Path) -> None:
    session_path = tmp_path / "sessions.sqlite"
    task_path = tmp_path / "tasks.sqlite"

    async def propose(app: CayuApp, task: Task, worker_id: str) -> TaskHandlerOutcome:
        outcome = await app.propose_tool_call(
            "scout",
            "file_issue",
            task.input,
            session_id="task-proposal",
            task_id=task.id,
            task_worker_id=worker_id,
            task_lease_expires_at=task.lease_expires_at,
        )
        assert outcome.status is ToolCallProposalStatus.PENDING_APPROVAL
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def scenario():
        first = _build(SQLiteSessionStore(session_path), SQLiteTaskStore(task_path))
        await first.create_task(
            TaskCreate(task_id="file-task", type="file", input={"title": "Stored title"})
        )
        handled = await run_task_worker(
            first,
            first.task_store,
            propose,
            worker_id="worker-a",
            query=TaskQuery(type="file"),
            max_tasks=1,
            poll_interval_s=0.05,
            reclaim=False,
        )
        assert handled == 1
        handed_off = await first.task_store.load_task("file-task")
        await first.aclose()

        second = _build(SQLiteSessionStore(session_path), SQLiteTaskStore(task_path))
        continuation = (
            await second.task_store.claim_interrupted_task_continuation(
                "control-plane", TaskQuery(type="file"), handoff_id=str(uuid4())
            )
        ).task
        assert continuation is not None
        [action] = (
            await second.session_store.query_pending_actions(
                PendingActionQuery(session_id="task-proposal")
            )
        ).actions
        await _drain(
            second.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="task-proposal",
                    task_worker_id="control-plane",
                    task_handoff_id=continuation.interrupted_handoff_id,
                    approval_id=action.approval_id,
                    tool_round_id=action.round_id,
                    tool_call_id=action.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        )
        completed = await second.task_store.load_task("file-task")
        await second.aclose()
        return handed_off, completed

    handed_off, completed = asyncio.run(scenario())
    assert handed_off is not None and handed_off.status == "running"
    assert handed_off.worker_id is None
    assert completed is not None and completed.status == "completed"
    assert FILED == [{"title": "Stored title"}]


def test_proposal_refuses_secrets_owned_fields_and_unknown_agents() -> None:
    app = _build(secret_redactor=SecretRedactor("sk-live-canary"))

    async def scenario():
        with pytest.raises(ValueError, match="registered secret"):
            await app.propose_tool_call("scout", "file_issue", {"title": "sk-live-canary"})
        with pytest.raises(ValueError, match="owns these RunRequest fields: max_steps"):
            await app.propose_tool_call("scout", "file_issue", {"title": "x"}, max_steps=5)
        with pytest.raises(KeyError):
            await app.propose_tool_call("nobody", "file_issue", {"title": "x"})
        await app.aclose()

    asyncio.run(scenario())
    with pytest.raises(ValueError, match="runtime-owned"):
        CayuApp(enable_logging=False).register_provider(ToolCallProposalProvider())


def test_proposal_request_composes_with_run_and_keeps_caller_fields() -> None:
    request = tool_call_proposal_request(
        "scout",
        "file_issue",
        {"title": "Stored title"},
        tool_call_id="call-1",
        session_id="composed",
        title="File finding 7",
        labels={"finding": "7"},
    )
    assert request.session_id == "composed" and request.title == "File finding 7"
    assert request.labels == {"finding": "7"}
    assert request.target is not None
    assert request.target.provider_name == TOOL_CALL_PROPOSAL_PROVIDER_NAME
    assert request.max_steps == 2

    async def scenario():
        app = _build()
        events = await _drain(app.run(request))
        await app.aclose()
        return ToolCallProposalOutcome.from_events(events)

    outcome = asyncio.run(scenario())
    assert outcome.status is ToolCallProposalStatus.PENDING_APPROVAL


def test_provider_raises_the_call_from_the_durable_transcript() -> None:
    """A replayed first step (for example after a crash) raises the same call."""

    provider = ToolCallProposalProvider()
    proposal = proposal_message(
        tool_name="file_issue", arguments={"title": "Stored title"}, tool_call_id="call-1"
    )

    async def stream(messages):
        request = ModelRequest(model="tool-proposal", messages=messages)
        return [event async for event in provider.stream(request)]

    first = asyncio.run(stream([proposal]))
    again = asyncio.run(stream([proposal]))
    assert first == again
    assert first[0] == ModelStreamEvent.tool_call(
        name="file_issue", arguments={"title": "Stored title"}, id="call-1"
    )
    result = Message(
        role="tool",
        content=[ToolResultPart(tool_call_id="call-1", tool_name="file_issue", content="ok")],
    )
    settled = asyncio.run(stream([proposal, result]))
    assert [event.type for event in settled][-1] == first[-1].type
    assert not any(event.type == first[0].type for event in settled)
    ordinary = asyncio.run(stream([Message.text("user", "Please file it.")]))
    assert not any(event.type == first[0].type for event in ordinary)


def test_watcher_reacts_to_a_denied_proposal() -> None:
    seen: list[tuple[str, str]] = []

    def record(context) -> None:
        seen.append((context.record.event.type.value, context.record.event.session_id))

    watcher = EventWatcher(
        name="approval-outcomes",
        query=EventQuery(
            event_types=(EventType.TOOL_CALL_APPROVED, EventType.TOOL_CALL_APPROVAL_DENIED)
        ),
        handler=record,
    )

    async def scenario():
        app = _build()
        assert app.register_event_watcher(watcher) is watcher
        with pytest.raises(ValueError, match="already registered"):
            app.register_event_watcher(watcher)
        stop = asyncio.Event()
        loop = asyncio.create_task(serve_event_watchers(app, stop, poll_interval_seconds=0.05))
        pending = await app.propose_tool_call("scout", "file_issue", {"title": "Stored title"})
        await _drain(app.resolve_tool_approval(_resolution(pending, ToolApprovalDecision.DENY)))
        async with asyncio.timeout(10):
            while not seen:
                await asyncio.sleep(0.02)
        stop.set()
        await asyncio.wait_for(loop, 5)
        await app.aclose()
        return pending

    pending = asyncio.run(scenario())
    assert seen == [("tool.call.approval_denied", pending.session_id)]
    assert FILED == []


def test_watcher_loop_returns_when_the_application_closes() -> None:
    async def scenario():
        app = _build()
        app.register_event_watcher(
            EventWatcher(name="noop", query=EventQuery(), handler=lambda context: None)
        )
        loop = asyncio.create_task(
            serve_event_watchers(app, asyncio.Event(), poll_interval_seconds=0.01)
        )
        await asyncio.sleep(0.05)
        await app.aclose()
        await asyncio.wait_for(loop, 5)
        with pytest.raises(RuntimeError, match="before the application shuts down"):
            app.register_event_watcher(
                EventWatcher(name="late", query=EventQuery(), handler=lambda context: None)
            )

    asyncio.run(scenario())


def test_secret_hidden_by_json_escaping_is_refused_before_any_session() -> None:
    secret = 'private"credential\\value'
    app = _build(secret_redactor=SecretRedactor(secret))

    async def scenario():
        for fields in (
            {"arguments": {"title": secret}},
            {"arguments": {"nested": [{"value": f"prefix {secret}\n"}]}},
            {"arguments": {secret: "key"}},
            {"arguments": {"title": "x"}, "tool_call_id": secret},
        ):
            with pytest.raises(ValueError, match="registered secret"):
                await app.propose_tool_call("scout", "file_issue", **fields)
        page = await app.session_store.list_sessions()
        await app.aclose()
        return page

    assert asyncio.run(scenario()).sessions == []


def test_expired_approval_is_reported_as_expired_not_denied() -> None:
    async def scenario():
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_agent(
            AgentSpec(name="scout", model="unconfigured-model"),
            tools=[FileIssue()],
            tool_policy=AlwaysRequireApprovalToolPolicy(expires_in_seconds=0.05),
        )
        pending = await app.propose_tool_call("scout", "file_issue", {"title": "Late"})
        await asyncio.sleep(0.2)
        await _drain(app.resolve_tool_approval(_resolution(pending, ToolApprovalDecision.APPROVE)))
        events = await app.session_store.load_events(pending.session_id)
        await app.aclose()
        return events

    events = asyncio.run(scenario())
    assert any(event.type is EventType.TOOL_CALL_APPROVAL_EXPIRED for event in events)
    assert ToolCallProposalOutcome.from_events(events).status is ToolCallProposalStatus.EXPIRED
    assert FILED == []


def test_approved_but_unsettled_call_is_no_longer_pending() -> None:
    async def scenario():
        app = _build()
        pending = await app.propose_tool_call("scout", "file_issue", {"title": "Stored title"})
        await _drain(app.resolve_tool_approval(_resolution(pending, ToolApprovalDecision.APPROVE)))
        events = await app.session_store.load_events(pending.session_id)
        await app.aclose()
        return events

    events = asyncio.run(scenario())
    for event_type in (EventType.TOOL_CALL_APPROVED, EventType.TOOL_CALL_STARTED):
        cut = next(index for index, event in enumerate(events) if event.type is event_type)
        outcome = ToolCallProposalOutcome.from_events(events[: cut + 1])
        assert outcome.status is ToolCallProposalStatus.UNSETTLED


def test_provider_finds_the_proposal_beside_context_policy_parts() -> None:
    from cayu.memory.context import _overlay_manifest

    provider = ToolCallProposalProvider()
    proposal = proposal_message(
        tool_name="file_issue", arguments={"title": "Stored title"}, tool_call_id="call-1"
    )
    [enriched] = _overlay_manifest([proposal], anchor_index=0, manifest_text="Recalled notes.")
    assert len(enriched.content) == 2

    async def stream():
        request = ModelRequest(model="tool-proposal", messages=[enriched])
        return [event async for event in provider.stream(request)]

    assert asyncio.run(stream())[0] == ModelStreamEvent.tool_call(
        name="file_issue", arguments={"title": "Stored title"}, id="call-1"
    )


def test_watcher_registration_refuses_subclasses_that_delivery_would_reject() -> None:
    class CustomWatcher(EventWatcher):
        pass

    custom = CustomWatcher(name="custom", query=EventQuery(), handler=lambda context: None)
    app = _build()
    with pytest.raises(TypeError, match="requires an EventWatcher"):
        app.register_event_watcher(custom)
    with pytest.raises(TypeError, match="EventWatcher instances"):
        asyncio.run(serve_event_watchers(app, asyncio.Event(), watchers=[custom]))
    asyncio.run(app.aclose())


def test_backlogged_watcher_does_not_starve_later_watchers() -> None:
    order: list[str] = []

    async def scenario():
        stop = asyncio.Event()

        def outcomes(context) -> None:
            order.append("outcomes")
            stop.set()

        app = _build()
        for _ in range(3):
            await app.propose_tool_call("scout", "file_issue", {"title": "Stored title"})
        total = len(await app.session_store.query_events(EventQuery()))
        app.register_event_watcher(
            EventWatcher(name="busy", query=EventQuery(), handler=lambda c: order.append("busy"))
        )
        app.register_event_watcher(
            EventWatcher(
                name="outcomes",
                query=EventQuery(event_types=(EventType.TOOL_CALL_APPROVAL_REQUESTED,)),
                handler=outcomes,
            )
        )
        await asyncio.wait_for(
            serve_event_watchers(app, stop, poll_interval_seconds=0.05, limit=2), 10
        )
        await app.aclose()
        return total

    total = asyncio.run(scenario())
    # The later watcher ran in the first pass, long before the backlog drained.
    assert order.index("outcomes") <= 2 < total


def test_failing_watcher_is_retried_at_the_poll_interval_while_others_progress() -> None:
    attempts: list[float] = []

    def flaky(context) -> None:
        attempts.append(time.monotonic())  # Synchronous handlers run in a thread.
        raise RuntimeError("downstream unavailable")

    async def scenario():
        app = _build()
        for _ in range(3):
            await app.propose_tool_call("scout", "file_issue", {"title": "Stored title"})
        app.register_event_watcher(
            EventWatcher(name="busy", query=EventQuery(), handler=lambda c: None, batch_size=1)
        )
        app.register_event_watcher(
            EventWatcher(
                name="flaky",
                query=EventQuery(event_types=(EventType.TOOL_CALL_APPROVAL_REQUESTED,)),
                handler=flaky,
            )
        )
        stop = asyncio.Event()
        loop = asyncio.create_task(serve_event_watchers(app, stop, poll_interval_seconds=0.3))
        async with asyncio.timeout(10):
            while len(attempts) < 2:
                await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(loop, 5)
        await app.aclose()

    asyncio.run(scenario())
    assert attempts[1] - attempts[0] >= 0.25


def test_secret_inside_an_immutable_json_array_is_refused() -> None:
    secret = "sk-live-canary"
    app = _build(secret_redactor=SecretRedactor(secret))
    # Tool results hold their JSON in Cayu's immutable mappings and arrays.
    frozen = ToolResult(content="x", structured={"items": [secret]}).structured
    assert frozen is not None and type(frozen["items"]) is not list

    async def scenario():
        with pytest.raises(ValueError, match="registered secret"):
            await app.propose_tool_call("scout", "file_issue", frozen)
        page = await app.session_store.list_sessions()
        await app.aclose()
        return page

    assert asyncio.run(scenario()).sessions == []
