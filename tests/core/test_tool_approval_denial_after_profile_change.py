"""A denial always settles a pending approval, even when the paused round can't continue."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_human_review import (
    CONTEXT,
    ApprovalPolicy,
    Provider,
    RecordingTool,
    ReviewPolicy,
    pause,
)
from tests.core.test_tool_approval_profile_adoption import (
    _app,
    _approve,
    _collect,
    _FileBug,
    _pause,
    _RedescribedFileBug,
    _still_pending,
    _use_cayu,
)
from tests.core.test_tool_call_proposals import (
    FILED,
    FileIssue,
    PinnedTitle,
    _build,
    _drain,
    _resolution,
)

from cayu import AgentSpec, CayuApp
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.events import EventType
from cayu.messages import Message
from cayu.runtime import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import ExecutionProfileMismatchError
from cayu.sessions import PendingActionQuery
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus, SessionStatusConflict
from cayu.sessions.requests import ResumeRequest
from cayu.tools import AlwaysRequireApprovalToolPolicy
from cayu.tools.policy import GuardedToolPolicy


def _deny(requested, reason: str = "Not a bug.") -> object:
    return _approve(requested).model_copy(
        update={"decision": ToolApprovalDecision.DENY, "reason": reason}
    )


def _upgraded(monkeypatch, tool: _FileBug):
    store = InMemorySessionStore()
    _use_cayu(monkeypatch, "0.15.0")
    requested = asyncio.run(_pause(_app(store)))
    _use_cayu(monkeypatch, "0.16.0")
    return store, requested, _app(store, tool=tool)


@pytest.mark.parametrize("tool", [_FileBug(behavior_version="2"), _RedescribedFileBug()])
def test_a_denial_ends_the_session_when_the_round_cant_continue(monkeypatch, tool) -> None:
    tool.calls.clear()
    store, requested, app = _upgraded(monkeypatch, tool)

    events = asyncio.run(_collect(app.resolve_tool_approval(_deny(requested))))

    assert tool.calls == []
    assert not _still_pending(store)
    denied, failed = events
    assert denied.type == EventType.TOOL_CALL_APPROVAL_DENIED
    assert denied.tool_name == "file_bug"
    assert denied.payload["reason"] == "Not a bug."
    assert failed.type == EventType.SESSION_FAILED
    assert failed.payload["error_type"] == "SessionAbandoned"
    assert failed.payload["reason"] == "tool_approval_denied_after_profile_change"
    session = asyncio.run(store.load("bug-report"))
    assert session is not None and session.status is SessionStatus.FAILED
    durable = asyncio.run(store.load_events("bug-report"))
    assert [e.type for e in durable[-2:]] == [
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.SESSION_FAILED,
    ]

    # The same denial again republishes the same evidence and nothing more.
    again = asyncio.run(_collect(app.resolve_tool_approval(_deny(requested))))
    assert [e.id for e in again] == [denied.id, failed.id]
    assert [e.id for e in asyncio.run(store.load_events("bug-report"))] == [e.id for e in durable]

    # A different decision, and continuing the session, are both refused.
    with pytest.raises(RuntimeError, match="already denied with a different request"):
        asyncio.run(_collect(app.resolve_tool_approval(_deny(requested, "Other reason."))))
    # Even the application version that proposed the call can't continue it now.
    _use_cayu(monkeypatch, "0.15.0")
    original = _FileBug()
    with pytest.raises(SessionStatusConflict, match="abandoned session cannot be continued"):
        asyncio.run(
            _collect(
                _app(store, tool=original).resume(
                    ResumeRequest(
                        session_id="bug-report", messages=[Message.text("user", "Go on.")]
                    )
                )
            )
        )
    assert original.calls == []
    assert tool.calls == []


def test_an_approval_still_needs_its_profile_or_an_adoption(monkeypatch) -> None:
    tool = _FileBug(behavior_version="2")
    store, requested, app = _upgraded(monkeypatch, tool)

    with pytest.raises(ExecutionProfileMismatchError):
        asyncio.run(_collect(app.resolve_tool_approval(_approve(requested))))

    assert tool.calls == []
    assert _still_pending(store)


def test_a_cayu_only_upgrade_still_continues_a_denied_round(monkeypatch) -> None:
    tool = _FileBug()
    store, requested, app = _upgraded(monkeypatch, tool)

    events = asyncio.run(_collect(app.resolve_tool_approval(_deny(requested))))

    assert tool.calls == []
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert asyncio.run(store.load("bug-report")).status is SessionStatus.COMPLETED


class _FileIssueV2(FileIssue):
    spec = FileIssue.spec.model_copy(
        update={
            "execution_profile_identity": ExecutionProfileBehaviorIdentity(
                name="tests:proposals:file-issue", behavior_version="2", implementation_version="1"
            )
        }
    )


def test_a_proposal_denied_after_its_tool_changed_is_recorded_as_denied() -> None:
    """The case a filing Agent hits: it proposes, then ships a new tool version."""

    FILED.clear()
    store = InMemorySessionStore()

    async def scenario():
        proposer = _build(store)
        pending = await proposer.propose_tool_call("scout", "file_issue", {"title": "Stored title"})
        await proposer.aclose()

        upgraded = CayuApp(session_store=store, enable_logging=False)
        upgraded.register_agent(
            AgentSpec(name="scout", model="unconfigured-model"),
            tools=[_FileIssueV2()],
            tool_policy=GuardedToolPolicy(
                guards=[PinnedTitle()], then=AlwaysRequireApprovalToolPolicy(tools=["file_issue"])
            ),
        )
        events = await _drain(
            upgraded.resolve_tool_approval(
                _resolution(pending, ToolApprovalDecision.DENY, reason="Duplicate.")
            )
        )
        denials = await store.query_events(
            EventQuery(
                session_id=pending.session_id,
                event_type=EventType.TOOL_CALL_APPROVAL_DENIED,
                agent_name="scout",
                tool_name="file_issue",
            )
        )
        pending_after = await store.query_pending_actions(
            PendingActionQuery(session_id=pending.session_id)
        )
        await upgraded.aclose()
        return events, denials, pending_after

    events, denials, pending_after = asyncio.run(scenario())
    assert [e.type for e in events] == [
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.SESSION_FAILED,
    ]
    [denial] = denials
    assert denial.event.payload["reason"] == "Duplicate."
    assert pending_after.actions == []
    assert FILED == []


class _ChangedRecordingTool(RecordingTool):
    execution_profile_identity = ExecutionProfileBehaviorIdentity(
        name="review-tool", behavior_version="2", implementation_version="1"
    )


def _reviewed_app(store, policy, tool) -> CayuApp:
    app = CayuApp(session_store=store, human_review_policy=policy, enable_logging=False)
    app.register_provider(Provider([("call-1", "side_effect", {"value": "morning"})]), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"), tools=[tool], tool_policy=ApprovalPolicy()
    )
    return app


@pytest.mark.parametrize("current_review", [True, False])
def test_the_denial_needs_a_review_of_the_paused_approval(current_review) -> None:
    store = InMemorySessionStore()
    policy = ReviewPolicy()

    async def scenario():
        await pause(_reviewed_app(store, policy, RecordingTool()))
        view = await _reviewed_app(store, policy, RecordingTool()).inspect_human_review(
            "review-session", context=CONTEXT
        )
        assert view.status == "permitted" and view.reference is not None
        if not current_review:
            # The reviewed content no longer matches what the reference signed.
            policy.version = "delivery-v2"
        tool = _ChangedRecordingTool()
        denial = ToolApprovalRequest(
            session_id="review-session",
            approval_id=view.interaction_id,
            tool_round_id=view.tool_round_id,
            tool_call_id=view.tool_call_id,
            decision=ToolApprovalDecision.DENY,
            reason="Not now.",
            review_reference=view.reference,
        )
        app = _reviewed_app(store, policy, tool)
        if current_review:
            events = await _collect(app.resolve_tool_approval(denial))
            assert [e.type for e in events] == [
                EventType.TOOL_CALL_APPROVAL_DENIED,
                EventType.SESSION_FAILED,
            ]
            # Replaying the same denial is accepted without a fresh review.
            policy.version = "delivery-v2"
            again = await _collect(app.resolve_tool_approval(denial))
            assert [e.id for e in again] == [e.id for e in events]
            assert (await store.load("review-session")).status is SessionStatus.FAILED
        else:
            with pytest.raises((ValueError, PermissionError)):
                await _collect(app.resolve_tool_approval(denial))
            assert (await store.load("review-session")).status is SessionStatus.INTERRUPTED
            pending = await store.query_pending_actions(
                PendingActionQuery(session_id="review-session")
            )
            assert [a.approval_id for a in pending.actions] == [view.interaction_id]
        assert tool.calls == []

    asyncio.run(scenario())
