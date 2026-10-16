"""Shared runtime flows for supervision-event regressions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _pending_tool_approval_from_public_event,
    collect_events,
)
from tests.runtime.runtime_tool_support import RequireApprovalPolicy, SideEffectTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import SupervisionEventPage, SupervisionEventQuery
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.requests import RunRequest

ARGUMENT_CANARY = "argument-canary-5d1c"

# Most runtime events are stamped with wall time, so windows are fixed around the
# test run; an injected app clock is used only where approval expiry needs it.
WINDOW_START = datetime.now(UTC) - timedelta(days=1)
WINDOW_END = datetime.now(UTC) + timedelta(days=1)


def supervision_window(**changes) -> SupervisionEventQuery:
    return SupervisionEventQuery.model_validate(
        {"start_at": WINDOW_START, "end_at": WINDOW_END, **changes}
    )


async def durable_timestamp(store, session_id: str, event_type: EventType) -> datetime:
    records = await store.query_events(EventQuery(session_id=session_id, event_type=event_type))
    return records[0].event.timestamp


async def supervision_page(
    store, *, include_content: bool = False, **changes
) -> SupervisionEventPage:
    return await store.query_supervision_events(
        supervision_window(**changes), include_content=include_content
    )


def two_call_provider() -> FakeProvider:
    return FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1", name="side_effect", arguments={"value": ARGUMENT_CANARY}
                ),
                ModelStreamEvent.tool_call(id="call_2", name="side_effect", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )


def approval_app(store, *, sessions: int, policy=None, resource_access_policy=None) -> CayuApp:
    """One app whose provider scripts a pause and a continuation per session, in order."""

    turns = []
    for _ in range(sessions):
        turns.append(
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        )
        turns.append(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
    app = CayuApp(session_store=store, resource_access_policy=resource_access_policy)
    app.register_provider(FakeProvider(turns), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SideEffectTool()],
        tool_policy=policy or RequireApprovalPolicy(),
    )
    return app


async def pause_for_approval(
    app: CayuApp, store, session_id: str, *, labels: dict[str, str] | None = None
):
    events = await collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "use the tool")],
            labels=labels or {},
        ),
    )
    requested = next(e for e in events if e.type == EventType.TOOL_CALL_APPROVAL_REQUESTED)
    return await _pending_tool_approval_from_public_event(store, requested)


def resolve_approval(
    session_id: str, pending, decision: ToolApprovalDecision, **changes
) -> ToolApprovalRequest:
    return ToolApprovalRequest(
        session_id=session_id,
        approval_id=pending.approval_id,
        tool_round_id=pending.tool_round_id,
        tool_call_id=pending.tool_call_id,
        decision=decision,
        **changes,
    )
