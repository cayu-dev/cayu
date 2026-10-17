"""Supervision events name the human-attention request they settled and declared edits."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from pydantic import ValidationError
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_recovery_support import (
    FailingTerminalToolEventStore,
    _crashed_tool_round_app,
)
from tests.runtime.runtime_session_support import (
    _reviewer_actor,
    collect_tool_approval_events,
    collect_tool_approval_recovery_events,
    collect_tool_round_recovery_events,
)
from tests.runtime.runtime_supervision_support import (
    approval_app,
    pause_for_approval,
    resolve_approval,
    supervision_page,
    supervision_window,
)
from tests.runtime.runtime_tool_support import (
    PureRecordingTool,
    RequireApprovalPolicy,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals import (
    APPROVAL_REVISION_METADATA_KEY,
    ApprovalRevision,
    approval_revision_metadata,
)
from cayu.approvals.tools import (
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.approvals.user_input import UserInputRecoveryRequest, UserInputResponse
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.budgets.usage import UsageMetrics
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.human_attention import HumanAttentionRequest
from cayu.sessions import (
    SupervisionEventKind,
    SupervisionEventOutcome,
    SupervisionEventPage,
    SupervisionGrantRole,
    price_supervision_events,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.tools.user_input import UserInputTool

_PROPOSED = "a" * 64
_APPROVED = "b" * 64


async def _attention_request(store, session_id: str) -> HumanAttentionRequest:
    pending = await store.query_pending_actions(PendingActionQuery(session_id=session_id))
    (action,) = pending.actions
    request = HumanAttentionRequest.from_pending_action(action)
    assert request is not None
    return request


def test_approval_touches_name_their_attention_request_and_settled_state():
    store = InMemorySessionStore()
    app = approval_app(store, sessions=2)

    async def run():
        attention = {}
        for session_id, decision in (
            ("sess_attention_approve", ToolApprovalDecision.APPROVE),
            ("sess_attention_deny", ToolApprovalDecision.DENY),
        ):
            pending = await pause_for_approval(app, store, session_id)
            attention[session_id] = await _attention_request(store, session_id)
            await collect_tool_approval_events(
                app,
                resolve_approval(session_id, pending, decision, resolved_by=_reviewer_actor()),
            )
        observed = {
            session_id: await app.get_human_attention_state(request.reference)
            for session_id, request in attention.items()
        }
        return attention, observed, await supervision_page(store)

    attention, observed, page = asyncio.run(run())

    approve, deny = page.events
    assert (approve.kind, deny.kind) == (SupervisionEventKind.APPROVE, SupervisionEventKind.DENY)
    for event, state in ((approve, "resolved"), (deny, "cancelled")):
        assert event.attention_id == attention[event.session_id].reference.attention_id
        assert event.attention_state == state
        assert observed[event.session_id].state == state
    assert approve.approval_revision is None
    assert approve.authorization_grant_id is None


def test_round_recovery_names_its_manual_recovery_request():
    session_id = "sess_attention_round_recovery"
    app, store, _tool, checkpoint = _crashed_tool_round_app(session_id, tool=PureRecordingTool())

    async def run():
        request = await _attention_request(store, session_id)
        await collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="exported by hand",
                resolved_by=_reviewer_actor(),
            ),
        )
        return (
            request,
            await app.get_human_attention_state(request.reference),
            await supervision_page(store),
        )

    request, observed, page = asyncio.run(run())

    (event,) = page.events
    assert event.kind is SupervisionEventKind.RECOVER
    assert event.attention_id == request.reference.attention_id
    assert event.attention_state == "resolved" == observed.state


def test_declared_revision_classifies_an_approval_as_an_edit():
    store = InMemorySessionStore()
    app = approval_app(store, sessions=2)

    async def run():
        edited = await pause_for_approval(app, store, "sess_revision_edit")
        await collect_tool_approval_events(
            app,
            resolve_approval(
                "sess_revision_edit",
                edited,
                ToolApprovalDecision.APPROVE,
                metadata={
                    "finance_digest": _PROPOSED,
                    **approval_revision_metadata(
                        proposed_digest=_PROPOSED, approved_digest=_APPROVED
                    ),
                },
            ),
        )
        unchanged = await pause_for_approval(app, store, "sess_revision_unchanged")
        await collect_tool_approval_events(
            app,
            resolve_approval(
                "sess_revision_unchanged",
                unchanged,
                ToolApprovalDecision.APPROVE,
                metadata=approval_revision_metadata(
                    proposed_digest=_PROPOSED, approved_digest=_PROPOSED
                ),
            ),
        )
        return (
            await supervision_page(store),
            await supervision_page(store, kinds=["approve"]),
            await supervision_page(store, kinds=["edit"]),
        )

    page, approvals, edits = asyncio.run(run())

    edit, unchanged = page.events
    assert edit.kind is SupervisionEventKind.EDIT
    assert edit.outcome is SupervisionEventOutcome.APPROVED
    assert edit.approval_revision == ApprovalRevision(
        proposed_digest=_PROPOSED, approved_digest=_APPROVED
    )
    assert edit.attention_state == "resolved"
    assert unchanged.kind is SupervisionEventKind.APPROVE
    assert unchanged.approval_revision is not None and not unchanged.approval_revision.edited
    assert [event.session_id for event in approvals.events] == ["sess_revision_unchanged"]
    assert [event.session_id for event in edits.events] == ["sess_revision_edit"]


@pytest.mark.parametrize(
    ("declaration", "decision"),
    [
        ({"proposed_digest": _PROPOSED}, ToolApprovalDecision.APPROVE),
        (
            {"proposed_digest": _PROPOSED, "approved_digest": "row 7: account 4010"},
            ToolApprovalDecision.APPROVE,
        ),
        (
            {"proposed_digest": _PROPOSED, "approved_digest": _APPROVED, "rows": []},
            ToolApprovalDecision.APPROVE,
        ),
        ({"proposed_digest": _PROPOSED, "approved_digest": _APPROVED}, ToolApprovalDecision.DENY),
    ],
)
def test_approval_requests_reject_malformed_or_misplaced_revisions(declaration, decision):
    with pytest.raises(ValidationError, match="revision"):
        ToolApprovalRequest(
            session_id="sess_revision_invalid",
            approval_id="approval",
            tool_round_id="round",
            tool_call_id="call",
            decision=decision,
            metadata={APPROVAL_REVISION_METADATA_KEY: declaration},
        )


def test_grant_linkage_and_usage_cost_are_read_from_recorded_events():
    store = InMemorySessionStore()
    usage = UsageMetrics(
        provider_name="fake", model="fake-model", input_tokens=1000, output_tokens=500
    )

    async def run():
        for session_id in ("sess_grant_sampled", "sess_grant_created"):
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "go")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
        step = {
            "model_step_id": "mstep_grant",
            "model_attempt_id": "matt_grant",
            "tool_round_id": "tround_grant",
            "tool_call_id": "call_1",
        }
        sampled_evidence = {"grant_id": "grant-sampled", "decision": "sampled_review"}
        for session_id, request_metadata, resolution_metadata in (
            ("sess_grant_sampled", {"cayu:authorization_grant": sampled_evidence}, {}),
            ("sess_grant_created", {}, {"cayu:authorization_grant": {"grant_id": "grant-new"}}),
        ):
            await store.append_events(
                session_id,
                [
                    Event(
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        payload={**step, "usage_metrics": usage.model_dump(mode="json")},
                    ),
                    Event(
                        type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                        session_id=session_id,
                        tool_name="side_effect",
                        payload={
                            **step,
                            "approval_id": f"approval-{session_id}",
                            "approval": {"metadata": request_metadata},
                        },
                    ),
                    Event(
                        type=EventType.TOOL_CALL_APPROVED,
                        session_id=session_id,
                        tool_name="side_effect",
                        payload={
                            **step,
                            "approval_id": f"approval-{session_id}",
                            "metadata": resolution_metadata,
                        },
                    ),
                ],
            )
        return await store.query_supervision_events(supervision_window())

    page = asyncio.run(run())

    sampled, created = page.events
    assert (sampled.authorization_grant_id, sampled.authorization_grant_role) == (
        "grant-sampled",
        SupervisionGrantRole.SAMPLED_REVIEW,
    )
    assert (created.authorization_grant_id, created.authorization_grant_role) == (
        "grant-new",
        SupervisionGrantRole.CREATED,
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("4"),
            ),
        )
    )
    costs = price_supervision_events(page, pricing)
    assert set(costs) == {sampled.supervision_id, created.supervision_id}
    assert costs[sampled.supervision_id].priced is True
    assert costs[sampled.supervision_id].total_cost == Decimal("0.004")
    empty = SupervisionEventPage(events=(), next_cursor=page.next_cursor, has_more=False)
    assert price_supervision_events(empty, pricing) == {}


def test_user_input_recovery_names_the_manual_recovery_request():
    from tests.core.test_user_input import (
        _build,
        _collect,
        _CountingTool,
        _crashed_user_input_resume_events,
        _drain,
        private_events_for_public_events,
    )

    session_id = "sess_attention_input_recovery"

    async def run():
        store = InMemorySessionStore()
        app, _ = _build(
            [("call_1", "count", {}), ("call_2", "ask_user", {"question": "q"})],
            tools=[UserInputTool(), _CountingTool()],
            store=store,
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        question = await _attention_request(store, session_id)
        await store.append_events(
            session_id,
            _crashed_user_input_resume_events(
                await private_events_for_public_events(store, paused),
                session_id=session_id,
                tool_call_id="call_1",
            ),
        )
        stuck = await _drain(
            app.resolve_user_input(
                UserInputResponse(
                    session_id=session_id, input_id=question.reference.action_id, answer="a"
                )
            )
        )
        assert stuck[-1].payload.get("manual_recovery_required") is True
        manual = await _attention_request(store, session_id)
        assert manual.reference.kind == "manual_recovery"
        await _drain(
            app.recover_user_input(
                UserInputRecoveryRequest(
                    session_id=session_id,
                    input_id=question.reference.action_id,
                    answer="a",
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="Independent external receipt verified.",
                )
            )
        )
        observed = await app.get_human_attention_state(manual.reference)
        return question, manual, observed, await supervision_page(store, kinds=["recover"])

    question, manual, observed, page = asyncio.run(run())

    (recovered,) = page.events
    assert recovered.tool_call_ids == ("call_1",)
    assert recovered.attention_id == manual.reference.attention_id
    assert recovered.attention_id != question.reference.attention_id
    assert recovered.attention_state == observed.state == "resolved"


def test_approval_recovery_names_the_open_approval_request():
    session_id = "sess_attention_approval_recovery"
    store = FailingTerminalToolEventStore()
    app = CayuApp(session_store=store)
    app.register_provider(
        FakeProvider(
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
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[PureRecordingTool()],
        tool_policy=RequireApprovalPolicy(),
    )

    async def run():
        pending = await pause_for_approval(app, store, session_id)
        # Without a manual-recovery handoff the open request stays the approval.
        request = await _attention_request(store, session_id)
        crashed = await collect_tool_approval_events(
            app, resolve_approval(session_id, pending, ToolApprovalDecision.APPROVE)
        )
        assert crashed[-1].payload.get("manual_recovery_required") is not True
        await collect_tool_approval_recovery_events(
            app,
            ToolApprovalRecoveryRequest(
                session_id=session_id,
                approval_id=pending.approval_id,
                tool_round_id=pending.tool_round_id,
                tool_call_id=pending.tool_call_id,
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="done outside",
                resolved_by=_reviewer_actor(),
            ),
        )
        observed = await app.get_human_attention_state(request.reference)
        return request, observed, await supervision_page(store, kinds=["recover"])

    request, observed, page = asyncio.run(run())

    (recovered,) = page.events
    assert recovered.attention_id == request.reference.attention_id
    assert recovered.attention_state == observed.state == "resolved"
