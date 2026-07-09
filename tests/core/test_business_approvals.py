"""Tests for the business approval adapter (tier routing + three outcomes)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cayu import (
    BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY,
    BUSINESS_APPROVAL_ROUTING_METADATA_KEY,
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    BusinessApprovalOutcome,
    BusinessApprovalRouting,
    BusinessApprovalRoutingMissing,
    BusinessApprovalTierMismatch,
    CayuApp,
    Message,
    ModelStreamEvent,
    TieredApprovalPolicy,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolPolicyDecision,
    ToolPolicyRequest,
    business_approval_audit,
    business_approval_routing,
    business_approval_routing_metadata,
    resolve_business_approval,
)
from cayu.core.events import Event, EventType
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.evals import ScriptedModelProvider
from cayu.runtime import (
    InMemorySessionStore,
    PendingToolApproval,
    ResolutionActor,
    ResolutionActorSource,
    RunRequest,
    Session,
    SessionStatus,
)

CHAIN = ("area", "national", "corporate")


class RecordingTool(Tool):
    spec = ToolSpec(
        name="route_package",
        description="Route a package for delivery.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []
        self.contexts: list[ToolContext] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        self.contexts.append(ctx)
        return ToolResult(content="routed")


def _routing(required_tier: str = "national") -> BusinessApprovalRouting:
    return BusinessApprovalRouting(
        required_tier=required_tier,
        chain=CHAIN,
        metadata={"package_id": "pkg_7"},
    )


def _policy_request(tool_name: str = "route_package") -> ToolPolicyRequest:
    return ToolPolicyRequest(
        session=Session(
            id="s",
            agent_name="router",
            provider_name="scripted",
            model="fake-model",
            status=SessionStatus.RUNNING,
        ),
        agent=AgentSpec(name="router", model="fake-model"),
        tool_name=tool_name,
        tool_call_id="call_1",
    )


def _paused_app(
    policy, session_id: str, clock=None
) -> tuple[CayuApp, InMemorySessionStore, RecordingTool, Event]:
    """Run one session until it pauses on approval; return the pause event."""
    store = InMemorySessionStore()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1", name="route_package", arguments={"package": "pkg_7"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    tool = RecordingTool()
    app = CayuApp(session_store=store, clock=clock)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="router", model="fake-model"),
        tools=[tool],
        tool_policy=policy,
    )

    async def run() -> list[Event]:
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="router",
                    session_id=session_id,
                    messages=[Message.text("user", "route pkg_7")],
                )
            )
        ]

    events = asyncio.run(run())
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    approval_event = next(
        event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    return app, store, tool, approval_event


def _tiered_policy(required_tier: str = "national") -> TieredApprovalPolicy:
    return TieredApprovalPolicy(
        compute_routing=lambda request: _routing(required_tier),
        reason="Routing requires business approval.",
    )


def _resolve(app: CayuApp, session_id: str, approval_event: Event, **overrides) -> list[Event]:
    pending = PendingToolApproval.from_event(approval_event)
    kwargs = {
        "session_id": session_id,
        "approval_id": pending.approval_id,
        "approver_id": "u_42",
        "approver_tier": "national",
        "outcome": BusinessApprovalOutcome.APPROVED,
    }
    kwargs.update(overrides)

    async def run() -> list[Event]:
        events = await resolve_business_approval(app, **kwargs)
        return [event async for event in events]

    return asyncio.run(run())


def test_tiered_policy_pauses_with_routing_and_allows_on_none() -> None:
    paused = asyncio.run(_tiered_policy().authorize(_policy_request()))
    assert paused.decision is ToolPolicyDecision.REQUIRE_APPROVAL
    assert BUSINESS_APPROVAL_ROUTING_METADATA_KEY in paused.metadata

    async def compute_none(request: ToolPolicyRequest) -> None:
        return None

    allowed = asyncio.run(
        TieredApprovalPolicy(compute_routing=compute_none).authorize(_policy_request())
    )
    assert allowed.decision is ToolPolicyDecision.ALLOW


def test_routing_carrier_round_trips_through_a_real_pause() -> None:
    _app, _store, tool, approval_event = _paused_app(_tiered_policy(), "sess_biz_roundtrip")
    pending = PendingToolApproval.from_event(approval_event)
    routing = business_approval_routing(pending)
    assert routing.required_tier == "national"
    assert routing.chain == CHAIN
    assert routing.metadata == {"package_id": "pkg_7"}
    assert tool.calls == []


def test_tier_gate_allows_at_or_above_and_rejects_below() -> None:
    app, store, tool, approval_event = _paused_app(_tiered_policy(), "sess_biz_tiers")

    with pytest.raises(BusinessApprovalTierMismatch):
        _resolve(app, "sess_biz_tiers", approval_event, approver_tier="area")
    with pytest.raises(BusinessApprovalTierMismatch):
        _resolve(app, "sess_biz_tiers", approval_event, approver_tier="intern")
    # Gate failures fire eagerly, before any event was consumed: nothing resumed.
    assert tool.calls == []
    session = asyncio.run(store.load("sess_biz_tiers"))
    assert session is not None and session.status == SessionStatus.INTERRUPTED

    events = _resolve(app, "sess_biz_tiers", approval_event, approver_tier="corporate")
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == [{"package": "pkg_7"}]


def test_tier_gate_reads_persisted_routing_not_a_caller_claim() -> None:
    # Regression (reviewer talgat0528): the gate must read the routing persisted
    # on the checkpoint, so a sub-tier approver cannot resolve a corporate-routed
    # approval. Resolution accepts no caller-supplied pending/routing.
    app, _store, tool, approval_event = _paused_app(
        _tiered_policy("corporate"), "sess_biz_persisted_routing"
    )
    pending = PendingToolApproval.from_event(approval_event)
    assert business_approval_routing(pending).required_tier == "corporate"

    for tier in ("area", "national"):
        with pytest.raises(BusinessApprovalTierMismatch):
            _resolve(app, "sess_biz_persisted_routing", approval_event, approver_tier=tier)
    assert tool.calls == []

    events = _resolve(app, "sess_biz_persisted_routing", approval_event, approver_tier="corporate")
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == [{"package": "pkg_7"}]


def test_approved_outcome_stamps_resolution_metadata() -> None:
    app, _store, tool, approval_event = _paused_app(_tiered_policy(), "sess_biz_approved")
    events = _resolve(app, "sess_biz_approved", approval_event)
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == [{"package": "pkg_7"}]

    approved = next(event for event in events if event.type == EventType.TOOL_CALL_APPROVED)
    stamp = approved.payload["metadata"][BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY]
    assert stamp["outcome"] == "approved"
    assert stamp["condition_text"] is None
    assert stamp["approver_id"] == "u_42"
    assert stamp["approver_tier"] == "national"
    assert stamp["required_tier"] == "national"
    assert stamp["chain"] == list(CHAIN)

    with pytest.raises(ValueError, match="condition_text is only valid"):
        _resolve(app, "sess_biz_approved", approval_event, condition_text="n/a")


def test_conditioned_outcome_requires_text_and_reaches_the_tool() -> None:
    app, _store, tool, approval_event = _paused_app(_tiered_policy(), "sess_biz_conditioned")

    with pytest.raises(ValueError, match="condition_text"):
        _resolve(
            app,
            "sess_biz_conditioned",
            approval_event,
            outcome=BusinessApprovalOutcome.CONDITIONED,
        )
    assert tool.calls == []

    events = _resolve(
        app,
        "sess_biz_conditioned",
        approval_event,
        outcome=BusinessApprovalOutcome.CONDITIONED,
        condition_text="Ship only after payment clears.",
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == [{"package": "pkg_7"}]
    # The tool observes the business record through its context metadata.
    stamp = tool.contexts[0].metadata[BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY]
    assert stamp["outcome"] == "conditioned"
    assert stamp["condition_text"] == "Ship only after payment clears."


def test_declined_outcome_denies_with_self_explaining_reason() -> None:
    app, store, tool, approval_event = _paused_app(_tiered_policy(), "sess_biz_declined")

    with pytest.raises(ValueError, match="condition_text is only valid"):
        _resolve(
            app,
            "sess_biz_declined",
            approval_event,
            outcome=BusinessApprovalOutcome.DECLINED,
            condition_text="not allowed here",
        )

    events = _resolve(
        app, "sess_biz_declined", approval_event, outcome=BusinessApprovalOutcome.DECLINED
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == []
    denied = next(event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_DENIED)
    assert denied.payload["reason"] == "Business approval declined by u_42 (national)."
    transcript = asyncio.run(store.load_transcript("sess_biz_declined"))
    blocked = next(
        part
        for message in transcript
        for part in message.content
        if type(part).__name__ == "ToolResultPart"
    )
    assert blocked.is_error is True
    assert "Business approval declined by u_42" in blocked.content


def test_missing_routing_refuses_loudly() -> None:
    app, _store, _tool, approval_event = _paused_app(
        AlwaysRequireApprovalToolPolicy(), "sess_biz_unrouted"
    )
    with pytest.raises(BusinessApprovalRoutingMissing, match="no business approval routing"):
        _resolve(app, "sess_biz_unrouted", approval_event)
    # The raw primitive still resolves such approvals.
    pending = PendingToolApproval.from_event(approval_event)

    async def resolve_raw() -> list[Event]:
        return [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="sess_biz_unrouted",
                    approval_id=pending.approval_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]

    events = asyncio.run(resolve_raw())
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_reserved_resolution_key_collision_is_rejected() -> None:
    app, _store, _tool, approval_event = _paused_app(_tiered_policy(), "sess_biz_collision")
    with pytest.raises(ValueError, match="reserved key"):
        _resolve(
            app,
            "sess_biz_collision",
            approval_event,
            metadata={BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY: {"outcome": "forged"}},
        )


def test_wrong_approval_id_is_rejected() -> None:
    app, _store, _tool, approval_event = _paused_app(_tiered_policy(), "sess_biz_wrong_id")
    with pytest.raises(ValueError, match="does not match pending approval"):
        _resolve(app, "sess_biz_wrong_id", approval_event, approval_id="appr_other")


def test_audit_projects_pending_adapter_and_raw_resolutions() -> None:
    # Pending-only session.
    _app_pending, store_pending, _tool, _event = _paused_app(_tiered_policy(), "sess_audit_pending")
    pending_records = business_approval_audit(
        asyncio.run(store_pending.load_events("sess_audit_pending"))
    )
    assert len(pending_records) == 1
    record = pending_records[0]
    assert record.pending and not record.resolved_via_adapter
    assert record.routing is not None and record.routing.required_tier == "national"
    assert record.tool_name == "route_package"
    assert record.requested_at is not None and record.resolved_at is None

    # Adapter-resolved session.
    app_adapter, store_adapter, _tool, approval_event = _paused_app(
        _tiered_policy(), "sess_audit_adapter"
    )
    _resolve(
        app_adapter,
        "sess_audit_adapter",
        approval_event,
        outcome=BusinessApprovalOutcome.CONDITIONED,
        condition_text="Cap at 500 USD.",
        approver_tier="corporate",
    )
    adapter_records = business_approval_audit(
        asyncio.run(store_adapter.load_events("sess_audit_adapter"))
    )
    assert len(adapter_records) == 1
    record = adapter_records[0]
    assert record.decision is ToolApprovalDecision.APPROVE
    assert record.outcome is BusinessApprovalOutcome.CONDITIONED
    assert record.condition_text == "Cap at 500 USD."
    assert record.approver_id == "u_42"
    assert record.approver_tier == "corporate"
    assert record.resolved_via_adapter
    assert record.resolved_at is not None

    # Raw-primitive resolution: the projection exposes the adapter bypass.
    app_raw, store_raw, _tool, approval_event = _paused_app(_tiered_policy(), "sess_audit_raw")
    pending = PendingToolApproval.from_event(approval_event)

    async def resolve_raw() -> None:
        async for _event in app_raw.resolve_tool_approval(
            ToolApprovalRequest(
                session_id="sess_audit_raw",
                approval_id=pending.approval_id,
                decision=ToolApprovalDecision.APPROVE,
            )
        ):
            pass

    asyncio.run(resolve_raw())
    raw_records = business_approval_audit(asyncio.run(store_raw.load_events("sess_audit_raw")))
    assert len(raw_records) == 1
    record = raw_records[0]
    assert record.decision is ToolApprovalDecision.APPROVE
    assert record.outcome is None and not record.resolved_via_adapter


def test_audit_is_order_insensitive_and_pairs_per_session() -> None:
    app_a, store_a, _tool, approval_event_a = _paused_app(_tiered_policy(), "sess_audit_order_a")
    _resolve(app_a, "sess_audit_order_a", approval_event_a)
    events_a = asyncio.run(store_a.load_events("sess_audit_order_a"))

    # Reversed input (e.g. query_events with a descending order): the resolution
    # arrives before its request, and the projection must still pair them.
    reversed_records = business_approval_audit(list(reversed(events_a)))
    assert len(reversed_records) == 1
    assert reversed_records[0].decision is ToolApprovalDecision.APPROVE
    assert reversed_records[0].outcome is BusinessApprovalOutcome.APPROVED
    assert not reversed_records[0].pending

    # A mixed multi-session feed pairs records per session.
    app_b, store_b, _tool, approval_event_b = _paused_app(_tiered_policy(), "sess_audit_order_b")
    _resolve(
        app_b,
        "sess_audit_order_b",
        approval_event_b,
        outcome=BusinessApprovalOutcome.DECLINED,
        approver_id="u_other",
    )
    events_b = asyncio.run(store_b.load_events("sess_audit_order_b"))
    mixed = business_approval_audit(events_b + events_a)
    by_session = {record.session_id: record for record in mixed}
    assert len(mixed) == 2
    assert by_session["sess_audit_order_a"].outcome is BusinessApprovalOutcome.APPROVED
    assert by_session["sess_audit_order_a"].approver_id == "u_42"
    assert by_session["sess_audit_order_b"].outcome is BusinessApprovalOutcome.DECLINED
    assert by_session["sess_audit_order_b"].approver_id == "u_other"


def test_lost_resolution_race_surfaces_the_primitives_error() -> None:
    # Pins the docstring contract: the primitive independently reloads and
    # re-validates when the returned stream is first iterated, so an adapter
    # stream that lost a resolution race fails cleanly — no double resolution.
    session_id = "sess_biz_race"
    app, store, tool, approval_event = _paused_app(_tiered_policy(), session_id)
    pending = PendingToolApproval.from_event(approval_event)

    async def run() -> None:
        stale = await resolve_business_approval(
            app,
            session_id=session_id,
            approval_id=pending.approval_id,
            approver_id="u_slow",
            approver_tier="national",
            outcome=BusinessApprovalOutcome.APPROVED,
        )
        # Another resolver wins before the stale stream is iterated.
        async for _event in app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=pending.approval_id,
                decision=ToolApprovalDecision.APPROVE,
            )
        ):
            pass
        with pytest.raises(RuntimeError, match="no pending tool approval"):
            async for _event in stale:
                pass

    asyncio.run(run())
    # The winning resolution ran the tool exactly once; the stale stream added
    # nothing to the ledger.
    assert tool.calls == [{"package": "pkg_7"}]
    events = asyncio.run(store.load_events(session_id))
    approved = [event for event in events if event.type == EventType.TOOL_CALL_APPROVED]
    assert len(approved) == 1


def test_tiered_policy_expiry_flows_to_the_pending_approval() -> None:
    clock = {"now": datetime(2026, 7, 9, 12, 0, tzinfo=UTC)}
    policy = TieredApprovalPolicy(
        compute_routing=lambda request: _routing(),
        expires_in_seconds=60,
    )
    _app, _store, _tool, approval_event = _paused_app(
        policy, "sess_biz_expiry_carrier", clock=lambda: clock["now"]
    )
    pending = PendingToolApproval.from_event(approval_event)
    assert pending.expires_at == datetime(2026, 7, 9, 12, 1, tzinfo=UTC)


def test_expired_adapter_resolution_is_coerced_and_audited() -> None:
    # Approval expiry composes decision-first: a tier-authorized CONDITIONED
    # resolution after the window closes is coerced by the runtime to a
    # denial; the audit preserves the asserted outcome next to expired=True.
    clock = {"now": datetime(2026, 7, 9, 12, 0, tzinfo=UTC)}
    session_id = "sess_biz_expired"
    policy = TieredApprovalPolicy(
        compute_routing=lambda request: _routing(),
        expires_in_seconds=60,
    )
    app, store, tool, approval_event = _paused_app(policy, session_id, clock=lambda: clock["now"])

    clock["now"] = datetime(2026, 7, 9, 12, 2, tzinfo=UTC)
    events = _resolve(
        app,
        session_id,
        approval_event,
        outcome=BusinessApprovalOutcome.CONDITIONED,
        condition_text="Ship only after payment clears.",
    )
    assert any(event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED for event in events)
    denied = next(event for event in events if event.type == EventType.TOOL_CALL_APPROVAL_DENIED)
    assert denied.payload["expired"] is True
    assert tool.calls == []

    records = business_approval_audit(asyncio.run(store.load_events(session_id)))
    assert len(records) == 1
    record = records[0]
    assert record.decision is ToolApprovalDecision.DENY
    assert record.outcome is BusinessApprovalOutcome.CONDITIONED
    assert record.condition_text == "Ship only after payment clears."
    assert record.expired is True
    # The runtime attributes the coerced denial to its system expiry actor.
    assert record.resolved_by is not None
    assert record.resolved_by.subject == "cayu:approval-expiry"


def test_resolved_by_provenance_reaches_the_audit() -> None:
    session_id = "sess_biz_provenance"
    app, store, _tool, approval_event = _paused_app(_tiered_policy(), session_id)
    actor = ResolutionActor(subject="maria@corp.example", source=ResolutionActorSource.REQUEST)
    events = _resolve(app, session_id, approval_event, resolved_by=actor)
    assert events[-1].type == EventType.SESSION_COMPLETED

    records = business_approval_audit(asyncio.run(store.load_events(session_id)))
    assert len(records) == 1
    record = records[0]
    assert record.resolved_by is not None
    assert record.resolved_by.subject == "maria@corp.example"
    assert record.resolved_by.source is ResolutionActorSource.REQUEST
    # Typed provenance sits next to the caller-asserted business fields.
    assert record.approver_id == "u_42"


def test_non_dict_metadata_is_rejected_at_entry() -> None:
    app, _store, _tool, approval_event = _paused_app(_tiered_policy(), "sess_biz_bad_metadata")
    with pytest.raises(TypeError, match="metadata must be a dict"):
        _resolve(app, "sess_biz_bad_metadata", approval_event, metadata=[1, 2, 3])


def test_routing_model_validation() -> None:
    with pytest.raises(ValidationError, match="member of the chain"):
        BusinessApprovalRouting(required_tier="global", chain=CHAIN)
    with pytest.raises(ValidationError, match="duplicate"):
        BusinessApprovalRouting(required_tier="area", chain=("area", "area"))
    with pytest.raises(ValidationError, match="kind"):
        BusinessApprovalRouting(
            kind="cayu.business_approval.routing.v2", required_tier="area", chain=CHAIN
        )
    carrier = business_approval_routing_metadata(required_tier="area", chain=CHAIN)
    assert set(carrier) == {BUSINESS_APPROVAL_ROUTING_METADATA_KEY}
