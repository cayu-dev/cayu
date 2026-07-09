"""Multi-tier business approval on top of the binary approval primitive.

Runs entirely offline (scripted provider, no API keys):

1. The agent calls a routing tool; a product-computed routing pauses the
   session for approval at the ``national`` tier.
2. An ``area``-tier approver is rejected by the tier gate.
3. A ``national`` approver resolves with the CONDITIONED outcome; the tool
   runs and sees the condition through its context metadata.
4. The audit projection prints the business outcome next to the raw decision.

Recipe: docs/recipes/business-approvals.md.  Run with:

    PYTHONPATH=src python examples/business_approval_tiers.py
"""

from __future__ import annotations

import asyncio

from cayu import (
    AgentSpec,
    BusinessApprovalOutcome,
    BusinessApprovalRouting,
    BusinessApprovalTierMismatch,
    CayuApp,
    Message,
    ModelStreamEvent,
    PendingToolApproval,
    RunRequest,
    TieredApprovalPolicy,
    Tool,
    ToolContext,
    ToolPolicyRequest,
    ToolResult,
    ToolSpec,
    business_approval_audit,
    business_approval_routing,
    resolve_business_approval,
)
from cayu.core.events import EventType
from cayu.evals import ScriptedModelProvider
from cayu.runtime import InMemorySessionStore

CHAIN = ("area", "national", "corporate")


class RoutePackageTool(Tool):
    spec = ToolSpec(
        name="route_package",
        description="Route a package for delivery.",
        input_schema={
            "type": "object",
            "properties": {
                "package_id": {"type": "string"},
                "amount_usd": {"type": "number"},
            },
            "required": ["package_id", "amount_usd"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        record = ctx.metadata.get("cayu:business_approval") or {}
        condition = record.get("condition_text")
        note = f" (condition: {condition})" if condition else ""
        return ToolResult(
            content=f"Package {args['package_id']} routed{note}.",
            structured={"package_id": args["package_id"], "condition": condition},
        )


def compute_routing(request: ToolPolicyRequest) -> BusinessApprovalRouting | None:
    """Product-owned tier computation: value decides the required tier."""
    if request.tool_name != "route_package":
        return None
    amount = request.arguments.get("amount_usd", 0)
    tier = "corporate" if amount > 100_000 else "national" if amount > 10_000 else "area"
    return BusinessApprovalRouting(
        required_tier=tier,
        chain=CHAIN,
        metadata={"package_id": request.arguments.get("package_id")},
    )


async def main() -> None:
    store = InMemorySessionStore()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="route_package",
                    arguments={"package_id": "pkg_7", "amount_usd": 42_000},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Package pkg_7 is on its way."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="router", model="scripted-model"),
        tools=[RoutePackageTool()],
        tool_policy=TieredApprovalPolicy(
            compute_routing=compute_routing,
            reason="Package routing requires business approval.",
        ),
    )

    # 1. Run until the routing pauses the session for approval.
    session_id = "sess_business_approval_demo"
    approval_event = None
    async for event in app.run(
        RunRequest(
            agent_name="router",
            session_id=session_id,
            messages=[Message.text("user", "Route package pkg_7 (42,000 USD).")],
        )
    ):
        if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED:
            approval_event = event
    assert approval_event is not None
    pending = PendingToolApproval.from_event(approval_event)
    routing = business_approval_routing(pending)
    print(f"paused for approval: required_tier={routing.required_tier} chain={routing.chain}")
    print(f"routing metadata for the approval UI: {routing.metadata}")

    # 2. An area-tier approver is below the required tier: the gate rejects.
    try:
        await resolve_business_approval(
            app,
            session_id=session_id,
            approval_id=pending.approval_id,
            approver_id="sam.lee",
            approver_tier="area",
            outcome=BusinessApprovalOutcome.APPROVED,
        )
    except BusinessApprovalTierMismatch as exc:
        print(f"tier gate rejected: {exc}")

    # 3. A national approver resolves with conditions; the session resumes.
    events = await resolve_business_approval(
        app,
        session_id=session_id,
        approval_id=pending.approval_id,
        approver_id="maria.k",
        approver_tier="national",
        outcome=BusinessApprovalOutcome.CONDITIONED,
        condition_text="Ship only after payment clears.",
    )
    async for event in events:
        if event.type == EventType.TOOL_CALL_COMPLETED:
            print(f"tool result: {event.payload['result']['content']}")
        if event.type == EventType.SESSION_COMPLETED:
            print("session completed")

    # 4. The audit trail pairs the business outcome with the raw decision.
    for record in business_approval_audit(await store.load_events(session_id)):
        print(
            "audit:",
            f"decision={record.decision and record.decision.value}",
            f"outcome={record.outcome and record.outcome.value}",
            f"condition={record.condition_text!r}",
            f"approver={record.approver_id} ({record.approver_tier})",
        )


if __name__ == "__main__":
    asyncio.run(main())
