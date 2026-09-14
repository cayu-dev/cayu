"""Self-hosted attention integration. Every CLI command can run in a fresh process."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from examples.human_attention.consumer import DurableAttentionSink, Inbox, reconcile

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    HumanAttentionRequest,
    InterruptSessionRequest,
    Message,
    ModelStreamEvent,
    PendingActionQuery,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolContext,
    ToolEffect,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    ToolResult,
    ToolSpec,
    UserInputResponse,
)
from cayu.tools.user_input import UserInputTool


class ApprovalTool(Tool):
    spec = ToolSpec(
        name="record",
        description="Record a local demonstration decision.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        effect=ToolEffect.IDEMPOTENT,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="attention-demo-record", behavior_version="1", implementation_version="1"
        ),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="Recorded.")


class ApprovalPolicy(ToolPolicy):
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="attention-demo-policy", behavior_version="1", implementation_version="1"
        )

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL
            if request.tool_name == "record"
            else ToolPolicyDecision.ALLOW
        )


def build(
    state: Path,
    *,
    phase: str,
    kind: str = "user_input",
    crash_before_delivery: bool = False,
    fail_after_accept: bool = False,
    store=None,
):
    state.mkdir(parents=True, exist_ok=True)
    inbox = Inbox(state / "destination.sqlite")
    store = SQLiteSessionStore(state / "runtime.sqlite") if store is None else store
    sink = DurableAttentionSink(
        inbox, crash_before_delivery=crash_before_delivery, fail_after_accept=fail_after_accept
    )
    app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
    batches = [
        [
            ModelStreamEvent.text_delta("Finished."),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    ]
    if phase == "pause":
        batches.insert(
            0,
            [
                ModelStreamEvent.tool_call(
                    id="attention-call",
                    name="ask_user" if kind == "user_input" else "record",
                    arguments={"question": "Which environment?"} if kind == "user_input" else {},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        )
    app.register_provider(ScriptedModelProvider(batches), default=True)
    app.register_agent(
        AgentSpec(name="attention-demo", model="fixture"),
        tools=[UserInputTool(), ApprovalTool()],
        tool_policy=ApprovalPolicy(),
    )
    return app, store, inbox


async def command(args):
    app, store, inbox = build(
        args.state,
        phase=args.phase,
        kind=args.kind,
        crash_before_delivery=args.crash_before_delivery,
        fail_after_accept=args.fail_after_accept,
    )
    try:
        if args.phase == "pause":
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="attention-demo",
                        session_id="attention-example",
                        messages=[Message.text("user", "Begin.")],
                    )
                )
            ]
            print(json.dumps({"last_event": str(events[-1].type)}))
        elif args.phase == "consume":
            # Recovery is bounded and preserves durable retry spacing. Repair by
            # authoritative scan also covers pauses predating consumer enrollment.
            await app.recover_persisted_event_side_effects(limit=200)
            print(
                json.dumps(
                    await reconcile(
                        app,
                        inbox,
                        max_pages=args.max_pages,
                        crash_after_accept=args.crash_after_accept,
                    )
                )
            )
        elif args.phase == "inspect":
            print(json.dumps(inbox.inspect()))
        elif args.phase == "late-hint":
            inbox.accept_hint("attention-example", "late-opening-observation")
            print(json.dumps(await reconcile(app, inbox)))
        elif args.phase == "interrupt":
            events = [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(
                        session_id="attention-example",
                        reason="Local operator cancelled the question.",
                    )
                )
            ]
            print(json.dumps({"events": len(events)}))
        else:
            pending = await store.query_pending_actions(
                PendingActionQuery(session_id="attention-example")
            )
            if pending.issues or pending.has_more or len(pending.actions) != 1:
                raise RuntimeError("One fully observed current action is required.")
            request = HumanAttentionRequest.from_pending_action(pending.actions[0])
            if (
                request is None
                or (await app.get_human_attention_state(request.reference)).state != "active"
            ):
                raise RuntimeError("The exact action is no longer available.")
            ref = request.reference
            if args.phase == "answer" and ref.kind == "user_input":
                stream = app.resolve_user_input(
                    UserInputResponse(
                        session_id=ref.session_id, input_id=ref.action_id, answer="staging"
                    )
                )
            elif args.phase in {"approve", "deny"} and ref.kind == "tool_approval":
                if ref.round_id is None or ref.tool_call_id is None:
                    raise RuntimeError("The exact approval round and call are unavailable.")
                stream = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=ref.session_id,
                        approval_id=ref.action_id,
                        tool_round_id=ref.round_id,
                        tool_call_id=ref.tool_call_id,
                        decision=ToolApprovalDecision.APPROVE
                        if args.phase == "approve"
                        else ToolApprovalDecision.DENY,
                        reason="Authorized local demo operator.",
                    )
                )
            else:
                raise RuntimeError("Command does not match the pending action kind.")
            events = [event async for event in stream]
            print(json.dumps({"last_event": str(events[-1].type)}))
    finally:
        await store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=[
            "pause",
            "consume",
            "inspect",
            "answer",
            "approve",
            "deny",
            "interrupt",
            "late-hint",
        ],
    )
    parser.add_argument("state", type=Path)
    parser.add_argument("--kind", choices=["user_input", "tool_approval"], default="user_input")
    parser.add_argument("--crash-before-delivery", action="store_true")
    parser.add_argument("--fail-after-accept", action="store_true")
    parser.add_argument("--crash-after-accept", action="store_true")
    parser.add_argument("--max-pages", type=int, default=32)
    asyncio.run(command(parser.parse_args()))


if __name__ == "__main__":
    main()
