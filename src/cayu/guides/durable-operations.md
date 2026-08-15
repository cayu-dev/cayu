# Durable operations lifecycle

Use this quickstart when an agent must observe a system, diagnose it, propose a
change, obtain authority, act, verify the result, and survive process restarts.
It is a paved composition of existing public Cayu APIs, not a second
orchestration engine. For individual contracts, keep using `cayu guide
references` and `cayu guide tool-effects`.

## Lifecycle map

| Phase | Durable/public Cayu seam | Application responsibility |
| --- | --- | --- |
| Observe and diagnose | ordinary model turns and `ToolEffect.NONE` tools | Return bounded evidence; distinguish unknown from healthy. |
| Propose | an application-owned durable proposal record, followed by a model tool call held by a proposal-binding `ToolPolicy` | Store bounded, non-secret review fields and put the same stable domain `action_id` in the proposed action. |
| Authorize | `PendingActionQuery` plus the application proposal record, then `ToolApprovalRequest` | Authenticate the operator, authorize them for the exact review fields, and resolve the durable approval/round/call IDs with typed provenance. |
| Act once | Cayu's durable tool-round ledger plus the tool's effect contract | Use `ToolContext.idempotency_key` or the stable domain `action_id` at the downstream system. |
| Verify | a separate read-only tool call and its durable result/event | Observe post-action state; report pass, fail, or inconclusive. |
| Inspect/recover | pending actions and `recover_incomplete_session` | Reconcile durable evidence before continuing; do not equate retry with recovery. |

Use `SQLiteSessionStore` and `SQLiteTaskStore` for a durable single-host
application. Use a deployment-appropriate shared store for multiple active
processes. In-memory stores are for hermetic tests, never production ownership.
The `CayuApp` object remains process-local; every replacement process rebuilds
it against the same stores.

If dispatch can commit before its success acknowledgement becomes durable, use
the focused `cayu guide tool-effects#act-once-recovery` protocol within the Act
once and Inspect/recover phases. It adds durable uncertainty and bounded
reconciliation dispositions; it is not a second lifecycle.

## Runnable public-API skeleton

This credential-free program performs no real external effect. Its proposal and
action tools write only to fake maps so the complete proposal, authorization,
one action receipt, verification, pending-action inspection, and recovery path
can be tested deterministically. In production, both maps must be durable,
tenant-qualified application stores. Copy the architecture, not the fake
storage or effect. All imports below are from the public `cayu` package.

```python
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from cayu import (
    AgentSpec,
    CayuApp,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    Message,
    ModelStreamEvent,
    PendingActionKind,
    PendingActionQuery,
    ResolutionActor,
    ResolutionActorSource,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    SQLiteTaskStore,
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
)

ACTION_ID_PATTERN = r"^change-[0-9]{4}$"
MAX_ACTION_ID_LENGTH = 11
ALLOWED_TARGET = "demo"
ALLOWED_REASON = "observed_drift"


def validate_action_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) > MAX_ACTION_ID_LENGTH
        or re.fullmatch(ACTION_ID_PATTERN, value, flags=re.ASCII) is None
    ):
        raise ValueError("action_id is not allowed")
    return value


def proposal_input_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "action_id": {
                "type": "string",
                "pattern": ACTION_ID_PATTERN,
                "maxLength": MAX_ACTION_ID_LENGTH,
            },
            "target": {"type": "string", "enum": [ALLOWED_TARGET]},
            "reason": {"type": "string", "enum": [ALLOWED_REASON]},
        },
        "required": ["action_id", "target", "reason"],
        "additionalProperties": False,
    }


def validate_proposal(args: dict) -> dict[str, str]:
    """Fail closed on anything outside this application's review vocabulary."""
    if set(args) != {"action_id", "target", "reason"}:
        raise ValueError("proposal fields are not allowed")
    action_id = validate_action_id(args.get("action_id"))
    if args.get("target") != ALLOWED_TARGET or args.get("reason") != ALLOWED_REASON:
        raise ValueError("proposal vocabulary is not allowed")
    return {"action_id": action_id, "target": ALLOWED_TARGET, "reason": ALLOWED_REASON}


def require_authorized_operator(
    actor: ResolutionActor,
    review: dict[str, str],
) -> ResolutionActor:
    """Represent a product handler's fail-closed authorization decision."""
    if actor.tenant != "demo-tenant":
        raise PermissionError("operator is outside the proposal tenant")
    roles = actor.claims.get("roles")
    if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
        raise PermissionError("operator roles must be an exact JSON string array")
    if "operations-approver" not in roles:
        raise PermissionError("operator lacks the required role")
    if review["target"] != ALLOWED_TARGET:
        raise PermissionError("operator is not authorized for this target")
    return actor


class ProposalStore:
    """A fake application-owned review projection for this test only."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, str]] = {}


class FakeSystem:
    """A fake system with one synchronous operation-record commit point."""

    def __init__(self) -> None:
        self.operations: dict[str, dict] = {}
        self.timeline: list[str] = []

    async def apply_once(self, action_id: str, target: str) -> dict[str, str]:
        self.timeline.append("apply_started")
        await asyncio.sleep(0)  # Make an ordering race observable in this test.
        receipt = {"action_id": action_id, "target": target, "status": "applied"}
        operation = {"target": target, "receipt": receipt}
        existing = self.operations.get(action_id)
        if existing is not None and existing != operation:
            raise ValueError("action identity conflicts with the durable operation")
        # One assignment is the fake transaction. A real downstream system must
        # durably reserve the identity before its effect, then reconcile and
        # publish a terminal receipt without claiming generic exactly-once safety.
        self.operations[action_id] = operation
        self.timeline.append("apply_finished")
        return receipt

    def observe_target(self, action_id: str) -> str | None:
        self.timeline.append("verify")
        operation = self.operations.get(action_id)
        return None if operation is None else operation["target"]


class RecordProposal(Tool):
    spec = ToolSpec(
        name="record_proposal",
        description="Record bounded non-secret fields for operator review.",
        input_schema=proposal_input_schema(),
        parallel_safe=False,
        effect=ToolEffect.IDEMPOTENT,
    )

    def __init__(self, proposal_store: ProposalStore) -> None:
        super().__init__()
        self.proposal_store = proposal_store

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            record = validate_proposal(args)
        except ValueError:
            return ToolResult(
                content="proposal rejected by the application vocabulary",
                structured={"recorded": False},
                is_error=True,
            )
        existing = self.proposal_store.records.get(ctx.session_id)
        if existing is not None and existing != record:
            return ToolResult(
                content="proposal identity conflicts with the durable record",
                structured={"recorded": False},
                is_error=True,
            )
        self.proposal_store.records[ctx.session_id] = record
        return ToolResult(content="proposal recorded", structured=record)


class ReviewedActionPolicy(ToolPolicy):
    """Bind approval to a previously recorded, application-validated proposal."""

    def __init__(self, proposal_store: ProposalStore) -> None:
        self.proposal_store = proposal_store

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name in {"record_proposal", "verify_change"}:
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)
        if request.tool_name != "apply_change":
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason="Tool is outside the reviewed operations lifecycle.",
            )
        recorded = self.proposal_store.records.get(request.session.id)
        if recorded is None:
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason="No pre-existing review record for this session.",
            )
        try:
            proposed = validate_proposal(request.arguments)
        except ValueError:
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason="Action is outside the application review vocabulary.",
            )
        if proposed != recorded:
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason="Action does not match the reviewed proposal.",
            )
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            reason="Application-validated proposal requires human authority.",
        )


class ApplyChange(Tool):
    spec = ToolSpec(
        name="apply_change",
        description="Apply one explicitly authorized change.",
        input_schema=proposal_input_schema(),
        parallel_safe=False,
        # Replays collapse through the stable domain action_id contract below.
        effect=ToolEffect.IDEMPOTENT,
    )

    def __init__(self, proposal_store: ProposalStore, system: FakeSystem) -> None:
        super().__init__()
        self.proposal_store = proposal_store
        self.system = system

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            proposed = validate_proposal(args)
        except ValueError:
            return ToolResult(
                content="action rejected by the application vocabulary",
                structured={"applied": False},
                is_error=True,
            )
        if self.proposal_store.records.get(ctx.session_id) != proposed:
            return ToolResult(
                content="action does not match the reviewed proposal",
                structured={"applied": False},
                is_error=True,
            )
        action_id = args["action_id"]
        try:
            receipt = await self.system.apply_once(action_id, args["target"])
        except ValueError:
            return ToolResult(
                content="action identity conflicts with the durable operation",
                structured={"applied": False},
                is_error=True,
            )
        return ToolResult(content="change receipt", structured=receipt)


class VerifyChange(Tool):
    spec = ToolSpec(
        name="verify_change",
        description="Observe post-action state for one stable action identity.",
        input_schema={
            "type": "object",
            "properties": {
                "action_id": {
                    "type": "string",
                    "pattern": ACTION_ID_PATTERN,
                    "maxLength": MAX_ACTION_ID_LENGTH,
                }
            },
            "required": ["action_id"],
            "additionalProperties": False,
        },
        effect=ToolEffect.NONE,
    )

    def __init__(self, system: FakeSystem) -> None:
        super().__init__()
        self.system = system

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            action_id = validate_action_id(args.get("action_id"))
        except ValueError:
            return ToolResult(
                content="verification identity rejected by the application vocabulary",
                structured={"verified": False},
                is_error=True,
            )
        # This is a fresh post-action observation, not the execution receipt.
        observed_target = self.system.observe_target(action_id)
        verified = observed_target == ALLOWED_TARGET
        return ToolResult(
            content="verification passed" if verified else "verification inconclusive",
            structured={
                "action_id": action_id,
                "observed_target": observed_target,
                "verified": verified,
            },
            is_error=not verified,
        )


async def scenario(database: Path) -> None:
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="proposal-call",
                    name="record_proposal",
                    arguments={
                        "action_id": "change-0001",
                        "target": "demo",
                        "reason": "observed_drift",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="action-call",
                    name="apply_change",
                    arguments={
                        "action_id": "change-0001",
                        "target": "demo",
                        "reason": "observed_drift",
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="verify-call",
                    name="verify_change",
                    arguments={"action_id": "change-0001"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Change change-0001 is verified."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    proposal_store = ProposalStore()
    system = FakeSystem()
    record_proposal = RecordProposal(proposal_store)
    action = ApplyChange(proposal_store, system)
    verify = VerifyChange(system)
    app = CayuApp(
        session_store=SQLiteSessionStore(str(database)),
        task_store=SQLiteTaskStore(str(database)),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="operator",
            model="scripted-model",
            workflow_tool_names=("record_proposal", "apply_change", "verify_change"),
        ),
        tools=[record_proposal, action, verify],
        # Proposal recording and verification are allowed; the consequential
        # action is the only operation that pauses for human authority.
        tool_policy=ReviewedActionPolicy(proposal_store),
    )

    # Observe/diagnose/propose. Execution stops before ApplyChange.run.
    _pause_events = [
        event
        async for event in app.run(
            RunRequest(
                session_id="ops-demo",  # stable across process restarts
                agent_name="operator",
                messages=[
                    Message.text(
                        "user",
                        "Observe the demo, diagnose drift, and propose a safe correction.",
                    )
                ],
            )
        )
    ]
    assert system.operations == {}

    pending = await app.session_store.query_pending_actions(
        PendingActionQuery(session_id="ops-demo")
    )
    assert not pending.issues and len(pending.actions) == 1
    proposal = pending.actions[0]
    assert proposal.kind is PendingActionKind.TOOL_APPROVAL
    # Public pending-action projections intentionally omit executable arguments.
    assert proposal.arguments is None

    # The product handler joins the pending runtime identity to its own bounded
    # review projection. Never infer authorization from the redacted runtime
    # projection alone, and never put secrets in the application review record.
    review = proposal_store.records[proposal.session.id]
    assert review == {
        "action_id": "change-0001",
        "target": "demo",
        "reason": "observed_drift",
    }

    # Safe to call after a replacement process starts. Recovery recognizes that
    # this session still needs its existing approval; it does not run the tool.
    recovery = await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(session_id="ops-demo")
    )
    assert IncompleteSessionRecoveryAction.PENDING_APPROVAL in recovery.actions
    assert system.operations == {}

    # This represents a trusted server-side handler after authentication and an
    # application authorization check over the exact bounded review record.
    # Direct SDK access is a trusted boundary: do not expose this method to an
    # untrusted client or accept caller-asserted roles without authentication.
    authenticated_operator = ResolutionActor(
        subject="operator@example.test",
        tenant="demo-tenant",
        source=ResolutionActorSource.REQUEST,
        claims={"roles": ["operations-approver"]},
    )
    authorized_operator = require_authorized_operator(authenticated_operator, review)

    # Authorization is bound to the exact durable runtime identity. Use these
    # IDs from the current pending record; do not regenerate or guess them, and
    # stamp the authenticated resolver into the durable audit trail.
    _resume_events = [
        event
        async for event in app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=proposal.session.id,
                approval_id=proposal.approval_id,
                tool_round_id=proposal.round_id,
                tool_call_id=proposal.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
                reason="reviewed exact proposal",
                resolved_by=authorized_operator,
            )
        )
    ]

    # The action ran once, then the separate verifier ran, then the model ended.
    assert system.operations == {
        "change-0001": {
            "target": "demo",
            "receipt": {
                "action_id": "change-0001",
                "target": "demo",
                "status": "applied",
            },
        }
    }
    # Action and verification came from one approved model round. The yielding
    # action is an ordering barrier, so the read cannot race ahead of its effect.
    assert system.timeline == ["apply_started", "apply_finished", "verify"]
    assert len(provider.requests) == 3
    assert not (
        await app.session_store.query_pending_actions(
            PendingActionQuery(session_id="ops-demo")
        )
    ).actions

    # A replacement process can inspect the terminal durable state.
    reopened = SQLiteSessionStore(str(database))
    state = await reopened.load_state("ops-demo")
    assert state is not None and state.status.value == "completed"


with TemporaryDirectory() as temporary:
    asyncio.run(scenario(Path(temporary) / "cayu.db"))
```

Run the skeleton as a test before replacing the scripted provider or fake tool.
For a real agent, persist the application review projection and action receipts
outside the process, preserve the `action_id` from proposal through action, and
put receipt/verification evidence in structured tool results. Keep
observations, diagnoses, review fields, and verification evidence bounded and
non-secret. Cayu persists the runtime transcript, checkpoints, events, policy
decision, resolver provenance, and tool-round identity; the application must
authenticate and authorize the resolver, and the external system must still
provide its own idempotency or receipt contract.

## Recovery decision table

Inspect `PendingActionQuery` first and call
`recover_incomplete_session(IncompleteSessionRecoveryRequest(...))` to classify
an incomplete session from durable evidence.

| Evidence | Safe disposition |
| --- | --- |
| Approval pending; no start evidence | approve or deny the existing exact proposal. |
| Non-effectful work incomplete | resume/recompute only through the public recovery API. |
| Terminal tool receipt is durable | let recovery reconcile/publicize that receipt; do not call the effect directly. |
| External action started but no trustworthy terminal receipt | require operator/downstream reconciliation; never automatically replay. |
| Verification failed or is inconclusive | retain evidence and require a new diagnosis; do not invent success. |

Cayu also exposes `ToolApprovalRecoveryRequest` and `ToolRoundRecoveryRequest`
for manual reconciliation when pending-action inspection identifies uncertain
tool evidence. Supply a terminal outcome only after an operator has obtained
trustworthy external evidence. A recovery request records that evidence; it is
not permission to guess that an effect succeeded.

The local operator surface uses these same durable APIs: run `cayu session` for
CLI inspection, or `cayu serve --dev` on a trusted local machine and open
`/cayu/`. Production control-plane access requires `AuthenticatedAccess(...)`.
Do not create a parallel approval database or expose private checkpoint IDs and
arguments in a product API.

## Unsafe shortcuts

- **Approval after execution:** authorization must pause the proposed tool call
  before the effect. A prompt saying "ask first" is not enforcement.
- **Approval without review evidence:** public pending-action projections omit
  executable arguments by design. Join their durable IDs to a bounded,
  application-owned proposal record, then authenticate and authorize the human
  against those exact fields before resolving the approval. Record the proposal
  in an earlier tool round: Cayu authorizes an entire round before executing any
  tool, so a proposal and consequential action emitted together must fail closed.
- **Caller-asserted authority:** direct SDK resolution is a trusted boundary.
  Gate it in an authenticated server-side handler and pass `ResolutionActor`
  provenance; a `reason` string is not identity or authorization.
- **Regenerated identities:** never generate a new action, approval, round, or
  call ID while resolving an existing proposal. Resolve IDs from the current
  pending-action record. Keep one stable domain `action_id` through proposal,
  receipt, verification, and downstream idempotency.
- **Process-local production state:** a global `CayuApp`, dict, lock, or in-memory
  store does not coordinate replacement workers. Rebuild the app over durable
  stores and put effect idempotency in durable downstream state.
- **Blind replay:** Cayu does not promise that an arbitrary external side effect
  is exactly once. If a process can die after starting an effect but before a
  terminal receipt is durable, classify the outcome as uncertain and reconcile
  it. Never turn "started without terminal evidence" into automatic retry or
  assumed success. `TaskStore.terminalize_task(...)` makes Cayu's own durable
  task completion/failure replay-safe after acknowledgement loss; its receipt
  proves only that task state commit. It does not prove that a provider, tool,
  webhook, payment, or other external effect happened exactly once.
- **Verification by narration:** a successful action receipt is not post-action
  verification. Use a separate read-only observation and retain pass, fail, or
  inconclusive evidence tied to the same stable `action_id`.
