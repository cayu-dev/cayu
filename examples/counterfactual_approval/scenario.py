from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from examples._advanced_support import (
    ScenarioResult,
    advanced_run_limits,
    collect_events,
    count_model_completions,
    session_evidence,
    stable_output_spec,
    validated_output,
)

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    Event,
    EventType,
    InMemorySessionStore,
    Message,
    RunRequest,
    SessionStore,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelProvider

ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "future": {"type": "string", "enum": ["approve", "deny", "explain"]},
        "outcome": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "alternatives": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "recommendation": {"type": "string"},
        "uncertainties": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "external_state_version": {"type": "integer"},
    },
    "required": [
        "future",
        "outcome",
        "risks",
        "alternatives",
        "evidence",
        "recommendation",
        "uncertainties",
        "external_state_version",
    ],
    "additionalProperties": False,
}
VERIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "confirmed": {"type": "boolean"},
        "observed_version": {"type": "integer"},
        "observed_mutations": {"type": "integer"},
        "evidence": {"type": "string"},
    },
    "required": ["confirmed", "observed_version", "observed_mutations", "evidence"],
    "additionalProperties": False,
}


@dataclass
class DeploymentState:
    version: int = 7
    mutation_count: int = 0
    deployed_release: str | None = None
    receipts: dict[str, ToolResult] = field(default_factory=dict)

    def deploy(
        self,
        *,
        service: str,
        release: str,
        expected_version: int,
        idempotency_key: str,
    ) -> ToolResult:
        existing = self.receipts.get(idempotency_key)
        if existing is not None:
            structured = dict(existing.structured or {})
            structured["receipt_reused"] = True
            return existing.model_copy(update={"structured": structured})
        if expected_version != self.version:
            return ToolResult(
                content="External state changed; deployment rejected.",
                structured={
                    "expected_version": expected_version,
                    "actual_version": self.version,
                },
                is_error=True,
            )
        self.mutation_count += 1
        self.version += 1
        self.deployed_release = release
        result = ToolResult(
            content=f"Deployed {release} to {service}.",
            structured={
                "service": service,
                "release": release,
                "version": self.version,
                "mutation_count": self.mutation_count,
            },
        )
        self.receipts[idempotency_key] = result
        return result


class DeployServiceTool(Tool):
    spec = ToolSpec(
        name="deploy_service",
        description="Deploy a release only when the expected external-state version still matches.",
        input_schema={
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "release": {"type": "string"},
                "expected_version": {"type": "integer"},
            },
            "required": ["service", "release", "expected_version"],
            "additionalProperties": False,
        },
        effect=ToolEffect.EXTERNAL,
    )

    def __init__(self, state: DeploymentState) -> None:
        self.state = state

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if ctx.idempotency_key is None:
            raise RuntimeError("deploy_service requires a runtime idempotency key")
        return self.state.deploy(
            service=args["service"],
            release=args["release"],
            expected_version=args["expected_version"],
            idempotency_key=ctx.idempotency_key,
        )


class CrashAfterPrimaryMutationStore(InMemorySessionStore):
    """Inject one durable-write failure after the protected external effect."""

    def __init__(self, primary_session_id: str) -> None:
        super().__init__()
        self.primary_session_id = primary_session_id
        self.failed_terminal_once = False

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        if (
            session_id == self.primary_session_id
            and not self.failed_terminal_once
            and any(event.type == EventType.TOOL_CALL_COMPLETED for event in events)
        ):
            self.failed_terminal_once = True
            raise RuntimeError("simulated crash before the terminal tool receipt was persisted")
        await super().append_events(session_id, events)


async def run_scenario(
    root: Path,
    *,
    provider: ModelProvider,
    model: str,
    mode: str,
) -> ScenarioResult:
    state = DeploymentState()
    causal_budget_id = "advanced-counterfactual-approval-budget"
    primary_id = "approval-primary"
    store = CrashAfterPrimaryMutationStore(primary_id)
    app = _build_app(provider=provider, model=model, state=state, session_store=store)
    snapshot = json.loads(
        (Path(__file__).with_name("fixtures") / "approval_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    initial_events = await collect_events(
        app.run(
            RunRequest(
                agent_name="approval-request",
                session_id=primary_id,
                causal_budget_id=causal_budget_id,
                messages=[Message.text("user", f"Propose this protected action: {snapshot!r}")],
                limits=advanced_run_limits(),
            )
        )
    )
    approval_events = [
        event for event in initial_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    ]
    if len(approval_events) != 1:
        raise RuntimeError(f"Expected one approval request, got {len(approval_events)}")
    approval_id = approval_events[0].payload["approval"]["approval_id"]

    analyses: dict[str, dict[str, Any]] = {}
    role_to_future = {
        "approve-future": "approve",
        "deny-future": "deny",
        "explainer": "explain",
    }
    for role, future in role_to_future.items():
        events = await collect_events(
            app.run(
                RunRequest(
                    agent_name=role,
                    session_id=f"approval-{future}",
                    parent_session_id=primary_id,
                    causal_budget_id=causal_budget_id,
                    messages=[
                        Message.text(
                            "user",
                            f"Analyze future={future!r} from immutable snapshot {snapshot!r}.",
                        )
                    ],
                    structured_output=stable_output_spec(f"{future}-analysis", ANALYSIS_SCHEMA),
                    limits=advanced_run_limits(),
                )
            )
        )
        analyses[future] = validated_output(events)

    state_revalidated = state.version == snapshot["external_state_version"]
    approval_resolution = ToolApprovalRequest(
        session_id=primary_id,
        approval_id=approval_id,
        decision=ToolApprovalDecision.APPROVE,
        metadata={
            "external_state_revalidated": state_revalidated,
            "external_state_version": state.version,
        },
    )
    crashed_events = await collect_events(app.resolve_tool_approval(approval_resolution))
    if crashed_events[-1].type != EventType.SESSION_INTERRUPTED or state.mutation_count != 1:
        raise RuntimeError("Approval did not stop at the injected post-mutation crash boundary.")
    receipt_key, receipt = next(iter(state.receipts.items()))
    tool_call_id = approval_events[0].payload["approval"]["tool_call_id"]

    # Rebuild the application around the same durable store, then reconcile the
    # unknown result from the external receipt without executing the tool again.
    app = _build_app(provider=provider, model=model, state=state, session_store=store)
    recovered_events = await collect_events(
        app.recover_tool_approval(
            ToolApprovalRecoveryRequest(
                session_id=primary_id,
                approval_id=approval_id,
                tool_call_id=tool_call_id,
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message=receipt.content,
                structured=receipt.structured,
                metadata={"receipt_id": receipt_key, "source": "external-reconciliation"},
                limits=advanced_run_limits(),
            )
        )
    )
    if not any(event.type == EventType.SESSION_COMPLETED for event in recovered_events):
        raise RuntimeError("Recovered approval session did not complete.")

    # Exercise staleness through a second genuinely paused Cayu approval. The
    # original snapshot expects version 7, but the recovered deployment made it 8.
    stale_id = "approval-stale-probe"
    stale_initial = await collect_events(
        app.run(
            RunRequest(
                agent_name="stale-approval-probe",
                session_id=stale_id,
                causal_budget_id=causal_budget_id,
                messages=[Message.text("user", f"Request from stale snapshot: {snapshot!r}")],
                limits=advanced_run_limits(),
            )
        )
    )
    stale_approval = next(
        event for event in stale_initial if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    ).payload["approval"]
    stale_events = await collect_events(
        app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=stale_id,
                approval_id=stale_approval["approval_id"],
                decision=ToolApprovalDecision.APPROVE,
                metadata={"external_state_version": state.version},
                limits=advanced_run_limits(),
            )
        )
    )
    stale_tool_result = next(
        event
        for event in stale_events
        if event.type == EventType.TOOL_CALL_FAILED and event.tool_name == "deploy_service"
    )

    verifier_events = await collect_events(
        app.run(
            RunRequest(
                agent_name="verifier",
                session_id="approval-verifier",
                parent_session_id=primary_id,
                causal_budget_id=causal_budget_id,
                messages=[
                    Message.text(
                        "user",
                        json.dumps(
                            {
                                "deployment_receipt": receipt.structured,
                                "observed_state": {
                                    "version": state.version,
                                    "mutations": state.mutation_count,
                                    "release": state.deployed_release,
                                },
                                "verification_rule": (
                                    "Confirm only when receipt and observed state agree on "
                                    "version 8, one mutation, and release 2026.07.11."
                                ),
                            },
                            sort_keys=True,
                        ),
                    )
                ],
                structured_output=stable_output_spec("approval-verification", VERIFIER_SCHEMA),
                limits=advanced_run_limits(),
            )
        )
    )
    verification = validated_output(verifier_events)

    brief = {
        **snapshot,
        "approve_outcome": analyses["approve"]["outcome"],
        "deny_outcome": analyses["deny"]["outcome"],
        "risks": analyses["explain"]["risks"],
        "alternatives": analyses["explain"]["alternatives"],
        "evidence": analyses["explain"]["evidence"],
        "recommendation": analyses["explain"]["recommendation"],
        "uncertainties": analyses["explain"]["uncertainties"],
    }
    expected_brief_fields = {
        "requested_action",
        "reason",
        "affected_resources",
        "approve_outcome",
        "deny_outcome",
        "risks",
        "reversibility",
        "alternatives",
        "evidence",
        "recommendation",
        "uncertainties",
        "external_state_version",
        "expires_at",
    }
    roles = {
        primary_id: "approval-request",
        "approval-approve": "approve-future",
        "approval-deny": "deny-future",
        "approval-explain": "explainer",
        "approval-verifier": "verifier",
        stale_id: "stale-approval-probe",
    }
    sessions = await session_evidence(app, roles)
    analyses_authority_free = all(not app.get_agent(role).tools for role in role_to_future)
    assertions = {
        "analyses_are_authority_free": analyses_authority_free,
        "decision_brief_complete": expected_brief_fields <= brief.keys(),
        "external_state_revalidated": state_revalidated,
        "exactly_one_mutation": state.mutation_count == 1,
        "losing_future_discarded": analyses["deny"]["future"] == "deny",
        "stale_snapshot_rejected": (
            stale_tool_result.payload["result"]["is_error"] is True
            and stale_tool_result.payload["result"]["structured"]["actual_version"] == state.version
        ),
        "approval_recovery_preserved_receipt": (
            state.mutation_count == 1 and len(state.receipts) == 1
        ),
        "approval_recovered_after_runtime_restart": (
            store.failed_terminal_once
            and any(event.payload.get("manual_recovery") is True for event in recovered_events)
        ),
        "verifier_confirmed_actual_state": (
            verification["confirmed"] is True
            and verification["observed_version"] == state.version
            and verification["observed_mutations"] == state.mutation_count
        ),
    }
    model_requests = await count_model_completions(
        app, [session.session_id for session in sessions]
    )
    result = ScenarioResult(
        scenario="counterfactual-approval",
        mode=mode,
        status="verified" if all(assertions.values()) else "failed",
        assertions=assertions,
        sessions=sessions,
        provider_name=provider.name,
        model=model,
        metrics={
            "model_requests": model_requests,
            "mutation_count": state.mutation_count,
            "retained_future": "approve",
            "discarded_futures": ["deny"],
            "approval_receipts": len(state.receipts),
        },
        outputs={
            "approval_snapshot": snapshot,
            "decision_brief": brief,
            "deployment_receipt": receipt.structured,
            "verification": verification,
        },
    )
    result.write(root)
    result.require_verified()
    return result


def _build_app(
    *,
    provider: ModelProvider,
    model: str,
    state: DeploymentState,
    session_store: SessionStore,
) -> CayuApp:
    app = CayuApp(enable_logging=False, session_store=session_store)
    app.register_provider(provider, default=True)
    policy = AlwaysRequireApprovalToolPolicy(tools=["deploy_service"])
    app.register_agent(
        AgentSpec(
            name="approval-request",
            model=model,
            system_prompt=(
                "Request deploy_service for payments release 2026.07.11 with "
                "expected_version 7. Do not perform an alternative mutation."
            ),
        ),
        tools=[DeployServiceTool(state)],
        tool_policy=policy,
    )
    app.register_agent(
        AgentSpec(
            name="stale-approval-probe",
            model=model,
            system_prompt=(
                "Request deploy_service for payments release 2026.07.12 with "
                "expected_version 7 exactly, so runtime execution can reject stale state."
            ),
        ),
        tools=[DeployServiceTool(state)],
        tool_policy=policy,
    )
    for role in ("approve-future", "deny-future", "explainer"):
        app.register_agent(
            AgentSpec(
                name=role,
                model=model,
                system_prompt=(
                    "You are an authority-free read-only analyst. You have no mutation tools. "
                    "Use only the supplied immutable approval snapshot."
                ),
            )
        )
    app.register_agent(
        AgentSpec(
            name="verifier",
            model=model,
            system_prompt=(
                "You are an authority-free read-only verifier. The user supplies a durable "
                "deployment receipt and an observed external-state snapshot. Treat those "
                "structured values as authoritative evidence. Set confirmed=true exactly when "
                "they agree with the supplied verification rule; explain the matching fields."
            ),
        )
    )
    return app
