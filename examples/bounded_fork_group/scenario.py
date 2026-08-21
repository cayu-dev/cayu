from __future__ import annotations

from pathlib import Path
from typing import Any

from examples._advanced_support import (
    ScenarioResult,
    advanced_run_limits,
    collect_events,
    count_model_completions,
    session_evidence,
)

from cayu import (
    AgentSpec,
    CayuApp,
    ForkGroupBranchSpec,
    ForkGroupCheckpointSelector,
    ForkGroupDisposition,
    ForkGroupEvaluatorSpec,
    ForkGroupFailureMode,
    ForkGroupFailurePolicy,
    ForkGroupGate,
    ForkGroupGateDecision,
    ForkGroupGateRequest,
    ForkGroupReplacementPlanner,
    ForkGroupReplacementPlannerRequest,
    ForkGroupReplacementSpec,
    ForkGroupRequest,
    ForkGroupState,
    Message,
    RunRequest,
    StructuredOutputSpec,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
    session_fork_profile_relationship,
)
from cayu.providers import ModelProvider

CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "proposal": {"type": "string", "minLength": 1, "maxLength": 500},
        "quality": {"type": "integer", "minimum": 1, "maximum": 10},
        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["proposal", "quality", "risk"],
    "additionalProperties": False,
}


class CandidateContractGate(ForkGroupGate):
    """Application-owned deterministic contract checked before model evaluation."""

    def __init__(self) -> None:
        self.attempts: list[tuple[str, int]] = []

    @property
    def identity(self) -> str:
        return "examples.bounded-fork-group.candidate-contract.v1"

    async def evaluate(self, request: ForkGroupGateRequest) -> ForkGroupGateDecision:
        self.attempts.append((request.branch.branch_id, request.branch.attempt_index))
        output = request.branch.structured_output
        contract_passed = (
            request.branch.has_structured_output
            and type(output) is dict
            and type(output.get("proposal")) is str
            and type(output.get("quality")) is int
            and 1 <= output["quality"] <= 10
            and output.get("risk") in {"low", "medium", "high"}
        )
        # The application, not the evaluator, rejects one seed attempt so the
        # example deterministically exercises bounded replacement in every mode.
        passed = contract_passed and not (
            request.branch.branch_id == "extensible" and request.branch.attempt_index == 0
        )
        return ForkGroupGateDecision(
            passed=passed,
            summary=(
                "candidate contract passed"
                if passed
                else "seed attempt rejected by application policy"
            ),
        )


class BoundedReplacementPlanner(ForkGroupReplacementPlanner):
    """Application-owned mutation content; Cayu owns attempt/session identity."""

    def __init__(self, output_spec: StructuredOutputSpec) -> None:
        self.output_spec = output_spec
        self.requests: list[ForkGroupReplacementPlannerRequest] = []

    @property
    def identity(self) -> str:
        return "examples.bounded-fork-group.replacement-planner.v1"

    async def plan(
        self,
        request: ForkGroupReplacementPlannerRequest,
    ) -> ForkGroupReplacementSpec:
        self.requests.append(request)
        return ForkGroupReplacementSpec(
            messages=(
                Message.text(
                    "user",
                    "Replace the rejected extensible seed with a safer bounded proposal. "
                    f"This is attempt {request.attempt_index}.",
                ),
            ),
            structured_output=self.output_spec,
            limits=advanced_run_limits(),
        )


class EvaluatorForbiddenTool(Tool):
    """Authority present on the registered evaluator but removed by the coordinator."""

    spec = ToolSpec(
        name="evaluator_forbidden_tool",
        description="Must never be exposed to the fork-group evaluator.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx, args
        raise AssertionError("Fork-group evaluator inherited application tool authority.")


async def run_scenario(
    root: Path,
    *,
    provider: ModelProvider,
    model: str,
    mode: str,
) -> ScenarioResult:
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="candidate",
            model=model,
            system_prompt="Produce the requested bounded proposal as structured output.",
        )
    )
    app.register_agent(
        AgentSpec(
            name="evaluator",
            model=model,
            system_prompt=(
                "Judge only the supplied fork-group evidence. Select exactly one eligible "
                "candidate slot and give every eligible slot one disposition."
            ),
            workflow_tool_names=("evaluator_forbidden_tool",),
        ),
        tools=[EvaluatorForbiddenTool()],
    )
    source_id = "bounded-group-source"
    causal_budget_id = "bounded-group-budget"
    await collect_events(
        app.run(
            RunRequest(
                agent_name="candidate",
                session_id=source_id,
                causal_budget_id=causal_budget_id,
                messages=[
                    Message.text(
                        "user",
                        "Prepare shared context for two implementation proposals.",
                    )
                ],
                limits=advanced_run_limits(),
            )
        )
    )
    output_spec = StructuredOutputSpec(
        name="bounded-candidate",
        json_schema=CANDIDATE_SCHEMA,
        max_retries=2,
        repair_prompt="Return one concise proposal that satisfies the candidate schema.",
    )
    gate = CandidateContractGate()
    gate_selection = app.register_fork_group_gate("candidate-contract-v1", gate)
    planner = BoundedReplacementPlanner(output_spec)
    planner_selection = app.register_fork_group_replacement_planner(
        "replacement-planner-v1",
        planner,
    )
    request = ForkGroupRequest(
        group_id="bounded-group",
        source_session_id=source_id,
        source_checkpoint=ForkGroupCheckpointSelector(),
        causal_budget_id=causal_budget_id,
        max_parallelism=1,
        branches=(
            ForkGroupBranchSpec(
                branch_id="focused",
                session_id="bounded-group-focused",
                messages=(
                    Message.text(
                        "user",
                        "Propose the smallest focused implementation with quality and risk.",
                    ),
                ),
                structured_output=output_spec,
                limits=advanced_run_limits(),
            ),
            ForkGroupBranchSpec(
                branch_id="extensible",
                session_id="bounded-group-extensible",
                messages=(
                    Message.text(
                        "user",
                        "Propose a broader extensible implementation with quality and risk.",
                    ),
                ),
                structured_output=output_spec,
                limits=advanced_run_limits(),
            ),
        ),
        gates=(gate_selection,),
        failure_policy=ForkGroupFailurePolicy(
            mode=ForkGroupFailureMode.EVALUATE_VIABLE,
            minimum_viable_branches=2,
            max_replacement_attempts=1,
            replacement_parallelism=1,
            replacement_planner=planner_selection,
        ),
        evaluator=ForkGroupEvaluatorSpec(
            session_id="bounded-group-evaluator",
            agent_name="evaluator",
            limits=advanced_run_limits(),
        ),
    )
    group = await app.run_fork_group(request)
    model_requests_before_replay = await count_model_completions(
        app,
        [
            source_id,
            *(branch.session_id for branch in group.branches),
            request.evaluator.session_id,
        ],
    )
    replay = await app.run_fork_group(request)
    model_requests_after_replay = await count_model_completions(
        app,
        [
            source_id,
            *(branch.session_id for branch in group.branches),
            request.evaluator.session_id,
        ],
    )
    selected = [
        item.branch_id
        for item in group.dispositions
        if item.disposition is ForkGroupDisposition.SELECTED
    ]
    focused = next(
        branch
        for branch in group.branches
        if branch.branch_id == "focused" and branch.attempt_index == 0
    )
    rejected_seed = next(
        branch
        for branch in group.branches
        if branch.branch_id == "extensible" and branch.attempt_index == 0
    )
    replacement = next(
        branch
        for branch in group.branches
        if branch.branch_id == "extensible" and branch.attempt_index == 1
    )
    sessions = await session_evidence(
        app,
        {
            source_id: "source",
            focused.session_id: "focused",
            rejected_seed.session_id: "extensible-seed",
            replacement.session_id: "extensible-replacement",
            "bounded-group-evaluator": "evaluator",
        },
    )
    candidate_sessions = [
        session
        for session in sessions
        if session.role in {"focused", "extensible-seed", "extensible-replacement"}
    ]
    source_session = await app.session_store.load(source_id)
    candidate_records = [
        await app.session_store.load(session.session_id) for session in candidate_sessions
    ]
    evaluator_session = await app.session_store.load(request.evaluator.session_id)
    if (
        source_session is None
        or evaluator_session is None
        or any(session is None for session in candidate_records)
    ):
        raise RuntimeError("Fork-group evidence sessions disappeared after completion.")
    relationships = [
        session_fork_profile_relationship(session)
        for session in candidate_records
        if session is not None
    ]
    manifest_agents = {agent.name: agent for agent in app.describe().agents}
    evaluator_manifest = manifest_agents[evaluator_session.agent_name]
    total_tokens = sum(session.usage["total_tokens"] for session in sessions)
    eligible = [branch for branch in group.branches if branch.eligible]
    assertions = {
        "application_gates_control_eligibility": (
            gate.attempts == [("focused", 0), ("extensible", 0), ("extensible", 1)]
            and focused.eligible
            and focused.gate_results[0].passed
            and not rejected_seed.eligible
            and not rejected_seed.gate_results[0].passed
            and replacement.eligible
            and replacement.gate_results[0].passed
        ),
        "bounded_group_completed": group.state is ForkGroupState.COMPLETED,
        "dispositions_cover_eligible_and_select_one": (
            len(selected) == 1
            and {(item.branch_id, item.attempt_id) for item in group.dispositions}
            == {(branch.branch_id, branch.attempt_id) for branch in eligible}
        ),
        "replacement_lineage_preserves_the_seed": (
            len(group.branches) == 3
            and rejected_seed.status.value == "completed"
            and rejected_seed.superseded_by_attempt_id == replacement.attempt_id
            and replacement.replaced_attempt_id == rejected_seed.attempt_id
            and replacement.session_id != rejected_seed.session_id
            and len(planner.requests) == 1
            and planner.requests[0].attempt_id == replacement.attempt_id
        ),
        "surviving_sibling_was_not_rerun": (
            focused.attempt_index == 0
            and focused.replaced_attempt_id is None
            and focused.superseded_by_attempt_id is None
        ),
        "economic_evidence_is_complete": (
            total_tokens > 0
            and all(session.model_steps == 1 for session in sessions)
            and all(session.usage["total_tokens"] > 0 for session in sessions)
        ),
        "evaluator_is_structurally_isolated": (
            evaluator_session.parent_session_id is None
            and evaluator_session.agent_name != request.evaluator.agent_name
            and not evaluator_manifest.tools
            and not evaluator_manifest.workflow_tool_names
            and not evaluator_manifest.runtime_hooks
            and not evaluator_manifest.loop_policies
        ),
        "exact_checkpoint_and_profile_are_frozen": (
            source_session.run_epoch == group.source.run_epoch
            and len(group.source.checkpoint_sha256) == 64
            and len(group.source.transcript_sha256) == 64
            and len(group.source.execution_profile_fingerprint) == 64
            and len(relationships) == len(candidate_sessions)
            and all(
                session is not None
                and session.metadata.get("cayu:fork_group_source_snapshot")
                == group.source.model_dump(mode="json")
                for session in candidate_records
            )
            and all(
                relationship is not None
                and relationship.source_session_id == source_id
                and relationship.source_run_epoch == group.source.run_epoch
                and relationship.source_profile.fingerprint
                == group.source.execution_profile_fingerprint
                for relationship in relationships
            )
        ),
        "replay_did_not_rerun_models": (
            replay.replayed
            and replay.dispositions == group.dispositions
            and model_requests_after_replay == model_requests_before_replay
        ),
        "siblings_share_causal_budget": all(
            session.parent_session_id == source_id and session.causal_budget_id == causal_budget_id
            for session in candidate_sessions
        ),
    }
    result = ScenarioResult(
        scenario="bounded-fork-group",
        mode=mode,
        status="verified" if all(assertions.values()) else "failed",
        assertions=assertions,
        sessions=sessions,
        provider_name=provider.name,
        model=model,
        metrics={
            "branch_count": len(group.branches),
            "model_requests": model_requests_before_replay,
            "selected_branch": selected[0] if len(selected) == 1 else None,
            "total_tokens": total_tokens,
        },
        outputs={
            "candidates": {
                branch.attempt_id: {
                    "branch_id": branch.branch_id,
                    "attempt_index": branch.attempt_index,
                    "eligible": branch.eligible,
                    "structured_output": branch.structured_output,
                }
                for branch in group.branches
            },
            "dispositions": [item.model_dump(mode="json") for item in group.dispositions],
        },
    )
    result.write(root)
    result.require_verified()
    return result
