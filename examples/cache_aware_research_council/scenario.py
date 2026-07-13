from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples._advanced_support import (
    ScenarioResult,
    advanced_run_limits,
    collect_events,
    count_model_completions,
    first_model_input_tokens,
    fork_session,
    paired_cost_evidence,
    session_evidence,
    stable_output_spec,
    validated_output,
)

from cayu import (
    AgentSpec,
    CayuApp,
    CheckpointCompactionContextPolicy,
    EventType,
    Message,
    ModelCatalog,
    ResumeRequest,
    RunRequest,
    SessionStore,
    TranscriptDigestCompactor,
    estimate_session_cost,
)
from cayu.providers import ModelProvider

_STRUCTURED_OUTPUT_TEXT_MAX_CHARS = 1024

# The live provider needs room for ordinary report prose without avoidable
# structured-output repair retries. Prompts and array bounds still keep the
# example concise; one shared cap prevents the schemas from drifting apart.
REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string", "maxLength": _STRUCTURED_OUTPUT_TEXT_MAX_CHARS},
        "claim": {"type": "string", "maxLength": _STRUCTURED_OUTPUT_TEXT_MAX_CHARS},
        "evidence": {
            "type": "array",
            "items": {
                "type": "string",
                "maxLength": _STRUCTURED_OUTPUT_TEXT_MAX_CHARS,
            },
            "minItems": 2,
            "maxItems": 3,
        },
        "uncertainties": {
            "type": "array",
            "items": {
                "type": "string",
                "maxLength": _STRUCTURED_OUTPUT_TEXT_MAX_CHARS,
            },
            "minItems": 1,
            "maxItems": 2,
        },
    },
    "required": ["strategy", "claim", "evidence", "uncertainties"],
    "additionalProperties": False,
}
TOPIC = (Path(__file__).with_name("fixtures") / "topic.md").read_text(encoding="utf-8").strip()
SOURCE_CONTEXT = (
    TOPIC
    + "\n\nEvidence notebook:\n"
    + "\n".join(
        f"Observation {index:03d}: durable lineage, recovery receipts, and cache accounting "
        "must be compared under the same research prompt and provider configuration."
        for index in range(1, 101)
    )
)
EVALUATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "maxLength": _STRUCTURED_OUTPUT_TEXT_MAX_CHARS},
        "weakness": {"type": "string", "maxLength": _STRUCTURED_OUTPUT_TEXT_MAX_CHARS},
        "repair_instruction": {
            "type": "string",
            "maxLength": _STRUCTURED_OUTPUT_TEXT_MAX_CHARS,
        },
    },
    "required": ["winner", "weakness", "repair_instruction"],
    "additionalProperties": False,
}
REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fixed_weakness": {
            "type": "string",
            "maxLength": _STRUCTURED_OUTPUT_TEXT_MAX_CHARS,
        },
        "added_evidence": {
            "type": "array",
            "items": {
                "type": "string",
                "maxLength": _STRUCTURED_OUTPUT_TEXT_MAX_CHARS,
            },
            "minItems": 1,
            "maxItems": 2,
        },
        "remaining_uncertainty": {
            "type": "string",
            "maxLength": _STRUCTURED_OUTPUT_TEXT_MAX_CHARS,
        },
    },
    "required": ["fixed_weakness", "added_evidence", "remaining_uncertainty"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class CacheWindowDecision:
    observed_at_s: float
    cache_expires_at_s: float
    safety_margin_s: float

    @property
    def compact_before_next_turn(self) -> bool:
        return self.cache_expires_at_s - self.observed_at_s <= self.safety_margin_s


async def run_scenario(
    root: Path,
    *,
    provider: ModelProvider,
    model: str,
    mode: str,
    model_catalog: ModelCatalog | None = None,
) -> ScenarioResult:
    prompts = {
        "source": SOURCE_CONTEXT,
        "primary-sources": "Prioritize primary technical evidence and quantify claims.",
        "contrarian": "Challenge the premise and look for counterexamples.",
        "practitioner": "Focus on operational constraints and failure recovery.",
        "evaluator": "Compare the reports and identify the seeded missing baseline.",
        "repair": "Repair only the evaluator's identified weakness with new evidence.",
    }
    app = _build_app(provider=provider, model=model, prompts=prompts)

    causal_budget_id = "advanced-research-council-budget"
    source_id = "research-source"
    source_events = await collect_events(
        app.run(
            RunRequest(
                agent_name="source",
                session_id=source_id,
                causal_budget_id=causal_budget_id,
                messages=[Message.text("user", prompts["source"])],
                limits=advanced_run_limits(),
            )
        )
    )
    if not any(event.type == EventType.SESSION_COMPLETED for event in source_events):
        raise RuntimeError("Research source session did not complete.")

    cache_decision = CacheWindowDecision(
        observed_at_s=970.0,
        cache_expires_at_s=1_000.0,
        safety_margin_s=45.0,
    )
    if not cache_decision.compact_before_next_turn:
        raise RuntimeError("Injected cache observation did not request compaction.")
    compaction_events = await collect_events(
        app.resume(
            ResumeRequest(
                session_id=source_id,
                messages=[
                    Message.text(
                        "user",
                        "The cache deadline is inside the safety margin. Persist a compacted "
                        "checkpoint before the research branches are created.",
                    )
                ],
                limits=advanced_run_limits(),
            )
        )
    )
    compaction_completed = [
        event for event in compaction_events if event.type == EventType.CONTEXT_COMPACTION_COMPLETED
    ]
    checkpoint = await app.session_store.load_checkpoint(source_id)
    compaction_checkpoint = (checkpoint or {}).get("context_compaction")
    if len(compaction_completed) != 1 or not isinstance(compaction_checkpoint, dict):
        raise RuntimeError("Research source did not persist one compaction checkpoint.")
    checkpoint_evidence = {
        "version": compaction_checkpoint.get("version"),
        "compacted_transcript_cursor": compaction_checkpoint.get("compacted_transcript_cursor"),
        "metadata": compaction_checkpoint.get("metadata"),
        "summary_chars": len(str(compaction_checkpoint.get("summary", ""))),
    }

    # Run the paired uncompacted branches from the exact same post-decision
    # source checkpoint used by the candidate below. These agents ignore the
    # stored compacted view, while the rebuilt candidate app activates it.
    baseline_reports: dict[str, dict[str, Any]] = {}
    baseline_ids: dict[str, str] = {}
    for role in ("primary-sources", "contrarian", "practitioner"):
        branch_id = f"research-baseline-{role}"
        baseline_ids[role] = branch_id
        await fork_session(
            app,
            source_session_id=source_id,
            session_id=branch_id,
            agent_name=role,
        )
        events = await collect_events(
            app.resume(
                ResumeRequest(
                    session_id=branch_id,
                    messages=[Message.text("user", prompts[role])],
                    structured_output=stable_output_spec(f"baseline-{role}-report", REPORT_SCHEMA),
                    limits=advanced_run_limits(),
                )
            )
        )
        baseline_reports[role] = validated_output(events)

    app = _build_app(
        provider=provider,
        model=model,
        prompts=prompts,
        session_store=app.session_store,
        compact_branch_context=True,
    )

    reports: dict[str, dict[str, Any]] = {}
    branch_ids: dict[str, str] = {}
    for role in ("primary-sources", "contrarian", "practitioner"):
        branch_id = f"research-{role}"
        branch_ids[role] = branch_id
        await fork_session(
            app,
            source_session_id=source_id,
            session_id=branch_id,
            agent_name=role,
        )
        events = await collect_events(
            app.resume(
                ResumeRequest(
                    session_id=branch_id,
                    messages=[Message.text("user", prompts[role])],
                    structured_output=stable_output_spec(f"{role}-report", REPORT_SCHEMA),
                    limits=advanced_run_limits(),
                )
            )
        )
        reports[role] = validated_output(events)

    evaluator_id = "research-evaluator"
    await fork_session(
        app,
        source_session_id=source_id,
        session_id=evaluator_id,
        agent_name="evaluator",
    )
    evaluator_events = await collect_events(
        app.resume(
            ResumeRequest(
                session_id=evaluator_id,
                messages=[Message.text("user", f"Evaluate these reports: {reports!r}")],
                structured_output=stable_output_spec("research-evaluation", EVALUATION_SCHEMA),
                limits=advanced_run_limits(),
            )
        )
    )
    evaluation = validated_output(evaluator_events)

    repair_id = "research-repair"
    await fork_session(
        app,
        source_session_id=evaluator_id,
        session_id=repair_id,
        agent_name="repair",
    )
    repair_events = await collect_events(
        app.resume(
            ResumeRequest(
                session_id=repair_id,
                messages=[Message.text("user", evaluation["repair_instruction"])],
                structured_output=stable_output_spec("research-repair", REPAIR_SCHEMA),
                limits=advanced_run_limits(),
            )
        )
    )
    repair = validated_output(repair_events)

    sessions = await session_evidence(
        app,
        {
            source_id: "source",
            **{session_id: f"baseline-{role}" for role, session_id in baseline_ids.items()},
            **{session_id: role for role, session_id in branch_ids.items()},
            evaluator_id: "evaluator",
            repair_id: "repair",
        },
    )
    strategy_names = {report["strategy"] for report in reports.values()}
    critique_terms = _semantic_terms(evaluation["weakness"])
    repair_terms = _semantic_terms(repair["fixed_weakness"])
    sessions_by_role = {session.role: session for session in sessions}
    research_roles = ("primary-sources", "contrarian", "practitioner")
    baseline_first_attempt_input_tokens = sum(
        [await first_model_input_tokens(app, baseline_ids[role]) for role in research_roles]
    )
    candidate_first_attempt_input_tokens = sum(
        [await first_model_input_tokens(app, branch_ids[role]) for role in research_roles]
    )
    baseline_total_input_tokens = sum(
        sessions_by_role[f"baseline-{role}"].usage["input_tokens"] for role in research_roles
    )
    candidate_total_input_tokens = sum(
        sessions_by_role[role].usage["input_tokens"] for role in research_roles
    )
    baseline_total_output_tokens = sum(
        sessions_by_role[f"baseline-{role}"].usage["output_tokens"] for role in research_roles
    )
    candidate_total_output_tokens = sum(
        sessions_by_role[role].usage["output_tokens"] for role in research_roles
    )
    baseline_model_steps = sum(
        sessions_by_role[f"baseline-{role}"].model_steps for role in research_roles
    )
    candidate_model_steps = sum(sessions_by_role[role].model_steps for role in research_roles)
    paired_branch_cost = await _paired_cost_evidence(
        app,
        baseline_session_ids=[baseline_ids[role] for role in research_roles],
        candidate_session_ids=[branch_ids[role] for role in research_roles],
        model_catalog=model_catalog,
    )
    assertions = {
        "causal_budget_shared": {item.causal_budget_id for item in sessions} == {causal_budget_id},
        "compaction_checkpoint_persisted": (
            cache_decision.compact_before_next_turn
            and compaction_checkpoint.get("compacted_transcript_cursor", 0) > 0
        ),
        "paired_baseline_recorded": (
            set(baseline_reports) == set(reports)
            and baseline_first_attempt_input_tokens > 0
            and candidate_first_attempt_input_tokens > 0
            and baseline_total_input_tokens > 0
            and candidate_total_input_tokens > 0
        ),
        "compacted_candidate_first_attempt_context_is_smaller": (
            candidate_first_attempt_input_tokens < baseline_first_attempt_input_tokens
        ),
        "compacted_candidate_used_fewer_input_tokens": (
            candidate_total_input_tokens < baseline_total_input_tokens
        ),
        "evaluator_found_material_weakness": len(critique_terms) >= 4,
        "repair_addressed_critique": (
            len(critique_terms & repair_terms) >= 2 and bool(repair["added_evidence"])
        ),
        "fork_lineage_persisted": all(
            session.parent_session_id is not None
            for session in sessions
            if session.role != "source"
        ),
        "strategies_are_distinct": len(strategy_names) == 3,
    }
    model_requests = await count_model_completions(
        app, [session.session_id for session in sessions]
    )
    result = ScenarioResult(
        scenario="cache-aware-research-council",
        mode=mode,
        status="verified" if all(assertions.values()) else "failed",
        assertions=assertions,
        sessions=sessions,
        provider_name=provider.name,
        model=model,
        metrics={
            "model_requests": model_requests,
            "cache_window": {
                "observed_at_s": cache_decision.observed_at_s,
                "cache_expires_at_s": cache_decision.cache_expires_at_s,
                "safety_margin_s": cache_decision.safety_margin_s,
                "decision": "compact-before-next-turn",
                "source": "injected-cache-observation",
                "checkpoint": checkpoint_evidence,
            },
            "paired_token_observation": {
                "baseline": "uncompacted-forks",
                "candidate": "checkpoint-compacted-forks",
                "baseline_input_tokens": baseline_total_input_tokens,
                "candidate_input_tokens": candidate_total_input_tokens,
                "input_token_delta": baseline_total_input_tokens - candidate_total_input_tokens,
                "baseline_output_tokens": baseline_total_output_tokens,
                "candidate_output_tokens": candidate_total_output_tokens,
                "measurement": "total-provider-input-with-first-attempt-control",
                "baseline_first_attempt_input_tokens": baseline_first_attempt_input_tokens,
                "candidate_first_attempt_input_tokens": candidate_first_attempt_input_tokens,
                "first_attempt_input_token_delta": baseline_first_attempt_input_tokens
                - candidate_first_attempt_input_tokens,
                "baseline_model_steps": baseline_model_steps,
                "candidate_model_steps": candidate_model_steps,
                "provider_reported": True,
                "same_source_and_branch_prompts": True,
            },
            "paired_cost_evidence": paired_branch_cost,
        },
        outputs={
            "baseline_reports": baseline_reports,
            "reports": reports,
            "evaluation": evaluation,
            "repair": repair,
        },
    )
    result.write(root)
    result.require_verified()
    return result


async def _paired_cost_evidence(
    app: CayuApp,
    *,
    baseline_session_ids: list[str],
    candidate_session_ids: list[str],
    model_catalog: ModelCatalog | None,
) -> dict[str, Any]:
    if model_catalog is None:
        return paired_cost_evidence(
            candidate=(),
            baseline=(),
            catalog=None,
            baseline_cost_field="paired_baseline_cost",
        )

    async def priced_sessions(session_ids: list[str]):
        return [
            estimate_session_cost(
                session_id=session_id,
                events=await app.session_store.load_events(session_id),
                catalog=model_catalog,
            )
            for session_id in session_ids
        ]

    baseline = await priced_sessions(baseline_session_ids)
    candidate = await priced_sessions(candidate_session_ids)
    return paired_cost_evidence(
        candidate=candidate,
        baseline=baseline,
        catalog=model_catalog,
        baseline_cost_field="paired_baseline_cost",
    )


def _build_app(
    *,
    provider: ModelProvider,
    model: str,
    prompts: dict[str, str],
    session_store: SessionStore | None = None,
    compact_branch_context: bool = False,
) -> CayuApp:
    app = CayuApp(enable_logging=False, session_store=session_store)
    app.register_provider(provider, default=True)
    for name, prompt in prompts.items():
        context_policy = None
        if name == "source" or (
            compact_branch_context and name in {"primary-sources", "contrarian", "practitioner"}
        ):
            context_policy = CheckpointCompactionContextPolicy(
                compactor=TranscriptDigestCompactor(max_summary_chars=2_000),
                max_user_turns=1,
                compact_after_messages=2,
            )
        app.register_agent(
            AgentSpec(
                name=name,
                model=model,
                system_prompt=(
                    prompt + " Be concise, use one sentence per string field, and stay within the "
                    "structured field limits."
                ),
            ),
            context_policy=context_policy,
        )
    return app


def _semantic_terms(value: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", value.lower()) if len(term) >= 5}
