"""The same conversation through the Cayu runtime, with automatic recall and tool exposure.

Variant ``recall_exposure`` runs the six questions as six interactions of one
session. Each new user message triggers automatic memory recall from a small
knowledge store, and an exposure policy changes the callable tools between
turns: ``controller_sync_work_plan`` becomes ``controller_update_work_plan``
once a plan exists, and ``lookup_policy_revision`` is hidden after its budget
is spent. The agent uses ``tool_exposure_mode="stable_catalogue"`` and its tool
schemas have ``number`` fields with integer bounds.

These are the three prefix-cache breakers found on the τ³ banking agent: a
recall block removed from an earlier user message, an exposure change rewriting
the tools array, and number bounds that render two ways upstream.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from uuid import uuid4

from examples.late_system_message_caching.scenario import (
    POLICY_NOTES,
    QUESTIONS,
    SYSTEM_PROMPT,
    TURNS,
    ConversationRun,
    TurnMeasurement,
)
from pydantic import SecretStr

from cayu import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
    AgentSpec,
    AutomaticRecallContextPolicy,
    AutomaticRecallPolicy,
    AutomaticRecallSourceConfig,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    Message,
    RequestFootprintConfig,
    ResumeRequest,
    RunRequest,
    Tool,
    ToolContext,
    ToolExposureDecision,
    ToolExposureMode,
    ToolExposurePolicy,
    ToolExposurePolicyRequest,
    ToolResult,
    ToolSpec,
    WeightedReciprocalRankFusionConfig,
    default_price_book,
    estimate_model_step_cost,
    usage_metrics_from_event_payload,
)
from cayu.providers import ModelProvider

RECALL_EXPOSURE_VARIANT = "recall_exposure"
_NAMESPACE = "bank-policy"
# Turn at which a work plan exists, and from which the lookup budget is spent.
_PLAN_EXISTS_FROM_TURN = 3
_LOOKUP_BUDGET_SPENT_FROM_TURN = 5
_REVISIONS = {
    12: "Revision: the daily transfer limit in policy note 12 is now 1,500 units.",
    40: "Revision: disputes in policy note 40 now take 5 business days.",
    77: "Revision: identity check 2 applies in policy note 77 from October.",
    101: "Revision: review queue 9 now flags policy note 101.",
    150: "Revision: policy note 150 now covers tier 3 accounts.",
    199: "Revision: disputes in policy note 199 now close in 4 business days.",
}
# Durable copies store 100.0 as 100. The OpenAI adapters send it as 100.0.
_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 100.0},
        "risk": {"type": "number", "minimum": 0, "maximum": 100},
    },
    "required": ["summary"],
}


class _WorkPlanTool(Tool):
    def __init__(self, name: str, description: str) -> None:
        self.spec = ToolSpec(name=name, description=description, input_schema=_SCORE_SCHEMA)
        super().__init__()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(content="Recorded.")


class _WorkPlanExposure(ToolExposurePolicy):
    """Swap the plan tool once a plan exists and hide lookups after their budget."""

    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        plan_exists = request.metadata.get("plan_exists") is True
        lookup_budget_spent = request.metadata.get("lookup_budget_spent") is True
        names = [
            "controller_update_work_plan" if plan_exists else "controller_sync_work_plan",
            *([] if lookup_budget_spent else ["lookup_policy_revision"]),
        ]
        return ToolExposureDecision(
            profile_id=f"plan-{int(plan_exists)}-lookup-{int(not lookup_budget_spent)}",
            tool_names=tuple(names),
        )


def _recall_policy() -> AutomaticRecallContextPolicy:
    fusion = WeightedReciprocalRankFusionConfig(
        configuration_version="late-system-example-v1",
        channel_weights={KNOWLEDGE_LEXICAL_CHANNEL: 1.0, KNOWLEDGE_SEMANTIC_CHANNEL: 1.0},
        max_candidates_per_channel=5,
        fused_head_limit=5,
    )
    return AutomaticRecallContextPolicy(
        admission_policy=AutomaticRecallPolicy(
            calibration_version="late-system-example-calibration-v1",
            fusion_strategy_version=WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
            fusion_configuration_version="late-system-example-v1",
            minimum_inject_score=0.01,
            minimum_offer_score=0.005,
        ),
        fusion_config=fusion,
        sources=AutomaticRecallSourceConfig(
            knowledge_namespace=_NAMESPACE,
            include_transcript=False,
            knowledge_candidate_limit=5,
        ),
    )


def _measure(
    *, provider_name: str, model: str, turn: int, payload: dict[str, Any]
) -> TurnMeasurement:
    metrics = usage_metrics_from_event_payload(payload)
    if metrics is None:
        raise RuntimeError(f"{provider_name} reported no usage for turn {turn}.")
    estimate = estimate_model_step_cost(
        metrics=metrics,
        pricing=default_price_book(),
        effective_on=date.today(),
    )
    return TurnMeasurement(
        variant=RECALL_EXPOSURE_VARIANT,
        turn=turn,
        input_tokens=metrics.input_tokens,
        cache_read_tokens=metrics.cache.read_tokens,
        cache_write_tokens=metrics.cache.write_tokens,
        uncached_input_tokens=metrics.cache.uncached_input_tokens,
        output_tokens=metrics.output_tokens,
        cost_usd=str(estimate.total_cost) if estimate.priced else None,
    )


async def run_recall_exposure_conversation(
    provider: ModelProvider,
    *,
    provider_name: str,
    model: str,
    options: dict[str, Any],
    nonce: str | None = None,
) -> ConversationRun:
    """Run six interactions through the runtime and measure every model call."""

    nonce = nonce or uuid4().hex[:12]
    scope = KnowledgeAccessScope.for_namespace(_NAMESPACE)
    knowledge = InMemoryKnowledgeStore(access_scope=scope)
    for note, text in _REVISIONS.items():
        await knowledge.create_entry(
            KnowledgeEntry(id=f"note-{note}-revision", namespace=_NAMESPACE, text=text)
        )
    app = CayuApp(
        # Automatic recall records keyed evidence for every provider attempt.
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="late-system-example",
            fingerprint_key=SecretStr(f"late-system-example-{nonce}"),
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="banking",
            model=model,
            # The nonce keeps each conversation's prefix unique across trials.
            system_prompt=(
                f"Conversation {nonce}. {SYSTEM_PROMPT} Revisions recalled from memory "
                "override the notes. Answer directly; do not call tools."
            ),
            provider_options=options,
        ),
        tools=[
            _WorkPlanTool("controller_sync_work_plan", "Create the work plan."),
            _WorkPlanTool("controller_update_work_plan", "Update the existing work plan."),
            _WorkPlanTool("lookup_policy_revision", "Look up a policy revision."),
        ],
        tool_exposure_policy=_WorkPlanExposure(),
        tool_exposure_mode=ToolExposureMode.STABLE_CATALOGUE,
        context_policy=_recall_policy(),
    )
    session_id = f"late-system-recall-exposure-{nonce}"
    run = ConversationRun(variant=RECALL_EXPOSURE_VARIANT)
    try:
        for turn in range(1, TURNS + 1):
            question = QUESTIONS[(turn - 1) % len(QUESTIONS)]
            # As in the other variants, the notes are conversation content. Turn
            # 1's recall block sits in front of them, so removing it later would
            # invalidate the cached notes whatever order the provider caches in.
            message = Message.text(
                "user",
                f"Here are the policy notes:\n{POLICY_NOTES}\n\n{question}"
                if turn == 1
                else question,
            )
            metadata = {
                "plan_exists": turn >= _PLAN_EXISTS_FROM_TURN,
                "lookup_budget_spent": turn >= _LOOKUP_BUDGET_SPENT_FROM_TURN,
            }
            events = (
                app.run(
                    RunRequest(
                        agent_name="banking",
                        session_id=session_id,
                        messages=[message],
                        metadata=metadata,
                    )
                )
                if turn == 1
                else app.resume(
                    ResumeRequest(session_id=session_id, messages=[message], metadata=metadata)
                )
            )
            async for event in events:
                if event.type is EventType.MODEL_COMPLETED:
                    run.turns.append(
                        _measure(
                            provider_name=provider_name,
                            model=model,
                            turn=turn,
                            payload=event.payload,
                        )
                    )
                elif event.type is EventType.SESSION_FAILED:
                    raise RuntimeError(f"{provider_name} recall/exposure turn {turn} failed.")
    finally:
        await app.aclose()
    return run
