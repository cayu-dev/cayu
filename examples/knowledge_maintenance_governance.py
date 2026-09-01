"""Govern evaluated maintenance without a provider, scheduler, or worker."""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from cayu import (
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeEntry,
    KnowledgeGovernanceConfig,
    KnowledgeGovernanceMode,
    KnowledgeMaintenanceCandidateSignal,
    KnowledgeMaintenanceEvaluationVerdict,
    KnowledgeMaintenanceEvaluatorDecision,
    KnowledgeMaintenanceEvaluatorOutput,
    KnowledgeMaintenanceEvidenceMapping,
    KnowledgeMaintenanceGovernanceDecision,
    KnowledgeMaintenanceGovernanceDisposition,
    KnowledgeMaintenanceGovernor,
    KnowledgeMaintenanceInferenceUsage,
    KnowledgeMaintenancePlanDraft,
    KnowledgeMaintenancePlanEndpoint,
    KnowledgeMaintenancePlanEndpointKind,
    KnowledgeMaintenancePlannerOutput,
    KnowledgeMaintenancePlanningConfig,
    KnowledgeMaintenancePlanningWorkflow,
    KnowledgeMaintenanceProposalPublisher,
    KnowledgeMaintenanceProposalPublisherConfig,
    KnowledgeMaintenanceRelationDraft,
    KnowledgeMaintenanceReplacementDraft,
    KnowledgeMaintenanceRouter,
    KnowledgeMaintenanceRouterConfig,
    KnowledgeMaintenanceRoutingRequest,
    KnowledgeMaintenanceSignalKind,
    KnowledgeRelationKind,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    KnowledgeVisibility,
    SQLiteKnowledgeStore,
)

_NOW = datetime(2026, 9, 1, 9, tzinfo=UTC)
_SOURCE_SCOPE = KnowledgeAccessScope.for_namespace(
    "example:maintenance",
    allowed_visibilities=[KnowledgeVisibility.PROJECT],
    allowed_statuses=[KnowledgeStatus.ACTIVE],
)
_MAINTENANCE_SCOPE = KnowledgeAccessScope.for_namespace(
    "example:maintenance",
    allowed_visibilities=[KnowledgeVisibility.PROJECT],
    allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
)


class DeterministicPlanner:
    async def propose_maintenance(self, request):
        sources = tuple(candidate.reference for candidate in request.snapshot.candidates)
        mappings = tuple(
            KnowledgeMaintenanceEvidenceMapping(
                id=f"claim:{index}",
                claim=f"The replacement preserves source {index}.",
                source_references=(source,),
            )
            for index, source in enumerate(sources)
        )
        relations = tuple(
            KnowledgeMaintenanceRelationDraft(
                id=f"relation:{index}",
                subject=KnowledgeMaintenancePlanEndpoint(
                    kind=KnowledgeMaintenancePlanEndpointKind.REPLACEMENT
                ),
                object=KnowledgeMaintenancePlanEndpoint(
                    kind=KnowledgeMaintenancePlanEndpointKind.SOURCE,
                    reference=source,
                ),
                kind=KnowledgeRelationKind.SUPERSEDES,
                evidence_mapping_ids=(mappings[index].id,),
            )
            for index, source in enumerate(sources)
        )
        return KnowledgeMaintenancePlannerOutput(
            plan=KnowledgeMaintenancePlanDraft(
                id="example-maintenance-plan",
                routing_request_fingerprint=request.snapshot.routing_request_fingerprint,
                routing_result_fingerprint=request.snapshot.routing_result_fingerprint,
                configuration_fingerprint=request.configuration_fingerprint,
                policy_id=request.snapshot.policy_id,
                source_references=sources,
                replacement=KnowledgeMaintenanceReplacementDraft(
                    text="The two exact source facts are consolidated without losing evidence.",
                    title="Consolidated example fact",
                    kind="fact",
                ),
                relations=relations,
                evidence_mappings=mappings,
                rationale="The deterministic example preserves both source claims.",
                evidence_summary="Every claim maps to one exact source revision.",
            ),
            usage=KnowledgeMaintenanceInferenceUsage(),
        )


class DeterministicEvaluator:
    async def evaluate_maintenance_plan(self, request):
        return KnowledgeMaintenanceEvaluatorOutput(
            decision=KnowledgeMaintenanceEvaluatorDecision(
                plan_fingerprint=request.plan.fingerprint,
                routing_result_fingerprint=(
                    request.planner_input.snapshot.routing_result_fingerprint
                ),
                configuration_fingerprint=request.planner_input.configuration_fingerprint,
                verdict=KnowledgeMaintenanceEvaluationVerdict.ACCEPTED,
            ),
            usage=KnowledgeMaintenanceInferenceUsage(),
        )


class ApplicationMaintenancePolicy:
    def __init__(self, disposition: KnowledgeMaintenanceGovernanceDisposition) -> None:
        self.disposition = disposition

    async def decide_maintenance(self, request):
        return KnowledgeMaintenanceGovernanceDecision(
            request_sha256=request.fingerprint,
            disposition=self.disposition,
            policy_identity="example.application-maintenance-policy",
            policy_version="1",
            code=f"example_{self.disposition.value}",
        )


async def publish_evaluated_proposal(store, prefix: str):
    source_ids = (f"{prefix}-source-a", f"{prefix}-source-b")
    for source_id in source_ids:
        await store.create_entry(
            KnowledgeEntry(
                id=source_id,
                text=f"Exact durable source fact {source_id}.",
                namespace="example:maintenance",
                visibility=KnowledgeVisibility.PROJECT,
                status=KnowledgeStatus.ACTIVE,
                created_by_type=KnowledgeActorType.APP,
                created_by="example",
                source_type="example",
                source_id=source_id,
                source_hash=f"sha256:{source_id}",
                created_at=_NOW,
                updated_at=_NOW,
            ),
            access_scope=_MAINTENANCE_SCOPE,
        )
    request = KnowledgeMaintenanceRoutingRequest(
        id=f"{prefix}-routing",
        policy_id="example-consolidation-v1",
        namespace="example:maintenance",
        access_scope=_SOURCE_SCOPE,
        signals=tuple(
            KnowledgeMaintenanceCandidateSignal(
                id=f"signal:{source_id}",
                kind=KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                references=(KnowledgeRevisionRef(entry_id=source_id, revision=1),),
                producer_id="example",
                producer_version="1",
                reason_code="deterministic_example",
                observed_at=_NOW,
            )
            for source_id in source_ids
        ),
        created_at=_NOW,
    )
    routing = await KnowledgeMaintenanceRouter(
        store,
        config=KnowledgeMaintenanceRouterConfig(max_candidates=2),
    ).route(request)
    planning = await KnowledgeMaintenancePlanningWorkflow(
        store,
        planner=DeterministicPlanner(),
        evaluator=DeterministicEvaluator(),
        config=KnowledgeMaintenancePlanningConfig(
            planner_id="example.planner",
            planner_version="1",
            evaluator_id="example.evaluator",
            evaluator_version="1",
            planner_model_ids=("example.no-provider",),
            evaluator_model_ids=("example.no-provider",),
        ),
        clock=lambda: _NOW,
    ).plan(request, routing)
    return await KnowledgeMaintenanceProposalPublisher(
        store,
        access_scope=_MAINTENANCE_SCOPE,
        config=KnowledgeMaintenanceProposalPublisherConfig(
            publisher_id="example.publisher",
            publisher_version="1",
        ),
    ).publish(request, routing, planning)


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cayu-maintenance-governance-") as directory:
        store = SQLiteKnowledgeStore(Path(directory) / "knowledge.db")
        try:
            reviewed = await publish_evaluated_proposal(store, "reviewed")
            reviewed_receipt = await KnowledgeMaintenanceGovernor(
                store,
                config=KnowledgeGovernanceConfig(mode=KnowledgeGovernanceMode.REVIEWED),
            ).govern(
                operation_id="reviewed-governance",
                proposal_id=reviewed.proposal.id,
                access_scope=_MAINTENANCE_SCOPE,
            )

            automatic = await publish_evaluated_proposal(store, "automatic")
            automatic_receipt = await KnowledgeMaintenanceGovernor(
                store,
                config=KnowledgeGovernanceConfig(
                    mode=KnowledgeGovernanceMode.POLICY_AUTOMATIC,
                    policy_identity="example.application-maintenance-policy",
                    policy_version="1",
                ),
                policy=ApplicationMaintenancePolicy(
                    KnowledgeMaintenanceGovernanceDisposition.APPROVE
                ),
            ).govern(
                operation_id="automatic-governance",
                proposal_id=automatic.proposal.id,
                access_scope=_MAINTENANCE_SCOPE,
            )

            autonomous = await publish_evaluated_proposal(store, "autonomous")
            autonomous_receipt = await KnowledgeMaintenanceGovernor(
                store,
                config=KnowledgeGovernanceConfig(
                    mode=KnowledgeGovernanceMode.AUTONOMOUS,
                    policy_identity="example.application-maintenance-policy",
                    policy_version="1",
                ),
                policy=ApplicationMaintenancePolicy(
                    KnowledgeMaintenanceGovernanceDisposition.REJECT
                ),
            ).govern(
                operation_id="autonomous-governance",
                proposal_id=autonomous.proposal.id,
                access_scope=_MAINTENANCE_SCOPE,
            )

            print(
                json.dumps(
                    {
                        "reviewed": reviewed_receipt.authority.decision.disposition.value,
                        "policy_automatic": (
                            automatic_receipt.authority.decision.disposition.value
                        ),
                        "autonomous": (autonomous_receipt.authority.decision.disposition.value),
                        "provider_calls": 0,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        finally:
            await store.close()


if __name__ == "__main__":
    asyncio.run(main())
