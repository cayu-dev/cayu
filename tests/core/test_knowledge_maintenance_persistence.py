from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import cayu
from cayu.knowledge_maintenance import (
    KnowledgeMaintenanceCandidateSignal,
    KnowledgeMaintenanceRouter,
    KnowledgeMaintenanceRouterConfig,
    KnowledgeMaintenanceRoutingRequest,
    KnowledgeMaintenanceSignalKind,
)
from cayu.knowledge_maintenance_persistence import (
    KnowledgeMaintenanceProposalPublicationConflict,
    KnowledgeMaintenanceProposalPublicationOutcome,
    KnowledgeMaintenanceProposalPublisher,
    KnowledgeMaintenanceProposalPublisherConfig,
)
from cayu.knowledge_maintenance_planning import (
    KnowledgeMaintenanceEvaluationVerdict,
    KnowledgeMaintenanceEvaluatorDecision,
    KnowledgeMaintenanceEvaluatorOutput,
    KnowledgeMaintenanceEvidenceMapping,
    KnowledgeMaintenanceInferenceUsage,
    KnowledgeMaintenancePlanDraft,
    KnowledgeMaintenancePlanEndpoint,
    KnowledgeMaintenancePlanEndpointKind,
    KnowledgeMaintenancePlannerOutput,
    KnowledgeMaintenancePlanningConfig,
    KnowledgeMaintenancePlanningOutcome,
    KnowledgeMaintenancePlanningWorkflow,
    KnowledgeMaintenanceRelationDraft,
    KnowledgeMaintenanceReplacementDraft,
)
from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeEntry,
    KnowledgeMaintenanceConflict,
    KnowledgeMaintenanceDecision,
    KnowledgeMaintenanceDecisionKind,
    KnowledgeMaintenanceStale,
    KnowledgeRelationKind,
    KnowledgeRelationQuery,
    KnowledgeReviewWorkflow,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    KnowledgeVisibility,
    SQLiteKnowledgeStore,
)

_NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
_SOURCE_TIME = _NOW - timedelta(days=30)
_PRIVILEGED = KnowledgeAccessScope.privileged()
_ROUTING_SCOPE = KnowledgeAccessScope.for_namespace(
    "project:cayu",
    required_labels={"project": "cayu"},
    allowed_visibilities=[KnowledgeVisibility.PROJECT],
    allowed_statuses=[KnowledgeStatus.ACTIVE],
    include_expired=True,
)
_REVIEW_SCOPE = KnowledgeAccessScope.for_namespace(
    "project:cayu",
    required_labels={"project": "cayu"},
    allowed_visibilities=[KnowledgeVisibility.PROJECT],
    allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
    include_expired=True,
)
_UNRELATED_SCOPE = KnowledgeAccessScope.for_namespace(
    "project:unrelated",
    allowed_visibilities=[KnowledgeVisibility.PROJECT],
    allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
    include_expired=True,
)


class _Planner:
    async def propose_maintenance(self, request):
        sources = tuple(candidate.reference for candidate in request.snapshot.candidates)
        mappings = tuple(
            KnowledgeMaintenanceEvidenceMapping(
                id=f"claim:{index}",
                claim=f"Replacement claim supported by source {index}.",
                source_references=(reference,),
            )
            for index, reference in enumerate(sources)
        )
        relations = tuple(
            KnowledgeMaintenanceRelationDraft(
                id=f"relation:{index}",
                subject=KnowledgeMaintenancePlanEndpoint(
                    kind=KnowledgeMaintenancePlanEndpointKind.REPLACEMENT
                ),
                object=KnowledgeMaintenancePlanEndpoint(
                    kind=KnowledgeMaintenancePlanEndpointKind.SOURCE,
                    reference=reference,
                ),
                kind=KnowledgeRelationKind.SUPERSEDES,
                evidence_mapping_ids=(mappings[index].id,),
            )
            for index, reference in enumerate(sources)
        )
        return KnowledgeMaintenancePlannerOutput(
            plan=KnowledgeMaintenancePlanDraft(
                id="accepted-maintenance-plan",
                routing_request_fingerprint=request.snapshot.routing_request_fingerprint,
                routing_result_fingerprint=request.snapshot.routing_result_fingerprint,
                configuration_fingerprint=request.configuration_fingerprint,
                policy_id=request.snapshot.policy_id,
                source_references=sources,
                replacement=KnowledgeMaintenanceReplacementDraft(
                    text="The exact routed facts are retained in one reviewed replacement.",
                    title="Reviewed consolidated fact",
                    kind="fact",
                    aspects=("maintenance",),
                ),
                relations=relations,
                evidence_mappings=mappings,
                rationale="The replacement retains every reviewed source claim.",
                evidence_summary="Every claim and relation maps to an exact source revision.",
            ),
            usage=KnowledgeMaintenanceInferenceUsage(),
        )


class _Evaluator:
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


def _entry(entry_id: str, *, expires_at: datetime | None = None) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        text=f"Exact reviewed source content for {entry_id}.",
        namespace="project:cayu",
        labels={"project": "cayu"},
        visibility=KnowledgeVisibility.PROJECT,
        status=KnowledgeStatus.ACTIVE,
        created_by_type=KnowledgeActorType.APP,
        created_by="test-suite",
        created_at=_SOURCE_TIME,
        updated_at=_SOURCE_TIME,
        source_type="artifact",
        source_id=f"artifact:{entry_id}",
        source_hash=f"sha256:{entry_id}:1",
        expires_at=expires_at,
    )


async def _accepted(
    store: Any,
    prefix: str,
    *,
    expires_at: datetime | None = None,
    source_count: int = 2,
):
    source_ids = tuple(f"{prefix}-source-{index:02d}" for index in range(source_count))
    for source_id in source_ids:
        await store.create_entry(
            _entry(source_id, expires_at=expires_at),
            access_scope=_PRIVILEGED,
        )
    request = KnowledgeMaintenanceRoutingRequest(
        id=f"{prefix}-routing",
        policy_id="reviewed-consolidation-v1",
        namespace="project:cayu",
        labels={"project": "cayu"},
        access_scope=_ROUTING_SCOPE,
        signals=tuple(
            KnowledgeMaintenanceCandidateSignal(
                id=f"signal:{source_id}",
                kind=KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                references=(KnowledgeRevisionRef(entry_id=source_id, revision=1),),
                producer_id="test-suite",
                producer_version="1",
                reason_code="explicit_review",
                observed_at=_NOW,
            )
            for source_id in source_ids
        ),
        created_at=_NOW,
    )
    routing = await KnowledgeMaintenanceRouter(
        store,
        config=KnowledgeMaintenanceRouterConfig(
            max_signals=source_count,
            max_candidate_reads=source_count,
            max_candidates=source_count,
            max_candidate_bytes=1_048_576,
            max_concurrency=min(source_count, 8),
        ),
    ).route(request)
    planning = await KnowledgeMaintenancePlanningWorkflow(
        store,
        planner=_Planner(),
        evaluator=_Evaluator(),
        config=KnowledgeMaintenancePlanningConfig(
            planner_id="test-planner",
            planner_version="1",
            evaluator_id="test-evaluator",
            evaluator_version="1",
            planner_model_ids=("test-model",),
            evaluator_model_ids=("test-model",),
        ),
        clock=lambda: _NOW,
    ).plan(request, routing)
    assert planning.outcome is KnowledgeMaintenancePlanningOutcome.ACCEPTED
    return request, routing, planning


def _publisher(store: Any) -> KnowledgeMaintenanceProposalPublisher:
    return KnowledgeMaintenanceProposalPublisher(
        store,
        access_scope=_REVIEW_SCOPE,
        config=KnowledgeMaintenanceProposalPublisherConfig(
            publisher_id="test-publisher",
            publisher_version="1",
        ),
    )


def _decision(proposal, *, kind: KnowledgeMaintenanceDecisionKind, suffix: str):
    return KnowledgeMaintenanceDecision(
        operation_id=f"decision:{suffix}",
        proposal_id=proposal.id,
        proposal_fingerprint=proposal.fingerprint,
        kind=kind,
        reviewer_type=KnowledgeActorType.USER,
        reviewer="reviewer-1",
        reason="The exact pending proposal and source evidence were reviewed.",
        decided_at=_NOW + timedelta(minutes=1),
    )


async def _assert_pending_replacement_mutation_fence(
    store: Any,
    publication: Any,
    *,
    prefix: str,
) -> None:
    replacement = publication.replacement
    successor = replacement.model_copy(
        update={
            "revision": 2,
            "updated_at": replacement.updated_at + timedelta(seconds=1),
        }
    )
    chunks = await store.read_chunks(
        replacement.id,
        revision=replacement.revision,
        access_scope=_REVIEW_SCOPE,
        max_chunks=100,
        max_bytes=100_000,
    )
    successor_chunks = [
        chunk.model_copy(
            update={
                "id": f"{chunk.id}:generic-r2",
                "entry_revision": successor.revision,
            }
        )
        for chunk in chunks
    ]

    with pytest.raises(KnowledgeMaintenanceConflict) as append_error:
        await store.append_entry_revision(
            successor,
            expected_revision=replacement.revision,
            access_scope=_REVIEW_SCOPE,
        )
    assert append_error.value.reason == "pending_replacement_lifecycle_owned"

    with pytest.raises(KnowledgeMaintenanceConflict) as transition_error:
        await store.transition_entry_status(
            replacement.id,
            expected_revision=replacement.revision,
            access_scope=_REVIEW_SCOPE,
            from_status=KnowledgeStatus.PENDING,
            to_status=KnowledgeStatus.ACTIVE,
        )
    assert transition_error.value.reason == "pending_replacement_lifecycle_owned"

    with pytest.raises(KnowledgeMaintenanceConflict) as soft_delete_error:
        await store.delete_entry(
            replacement.id,
            expected_revision=replacement.revision,
            access_scope=_REVIEW_SCOPE,
        )
    assert soft_delete_error.value.reason == "pending_replacement_lifecycle_owned"

    with pytest.raises(KnowledgeMaintenanceConflict) as hard_delete_error:
        await store.delete_entry(
            replacement.id,
            expected_revision=replacement.revision,
            access_scope=_REVIEW_SCOPE,
            hard=True,
        )
    assert hard_delete_error.value.reason == "maintenance_replacement_history_owned"

    with pytest.raises(KnowledgeMaintenanceConflict) as publication_error:
        await store.publish_entry_revision(
            successor,
            successor_chunks,
            expected_revision=replacement.revision,
            operation_id=f"{prefix}-generic-publication",
            access_scope=_REVIEW_SCOPE,
        )
    assert publication_error.value.reason == "pending_replacement_lifecycle_owned"

    assert await store.get_entry(replacement.id, access_scope=_REVIEW_SCOPE) == replacement
    loaded = await store.load_maintenance_proposal_publication(
        publication.proposal.id,
        access_scope=_REVIEW_SCOPE,
    )
    assert loaded is not None
    assert loaded.replacement == replacement


async def _assert_hidden_replacement_occupancy_fails_closed(
    store: Any,
    *,
    prefix: str,
) -> None:
    reference = InMemoryKnowledgeStore()
    reference_request, reference_routing, reference_planning = await _accepted(
        reference,
        prefix,
    )
    expected = await _publisher(reference).publish(
        reference_request,
        reference_routing,
        reference_planning,
    )

    request, routing, planning = await _accepted(store, prefix)
    await store.create_entry(
        KnowledgeEntry(
            id=expected.replacement.id,
            text="A replacement identity occupied outside the review scope.",
            namespace="project:hidden",
            labels={"project": "hidden"},
            visibility=KnowledgeVisibility.PROJECT,
            status=KnowledgeStatus.ACTIVE,
        ),
        access_scope=_PRIVILEGED,
    )
    with pytest.raises(KnowledgeAccessDenied) as denied:
        await _publisher(store).publish(request, routing, planning)
    assert denied.value.operation == "publish_maintenance_proposal"


async def _assert_publication_conformance(store: Any, prefix: str) -> None:
    request, routing, planning = await _accepted(store, prefix)
    baseline = (
        await store.read_changes(
            after_sequence=0,
            limit=100,
            access_scope=_PRIVILEGED,
        )
    ).high_water_sequence
    publisher = _publisher(store)

    concurrent = await asyncio.gather(
        publisher.publish(request, routing, planning),
        publisher.publish(request, routing, planning),
    )
    assert [item.outcome for item in concurrent].count(
        KnowledgeMaintenanceProposalPublicationOutcome.PENDING_PERSISTED
    ) == 1
    assert [item.outcome for item in concurrent].count(
        KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_PENDING
    ) == 1
    publication = next(
        item
        for item in concurrent
        if item.outcome is KnowledgeMaintenanceProposalPublicationOutcome.PENDING_PERSISTED
    )
    concurrent_replay = next(item for item in concurrent if item is not publication)
    assert concurrent_replay.proposal == publication.proposal
    assert concurrent_replay.accepted_plan == publication.accepted_plan
    assert concurrent_replay.replacement == publication.replacement
    assert concurrent_replay.receipt == publication.receipt.model_copy(update={"replayed": True})
    assert publication.outcome is KnowledgeMaintenanceProposalPublicationOutcome.PENDING_PERSISTED
    assert publication.receipt.replayed is False
    assert publication.replacement.status is KnowledgeStatus.PENDING
    assert publication.proposal.replacement == KnowledgeRevisionRef(
        entry_id=publication.replacement.id,
        revision=1,
    )
    assert publication.proposal.metadata["accepted_plan_fingerprint"] == (
        publication.accepted_plan.fingerprint
    )
    assert (
        await store.load_maintenance_proposal(
            publication.proposal.id,
            access_scope=_REVIEW_SCOPE,
        )
        == publication.proposal
    )
    assert (
        await store.load_maintenance_proposal_publication(
            publication.proposal.id,
            access_scope=_UNRELATED_SCOPE,
        )
        is None
    )
    evidence = await store.read_evidence(
        publication.replacement.id,
        revision=1,
        access_scope=_REVIEW_SCOPE,
        max_records=10,
        max_bytes=100_000,
    )
    assert evidence is not None
    assert {(item.source_id, item.source_revision) for item in evidence.evidence} == {
        (reference.entry_id, str(reference.revision)) for reference in publication.proposal.sources
    }
    relations = await store.read_relations(
        KnowledgeRelationQuery(reference=publication.proposal.sources[0], limit=10),
        access_scope=_REVIEW_SCOPE,
    )
    assert relations is not None
    assert relations.relations == []
    await _assert_pending_replacement_mutation_fence(
        store,
        publication,
        prefix=prefix,
    )

    equivalent_attempt = planning.model_copy(
        update={"processed_at": planning.processed_at + timedelta(hours=1)}
    )
    replay = await publisher.publish(request, routing, equivalent_attempt)
    assert replay.proposal == publication.proposal
    assert replay.accepted_plan == publication.accepted_plan
    assert replay.replacement == publication.replacement
    assert replay.receipt == publication.receipt.model_copy(update={"replayed": True})
    assert replay.outcome is KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_PENDING
    changes = (
        await store.read_changes(
            after_sequence=baseline,
            limit=100,
            access_scope=_PRIVILEGED,
        )
    ).changes
    assert (
        len(
            [
                change
                for change in changes
                if change.operation_id == publication.receipt.operation_id
            ]
        )
        == 1
    )

    altered = publication.proposal.model_copy(
        update={"rationale": "A different proposal must not replace the stored authority."}
    )
    with pytest.raises(KnowledgeMaintenanceConflict, match="durable state"):
        await store.apply_maintenance_decision(
            altered,
            _decision(
                altered,
                kind=KnowledgeMaintenanceDecisionKind.APPROVE,
                suffix=f"{prefix}-altered",
            ),
            access_scope=_REVIEW_SCOPE,
        )

    alternate = publication.proposal.model_copy(
        update={
            "id": f"{prefix}-alternate-proposal",
            "rationale": "A new proposal identity cannot claim a published replacement.",
        }
    )
    with pytest.raises(KnowledgeMaintenanceConflict) as ownership_error:
        await store.apply_maintenance_decision(
            alternate,
            _decision(
                alternate,
                kind=KnowledgeMaintenanceDecisionKind.APPROVE,
                suffix=f"{prefix}-alternate",
            ),
            access_scope=_REVIEW_SCOPE,
        )
    assert ownership_error.value.reason == "proposal_publication_mismatch"
    assert (
        await store.get_entry(publication.replacement.id, access_scope=_REVIEW_SCOPE)
        == publication.replacement
    )
    still_pending = await publisher.load(publication.proposal.id)
    assert still_pending is not None
    assert still_pending.outcome is (
        KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_PENDING
    )

    receipt = await KnowledgeReviewWorkflow(
        store,
        access_scope=_REVIEW_SCOPE,
        namespace="project:cayu",
        labels={"project": "cayu"},
    ).decide_maintenance(
        publication.proposal,
        _decision(
            publication.proposal,
            kind=KnowledgeMaintenanceDecisionKind.APPROVE,
            suffix=f"{prefix}-approve",
        ),
    )
    assert receipt.replacement is not None
    decided = await publisher.load(publication.proposal.id)
    assert decided is not None
    assert decided.outcome is KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_DECIDED
    assert decided.replacement.status is KnowledgeStatus.PENDING
    active_replacement = await store.get_entry(
        publication.replacement.id,
        access_scope=_REVIEW_SCOPE,
    )
    assert active_replacement is not None
    with pytest.raises(KnowledgeMaintenanceConflict) as approved_hard_delete_error:
        await store.delete_entry(
            active_replacement.id,
            expected_revision=active_replacement.revision,
            access_scope=_REVIEW_SCOPE,
            hard=True,
        )
    assert approved_hard_delete_error.value.reason == ("maintenance_replacement_history_owned")
    archived_replacement = await store.transition_entry_status(
        active_replacement.id,
        expected_revision=active_replacement.revision,
        access_scope=_REVIEW_SCOPE,
        from_status=KnowledgeStatus.ACTIVE,
        to_status=KnowledgeStatus.ARCHIVED,
    )
    assert archived_replacement.revision == active_replacement.revision + 1
    for source in publication.proposal.sources:
        current = await store.get_entry(source.entry_id, access_scope=_PRIVILEGED)
        assert current is not None
        assert current.status is KnowledgeStatus.ARCHIVED

    reject_request, reject_routing, reject_planning = await _accepted(
        store,
        f"{prefix}-rejected",
    )
    rejected_publication = await publisher.publish(
        reject_request,
        reject_routing,
        reject_planning,
    )
    rejected_receipt = await KnowledgeReviewWorkflow(
        store,
        access_scope=_REVIEW_SCOPE,
        namespace="project:cayu",
        labels={"project": "cayu"},
    ).decide_maintenance(
        rejected_publication.proposal,
        _decision(
            rejected_publication.proposal,
            kind=KnowledgeMaintenanceDecisionKind.REJECT,
            suffix=f"{prefix}-reject",
        ),
    )
    assert rejected_receipt.replacement is None
    with pytest.raises(KnowledgeMaintenanceConflict) as rejected_transition_error:
        await store.transition_entry_status(
            rejected_publication.replacement.id,
            expected_revision=rejected_publication.replacement.revision,
            access_scope=_REVIEW_SCOPE,
            from_status=KnowledgeStatus.PENDING,
            to_status=KnowledgeStatus.ACTIVE,
        )
    assert rejected_transition_error.value.reason == "rejected_replacement_lifecycle_owned"
    for source in rejected_publication.proposal.sources:
        current = await store.get_entry(source.entry_id, access_scope=_PRIVILEGED)
        assert current is not None
        assert current.revision == source.revision
        assert current.status is KnowledgeStatus.ACTIVE

    archived_rejected = await store.transition_entry_status(
        rejected_publication.replacement.id,
        expected_revision=rejected_publication.replacement.revision,
        access_scope=_REVIEW_SCOPE,
        from_status=KnowledgeStatus.PENDING,
        to_status=KnowledgeStatus.ARCHIVED,
    )
    assert archived_rejected.status is KnowledgeStatus.ARCHIVED
    with pytest.raises(KnowledgeMaintenanceConflict) as rejected_content_error:
        await store.append_entry_revision(
            archived_rejected.model_copy(
                update={
                    "revision": archived_rejected.revision + 1,
                    "text": "A rejection cannot be bypassed through a content revision.",
                    "updated_at": archived_rejected.updated_at + timedelta(seconds=1),
                }
            ),
            expected_revision=archived_rejected.revision,
            access_scope=_PRIVILEGED,
        )
    assert rejected_content_error.value.reason == "rejected_replacement_lifecycle_owned"
    deleted_rejected = await store.delete_entry(
        archived_rejected.id,
        expected_revision=archived_rejected.revision,
        access_scope=_PRIVILEGED,
    )
    assert deleted_rejected is not None
    assert deleted_rejected.status is KnowledgeStatus.DELETED
    with pytest.raises(KnowledgeMaintenanceConflict) as retirement_reversal_error:
        await store.transition_entry_status(
            deleted_rejected.id,
            expected_revision=deleted_rejected.revision,
            access_scope=_PRIVILEGED,
            from_status=KnowledgeStatus.DELETED,
            to_status=KnowledgeStatus.ARCHIVED,
        )
    assert retirement_reversal_error.value.reason == ("rejected_replacement_lifecycle_owned")
    with pytest.raises(KnowledgeMaintenanceConflict) as reactivation_error:
        await store.transition_entry_status(
            deleted_rejected.id,
            expected_revision=deleted_rejected.revision,
            access_scope=_PRIVILEGED,
            from_status=KnowledgeStatus.DELETED,
            to_status=KnowledgeStatus.ACTIVE,
        )
    assert reactivation_error.value.reason == "rejected_replacement_lifecycle_owned"
    with pytest.raises(KnowledgeMaintenanceConflict) as rejected_hard_delete_error:
        await store.delete_entry(
            deleted_rejected.id,
            expected_revision=deleted_rejected.revision,
            access_scope=_PRIVILEGED,
            hard=True,
        )
    assert rejected_hard_delete_error.value.reason == ("maintenance_replacement_history_owned")
    retained_rejection = await publisher.load(rejected_publication.proposal.id)
    assert retained_rejection is not None
    assert retained_rejection.outcome is (
        KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_DECIDED
    )
    assert retained_rejection.replacement == rejected_publication.replacement

    stale_request, stale_routing, stale_planning = await _accepted(
        store,
        f"{prefix}-stale-rejection",
    )
    stale_publication = await publisher.publish(
        stale_request,
        stale_routing,
        stale_planning,
    )
    stale_source = await store.get_entry(
        stale_publication.proposal.sources[0].entry_id,
        access_scope=_PRIVILEGED,
    )
    assert stale_source is not None
    advanced_source = stale_source.model_copy(
        update={
            "revision": stale_source.revision + 1,
            "text": stale_source.text + " Advanced after proposal publication.",
            "updated_at": stale_source.updated_at + timedelta(minutes=1),
        }
    )
    assert (
        await store.append_entry_revision(
            advanced_source,
            expected_revision=stale_source.revision,
            access_scope=_PRIVILEGED,
        )
        == advanced_source
    )
    stale_rejection = await KnowledgeReviewWorkflow(
        store,
        access_scope=_REVIEW_SCOPE,
        namespace="project:cayu",
        labels={"project": "cayu"},
    ).decide_maintenance(
        stale_publication.proposal,
        _decision(
            stale_publication.proposal,
            kind=KnowledgeMaintenanceDecisionKind.REJECT,
            suffix=f"{prefix}-stale-reject",
        ),
    )
    assert stale_rejection.outcome is cayu.KnowledgeMaintenanceOutcome.REJECTED
    assert await store.get_entry(advanced_source.id, access_scope=_PRIVILEGED) == advanced_source
    archived_stale_replacement = await store.transition_entry_status(
        stale_publication.replacement.id,
        expected_revision=stale_publication.replacement.revision,
        access_scope=_REVIEW_SCOPE,
        from_status=KnowledgeStatus.PENDING,
        to_status=KnowledgeStatus.ARCHIVED,
    )
    assert archived_stale_replacement.status is KnowledgeStatus.ARCHIVED

    expired_request, expired_routing, expired_planning = await _accepted(
        store,
        f"{prefix}-expired",
        expires_at=_NOW - timedelta(days=1),
    )
    expired_publication = await publisher.publish(
        expired_request,
        expired_routing,
        expired_planning,
    )
    assert expired_publication.replacement.expires_at == _NOW - timedelta(days=1)
    assert await store.prune_expired(access_scope=_PRIVILEGED, now=_NOW) == 2
    assert (
        await store.get_entry(
            expired_publication.replacement.id,
            access_scope=_REVIEW_SCOPE,
        )
        == expired_publication.replacement
    )
    preserved_publication = await publisher.load(expired_publication.proposal.id)
    assert preserved_publication is not None
    assert preserved_publication.replacement == expired_publication.replacement
    expired_rejection = await KnowledgeReviewWorkflow(
        store,
        access_scope=_REVIEW_SCOPE,
        namespace="project:cayu",
        labels={"project": "cayu"},
    ).decide_maintenance(
        expired_publication.proposal,
        _decision(
            expired_publication.proposal,
            kind=KnowledgeMaintenanceDecisionKind.REJECT,
            suffix=f"{prefix}-expired-reject",
        ),
    )
    assert expired_rejection.outcome is cayu.KnowledgeMaintenanceOutcome.REJECTED
    archived_expired_replacement = await store.transition_entry_status(
        expired_publication.replacement.id,
        expected_revision=expired_publication.replacement.revision,
        access_scope=_REVIEW_SCOPE,
        from_status=KnowledgeStatus.PENDING,
        to_status=KnowledgeStatus.ARCHIVED,
    )
    assert archived_expired_replacement.status is KnowledgeStatus.ARCHIVED

    await _assert_hidden_replacement_occupancy_fails_closed(
        store,
        prefix=f"{prefix}-hidden-occupancy",
    )


def test_public_maintenance_proposal_persistence_surface_is_exported() -> None:
    for name in (
        "KnowledgeMaintenanceAcceptedPlan",
        "KnowledgeMaintenanceProposalPublication",
        "KnowledgeMaintenanceProposalPublicationConflict",
        "KnowledgeMaintenanceProposalPublicationOutcome",
        "KnowledgeMaintenanceProposalPublicationReceipt",
        "KnowledgeMaintenanceProposalPublisher",
        "KnowledgeMaintenanceProposalPublisherConfig",
    ):
        assert getattr(cayu, name).__name__ == name


def test_inmemory_accepted_plan_publication_and_review_handoff() -> None:
    asyncio.run(_assert_publication_conformance(InMemoryKnowledgeStore(), "memory"))


def test_sqlite_accepted_plan_publication_and_review_handoff(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = SQLiteKnowledgeStore(tmp_path / "maintenance.db")
        try:
            await _assert_publication_conformance(store, "sqlite")
        finally:
            await store.close()

    asyncio.run(scenario())


def test_publication_rejects_a_source_that_advanced_after_evaluation() -> None:
    async def scenario() -> None:
        store = InMemoryKnowledgeStore()
        request, routing, planning = await _accepted(store, "stale")
        source = routing.candidates[0].entry
        await store.append_entry_revision(
            source.model_copy(
                update={
                    "revision": 2,
                    "text": source.text + " Updated after evaluation.",
                    "updated_at": _NOW + timedelta(minutes=1),
                }
            ),
            expected_revision=1,
            access_scope=_PRIVILEGED,
        )
        with pytest.raises(KnowledgeMaintenanceStale):
            await _publisher(store).publish(request, routing, planning)
        entries = await store.list_entries(
            cayu.KnowledgeListQuery(
                namespace="project:cayu",
                statuses=[KnowledgeStatus.PENDING],
                limit=10,
            ),
            access_scope=_PRIVILEGED,
        )
        assert entries.entries == []

    asyncio.run(scenario())


def test_publication_retry_recovers_after_commit_then_cancellation() -> None:
    class _CommitThenCancel(InMemoryKnowledgeStore):
        cancelled = False

        async def publish_maintenance_proposal(self, *args, **kwargs):
            receipt = await super().publish_maintenance_proposal(*args, **kwargs)
            if not self.cancelled:
                self.cancelled = True
                raise asyncio.CancelledError
            return receipt

    async def scenario() -> None:
        store = _CommitThenCancel()
        request, routing, planning = await _accepted(store, "cancelled")
        publisher = _publisher(store)
        with pytest.raises(asyncio.CancelledError):
            await publisher.publish(request, routing, planning)
        replay = await publisher.publish(request, routing, planning)
        assert replay.outcome is KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_PENDING
        assert replay.receipt.replayed is True

    asyncio.run(scenario())


def test_publication_request_reuse_rejects_different_material() -> None:
    async def scenario() -> None:
        store = InMemoryKnowledgeStore()
        request, routing, planning = await _accepted(store, "conflict")
        publication = await _publisher(store).publish(request, routing, planning)
        chunks = await store.read_chunks(
            publication.replacement.id,
            revision=1,
            access_scope=_REVIEW_SCOPE,
            max_chunks=100,
            max_bytes=100_000,
        )
        evidence = await store.read_evidence(
            publication.replacement.id,
            revision=1,
            access_scope=_REVIEW_SCOPE,
            max_records=100,
            max_bytes=100_000,
        )
        assert evidence is not None
        conflicting_chunks = [
            chunks[0].model_copy(
                update={"metadata": {**chunks[0].metadata, "conflicting_retry": True}}
            ),
            *chunks[1:],
        ]
        with pytest.raises(KnowledgeMaintenanceProposalPublicationConflict):
            await store.publish_maintenance_proposal(
                publication.replacement,
                conflicting_chunks,
                evidence=evidence.evidence,
                proposal=publication.proposal,
                accepted_plan=publication.accepted_plan,
                operation_id=publication.receipt.operation_id,
                access_scope=_REVIEW_SCOPE,
            )

    asyncio.run(scenario())


def test_sqlite_publication_load_validates_the_decision_record(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = SQLiteKnowledgeStore(tmp_path / "malformed-decision.db")
        try:
            request, routing, planning = await _accepted(store, "malformed-decision")
            publisher = _publisher(store)
            publication = await publisher.publish(request, routing, planning)
            decision = _decision(
                publication.proposal,
                kind=KnowledgeMaintenanceDecisionKind.REJECT,
                suffix="malformed-decision",
            )
            await store.apply_maintenance_decision(
                publication.proposal,
                decision,
                access_scope=_REVIEW_SCOPE,
            )
            async with store._lock:
                store._connection.execute(
                    "UPDATE cayu_knowledge_maintenance_decisions "
                    "SET decision_json = '{}' WHERE operation_id = ?",
                    (decision.operation_id,),
                )
                store._connection.commit()
            with pytest.raises(KnowledgeMaintenanceProposalPublicationConflict) as error:
                await publisher.load(publication.proposal.id)
            assert error.value.reason == "malformed_receipt"
        finally:
            await store.close()

    asyncio.run(scenario())


def test_max_source_publication_fingerprint_work_is_constant(monkeypatch) -> None:
    import cayu.knowledge_maintenance_persistence as persistence

    async def scenario() -> None:
        store = InMemoryKnowledgeStore()
        request, routing, planning = await _accepted(
            store,
            "max-source-fingerprint",
            source_count=50,
        )
        original_fingerprint = persistence._fingerprint
        accepted_plan_fingerprint_calls = 0
        publisher_config_fingerprint_calls = 0

        def tracked_fingerprint(value: object, field_name: str) -> str:
            nonlocal accepted_plan_fingerprint_calls, publisher_config_fingerprint_calls
            if field_name == "knowledge maintenance accepted plan fingerprint":
                accepted_plan_fingerprint_calls += 1
            elif field_name == "knowledge maintenance proposal publisher configuration":
                publisher_config_fingerprint_calls += 1
            return original_fingerprint(value, field_name)

        monkeypatch.setattr(persistence, "_fingerprint", tracked_fingerprint)
        publisher = _publisher(store)
        await publisher.publish(request, routing, planning)
        assert accepted_plan_fingerprint_calls <= 4
        assert publisher_config_fingerprint_calls <= 2

        accepted_plan_fingerprint_calls = 0
        publisher_config_fingerprint_calls = 0
        replay = await publisher.publish(request, routing, planning)
        assert replay.outcome is KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_PENDING
        assert accepted_plan_fingerprint_calls <= 4
        assert publisher_config_fingerprint_calls <= 2

    asyncio.run(scenario())
