from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from cayu.storage import (
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeChangeKind,
    KnowledgeEntry,
    KnowledgeMaintenanceConflict,
    KnowledgeMaintenanceDecision,
    KnowledgeMaintenanceDecisionKind,
    KnowledgeMaintenanceDecisionReceipt,
    KnowledgeMaintenanceOutcome,
    KnowledgeMaintenanceProposal,
    KnowledgeMaintenanceStale,
    KnowledgeQuery,
    KnowledgeRelation,
    KnowledgeRelationKind,
    KnowledgeRelationQuery,
    KnowledgeRevisionRef,
    KnowledgeStatus,
)

_NOW = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)
_SCOPE = KnowledgeAccessScope.privileged()


def maintenance_entry(
    entry_id: str,
    *,
    status: KnowledgeStatus,
    offset: int = 0,
) -> KnowledgeEntry:
    timestamp = _NOW + timedelta(seconds=offset)
    return KnowledgeEntry(
        id=entry_id,
        text=f"reviewed content for {entry_id}",
        namespace="project:cayu",
        labels={"project": "cayu"},
        status=status,
        created_at=timestamp,
        updated_at=timestamp,
    )


def maintenance_proposal(
    prefix: str,
    *,
    kind: KnowledgeRelationKind = KnowledgeRelationKind.SUPERSEDES,
) -> KnowledgeMaintenanceProposal:
    replacement = KnowledgeRevisionRef(entry_id=f"{prefix}-replacement", revision=1)
    source = KnowledgeRevisionRef(entry_id=f"{prefix}-source", revision=1)
    active_replacement = replacement.model_copy(update={"revision": 2})
    return KnowledgeMaintenanceProposal(
        id=f"{prefix}-proposal",
        replacement=replacement,
        sources=[source],
        relations=[
            KnowledgeRelation(
                id=f"{prefix}-relation",
                subject=active_replacement,
                object=source,
                kind=kind,
                created_by_type=KnowledgeActorType.APP,
                created_by="maintenance-planner",
                policy_id="reviewed-maintenance-v1",
                created_at=_NOW,
                metadata={"safe_code": prefix},
            )
        ],
        access_scope=_SCOPE,
        policy_id="reviewed-maintenance-v1",
        proposed_by_type=KnowledgeActorType.APP,
        proposed_by="maintenance-planner",
        created_at=_NOW,
        rationale="The replacement retains the reviewed source meaning.",
        evidence_summary="The proposal references bounded external evidence identities.",
        metadata={"safe_code": prefix},
    )


def maintenance_decision(
    proposal: KnowledgeMaintenanceProposal,
    *,
    operation_id: str,
    kind: KnowledgeMaintenanceDecisionKind,
) -> KnowledgeMaintenanceDecision:
    return KnowledgeMaintenanceDecision(
        operation_id=operation_id,
        proposal_id=proposal.id,
        proposal_fingerprint=proposal.fingerprint,
        kind=kind,
        reviewer_type=KnowledgeActorType.USER,
        reviewer="reviewer-7",
        reason="The exact proposal and its bounded evidence were reviewed.",
        decided_at=_NOW + timedelta(minutes=1),
        metadata={"review_queue": "knowledge"},
    )


async def _create_proposal_entries(store: Any, proposal: KnowledgeMaintenanceProposal) -> None:
    await store.create_entry(
        maintenance_entry(proposal.replacement.entry_id, status=KnowledgeStatus.PENDING)
    )
    for source in proposal.sources:
        await store.create_entry(maintenance_entry(source.entry_id, status=KnowledgeStatus.ACTIVE))


async def assert_knowledge_maintenance_conformance(store: Any) -> None:
    proposal = maintenance_proposal("maintenance-approval")
    await _create_proposal_entries(store, proposal)
    recall_query = KnowledgeQuery(
        text="reviewed content",
        namespace="project:cayu",
        limit=10,
    )
    pending_recall = await store.search(recall_query)
    assert all(hit.entry.id != proposal.replacement.entry_id for hit in pending_recall.hits)
    baseline = (await store.read_changes(after_sequence=0, limit=100)).high_water_sequence
    decision = maintenance_decision(
        proposal,
        operation_id="maintenance-approval-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    )

    receipt = await store.apply_maintenance_decision(proposal, decision)
    assert receipt.outcome is KnowledgeMaintenanceOutcome.APPLIED
    assert receipt.replayed is False
    assert receipt.replacement == KnowledgeRevisionRef(
        entry_id=proposal.replacement.entry_id,
        revision=2,
    )
    assert receipt.archived_revisions == [
        KnowledgeRevisionRef(entry_id=proposal.sources[0].entry_id, revision=2)
    ]
    assert receipt.relation_ids == [proposal.relations[0].id]

    replacement = await store.get_entry(proposal.replacement.entry_id)
    source = await store.get_entry(proposal.sources[0].entry_id)
    assert replacement is not None
    assert replacement.revision == 2
    assert replacement.status is KnowledgeStatus.ACTIVE
    assert source is not None
    assert source.revision == 2
    assert source.status is KnowledgeStatus.ARCHIVED
    active_recall = await store.search(recall_query)
    assert [hit.entry.id for hit in active_recall.hits] == [proposal.replacement.entry_id]
    relations = await store.read_relations(
        KnowledgeRelationQuery(reference=receipt.replacement, limit=10)
    )
    assert relations is not None
    assert relations.relations == proposal.relations

    changes = (await store.read_changes(after_sequence=baseline, limit=100)).changes
    operation_changes = [
        change for change in changes if change.operation_id == decision.operation_id
    ]
    assert [change.kind for change in operation_changes] == [
        KnowledgeChangeKind.STATUS_TRANSITIONED,
        KnowledgeChangeKind.STATUS_TRANSITIONED,
        KnowledgeChangeKind.RELATION_PUBLISHED,
    ]
    assert await store.load_maintenance_proposal(proposal.id) == proposal
    assert await store.load_maintenance_decision(decision.operation_id) == decision
    assert await store.load_maintenance_decision_receipt(decision.operation_id) == receipt

    replay = await store.apply_maintenance_decision(proposal, decision)
    assert replay == receipt.model_copy(update={"replayed": True})
    replay_changes = (await store.read_changes(after_sequence=baseline, limit=100)).changes
    assert replay_changes == changes

    retirement_scope = KnowledgeAccessScope.for_namespace(
        "project:cayu",
        allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
        include_expired=True,
    )
    retirement_template = maintenance_proposal("maintenance-retirement-scope")
    retirement = KnowledgeMaintenanceProposal.model_validate(
        {
            **retirement_template.model_dump(mode="python"),
            "access_scope": retirement_scope,
        }
    )
    retirement_decision = maintenance_decision(
        retirement,
        operation_id="maintenance-retirement-scope-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    )
    bound_scope = store._default_access_scope
    store._default_access_scope = None
    try:
        await store.create_entry(
            maintenance_entry(
                retirement.replacement.entry_id,
                status=KnowledgeStatus.PENDING,
            ),
            access_scope=retirement_scope,
        )
        for source in retirement.sources:
            await store.create_entry(
                maintenance_entry(source.entry_id, status=KnowledgeStatus.ACTIVE),
                access_scope=retirement_scope,
            )
        retirement_receipt = await store.apply_maintenance_decision(
            retirement,
            retirement_decision,
            access_scope=retirement_scope,
        )
        assert retirement_receipt.outcome is KnowledgeMaintenanceOutcome.APPLIED
        retired_source = retirement.sources[0]
        assert (
            await store.get_entry(
                retired_source.entry_id,
                access_scope=retirement_scope,
            )
            is None
        )
        assert (
            await store.get_entry(
                retired_source.entry_id,
                revision=retired_source.revision + 1,
                access_scope=retirement_scope,
            )
            is None
        )
        assert (
            await store.load_maintenance_proposal(
                retirement.id,
                access_scope=retirement_scope,
            )
            == retirement
        )
        assert (
            await store.load_maintenance_decision(
                retirement_decision.operation_id,
                access_scope=retirement_scope,
            )
            == retirement_decision
        )
        assert (
            await store.load_maintenance_decision_receipt(
                retirement_decision.operation_id,
                access_scope=retirement_scope,
            )
            == retirement_receipt
        )
        assert await store.apply_maintenance_decision(
            retirement,
            retirement_decision,
            access_scope=retirement_scope,
        ) == retirement_receipt.model_copy(update={"replayed": True})
    finally:
        store._default_access_scope = bound_scope

    rejected = maintenance_proposal("maintenance-rejection")
    await _create_proposal_entries(store, rejected)
    rejected_baseline = (await store.read_changes(after_sequence=0, limit=100)).high_water_sequence
    rejection = maintenance_decision(
        rejected,
        operation_id="maintenance-rejection-operation",
        kind=KnowledgeMaintenanceDecisionKind.REJECT,
    )
    rejected_receipt = await store.apply_maintenance_decision(rejected, rejection)
    assert rejected_receipt.outcome is KnowledgeMaintenanceOutcome.REJECTED
    assert rejected_receipt.replacement is None
    assert rejected_receipt.archived_revisions == []
    assert rejected_receipt.relation_ids == []
    rejected_replacement = await store.get_entry(rejected.replacement.entry_id)
    rejected_source = await store.get_entry(rejected.sources[0].entry_id)
    assert rejected_replacement is not None
    assert rejected_replacement.revision == 1
    assert rejected_replacement.status is KnowledgeStatus.PENDING
    assert rejected_source is not None
    assert rejected_source.revision == 1
    assert rejected_source.status is KnowledgeStatus.ACTIVE
    assert (await store.read_changes(after_sequence=0, limit=100)).high_water_sequence == (
        rejected_baseline
    )
    assert await store.load_maintenance_proposal(rejected.id) == rejected
    assert await store.load_maintenance_decision(rejection.operation_id) == rejection
    assert await store.apply_maintenance_decision(rejected, rejection) == (
        rejected_receipt.model_copy(update={"replayed": True})
    )
    assert (
        await store.read_relations(
            KnowledgeRelationQuery(
                reference=KnowledgeRevisionRef(
                    entry_id=rejected.replacement.entry_id,
                    revision=1,
                )
            )
        )
    ).relations == []

    stale = maintenance_proposal("maintenance-stale")
    await _create_proposal_entries(store, stale)
    stale_source = await store.get_entry(stale.sources[0].entry_id)
    assert stale_source is not None
    await store.append_entry_revision(
        stale_source.model_copy(
            update={
                "revision": 2,
                "text": "A concurrent reviewer advanced this exact source.",
                "updated_at": _NOW + timedelta(minutes=2),
            }
        ),
        expected_revision=1,
    )
    stale_decision = maintenance_decision(
        stale,
        operation_id="maintenance-stale-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    )
    try:
        await store.apply_maintenance_decision(stale, stale_decision)
    except KnowledgeMaintenanceStale as exc:
        assert exc.reason == "source_revision"
    else:  # pragma: no cover - conformance assertion
        raise AssertionError("A stale maintenance proposal must fail closed.")
    stale_replacement = await store.get_entry(stale.replacement.entry_id)
    assert stale_replacement is not None
    assert stale_replacement.revision == 1
    assert stale_replacement.status is KnowledgeStatus.PENDING
    assert await store.load_maintenance_decision_receipt(stale_decision.operation_id) is None

    stale_replacement_proposal = maintenance_proposal("maintenance-stale-replacement")
    await _create_proposal_entries(store, stale_replacement_proposal)
    pending = await store.get_entry(stale_replacement_proposal.replacement.entry_id)
    assert pending is not None
    await store.append_entry_revision(
        pending.model_copy(
            update={
                "revision": 2,
                "text": "A concurrent editor changed the pending replacement.",
                "updated_at": _NOW + timedelta(minutes=2),
            }
        ),
        expected_revision=1,
    )
    stale_replacement_decision = maintenance_decision(
        stale_replacement_proposal,
        operation_id="maintenance-stale-replacement-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    )
    try:
        await store.apply_maintenance_decision(
            stale_replacement_proposal,
            stale_replacement_decision,
        )
    except KnowledgeMaintenanceStale as exc:
        assert exc.reason == "replacement_revision"
    else:  # pragma: no cover - conformance assertion
        raise AssertionError("A changed pending replacement must fail closed.")
    assert (
        await store.load_maintenance_decision_receipt(stale_replacement_decision.operation_id)
        is None
    )

    contradiction = maintenance_proposal(
        "maintenance-contradiction",
        kind=KnowledgeRelationKind.CONTRADICTS,
    )
    await _create_proposal_entries(store, contradiction)
    contradiction_decision = maintenance_decision(
        contradiction,
        operation_id="maintenance-contradiction-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    )
    contradiction_receipt = await store.apply_maintenance_decision(
        contradiction,
        contradiction_decision,
    )
    assert contradiction_receipt.archived_revisions == []
    contradiction_source = await store.get_entry(contradiction.sources[0].entry_id)
    assert contradiction_source is not None
    assert contradiction_source.revision == 1
    assert contradiction_source.status is KnowledgeStatus.ACTIVE

    derived = maintenance_proposal(
        "maintenance-derived",
        kind=KnowledgeRelationKind.DERIVED_FROM,
    )
    await _create_proposal_entries(store, derived)
    derived_decision = maintenance_decision(
        derived,
        operation_id="maintenance-derived-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    )
    derived_receipt = await store.apply_maintenance_decision(derived, derived_decision)
    assert derived_receipt.archived_revisions == []
    derived_source = await store.get_entry(derived.sources[0].entry_id)
    assert derived_source is not None
    assert derived_source.revision == 1
    assert derived_source.status is KnowledgeStatus.ACTIVE

    concurrent = maintenance_proposal("maintenance-concurrent")
    await _create_proposal_entries(store, concurrent)
    concurrent_decision = maintenance_decision(
        concurrent,
        operation_id="maintenance-concurrent-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    )
    concurrent_receipts = await asyncio.gather(
        store.apply_maintenance_decision(concurrent, concurrent_decision),
        store.apply_maintenance_decision(concurrent, concurrent_decision),
    )
    assert sorted(receipt.replayed for receipt in concurrent_receipts) == [False, True]

    conflicting_decision = concurrent_decision.model_copy(
        update={"operation_id": "maintenance-concurrent-conflict", "reason": "changed"}
    )
    try:
        await store.apply_maintenance_decision(concurrent, conflicting_decision)
    except KnowledgeMaintenanceConflict as exc:
        assert exc.reason == "proposal_already_decided"
    else:  # pragma: no cover - conformance assertion
        raise AssertionError("One proposal cannot receive two durable decisions.")

    incompatible = maintenance_proposal("maintenance-incompatible-reviewers")
    await _create_proposal_entries(store, incompatible)
    approve = maintenance_decision(
        incompatible,
        operation_id="maintenance-incompatible-approve",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    )
    reject = maintenance_decision(
        incompatible,
        operation_id="maintenance-incompatible-reject",
        kind=KnowledgeMaintenanceDecisionKind.REJECT,
    )

    async def decide(
        candidate: KnowledgeMaintenanceDecision,
    ) -> KnowledgeMaintenanceDecisionReceipt | KnowledgeMaintenanceConflict:
        try:
            return await store.apply_maintenance_decision(incompatible, candidate)
        except KnowledgeMaintenanceConflict as exc:
            return exc

    incompatible_outcomes = await asyncio.gather(decide(approve), decide(reject))
    assert (
        sum(
            isinstance(outcome, KnowledgeMaintenanceDecisionReceipt)
            for outcome in incompatible_outcomes
        )
        == 1
    )
    assert (
        sum(isinstance(outcome, KnowledgeMaintenanceConflict) for outcome in incompatible_outcomes)
        == 1
    )
    winning_receipt = next(
        outcome
        for outcome in incompatible_outcomes
        if isinstance(outcome, KnowledgeMaintenanceDecisionReceipt)
    )
    losing_decision = reject if winning_receipt.operation_id == approve.operation_id else approve
    assert await store.load_maintenance_decision(winning_receipt.operation_id) is not None
    assert await store.load_maintenance_decision_receipt(losing_decision.operation_id) is None

    denied = maintenance_proposal("maintenance-denied")
    await _create_proposal_entries(store, denied)
    denied_decision = maintenance_decision(
        denied,
        operation_id="maintenance-denied-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    )
    unrelated_scope = KnowledgeAccessScope.for_namespace("project:unrelated")
    denied_baseline = (await store.read_changes(after_sequence=0, limit=100)).high_water_sequence
    private_scope = KnowledgeAccessScope.for_namespace(
        "project:cayu",
        allowed_statuses=list(KnowledgeStatus),
        include_expired=True,
    )
    bound_scope = store._default_access_scope
    store._default_access_scope = None
    try:
        try:
            await store.apply_maintenance_decision(
                denied,
                denied_decision,
                access_scope=unrelated_scope,
            )
        except KnowledgeAccessDenied as exc:
            assert exc.operation == "apply_maintenance_decision"
            assert denied.rationale not in str(exc)
            assert denied.evidence_summary not in str(exc)
        else:  # pragma: no cover - conformance assertion
            raise AssertionError("An unrelated scope authorized a maintenance decision.")
        assert (
            await store.load_maintenance_decision_receipt(
                denied_decision.operation_id,
                access_scope=unrelated_scope,
            )
            is None
        )
        assert (
            await store.read_changes(
                after_sequence=0,
                limit=100,
                access_scope=_SCOPE,
            )
        ).high_water_sequence == denied_baseline
        assert (
            await store.load_maintenance_proposal(
                proposal.id,
                access_scope=unrelated_scope,
            )
            is None
        )
        assert (
            await store.load_maintenance_decision(
                decision.operation_id,
                access_scope=unrelated_scope,
            )
            is None
        )
        assert (
            await store.load_maintenance_decision_receipt(
                decision.operation_id,
                access_scope=unrelated_scope,
            )
            is None
        )
        assert (
            await store.load_maintenance_decision_receipt(
                decision.operation_id,
                access_scope=private_scope,
            )
            is not None
        )
    finally:
        store._default_access_scope = bound_scope

    cancelled = maintenance_proposal("maintenance-cancelled-before-start")
    await _create_proposal_entries(store, cancelled)
    cancelled_decision = maintenance_decision(
        cancelled,
        operation_id="maintenance-cancelled-before-start-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    )
    cancellation = asyncio.create_task(
        store.apply_maintenance_decision(cancelled, cancelled_decision)
    )
    cancellation.cancel()
    try:
        await cancellation
    except asyncio.CancelledError:
        pass
    else:  # pragma: no cover - conformance assertion
        raise AssertionError("A pre-start cancellation unexpectedly committed.")
    cancelled_replacement = await store.get_entry(cancelled.replacement.entry_id)
    assert cancelled_replacement is not None
    assert cancelled_replacement.revision == 1
    assert cancelled_replacement.status is KnowledgeStatus.PENDING
    assert await store.load_maintenance_decision_receipt(cancelled_decision.operation_id) is None
