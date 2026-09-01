from __future__ import annotations

import asyncio

from tests.core.test_knowledge_maintenance_persistence import (
    _accepted,
    _decision,
    _publisher,
)

from cayu import (
    KnowledgeAccessScope,
    KnowledgeGovernanceConfig,
    KnowledgeGovernanceMode,
    KnowledgeMaintenanceConflict,
    KnowledgeMaintenanceDecision,
    KnowledgeMaintenanceDecisionKind,
    KnowledgeMaintenanceGovernanceAuthority,
    KnowledgeMaintenanceGovernanceDecision,
    KnowledgeMaintenanceGovernanceDisposition,
    KnowledgeMaintenanceGovernor,
    KnowledgeMaintenanceOutcome,
    KnowledgeReviewWorkflow,
    KnowledgeStatus,
    KnowledgeStore,
    decide_knowledge_maintenance_governance,
    load_knowledge_maintenance_governance_receipt,
    prepare_knowledge_maintenance_governance_request,
)
from cayu.knowledge_maintenance_governance import maintenance_decision_from_governance


class MaintenanceGovernancePolicy:
    """Deterministic provider-free application policy used across backends."""

    def __init__(
        self,
        disposition: KnowledgeMaintenanceGovernanceDisposition,
        *,
        identity: str = "conformance.maintenance-policy",
        version: str = "3",
    ) -> None:
        self.disposition = disposition
        self.identity = identity
        self.version = version
        self.calls = 0

    async def decide_maintenance(self, request):
        self.calls += 1
        return KnowledgeMaintenanceGovernanceDecision(
            request_sha256=request.fingerprint,
            disposition=self.disposition,
            policy_identity=self.identity,
            policy_version=self.version,
            code=f"policy_{self.disposition.value}",
            annotations={"risk_tier": "bounded"},
        )


async def maintenance_governance_publication(store: KnowledgeStore, prefix: str):
    request, routing, planning = await _accepted(store, prefix)
    return await _publisher(store).publish(request, routing, planning)


def maintenance_governance_config(
    mode: KnowledgeGovernanceMode,
    *,
    policy: MaintenanceGovernancePolicy | None = None,
) -> KnowledgeGovernanceConfig:
    if mode is KnowledgeGovernanceMode.REVIEWED:
        return KnowledgeGovernanceConfig(mode=mode)
    policy = policy or MaintenanceGovernancePolicy(KnowledgeMaintenanceGovernanceDisposition.REJECT)
    return KnowledgeGovernanceConfig(
        mode=mode,
        policy_identity=policy.identity,
        policy_version=policy.version,
    )


async def assert_knowledge_maintenance_governance_conformance(
    store: KnowledgeStore,
    *,
    access_scope: KnowledgeAccessScope,
    prefix: str,
) -> None:
    """Exercise the same authority, replay, and review contract on every backend."""

    reviewed = await maintenance_governance_publication(store, f"{prefix}-reviewed")
    reviewed_governor = KnowledgeMaintenanceGovernor(
        store,
        config=maintenance_governance_config(KnowledgeGovernanceMode.REVIEWED),
    )
    reviewed_receipt = await reviewed_governor.govern(
        operation_id=f"{prefix}-reviewed-route",
        proposal_id=reviewed.proposal.id,
        access_scope=access_scope,
    )
    assert reviewed_receipt.authority.decision.disposition is (
        KnowledgeMaintenanceGovernanceDisposition.ROUTE_TO_REVIEW
    )
    assert reviewed_receipt.maintenance_receipt is None
    assert reviewed_receipt.committed_at >= reviewed.receipt.committed_at
    reviewed_replacement = await store.get_entry(
        reviewed.replacement.id,
        access_scope=access_scope,
    )
    assert reviewed_replacement is not None
    assert reviewed_replacement.status is KnowledgeStatus.PENDING
    reviewed_replay = await reviewed_governor.govern(
        operation_id=reviewed_receipt.operation_id,
        proposal_id=reviewed.proposal.id,
        access_scope=access_scope,
    )
    assert reviewed_replay.replayed is True
    conflicting_route_authority = KnowledgeMaintenanceGovernanceAuthority(
        request=reviewed_receipt.authority.request,
        decision=reviewed_receipt.authority.decision.model_copy(
            update={"code": "changed_review_route"}
        ),
    )
    try:
        await store.record_maintenance_governance_route(
            conflicting_route_authority,
            access_scope=access_scope,
        )
    except KnowledgeMaintenanceConflict as exc:
        assert exc.reason == "governance_operation_reuse"
    else:
        raise AssertionError("Route replay accepted a different policy decision.")

    automatic_after_route_policy = MaintenanceGovernancePolicy(
        KnowledgeMaintenanceGovernanceDisposition.APPROVE
    )
    try:
        await KnowledgeMaintenanceGovernor(
            store,
            config=maintenance_governance_config(
                KnowledgeGovernanceMode.AUTONOMOUS,
                policy=automatic_after_route_policy,
            ),
            policy=automatic_after_route_policy,
        ).govern(
            operation_id=f"{prefix}-automatic-after-reviewed-route",
            proposal_id=reviewed.proposal.id,
            access_scope=access_scope,
        )
    except KnowledgeMaintenanceConflict as exc:
        assert exc.reason == "proposal_already_governed"
    else:
        raise AssertionError("Automatic governance superseded a routed proposal.")
    assert automatic_after_route_policy.calls == 1

    review_receipt = await KnowledgeReviewWorkflow(
        store,
        access_scope=access_scope,
    ).decide_maintenance(
        reviewed.proposal,
        _decision(
            reviewed.proposal,
            kind=KnowledgeMaintenanceDecisionKind.APPROVE,
            suffix=f"{prefix}-explicit-review",
        ),
    )
    assert review_receipt.outcome is KnowledgeMaintenanceOutcome.APPLIED
    assert review_receipt.operation_id != reviewed_receipt.operation_id
    assert (
        await store.load_maintenance_governance_route(
            reviewed_receipt.operation_id,
            access_scope=access_scope,
        )
        == reviewed_receipt
    )

    outcomes = (
        (
            KnowledgeGovernanceMode.POLICY_AUTOMATIC,
            KnowledgeMaintenanceGovernanceDisposition.APPROVE,
            KnowledgeMaintenanceOutcome.APPLIED,
        ),
        (
            KnowledgeGovernanceMode.POLICY_AUTOMATIC,
            KnowledgeMaintenanceGovernanceDisposition.REJECT,
            KnowledgeMaintenanceOutcome.REJECTED,
        ),
        (
            KnowledgeGovernanceMode.POLICY_AUTOMATIC,
            KnowledgeMaintenanceGovernanceDisposition.ROUTE_TO_REVIEW,
            None,
        ),
        (
            KnowledgeGovernanceMode.AUTONOMOUS,
            KnowledgeMaintenanceGovernanceDisposition.APPROVE,
            KnowledgeMaintenanceOutcome.APPLIED,
        ),
        (
            KnowledgeGovernanceMode.AUTONOMOUS,
            KnowledgeMaintenanceGovernanceDisposition.REJECT,
            KnowledgeMaintenanceOutcome.REJECTED,
        ),
        (
            KnowledgeGovernanceMode.AUTONOMOUS,
            KnowledgeMaintenanceGovernanceDisposition.ROUTE_TO_REVIEW,
            None,
        ),
    )
    first_operation_id: str | None = None
    for mode, disposition, expected_outcome in outcomes:
        case = f"{prefix}-{mode.value}-{disposition.value}"
        publication = await maintenance_governance_publication(store, case)
        policy = MaintenanceGovernancePolicy(disposition)
        governor = KnowledgeMaintenanceGovernor(
            store,
            config=maintenance_governance_config(mode, policy=policy),
            policy=policy,
        )
        receipt = await governor.govern(
            operation_id=f"{case}-operation",
            proposal_id=publication.proposal.id,
            access_scope=access_scope,
        )
        request = receipt.authority.request
        assert request.proposal == publication.proposal
        assert request.accepted_plan_fingerprint == publication.accepted_plan.fingerprint
        assert request.routing_request_fingerprint == publication.accepted_plan.request_fingerprint
        assert request.routing_result_fingerprint == (
            publication.accepted_plan.routing_result_fingerprint
        )
        assert request.routing_configuration_fingerprint == (
            publication.accepted_plan.routing_configuration_fingerprint
        )
        assert request.planning_configuration_fingerprint == (
            publication.accepted_plan.configuration_fingerprint
        )
        assert request.plan_fingerprint == publication.accepted_plan.plan.fingerprint
        assert request.evaluation_fingerprint == publication.accepted_plan.evaluation.fingerprint
        assert request.access_scope == access_scope
        assert receipt.authority.decision.disposition is disposition
        assert receipt.authority.decision.policy_identity == policy.identity
        assert receipt.authority.decision.policy_version == policy.version
        if expected_outcome is None:
            assert receipt.maintenance_receipt is None
            pending = await store.get_entry(
                publication.replacement.id,
                access_scope=access_scope,
            )
            assert pending is not None
            assert pending.status is KnowledgeStatus.PENDING
        else:
            assert receipt.maintenance_receipt is not None
            assert receipt.maintenance_receipt.outcome is expected_outcome
            current_replacement = await store.get_entry(
                publication.replacement.id,
                access_scope=access_scope,
            )
            if expected_outcome is KnowledgeMaintenanceOutcome.APPLIED:
                assert current_replacement is not None
                assert current_replacement.status is KnowledgeStatus.ACTIVE
                assert receipt.maintenance_receipt.archived_revisions == [
                    source.model_copy(update={"revision": source.revision + 1})
                    for source in publication.proposal.sources
                ]
                assert len(receipt.maintenance_receipt.relation_ids) == len(
                    publication.proposal.relations
                )
            else:
                assert current_replacement is not None
                assert current_replacement.status is KnowledgeStatus.PENDING
                for source in publication.proposal.sources:
                    current_source = await store.get_entry(
                        source.entry_id,
                        access_scope=access_scope,
                    )
                    assert current_source is not None
                    assert current_source.revision == source.revision
                    assert current_source.status is KnowledgeStatus.ACTIVE

        loaded = await load_knowledge_maintenance_governance_receipt(
            store,
            operation_id=receipt.operation_id,
            access_scope=access_scope,
        )
        assert loaded == receipt
        replay = await governor.govern(
            operation_id=receipt.operation_id,
            proposal_id=publication.proposal.id,
            access_scope=access_scope,
        )
        assert replay.replayed is True
        assert policy.calls == 1

        changed_config = KnowledgeGovernanceConfig(
            mode=mode,
            policy_identity="conformance.changed-policy",
            policy_version="9",
        )
        try:
            await KnowledgeMaintenanceGovernor(
                store,
                config=changed_config,
            ).govern(
                operation_id=receipt.operation_id,
                proposal_id=publication.proposal.id,
                access_scope=access_scope,
            )
        except KnowledgeMaintenanceConflict as exc:
            assert exc.reason == "governance_operation_reuse"
        else:
            raise AssertionError("Replay accepted a different policy identity.")

        if (
            disposition is KnowledgeMaintenanceGovernanceDisposition.ROUTE_TO_REVIEW
            and mode is KnowledgeGovernanceMode.POLICY_AUTOMATIC
        ):
            routed_review = await KnowledgeReviewWorkflow(
                store,
                access_scope=access_scope,
            ).decide_maintenance(
                publication.proposal,
                _decision(
                    publication.proposal,
                    kind=KnowledgeMaintenanceDecisionKind.APPROVE,
                    suffix=f"{case}-explicit-review",
                ),
            )
            assert routed_review.outcome is KnowledgeMaintenanceOutcome.APPLIED
            assert routed_review.operation_id != receipt.operation_id
            assert (
                await store.load_maintenance_governance_route(
                    receipt.operation_id,
                    access_scope=access_scope,
                )
                == receipt
            )

        if first_operation_id is None:
            first_operation_id = receipt.operation_id

    assert first_operation_id is not None
    conflicting = await maintenance_governance_publication(store, f"{prefix}-operation-conflict")
    try:
        await KnowledgeMaintenanceGovernor(
            store,
            config=maintenance_governance_config(KnowledgeGovernanceMode.REVIEWED),
        ).govern(
            operation_id=first_operation_id,
            proposal_id=conflicting.proposal.id,
            access_scope=access_scope,
        )
    except KnowledgeMaintenanceConflict as exc:
        assert exc.reason == "governance_operation_reuse"
    else:
        raise AssertionError("A governance operation was reused for a different proposal.")

    forged_route_publication = await maintenance_governance_publication(
        store,
        f"{prefix}-forged-route",
    )
    route_request = prepare_knowledge_maintenance_governance_request(
        forged_route_publication,
        operation_id=f"{prefix}-forged-route-operation",
        mode=KnowledgeGovernanceMode.REVIEWED,
    )
    route_authority = await decide_knowledge_maintenance_governance(
        route_request,
        config=maintenance_governance_config(KnowledgeGovernanceMode.REVIEWED),
    )
    forged_request = route_request.model_copy(
        update={"routing_configuration_fingerprint": "0" * 64}
    )
    forged_route_authority = KnowledgeMaintenanceGovernanceAuthority(
        request=forged_request,
        decision=route_authority.decision.model_copy(
            update={"request_sha256": forged_request.fingerprint}
        ),
    )
    try:
        await store.record_maintenance_governance_route(
            forged_route_authority,
            access_scope=access_scope,
        )
    except KnowledgeMaintenanceConflict as exc:
        assert exc.reason == "governance_request_mismatch"
    else:
        raise AssertionError("The store accepted governance over forged planning records.")
    assert (
        await store.load_maintenance_governance_route(
            forged_request.operation_id,
            access_scope=access_scope,
        )
        is None
    )

    forged_terminal_publication = await maintenance_governance_publication(
        store,
        f"{prefix}-forged-terminal",
    )
    terminal_policy = MaintenanceGovernancePolicy(KnowledgeMaintenanceGovernanceDisposition.APPROVE)
    terminal_request = prepare_knowledge_maintenance_governance_request(
        forged_terminal_publication,
        operation_id=f"{prefix}-forged-terminal-operation",
        mode=KnowledgeGovernanceMode.AUTONOMOUS,
    )
    terminal_authority = await decide_knowledge_maintenance_governance(
        terminal_request,
        config=maintenance_governance_config(
            KnowledgeGovernanceMode.AUTONOMOUS,
            policy=terminal_policy,
        ),
        policy=terminal_policy,
    )
    terminal_decision = maintenance_decision_from_governance(
        terminal_authority,
        decided_at=forged_terminal_publication.receipt.committed_at,
    )
    forged_values = terminal_decision.model_dump(mode="python")
    forged_values["metadata"]["cayu_knowledge_maintenance_governance"]["decision"][
        "request_sha256"
    ] = "0" * 64
    forged_terminal_decision = KnowledgeMaintenanceDecision.model_validate(forged_values)
    try:
        await store.apply_maintenance_decision(
            forged_terminal_publication.proposal,
            forged_terminal_decision,
            access_scope=access_scope,
        )
    except KnowledgeMaintenanceConflict as exc:
        assert exc.reason == "malformed_governance_attribution"
    else:
        raise AssertionError("The store accepted forged automatic-governance attribution.")
    assert (
        await store.load_maintenance_decision_receipt(
            forged_terminal_decision.operation_id,
            access_scope=access_scope,
        )
        is None
    )
    untouched = await store.get_entry(
        forged_terminal_publication.replacement.id,
        access_scope=access_scope,
    )
    assert untouched is not None
    assert untouched.status is KnowledgeStatus.PENDING

    concurrent_publication = await maintenance_governance_publication(
        store,
        f"{prefix}-concurrent-conflict",
    )
    approve_policy = MaintenanceGovernancePolicy(KnowledgeMaintenanceGovernanceDisposition.APPROVE)
    reject_policy = MaintenanceGovernancePolicy(KnowledgeMaintenanceGovernanceDisposition.REJECT)
    concurrent_config = maintenance_governance_config(
        KnowledgeGovernanceMode.AUTONOMOUS,
        policy=approve_policy,
    )
    concurrent_operation = f"{prefix}-concurrent-conflict-operation"
    results = await asyncio.gather(
        KnowledgeMaintenanceGovernor(
            store,
            config=concurrent_config,
            policy=approve_policy,
        ).govern(
            operation_id=concurrent_operation,
            proposal_id=concurrent_publication.proposal.id,
            access_scope=access_scope,
        ),
        KnowledgeMaintenanceGovernor(
            store,
            config=concurrent_config,
            policy=reject_policy,
        ).govern(
            operation_id=concurrent_operation,
            proposal_id=concurrent_publication.proposal.id,
            access_scope=access_scope,
        ),
        return_exceptions=True,
    )
    successes = [result for result in results if not isinstance(result, BaseException)]
    errors = [result for result in results if isinstance(result, BaseException)]
    assert successes
    assert all(isinstance(error, KnowledgeMaintenanceConflict) for error in errors)
    committed = await load_knowledge_maintenance_governance_receipt(
        store,
        operation_id=concurrent_operation,
        access_scope=access_scope,
    )
    assert committed is not None
    assert all(result.authority == committed.authority for result in successes)
    assert approve_policy.calls + reject_policy.calls in (1, 2)

    identical_publication = await maintenance_governance_publication(
        store,
        f"{prefix}-concurrent-identical",
    )
    identical_policy = MaintenanceGovernancePolicy(
        KnowledgeMaintenanceGovernanceDisposition.APPROVE
    )
    identical_governor = KnowledgeMaintenanceGovernor(
        store,
        config=maintenance_governance_config(
            KnowledgeGovernanceMode.AUTONOMOUS,
            policy=identical_policy,
        ),
        policy=identical_policy,
    )
    identical_operation = f"{prefix}-concurrent-identical-operation"
    identical_receipts = await asyncio.gather(
        *(
            identical_governor.govern(
                operation_id=identical_operation,
                proposal_id=identical_publication.proposal.id,
                access_scope=access_scope,
            )
            for _ in range(2)
        )
    )
    assert identical_receipts[0].authority == identical_receipts[1].authority
    assert {receipt.committed_at for receipt in identical_receipts} == {
        identical_receipts[0].committed_at
    }
    assert sum(receipt.replayed for receipt in identical_receipts) == 1
    assert identical_policy.calls in (1, 2)
