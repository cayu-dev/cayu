from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from tests.core.knowledge_maintenance_conformance import (
    maintenance_decision,
    maintenance_proposal,
)

import cayu
from cayu.storage import (
    MAX_KNOWLEDGE_MAINTENANCE_METADATA_BYTES,
    MAX_KNOWLEDGE_MAINTENANCE_SOURCES,
    MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES,
    MAX_KNOWLEDGE_REVISION,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeMaintenanceDecision,
    KnowledgeMaintenanceDecisionKind,
    KnowledgeMaintenanceDecisionReceipt,
    KnowledgeMaintenanceOutcome,
    KnowledgeMaintenanceProposal,
    KnowledgeRelationKind,
    KnowledgeRevisionRef,
    prepare_knowledge_maintenance_decision,
)

_NOW = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)


def test_knowledge_maintenance_public_contract_is_exported() -> None:
    for name in (
        "KnowledgeMaintenanceConflict",
        "KnowledgeMaintenanceDecision",
        "KnowledgeMaintenanceDecisionKind",
        "KnowledgeMaintenanceDecisionReceipt",
        "KnowledgeMaintenanceOutcome",
        "KnowledgeMaintenanceProposal",
        "KnowledgeMaintenanceStale",
        "prepare_knowledge_maintenance_decision",
    ):
        assert name in cayu.__all__
        assert getattr(cayu, name) is not None


def test_knowledge_maintenance_models_copy_and_fingerprint_nested_material() -> None:
    proposal = maintenance_proposal("model-copy")
    material = proposal.model_dump(mode="python")
    copied = KnowledgeMaintenanceProposal.model_validate(material)
    material["metadata"]["safe_code"] = "mutated"
    material["relations"][0]["metadata"]["safe_code"] = "mutated"
    assert copied == proposal
    assert copied.fingerprint == proposal.fingerprint
    assert len(proposal.fingerprint) == 64


def test_knowledge_maintenance_models_reject_nested_model_subclasses() -> None:
    class PrivateRevisionRef(KnowledgeRevisionRef):
        pass

    class PrivateAccessScope(KnowledgeAccessScope):
        pass

    proposal = maintenance_proposal("model-subclasses")
    proposal_material = proposal.model_dump(mode="python")
    private_replacement = PrivateRevisionRef.model_validate(
        proposal.replacement.model_dump(mode="python")
    )
    private_source = PrivateRevisionRef.model_validate(
        proposal.sources[0].model_dump(mode="python")
    )
    private_scope = PrivateAccessScope.model_validate(
        proposal.access_scope.model_dump(mode="python")
    )

    for field, value in (
        ("replacement", private_replacement),
        ("sources", [private_source]),
        ("access_scope", private_scope),
    ):
        with pytest.raises(TypeError, match="instances must not be subclasses"):
            KnowledgeMaintenanceProposal.model_validate({**proposal_material, field: value})

    decision = maintenance_decision(
        proposal,
        operation_id="model-subclasses-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    )
    _, _, request_sha256 = prepare_knowledge_maintenance_decision(proposal, decision)
    receipt_material = {
        "operation_id": decision.operation_id,
        "proposal_id": proposal.id,
        "proposal_fingerprint": proposal.fingerprint,
        "request_sha256": request_sha256,
        "outcome": KnowledgeMaintenanceOutcome.APPLIED,
        "replacement": proposal.replacement.model_copy(update={"revision": 2}),
        "archived_revisions": [proposal.sources[0].model_copy(update={"revision": 2})],
        "relation_ids": [proposal.relations[0].id],
        "committed_at": decision.decided_at,
    }
    for field, value in (
        ("replacement", private_replacement.model_copy(update={"revision": 2})),
        ("archived_revisions", [private_source.model_copy(update={"revision": 2})]),
    ):
        with pytest.raises(TypeError, match="KnowledgeRevisionRef instances"):
            KnowledgeMaintenanceDecisionReceipt.model_validate({**receipt_material, field: value})


def test_knowledge_maintenance_proposal_rejects_unreviewed_relation_endpoints() -> None:
    proposal = maintenance_proposal("wrong-endpoint")
    relation = proposal.relations[0].model_copy(
        update={"object": proposal.sources[0].model_copy(update={"entry_id": "not-reviewed"})}
    )
    with pytest.raises(ValidationError, match="reviewed source"):
        KnowledgeMaintenanceProposal.model_validate(
            {**proposal.model_dump(mode="python"), "relations": [relation]}
        )


def test_knowledge_maintenance_proposal_requires_active_successor_relations() -> None:
    proposal = maintenance_proposal("pending-relation")
    relation = proposal.relations[0].model_copy(update={"subject": proposal.replacement})
    with pytest.raises(ValidationError, match="active replacement"):
        KnowledgeMaintenanceProposal.model_validate(
            {**proposal.model_dump(mode="python"), "relations": [relation]}
        )


def test_knowledge_maintenance_proposal_rejects_exhausted_superseded_source() -> None:
    proposal = maintenance_proposal("exhausted-source")
    exhausted = proposal.sources[0].model_copy(update={"revision": MAX_KNOWLEDGE_REVISION})
    relation = proposal.relations[0].model_copy(update={"object": exhausted})
    with pytest.raises(ValidationError, match="superseded source revision.*advance"):
        KnowledgeMaintenanceProposal.model_validate(
            {
                **proposal.model_dump(mode="python"),
                "sources": [exhausted],
                "relations": [relation],
            }
        )


def test_knowledge_maintenance_proposal_requires_one_disposition_per_source() -> None:
    proposal = maintenance_proposal("ambiguous-disposition")
    second = proposal.relations[0].model_copy(
        update={
            "id": "ambiguous-disposition-derived",
            "kind": KnowledgeRelationKind.DERIVED_FROM,
        }
    )
    with pytest.raises(ValidationError, match="exactly one maintenance disposition"):
        KnowledgeMaintenanceProposal.model_validate(
            {**proposal.model_dump(mode="python"), "relations": [*proposal.relations, second]}
        )


def test_knowledge_maintenance_temporal_authority_cannot_run_backwards() -> None:
    proposal = maintenance_proposal("temporal-authority")
    future_relation = proposal.relations[0].model_copy(
        update={"created_at": proposal.created_at.replace(year=proposal.created_at.year + 1)}
    )
    with pytest.raises(ValidationError, match="cannot postdate"):
        KnowledgeMaintenanceProposal.model_validate(
            {**proposal.model_dump(mode="python"), "relations": [future_relation]}
        )

    decision = maintenance_decision(
        proposal,
        operation_id="temporal-authority-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    ).model_copy(update={"decided_at": proposal.created_at.replace(year=2025)})
    with pytest.raises(ValueError, match="cannot predate"):
        prepare_knowledge_maintenance_decision(proposal, decision)


def test_knowledge_maintenance_proposal_enforces_source_and_payload_bounds() -> None:
    proposal = maintenance_proposal("proposal-bounds")
    material = proposal.model_dump(mode="python")
    for schema_version in (True, 1.0, "1", 0, 2):
        with pytest.raises(ValidationError, match="schema_version.*integer 1"):
            KnowledgeMaintenanceProposal.model_validate(
                {**material, "schema_version": schema_version}
            )
    with pytest.raises(ValidationError, match="sources.*between 1"):
        KnowledgeMaintenanceProposal.model_validate(
            {
                **material,
                "sources": [
                    {"entry_id": f"bounded-source-{index}", "revision": 1}
                    for index in range(MAX_KNOWLEDGE_MAINTENANCE_SOURCES + 1)
                ],
            }
        )
    with pytest.raises(ValidationError, match="rationale.*at most"):
        KnowledgeMaintenanceProposal.model_validate(
            {
                **material,
                "rationale": "x" * (MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES + 1),
            }
        )
    with pytest.raises(ValidationError, match="metadata.*budget"):
        KnowledgeMaintenanceProposal.model_validate(
            {
                **material,
                "metadata": {"value": "x" * MAX_KNOWLEDGE_MAINTENANCE_METADATA_BYTES},
            }
        )


def test_knowledge_maintenance_decision_rejects_model_authority_and_unbounded_reason() -> None:
    proposal = maintenance_proposal("reviewer-authority")
    base = maintenance_decision(
        proposal,
        operation_id="reviewer-authority-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    ).model_dump(mode="python")
    for schema_version in (True, 1.0, "1", 0, 2):
        with pytest.raises(ValidationError, match="schema_version.*integer 1"):
            KnowledgeMaintenanceDecision.model_validate({**base, "schema_version": schema_version})
    with pytest.raises(ValidationError, match="Model output cannot authorize"):
        KnowledgeMaintenanceDecision.model_validate(
            {**base, "reviewer_type": KnowledgeActorType.MODEL}
        )
    with pytest.raises(ValidationError, match="reason.*at most"):
        KnowledgeMaintenanceDecision.model_validate(
            {**base, "reason": "x" * (MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES + 1)}
        )


def test_knowledge_maintenance_receipt_enforces_source_bounds() -> None:
    proposal = maintenance_proposal("receipt-bounds")
    decision = maintenance_decision(
        proposal,
        operation_id="receipt-bounds-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    )
    _, _, request_sha256 = prepare_knowledge_maintenance_decision(proposal, decision)
    material = {
        "operation_id": decision.operation_id,
        "proposal_id": proposal.id,
        "proposal_fingerprint": proposal.fingerprint,
        "request_sha256": request_sha256,
        "outcome": KnowledgeMaintenanceOutcome.APPLIED,
        "replacement": proposal.replacement.model_copy(update={"revision": 2}),
        "archived_revisions": [],
        "relation_ids": [proposal.relations[0].id],
        "committed_at": decision.decided_at,
    }
    with pytest.raises(ValidationError, match="archived_revisions.*source bound"):
        KnowledgeMaintenanceDecisionReceipt.model_validate(
            {
                **material,
                "archived_revisions": [
                    {"entry_id": f"bounded-archive-{index}", "revision": 2}
                    for index in range(MAX_KNOWLEDGE_MAINTENANCE_SOURCES + 1)
                ],
            }
        )
    with pytest.raises(ValidationError, match="relation_ids.*source bound"):
        KnowledgeMaintenanceDecisionReceipt.model_validate(
            {
                **material,
                "relation_ids": [
                    f"bounded-relation-{index}"
                    for index in range(MAX_KNOWLEDGE_MAINTENANCE_SOURCES + 1)
                ],
            }
        )
    with pytest.raises(ValidationError, match="archived_revisions.*logical entry"):
        KnowledgeMaintenanceDecisionReceipt.model_validate(
            {
                **material,
                "archived_revisions": [
                    {"entry_id": "duplicate-archive", "revision": 1},
                    {"entry_id": "duplicate-archive", "revision": 2},
                ],
                "relation_ids": ["duplicate-relation-1", "duplicate-relation-2"],
            }
        )
    with pytest.raises(ValidationError, match="replacement.*archived"):
        KnowledgeMaintenanceDecisionReceipt.model_validate(
            {
                **material,
                "archived_revisions": [
                    {
                        "entry_id": proposal.replacement.entry_id,
                        "revision": proposal.replacement.revision + 2,
                    }
                ],
            }
        )
    with pytest.raises(ValidationError, match="cannot outnumber approved relations"):
        KnowledgeMaintenanceDecisionReceipt.model_validate(
            {
                **material,
                "archived_revisions": [
                    {"entry_id": "archive-one", "revision": 2},
                    {"entry_id": "archive-two", "revision": 2},
                ],
            }
        )


def test_prepare_knowledge_maintenance_decision_binds_exact_proposal() -> None:
    proposal = maintenance_proposal(
        "decision-binding",
        kind=KnowledgeRelationKind.DERIVED_FROM,
    )
    decision = maintenance_decision(
        proposal,
        operation_id="decision-binding-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    )
    copied_proposal, copied_decision, request_sha256 = prepare_knowledge_maintenance_decision(
        proposal, decision
    )
    assert copied_proposal == proposal
    assert copied_decision == decision
    assert len(request_sha256) == 64

    changed = proposal.model_copy(update={"rationale": "Another reviewed rationale."})
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_knowledge_maintenance_decision(changed, decision)


def test_knowledge_maintenance_models_have_stable_json_round_trips() -> None:
    proposal = maintenance_proposal("stable-round-trip")
    decision = maintenance_decision(
        proposal,
        operation_id="stable-round-trip-operation",
        kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    )
    _, _, request_sha256 = prepare_knowledge_maintenance_decision(proposal, decision)
    receipt = KnowledgeMaintenanceDecisionReceipt(
        operation_id=decision.operation_id,
        proposal_id=proposal.id,
        proposal_fingerprint=proposal.fingerprint,
        request_sha256=request_sha256,
        outcome=KnowledgeMaintenanceOutcome.APPLIED,
        replacement=proposal.replacement.model_copy(
            update={"revision": proposal.replacement.revision + 1}
        ),
        archived_revisions=[
            proposal.sources[0].model_copy(update={"revision": proposal.sources[0].revision + 1})
        ],
        relation_ids=[proposal.relations[0].id],
        committed_at=decision.decided_at,
    )
    for model, model_type in (
        (proposal, KnowledgeMaintenanceProposal),
        (decision, KnowledgeMaintenanceDecision),
        (receipt, KnowledgeMaintenanceDecisionReceipt),
    ):
        assert model_type.model_validate_json(model.model_dump_json()) == model
