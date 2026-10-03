"""Shared activation authority, approval receipts and exact replay rules."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes
from cayu.knowledge import _revision_rules
from cayu.knowledge.activation_contracts import (
    KnowledgeActivationAuthority,
    KnowledgeActivationDisposition,
    KnowledgeActivationReceipt,
    KnowledgeActivationSource,
    KnowledgeGovernanceMode,
    KnowledgeReviewApproval,
    copy_knowledge_activation_authority,
    copy_knowledge_activation_receipt,
)
from cayu.knowledge.publication_contracts import (
    KnowledgePublicationReceipt,
    copy_knowledge_publication_receipt,
)
from cayu.knowledge.records import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeStatus,
    copy_knowledge_chunk,
    copy_knowledge_entry,
    copy_knowledge_evidence,
)
from cayu.knowledge.scopes import KnowledgeAccessScope, knowledge_access_scope_sha256


def _validate_review_approval_authority(
    authority: KnowledgeActivationAuthority,
    *,
    access_scope: KnowledgeAccessScope,
) -> None:
    request = authority.request
    decision = authority.decision
    if (
        request.mode is not KnowledgeGovernanceMode.REVIEWED
        or request.source is not KnowledgeActivationSource.REVIEW_APPROVAL
        or decision.disposition is not KnowledgeActivationDisposition.ACTIVATE
        or request.access_scope_sha256 != knowledge_access_scope_sha256(access_scope)
    ):
        raise ValueError("Reviewed activation authority is invalid for this operation.")


def _validate_review_approval_scope(
    entry: KnowledgeEntry,
    *,
    expected_namespace: str | None,
    expected_labels: dict[str, str],
) -> None:
    """Require one fresh or replayed approval to remain in its review scope."""

    if expected_namespace is not None and entry.namespace != expected_namespace:
        raise ValueError("Knowledge entry does not match expected namespace.")
    for key, value in expected_labels.items():
        if entry.labels.get(key) != value:
            raise ValueError("Knowledge entry does not match expected labels.")


def _activation_receipt_matches(
    receipt: KnowledgeActivationReceipt,
    *,
    authority: KnowledgeActivationAuthority,
    publication_request_sha256: str,
    publication_committed_at: datetime,
) -> bool:
    try:
        copied = copy_knowledge_activation_receipt(receipt)
        return (
            copied.operation_id == authority.request.operation_id
            and copied.expected_revision == authority.request.expected_revision
            and copied.entry_id == authority.request.candidate_entry.id
            and copied.entry_revision == authority.request.target_revision
            and copied.publication_request_sha256 == publication_request_sha256
            and copied.committed_at == publication_committed_at
            and copied.authority == authority
        )
    except (TypeError, ValueError):
        return False


def _review_approval_publication_request_sha256(
    before: KnowledgeEntry,
    after: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    evidence: list[KnowledgeEvidence],
    authority: KnowledgeActivationAuthority,
) -> str:
    return sha256(
        canonical_durable_json_bytes(
            {
                "contract": "cayu-knowledge-reviewed-activation-v1",
                "before": before.model_dump(mode="json"),
                "after": after.model_dump(mode="json"),
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "activation_authority": authority.model_dump(mode="json"),
            },
            "reviewed knowledge activation",
        )
    ).hexdigest()


def _prepare_review_approval_receipts(
    before: KnowledgeEntry,
    after: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    evidence: list[KnowledgeEvidence],
    authority: KnowledgeActivationAuthority,
    *,
    committed_at: datetime,
) -> tuple[KnowledgePublicationReceipt, KnowledgeActivationReceipt]:
    request_sha256 = _review_approval_publication_request_sha256(
        before,
        after,
        chunks,
        evidence,
        authority,
    )
    publication = KnowledgePublicationReceipt(
        operation_id=authority.request.operation_id,
        entry_id=after.id,
        entry_revision=after.revision,
        expected_revision=before.revision,
        request_sha256=request_sha256,
        entry_created_at=after.created_at,
        entry_updated_at=after.updated_at,
        committed_at=committed_at,
    )
    activation = KnowledgeActivationReceipt(
        operation_id=authority.request.operation_id,
        entry_id=after.id,
        entry_revision=after.revision,
        expected_revision=before.revision,
        publication_request_sha256=request_sha256,
        authority=authority,
        committed_at=committed_at,
    )
    return publication, activation


def _review_approval_receipts_match(
    publication: KnowledgePublicationReceipt,
    activation: KnowledgeActivationReceipt,
    *,
    authority: KnowledgeActivationAuthority,
    before: KnowledgeEntry,
    after: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    evidence: list[KnowledgeEvidence],
) -> bool:
    try:
        publication = copy_knowledge_publication_receipt(publication)
        activation = copy_knowledge_activation_receipt(activation)
        expected_publication, expected_activation = _prepare_review_approval_receipts(
            before,
            after,
            chunks,
            evidence,
            authority,
            committed_at=activation.committed_at,
        )
        return (
            publication == expected_publication
            and activation == expected_activation
            and publication.committed_at == activation.committed_at
        )
    except (TypeError, ValueError):
        return False


def _replay_review_approval_from_receipts(
    publication: KnowledgePublicationReceipt,
    activation: KnowledgeActivationReceipt,
    *,
    authority: KnowledgeActivationAuthority,
) -> KnowledgeReviewApproval | None:
    """Authenticate and reconstruct an exact reviewed approval from durable receipts."""

    try:
        publication = copy_knowledge_publication_receipt(publication, replayed=False)
        activation = copy_knowledge_activation_receipt(activation, replayed=False)
        authority = copy_knowledge_activation_authority(authority)
        request = authority.request
        if (
            request.mode is not KnowledgeGovernanceMode.REVIEWED
            or request.source is not KnowledgeActivationSource.REVIEW_APPROVAL
            or authority.decision.disposition is not KnowledgeActivationDisposition.ACTIVATE
        ):
            return None
        before = copy_knowledge_entry(request.candidate_entry)
        before_chunks = [copy_knowledge_chunk(chunk) for chunk in request.chunks]
        before_evidence = [copy_knowledge_evidence(item) for item in request.evidence]
        after = before.model_copy(
            update={
                "revision": request.target_revision,
                "status": KnowledgeStatus.ACTIVE,
                "updated_at": publication.entry_updated_at,
            }
        )
        target_chunks = (
            [_revision_rules._default_chunk_for_entry(after)]
            if _revision_rules._has_only_default_chunk(before, before_chunks)
            else _revision_rules._copy_chunks_for_revision(before_chunks, after)
        )
        target_evidence = _revision_rules._copy_evidence_for_revision(
            before_evidence,
            entry=after,
            previous_chunks=before_chunks,
            chunks=target_chunks,
        )
        if not _review_approval_receipts_match(
            publication,
            activation,
            authority=authority,
            before=before,
            after=after,
            chunks=target_chunks,
            evidence=target_evidence,
        ):
            return None
        return KnowledgeReviewApproval(
            entry=copy_knowledge_entry(after),
            receipt=copy_knowledge_activation_receipt(activation, replayed=True),
        )
    except (RuntimeError, TypeError, ValueError):
        return None
