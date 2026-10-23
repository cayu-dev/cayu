"""Reviewed rejection through the existing atomic revision-publication owner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_metadata,
    require_durable_clean_nonblank,
    require_durable_nonblank,
)
from cayu.knowledge._revision_rules import _copy_chunks_for_revision, _copy_evidence_for_revision
from cayu.knowledge.activation_contracts import (
    MAX_KNOWLEDGE_ACTIVATION_CHUNKS,
    MAX_KNOWLEDGE_ACTIVATION_EVIDENCE_RECORDS,
    MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES,
)
from cayu.knowledge.publication_contracts import (
    KnowledgePublicationConflict,
    KnowledgePublicationReceipt,
    copy_knowledge_publication_receipt,
    prepare_knowledge_publication,
)
from cayu.knowledge.records import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeStatus,
    copy_knowledge_entry,
)

if TYPE_CHECKING:
    from cayu.storage.knowledge_review import KnowledgeReviewWorkflow

REJECTION_METADATA_KEY = "cayu:review_rejection"


class KnowledgeRejectionDecision(BaseModel):
    """Caller attribution and rationale bound to one archived successor revision."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    reviewer_identity: str = Field(max_length=256)
    reviewer_version: str = Field(max_length=256)
    reason: str | None = Field(default=None, max_length=4096)
    annotations: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reviewer_identity", "reviewer_version")
    @classmethod
    def clean_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def clean_reason(cls, value: str | None) -> str | None:
        return None if value is None else require_durable_nonblank(value, "reason")

    @field_validator("annotations", mode="before")
    @classmethod
    def copy_annotations(cls, value: Any) -> dict[str, Any]:
        return copy_durable_metadata(value, "reviewer_annotations")


@dataclass(frozen=True, slots=True)
class KnowledgeReviewRejection:
    entry: KnowledgeEntry
    decision: KnowledgeRejectionDecision
    receipt: KnowledgePublicationReceipt


async def _prepare(
    workflow: KnowledgeReviewWorkflow,
    entry_id: str,
    revision: int,
    decision: KnowledgeRejectionDecision,
    updated_at: datetime,
    operation_id: str,
) -> tuple[str, KnowledgeEntry, list[KnowledgeChunk], list[KnowledgeEvidence], str]:
    proposal = await workflow.get_entry(entry_id, revision=revision)
    if proposal is not None:
        proposal = copy_knowledge_entry(proposal)
    if proposal is None or proposal.status is not KnowledgeStatus.PENDING:
        raise KnowledgePublicationConflict("proposal_unavailable")
    source_chunks = await workflow.store.read_chunks(
        proposal.id,
        revision=proposal.revision,
        access_scope=workflow.access_scope,
        # Retrieval truncates rather than reporting overflow. Probe one extra
        # chunk and four extra bytes (the maximum UTF-8 codepoint width). Any
        # truncated prefix must then exceed the accepted bound, even if its last
        # incomplete codepoint was omitted. Accepted material is therefore whole.
        max_chunks=MAX_KNOWLEDGE_ACTIVATION_CHUNKS + 1,
        max_bytes=MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES + 4,
    )
    if (
        len(source_chunks) > MAX_KNOWLEDGE_ACTIVATION_CHUNKS
        or sum(len(chunk.text.encode("utf-8")) for chunk in source_chunks)
        > MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES
    ):
        raise ValueError("Pending knowledge chunks exceed the review publication bound.")
    source_evidence = await workflow.store.read_evidence(
        proposal.id,
        revision=proposal.revision,
        access_scope=workflow.access_scope,
        max_records=MAX_KNOWLEDGE_ACTIVATION_EVIDENCE_RECORDS,
        max_bytes=MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES,
    )
    if source_evidence is None or source_evidence.truncated:
        raise ValueError("Pending knowledge evidence exceeds the review publication bound.")
    successor = KnowledgeEntry.model_validate(
        {
            **proposal.model_dump(),
            "revision": revision + 1,
            "status": KnowledgeStatus.ARCHIVED,
            "updated_at": max(updated_at, proposal.updated_at),
            "metadata": {
                **proposal.metadata,
                REJECTION_METADATA_KEY: decision.model_dump(mode="json"),
            },
        }
    )
    chunks = _copy_chunks_for_revision(source_chunks, successor)
    evidence = _copy_evidence_for_revision(
        source_evidence.evidence, entry=successor, previous_chunks=source_chunks, chunks=chunks
    )
    return prepare_knowledge_publication(
        successor, chunks, evidence=evidence, operation_id=operation_id, expected_revision=revision
    )


async def load_review_rejection(
    workflow: KnowledgeReviewWorkflow, entry_id: str, revision: int, operation_id: str
) -> KnowledgeReviewRejection | None:
    raw = await workflow.store.load_entry_publication_receipt(
        operation_id, access_scope=workflow.access_scope
    )
    if raw is None:
        return None
    receipt = copy_knowledge_publication_receipt(raw)
    if (
        receipt.operation_id != operation_id
        or receipt.entry_id != entry_id
        or receipt.expected_revision != revision
        or receipt.entry_revision != revision + 1
    ):
        raise KnowledgePublicationConflict("operation_mismatch")
    successor = await workflow.get_entry(entry_id, revision=revision + 1)
    if successor is not None:
        successor = copy_knowledge_entry(successor)
    if successor is None or successor.status is not KnowledgeStatus.ARCHIVED:
        raise KnowledgePublicationConflict("rejection_unavailable")
    try:
        decision = KnowledgeRejectionDecision.model_validate(
            successor.metadata.get(REJECTION_METADATA_KEY)
        )
        _, expected, _chunks, _evidence, digest = await _prepare(
            workflow, entry_id, revision, decision, receipt.entry_updated_at, operation_id
        )
    except (TypeError, ValueError):
        raise KnowledgePublicationConflict("malformed_rejection") from None
    if (
        digest != receipt.request_sha256
        or expected != successor
        or receipt.entry_created_at != successor.created_at
        or receipt.entry_updated_at != successor.updated_at
    ):
        raise KnowledgePublicationConflict("operation_mismatch")
    return KnowledgeReviewRejection(copy_knowledge_entry(successor), decision, receipt)


def _same_decision(left: KnowledgeRejectionDecision, right: KnowledgeRejectionDecision) -> bool:
    return canonical_durable_json_bytes(
        left.model_dump(mode="json"), "knowledge rejection decision"
    ) == canonical_durable_json_bytes(right.model_dump(mode="json"), "knowledge rejection decision")


async def reject_reviewed(
    workflow: KnowledgeReviewWorkflow,
    entry_id: str,
    revision: int,
    operation_id: str,
    decision: KnowledgeRejectionDecision,
) -> KnowledgeReviewRejection:
    # Revalidate before serialization: post-construction mutation must not be
    # serialized into warnings or alter the exact request behind the caller.
    if type(decision) is not KnowledgeRejectionDecision:
        raise TypeError("Expected a KnowledgeRejectionDecision.")
    decision = KnowledgeRejectionDecision(
        reviewer_identity=decision.reviewer_identity,
        reviewer_version=decision.reviewer_version,
        reason=decision.reason,
        annotations=decision.annotations,
    )
    prior = await load_review_rejection(workflow, entry_id, revision, operation_id)
    if prior is not None:
        if not _same_decision(prior.decision, decision):
            raise KnowledgePublicationConflict("operation_mismatch")
        return KnowledgeReviewRejection(
            prior.entry,
            prior.decision,
            copy_knowledge_publication_receipt(prior.receipt, replayed=True),
        )
    if not callable(getattr(workflow.store, "publish_entry_revision", None)):
        raise NotImplementedError("Audited rejection requires atomic revision publication.")
    _, successor, chunks, evidence, digest = await _prepare(
        workflow, entry_id, revision, decision, datetime.now(UTC), operation_id
    )
    try:
        receipt = await workflow.store.publish_entry_revision(
            successor,
            chunks,
            evidence=evidence,
            access_scope=workflow.access_scope,
            operation_id=operation_id,
            expected_revision=revision,
        )
    except KnowledgePublicationConflict:
        # Concurrent identical decisions can choose different server timestamps.
        # Reconcile only a fully verified receipt with the same caller tuple.
        prior = await load_review_rejection(workflow, entry_id, revision, operation_id)
        if prior is None or not _same_decision(prior.decision, decision):
            raise
        return KnowledgeReviewRejection(
            prior.entry,
            prior.decision,
            copy_knowledge_publication_receipt(prior.receipt, replayed=True),
        )
    receipt = copy_knowledge_publication_receipt(receipt)
    if receipt.request_sha256 != digest:
        raise KnowledgePublicationConflict("operation_mismatch")
    result = await load_review_rejection(workflow, entry_id, revision, operation_id)
    if result is None:
        raise KnowledgePublicationConflict("missing_receipt")
    return result
