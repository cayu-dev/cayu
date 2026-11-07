"""Reviewed rejection and archive through the existing atomic revision-publication owner.

A reviewed rejection archives a pending proposal; a reviewed archive takes a kept (active)
entry out of use. Both publish an archived successor revision that carries the reviewer's
decision in its metadata, bound to the publication receipt of the operation that wrote it.
"""

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
ARCHIVE_METADATA_KEY = "cayu:review_archive"
#: The decision annotation a reviewed drop through the knowledge routes names its operation
#: in, so the inbox can find and verify that operation's receipt.
DROP_OPERATION_ANNOTATION = "operation_id"
#: The most chunk text a reviewed archive copies into the archived revision. A kept entry
#: can be far larger than a proposal (an indexed document keeps up to 1,000 chunks of
#: about 4,000 bytes by default), so the archive isn't held to the proposal bound.
MAX_KNOWLEDGE_ARCHIVE_MATERIAL_BYTES = 64 * 1024 * 1024


class KnowledgeReviewMaterialTooLarge(ValueError):
    """The entry's chunks or evidence exceed what a reviewed decision copies."""


@dataclass(frozen=True, slots=True)
class _ReviewedKind:
    """What a reviewed decision archives, where it records itself, and how much it copies."""

    from_status: KnowledgeStatus
    metadata_key: str
    max_bytes: int
    unavailable: str


_REJECTION = _ReviewedKind(
    KnowledgeStatus.PENDING,
    REJECTION_METADATA_KEY,
    MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES,
    "proposal_unavailable",
)
_ARCHIVE = _ReviewedKind(
    KnowledgeStatus.ACTIVE,
    ARCHIVE_METADATA_KEY,
    MAX_KNOWLEDGE_ARCHIVE_MATERIAL_BYTES,
    "entry_unavailable",
)


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
    kind: _ReviewedKind = _REJECTION,
) -> tuple[str, KnowledgeEntry, list[KnowledgeChunk], list[KnowledgeEvidence], str]:
    proposal = await workflow.get_entry(entry_id, revision=revision)
    if proposal is not None:
        proposal = copy_knowledge_entry(proposal)
    if proposal is None or proposal.status is not kind.from_status:
        raise KnowledgePublicationConflict(kind.unavailable)
    source_chunks = await workflow.store.read_chunks(
        proposal.id,
        revision=proposal.revision,
        access_scope=workflow.access_scope,
        # Retrieval truncates rather than reporting overflow. Probe one extra
        # chunk and four extra bytes (the maximum UTF-8 codepoint width). Any
        # truncated prefix must then exceed the accepted bound, even if its last
        # incomplete codepoint was omitted. Accepted material is therefore whole.
        max_chunks=MAX_KNOWLEDGE_ACTIVATION_CHUNKS + 1,
        max_bytes=kind.max_bytes + 4,
    )
    if (
        len(source_chunks) > MAX_KNOWLEDGE_ACTIVATION_CHUNKS
        or sum(len(chunk.text.encode("utf-8")) for chunk in source_chunks) > kind.max_bytes
    ):
        raise KnowledgeReviewMaterialTooLarge(
            "Knowledge chunks exceed the review publication bound."
        )
    source_evidence = await workflow.store.read_evidence(
        proposal.id,
        revision=proposal.revision,
        access_scope=workflow.access_scope,
        max_records=MAX_KNOWLEDGE_ACTIVATION_EVIDENCE_RECORDS,
        max_bytes=kind.max_bytes,
    )
    if source_evidence is None:
        raise ValueError("Knowledge evidence is unavailable for review publication.")
    if source_evidence.truncated:
        raise KnowledgeReviewMaterialTooLarge(
            "Knowledge evidence exceeds the review publication bound."
        )
    successor = KnowledgeEntry.model_validate(
        {
            **proposal.model_dump(),
            "revision": revision + 1,
            "status": KnowledgeStatus.ARCHIVED,
            "updated_at": max(updated_at, proposal.updated_at),
            "metadata": {
                **proposal.metadata,
                kind.metadata_key: decision.model_dump(mode="json"),
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
    workflow: KnowledgeReviewWorkflow,
    entry_id: str,
    revision: int,
    operation_id: str,
    kind: _ReviewedKind = _REJECTION,
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
            successor.metadata.get(kind.metadata_key)
        )
        _, expected, _chunks, _evidence, digest = await _prepare(
            workflow, entry_id, revision, decision, receipt.entry_updated_at, operation_id, kind
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


async def load_review_archive(
    workflow: KnowledgeReviewWorkflow, entry_id: str, revision: int, operation_id: str
) -> KnowledgeReviewRejection | None:
    """The reviewed archive ``operation_id`` recorded for kept revision ``revision``."""

    return await load_review_rejection(workflow, entry_id, revision, operation_id, _ARCHIVE)


async def archive_reviewed(
    workflow: KnowledgeReviewWorkflow,
    entry_id: str,
    revision: int,
    operation_id: str,
    decision: KnowledgeRejectionDecision,
) -> KnowledgeReviewRejection:
    """Archive kept revision ``revision`` with the reviewer's decision bound to it."""

    return await reject_reviewed(workflow, entry_id, revision, operation_id, decision, _ARCHIVE)


async def reject_reviewed(
    workflow: KnowledgeReviewWorkflow,
    entry_id: str,
    revision: int,
    operation_id: str,
    decision: KnowledgeRejectionDecision,
    kind: _ReviewedKind = _REJECTION,
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
    prior = await load_review_rejection(workflow, entry_id, revision, operation_id, kind)
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
        workflow, entry_id, revision, decision, datetime.now(UTC), operation_id, kind
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
        prior = await load_review_rejection(workflow, entry_id, revision, operation_id, kind)
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
    # The receipt's digest covers exactly the successor this call prepared, so that is the
    # revision the store holds. It isn't read back: a scope that admits only the statuses an
    # agent recalls (active, pending) may archive an entry it sees without being able to read
    # archived entries, as for an unreviewed retirement.
    return KnowledgeReviewRejection(copy_knowledge_entry(successor), decision, receipt)
