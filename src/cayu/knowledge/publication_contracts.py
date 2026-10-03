"""Shared preparation and receipt contracts for atomic knowledge publications."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import canonical_durable_json_bytes
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.knowledge.activation_contracts import (
    KnowledgeActivationAuthority,
    KnowledgeActivationDisposition,
    KnowledgeActivationSource,
    _require_knowledge_activation_retirement_capacity,
    copy_knowledge_activation_authority,
)
from cayu.knowledge.records import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeStatus,
    _copy_entry_chunks,
    _copy_entry_evidence,
    _knowledge_entry_id,
    _knowledge_publication_operation_id,
    _next_knowledge_revision,
    _validate_knowledge_revision,
    copy_knowledge_entry,
)
from cayu.knowledge.scopes import KnowledgeAccessScope, knowledge_access_scope_sha256

_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")


class KnowledgePublicationConflict(RuntimeError):
    """An idempotent knowledge publication conflicts with durable state."""

    def __init__(self, reason: str) -> None:
        self.reason = require_clean_nonblank(reason, "reason")
        super().__init__("Knowledge publication conflicts with durable state.")


class KnowledgePublicationReceipt(BaseModel):
    """Bounded immutable evidence for one atomic entry-and-chunks publication."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    operation_id: str = Field(max_length=256)
    entry_id: str
    entry_revision: int
    expected_revision: int | None
    request_sha256: str
    entry_created_at: datetime
    entry_updated_at: datetime
    committed_at: datetime
    replayed: bool = False

    @field_validator("operation_id")
    @classmethod
    def validate_clean_ids(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 256:
            raise ValueError(f"`{info.field_name}` must be at most 256 UTF-8 bytes.")
        return value

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _knowledge_entry_id(value)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        if type(value) is not str or _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("`request_sha256` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("entry_revision")
    @classmethod
    def validate_entry_revision(cls, value: int) -> int:
        _validate_knowledge_revision(value, "entry_revision")
        return value

    @field_validator("expected_revision")
    @classmethod
    def validate_expected_revision(cls, value: int | None) -> int | None:
        if value is not None:
            _validate_knowledge_revision(value, "expected_revision")
        return value

    @model_validator(mode="after")
    def validate_revision_transition(self) -> KnowledgePublicationReceipt:
        expected_entry_revision = (
            1 if self.expected_revision is None else self.expected_revision + 1
        )
        if self.entry_revision != expected_entry_revision:
            raise ValueError("`entry_revision` must follow `expected_revision`.")
        return self

    @field_validator("entry_created_at", "entry_updated_at", "committed_at")
    @classmethod
    def validate_receipt_datetime(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"`{info.field_name}` must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("replayed")
    @classmethod
    def validate_replayed(cls, value: bool) -> bool:
        if type(value) is not bool:
            raise ValueError("`replayed` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_entry_timestamp_order(self) -> KnowledgePublicationReceipt:
        if self.entry_updated_at < self.entry_created_at:
            raise ValueError("`entry_updated_at` cannot precede `entry_created_at`.")
        return self


def copy_knowledge_publication_receipt(
    receipt: KnowledgePublicationReceipt,
    *,
    replayed: bool | None = None,
) -> KnowledgePublicationReceipt:
    if type(receipt) is not KnowledgePublicationReceipt:
        raise TypeError("KnowledgePublicationReceipt instances must not be subclasses.")
    return KnowledgePublicationReceipt(
        operation_id=receipt.operation_id,
        entry_id=receipt.entry_id,
        entry_revision=receipt.entry_revision,
        expected_revision=receipt.expected_revision,
        request_sha256=receipt.request_sha256,
        entry_created_at=receipt.entry_created_at,
        entry_updated_at=receipt.entry_updated_at,
        committed_at=receipt.committed_at,
        replayed=receipt.replayed if replayed is None else replayed,
    )


def prepare_knowledge_publication(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    *,
    evidence: list[KnowledgeEvidence] | None = None,
    operation_id: str,
    expected_revision: int | None = None,
    activation_authority: KnowledgeActivationAuthority | None = None,
) -> tuple[
    str,
    KnowledgeEntry,
    list[KnowledgeChunk],
    list[KnowledgeEvidence],
    str,
]:
    """Copy and bind one complete revision-publication authority tuple."""

    clean_operation_id = _knowledge_publication_operation_id(operation_id)
    copied_entry = copy_knowledge_entry(entry)
    _validate_revision_append(copied_entry, expected_revision=expected_revision)
    copied_chunks = _copy_entry_chunks(
        copied_entry.id,
        copied_entry.revision,
        chunks,
    )
    copied_evidence = _copy_entry_evidence(
        copied_entry.id,
        copied_entry.revision,
        evidence or [],
        chunks=copied_chunks,
    )
    copied_authority = (
        None
        if activation_authority is None
        else copy_knowledge_activation_authority(activation_authority)
    )
    if copied_authority is not None:
        _validate_activation_publication_material(
            copied_authority,
            operation_id=clean_operation_id,
            entry=copied_entry,
            chunks=copied_chunks,
            evidence=copied_evidence,
            expected_revision=expected_revision,
        )
    request_sha256 = _knowledge_publication_request_sha256(
        copied_entry,
        copied_chunks,
        copied_evidence,
        expected_revision=expected_revision,
        activation_authority=copied_authority,
    )
    return (
        clean_operation_id,
        copied_entry,
        copied_chunks,
        copied_evidence,
        request_sha256,
    )


def _validate_knowledge_publication_replay(
    receipt: KnowledgePublicationReceipt,
    *,
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    evidence: list[KnowledgeEvidence],
    expected_revision: int | None,
    request_sha256: str,
    activation_authority: KnowledgeActivationAuthority | None = None,
) -> None:
    receipt = copy_knowledge_publication_receipt(receipt)
    accepted_request_sha256s = {request_sha256}
    if not evidence and activation_authority is None:
        # Revision 42 receipts bind the same entry-and-chunks authority tuple
        # under the v1 digest contract. Revision 43 preserves those receipts,
        # so an exact empty-evidence retry must remain idempotent after migration.
        # Never permit the weaker digest when the new request carries evidence.
        accepted_request_sha256s.add(
            _knowledge_publication_v1_request_sha256(
                entry,
                chunks,
                expected_revision=expected_revision,
            )
        )
    if (
        receipt.entry_id != entry.id
        or receipt.entry_revision != entry.revision
        or receipt.request_sha256 not in accepted_request_sha256s
        or receipt.entry_created_at != entry.created_at
        or receipt.entry_updated_at != entry.updated_at
    ):
        raise KnowledgePublicationConflict("operation_mismatch")


def _validate_activation_publication_material(
    authority: KnowledgeActivationAuthority,
    *,
    operation_id: str,
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    evidence: list[KnowledgeEvidence],
    expected_revision: int | None,
    access_scope: KnowledgeAccessScope | None = None,
) -> None:
    authority = copy_knowledge_activation_authority(authority)
    _require_knowledge_activation_retirement_capacity(entry)
    request = authority.request
    decision = authority.decision
    if request.source is KnowledgeActivationSource.REVIEW_APPROVAL:
        raise ValueError("Review approval cannot use generated revision publication.")
    if (
        request.operation_id != operation_id
        or request.expected_revision != expected_revision
        or request.target_revision != entry.revision
    ):
        raise ValueError("Activation authority does not bind the publication operation.")
    candidate_entry = entry.model_copy(update={"status": KnowledgeStatus.PENDING})
    if request.candidate_entry != candidate_entry:
        raise ValueError("Activation authority does not bind the publication entry material.")
    if list(request.chunks) != chunks or list(request.evidence) != evidence:
        raise ValueError("Activation authority does not bind publication chunks and evidence.")
    if access_scope is not None and request.access_scope_sha256 != knowledge_access_scope_sha256(
        access_scope
    ):
        raise ValueError("Activation authority does not bind the publication access scope.")
    required_status = (
        KnowledgeStatus.ACTIVE
        if decision.disposition is KnowledgeActivationDisposition.ACTIVATE
        else KnowledgeStatus.PENDING
    )
    if decision.disposition is KnowledgeActivationDisposition.REJECT:
        raise ValueError("Rejected activation requests cannot be published.")
    if entry.status is not required_status:
        raise ValueError("Publication status conflicts with activation disposition.")


def _knowledge_publication_request_sha256(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    evidence: list[KnowledgeEvidence],
    *,
    expected_revision: int | None,
    activation_authority: KnowledgeActivationAuthority | None = None,
) -> str:
    if activation_authority is None:
        material = {
            "contract": "cayu-knowledge-revision-publication-v2",
            "expected_revision": expected_revision,
            "entry": entry.model_dump(mode="json"),
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            "evidence": [item.model_dump(mode="json") for item in evidence],
        }
    else:
        material = {
            "contract": "cayu-knowledge-revision-publication-v3",
            "expected_revision": expected_revision,
            "entry": entry.model_dump(mode="json"),
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "activation_authority": activation_authority.model_dump(mode="json"),
        }
    return sha256(
        canonical_durable_json_bytes(
            material,
            "knowledge publication",
        )
    ).hexdigest()


def _knowledge_publication_v1_request_sha256(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    *,
    expected_revision: int | None,
) -> str:
    """Reproduce the revision-42 receipt digest for migration-safe replay."""

    return sha256(
        canonical_durable_json_bytes(
            {
                "contract": "cayu-knowledge-revision-publication-v1",
                "expected_revision": expected_revision,
                "entry": entry.model_dump(mode="json"),
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            },
            "knowledge publication",
        )
    ).hexdigest()


def _validate_revision_append(
    entry: KnowledgeEntry,
    *,
    expected_revision: int | None,
) -> None:
    target_revision = (
        1 if expected_revision is None else _next_knowledge_revision(expected_revision)
    )
    _validate_knowledge_revision(entry.revision, "entry.revision")
    if entry.revision != target_revision:
        raise ValueError(
            f"Knowledge revision must be {target_revision} for expected_revision "
            f"{expected_revision!r}."
        )
