"""Backend-independent knowledge change-feed records and consumer contracts."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import canonical_durable_json_bytes, require_finite
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.knowledge.records import (
    _knowledge_entry_id,
    _validate_knowledge_revision,
    _validate_nonnegative_int,
    _validate_positive_int,
)
from cayu.knowledge.relations import _knowledge_relation_identity

_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")

MAX_KNOWLEDGE_CHANGE_LIMIT = 1_000

MAX_KNOWLEDGE_CHANGE_SEQUENCE = 2**63 - 1


class KnowledgeChangeKind(StrEnum):
    CREATED = "created"
    REVISION_APPENDED = "revision_appended"
    STATUS_TRANSITIONED = "status_transitioned"
    TOMBSTONED = "tombstoned"
    HARD_DELETED = "hard_deleted"
    EXPIRED = "expired"
    RELATION_PUBLISHED = "relation_published"


class KnowledgeChangeConsumerConflict(RuntimeError):
    """A knowledge-change consumer or lease fence conflicts with durable state."""

    def __init__(self, reason: str) -> None:
        self.reason = require_clean_nonblank(reason, "reason")
        super().__init__("Knowledge change consumer conflicts with durable state.")


class KnowledgeChange(BaseModel):
    """Metadata-only canonical knowledge mutation published in commit order."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    sequence: int
    kind: KnowledgeChangeKind
    entry_id: str
    entry_revision: int
    committed_at: datetime
    operation_id: str | None = None
    relation_id: str | None = None

    @field_validator("id", "operation_id")
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 256:
            raise ValueError(f"`{info.field_name}` must be at most 256 UTF-8 bytes.")
        return value

    @field_validator("relation_id")
    @classmethod
    def validate_relation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _knowledge_relation_identity(value, "relation_id")

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _knowledge_entry_id(value)

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: int) -> int:
        _validate_knowledge_change_sequence(value, "sequence")
        if value == 0:
            raise ValueError("`sequence` must be greater than 0.")
        return value

    @field_validator("entry_revision")
    @classmethod
    def validate_entry_revision(cls, value: int) -> int:
        _validate_knowledge_revision(value, "entry_revision")
        return value

    @field_validator("committed_at")
    @classmethod
    def validate_committed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`committed_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_relation_change(self) -> KnowledgeChange:
        if self.kind is KnowledgeChangeKind.RELATION_PUBLISHED:
            if self.relation_id is None:
                raise ValueError("Relation publication changes require `relation_id`.")
        elif self.relation_id is not None:
            raise ValueError("Only relation publication changes may carry `relation_id`.")
        return self


class KnowledgeChangeBatch(BaseModel):
    """One bounded ordered change page and its captured accessible high-water mark."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    changes: list[KnowledgeChange] = Field(default_factory=list)
    after_sequence: int = 0
    next_after_sequence: int = 0
    high_water_sequence: int = 0
    truncated: bool = False
    limit: int

    @field_validator("changes", mode="before")
    @classmethod
    def copy_changes(cls, value) -> list[KnowledgeChange]:
        return [copy_knowledge_change(change) for change in value]

    @field_validator("after_sequence", "next_after_sequence", "high_water_sequence")
    @classmethod
    def validate_sequences(cls, value: int, info) -> int:
        _validate_knowledge_change_sequence(value, info.field_name)
        return value

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: int) -> int:
        _validate_knowledge_change_limit(value)
        return value

    @field_validator("truncated", mode="before")
    @classmethod
    def validate_truncated(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`truncated` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_page(self) -> KnowledgeChangeBatch:
        if len(self.changes) > self.limit:
            raise ValueError("`changes` cannot contain more records than `limit`.")
        sequences = [change.sequence for change in self.changes]
        if sequences != sorted(set(sequences)):
            raise ValueError("Knowledge changes must have unique increasing sequences.")
        if any(sequence <= self.after_sequence for sequence in sequences):
            raise ValueError("Knowledge changes must follow `after_sequence`.")
        if sequences and sequences[-1] > self.high_water_sequence:
            raise ValueError("Knowledge changes cannot exceed `high_water_sequence`.")
        expected_next = (
            sequences[-1]
            if self.truncated and sequences
            else max(self.after_sequence, self.high_water_sequence)
        )
        if self.next_after_sequence != expected_next:
            raise ValueError("`next_after_sequence` does not match the bounded page.")
        if self.truncated and not sequences:
            raise ValueError("A truncated knowledge-change page cannot be empty.")
        return self


class KnowledgeChangeClaim(BaseModel):
    """One fenced at-least-once lease over an ordered knowledge change."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    consumer_id: str
    worker_id: str
    claim_id: str
    change: KnowledgeChange
    attempt: int
    claimed_at: datetime
    lease_expires_at: datetime

    @field_validator("consumer_id", "worker_id", "claim_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 256:
            raise ValueError(f"`{info.field_name}` must be at most 256 UTF-8 bytes.")
        return value

    @field_validator("change")
    @classmethod
    def copy_change(cls, value: KnowledgeChange) -> KnowledgeChange:
        return copy_knowledge_change(value)

    @field_validator("attempt")
    @classmethod
    def validate_attempt(cls, value: int) -> int:
        _validate_positive_int(value, "attempt")
        return value

    @field_validator("claimed_at", "lease_expires_at")
    @classmethod
    def validate_datetime(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"`{info.field_name}` must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_lease_window(self) -> KnowledgeChangeClaim:
        if self.lease_expires_at <= self.claimed_at:
            raise ValueError("`lease_expires_at` must follow `claimed_at`.")
        return self


class KnowledgeChangeConsumerState(BaseModel):
    """Durable cursor and active lease state for one scope-bound consumer."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    consumer_id: str
    access_scope_sha256: str
    cursor_sequence: int = 0
    pending_change_sequence: int | None = None
    pending_claim_id: str | None = None
    pending_worker_id: str | None = None
    pending_attempt: int = 0
    claimed_at: datetime | None = None
    lease_expires_at: datetime | None = None
    last_acknowledged_claim_id: str | None = None
    updated_at: datetime

    @field_validator(
        "consumer_id",
        "pending_claim_id",
        "pending_worker_id",
        "last_acknowledged_claim_id",
    )
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 256:
            raise ValueError(f"`{info.field_name}` must be at most 256 UTF-8 bytes.")
        return value

    @field_validator("access_scope_sha256")
    @classmethod
    def validate_scope_digest(cls, value: str) -> str:
        if type(value) is not str or _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("`access_scope_sha256` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("cursor_sequence")
    @classmethod
    def validate_cursor(cls, value: int) -> int:
        _validate_knowledge_change_sequence(value, "cursor_sequence")
        return value

    @field_validator("pending_change_sequence")
    @classmethod
    def validate_pending_sequence(cls, value: int | None) -> int | None:
        if value is not None:
            _validate_knowledge_change_sequence(value, "pending_change_sequence")
            if value == 0:
                raise ValueError("`pending_change_sequence` must be greater than 0.")
        return value

    @field_validator("pending_attempt")
    @classmethod
    def validate_pending_attempt(cls, value: int) -> int:
        _validate_nonnegative_int(value, "pending_attempt")
        return value

    @field_validator("claimed_at", "lease_expires_at", "updated_at")
    @classmethod
    def validate_datetime(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"`{info.field_name}` must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_pending_lease(self) -> KnowledgeChangeConsumerState:
        pending = (
            self.pending_change_sequence,
            self.pending_claim_id,
            self.pending_worker_id,
            self.claimed_at,
            self.lease_expires_at,
        )
        if any(value is not None for value in pending) and not all(
            value is not None for value in pending
        ):
            raise ValueError("Knowledge change pending-lease fields must be set together.")
        if self.pending_change_sequence is not None:
            if self.pending_change_sequence <= self.cursor_sequence:
                raise ValueError("A pending change must follow the consumer cursor.")
            if self.pending_attempt <= 0:
                raise ValueError("An active knowledge change lease requires a positive attempt.")
            assert self.claimed_at is not None
            assert self.lease_expires_at is not None
            if self.lease_expires_at <= self.claimed_at:
                raise ValueError("An active knowledge change lease must expire after claim time.")
        return self


def _knowledge_change_claim_sha256(claim: KnowledgeChangeClaim) -> str:
    claim = copy_knowledge_change_claim(claim)
    return sha256(
        canonical_durable_json_bytes(
            claim.model_dump(mode="json"),
            "knowledge change claim",
        )
    ).hexdigest()


def copy_knowledge_change(change: KnowledgeChange) -> KnowledgeChange:
    if type(change) is not KnowledgeChange:
        raise TypeError("KnowledgeChange instances must not be subclasses.")
    return KnowledgeChange(
        id=change.id,
        sequence=change.sequence,
        kind=change.kind,
        entry_id=change.entry_id,
        entry_revision=change.entry_revision,
        committed_at=change.committed_at,
        operation_id=change.operation_id,
        relation_id=change.relation_id,
    )


def copy_knowledge_change_claim(claim: KnowledgeChangeClaim) -> KnowledgeChangeClaim:
    if type(claim) is not KnowledgeChangeClaim:
        raise TypeError("KnowledgeChangeClaim instances must not be subclasses.")
    return KnowledgeChangeClaim(
        consumer_id=claim.consumer_id,
        worker_id=claim.worker_id,
        claim_id=claim.claim_id,
        change=copy_knowledge_change(claim.change),
        attempt=claim.attempt,
        claimed_at=claim.claimed_at,
        lease_expires_at=claim.lease_expires_at,
    )


def copy_knowledge_change_consumer_state(
    state: KnowledgeChangeConsumerState,
) -> KnowledgeChangeConsumerState:
    if type(state) is not KnowledgeChangeConsumerState:
        raise TypeError("KnowledgeChangeConsumerState instances must not be subclasses.")
    return KnowledgeChangeConsumerState(**state.model_dump())


def _validate_knowledge_change_limit(value: int) -> None:
    _validate_positive_int(value, "limit")
    if value > MAX_KNOWLEDGE_CHANGE_LIMIT:
        raise ValueError(f"`limit` must be less than or equal to {MAX_KNOWLEDGE_CHANGE_LIMIT}.")


def _validate_knowledge_change_sequence(value: int, field_name: str) -> None:
    _validate_nonnegative_int(value, field_name)
    if value > MAX_KNOWLEDGE_CHANGE_SEQUENCE:
        raise ValueError(f"`{field_name}` must be at most {MAX_KNOWLEDGE_CHANGE_SEQUENCE}.")


def _knowledge_change_lease_seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("`lease_seconds` must be a number.")
    result = require_finite(float(value), "lease_seconds")
    if result <= 0.0 or result > 86_400.0:
        raise ValueError("`lease_seconds` must be greater than 0 and at most 86400.")
    return result


def _knowledge_change_identity(value: str, field_name: str) -> str:
    clean = require_clean_nonblank(value, field_name)
    if len(clean.encode("utf-8")) > 256:
        raise ValueError(f"`{field_name}` must be at most 256 UTF-8 bytes.")
    return clean


def _initialize_knowledge_change_consumer_state(
    state: KnowledgeChangeConsumerState | None,
    *,
    consumer_id: str,
    access_scope_sha256: str,
    baseline_sequence: int,
    now: datetime,
) -> KnowledgeChangeConsumerState:
    if state is None:
        return KnowledgeChangeConsumerState(
            consumer_id=consumer_id,
            access_scope_sha256=access_scope_sha256,
            cursor_sequence=baseline_sequence,
            updated_at=now,
        )
    if state.access_scope_sha256 != access_scope_sha256:
        raise KnowledgeChangeConsumerConflict("access_scope_mismatch")
    if state.pending_change_sequence is not None:
        raise KnowledgeChangeConsumerConflict("consumer_has_active_claim")
    if state.cursor_sequence >= baseline_sequence:
        return copy_knowledge_change_consumer_state(state)
    if (
        state.cursor_sequence != 0
        or state.pending_attempt != 0
        or state.last_acknowledged_claim_id is not None
    ):
        raise KnowledgeChangeConsumerConflict("consumer_already_started")
    return state.model_copy(
        update={
            "cursor_sequence": baseline_sequence,
            "updated_at": now,
        }
    )
