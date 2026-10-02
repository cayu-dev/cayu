"""Backend-independent knowledge embedding and index-readiness contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import canonical_durable_json_bytes, require_finite
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.knowledge.changes import (
    MAX_KNOWLEDGE_CHANGE_SEQUENCE,
    _knowledge_change_identity,
    _validate_knowledge_change_limit,
)
from cayu.knowledge.records import (
    KnowledgeChunk,
    _bounded_knowledge_identity,
    _knowledge_chunk_id,
    _knowledge_entry_id,
    _validate_knowledge_revision,
    _validate_nonnegative_int,
    _validate_positive_int,
    copy_knowledge_chunk,
)

DEFAULT_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT = 500


MAX_KNOWLEDGE_EMBEDDING_DIMENSIONS = 2**31 - 1


MAX_KNOWLEDGE_INDEX_READINESS_LIMIT = 1_000


MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT = 10_000


_MAX_KNOWLEDGE_EMBEDDING_BACKFILL_CURSOR_BYTES = 2_048


KNOWLEDGE_CHUNK_TEXT_PROJECTION = "knowledge_chunk_text"


KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION = "cayu:knowledge-chunk-text:v1"


KNOWLEDGE_CHUNK_TEXT_GENERATOR = "cayu:canonical-knowledge-chunk"


KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION = "1"


KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION = "float32-cosine-v1"


class KnowledgeIndexState(StrEnum):
    """Publication state for one exact derived-index identity."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class KnowledgeIndexReadinessConflict(RuntimeError):
    """A readiness publication conflicts with its identity or sequence fence."""

    def __init__(self, reason: str) -> None:
        self.reason = require_clean_nonblank(reason, "reason")
        super().__init__("Knowledge index readiness conflicts with durable state.")


class KnowledgeEmbeddingProjectionConflict(RuntimeError):
    """A projection attempt was reused with a different immutable vector payload."""

    def __init__(self, reason: str) -> None:
        self.reason = require_clean_nonblank(reason, "reason")
        super().__init__("Knowledge embedding projection conflicts with durable state.")


class KnowledgeEmbeddingIdentity(BaseModel):
    """Complete identity of one durable knowledge embedding projection.

    Content hashes alone are not sufficient reuse keys. Comparable vectors must
    agree on the canonical revision, projected content, embedding space, the
    projection generator, preprocessing, and the stored index representation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    entry_id: str
    entry_revision: int
    chunk_id: str | None = None
    projection_type: str
    projection_content_hash: str
    embedding_model: str
    dimensions: int
    preprocessing_version: str
    generator: str
    generator_version: str
    index_representation_version: str

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _knowledge_entry_id(value)

    @field_validator("entry_revision")
    @classmethod
    def validate_entry_revision(cls, value: int) -> int:
        _validate_knowledge_revision(value, "entry_revision")
        return value

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _knowledge_chunk_id(value)

    @field_validator(
        "projection_type",
        "projection_content_hash",
        "embedding_model",
        "preprocessing_version",
        "generator",
        "generator_version",
        "index_representation_version",
    )
    @classmethod
    def validate_identity_component(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 512:
            raise ValueError(f"`{info.field_name}` must be at most 512 UTF-8 bytes.")
        return value

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: int) -> int:
        _validate_positive_int(value, "dimensions")
        if value > MAX_KNOWLEDGE_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"`dimensions` must be less than or equal to {MAX_KNOWLEDGE_EMBEDDING_DIMENSIONS}."
            )
        return value


class KnowledgeEmbeddingProjection(BaseModel):
    """One externally computed vector fenced to an exact pending projection attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    identity: KnowledgeEmbeddingIdentity
    readiness_sequence: int
    attempt_id: str
    vector: list[float]

    @field_validator("identity", mode="before")
    @classmethod
    def copy_identity(cls, value: KnowledgeEmbeddingIdentity) -> KnowledgeEmbeddingIdentity:
        return copy_knowledge_embedding_identity(value)

    @field_validator("readiness_sequence")
    @classmethod
    def validate_readiness_sequence(cls, value: int) -> int:
        _validate_knowledge_index_sequence(
            value,
            "readiness_sequence",
            allow_zero=False,
        )
        return value

    @field_validator("attempt_id")
    @classmethod
    def validate_attempt_id(cls, value: str) -> str:
        return _bounded_knowledge_index_identity(value, "attempt_id")

    @field_validator("vector", mode="before")
    @classmethod
    def copy_vector(cls, value) -> list[float]:
        if type(value) is not list:
            raise ValueError("`vector` must be a list.")
        result: list[float] = []
        for index, component in enumerate(value):
            if isinstance(component, bool) or not isinstance(component, int | float):
                raise ValueError(f"`vector[{index}]` must be a number.")
            result.append(require_finite(float(component), f"vector[{index}]"))
        return result

    @model_validator(mode="after")
    def validate_vector_dimensions(self) -> KnowledgeEmbeddingProjection:
        if len(self.vector) != self.identity.dimensions:
            raise ValueError("`vector` length must equal `identity.dimensions`.")
        return self


class KnowledgeEmbeddingProjectionWriteResult(BaseModel):
    """Accepted identities from one bounded projection persistence request."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    submitted_records: int
    stored_identities: list[KnowledgeEmbeddingIdentity] = Field(default_factory=list)

    @field_validator("submitted_records")
    @classmethod
    def validate_submitted_records(cls, value: int) -> int:
        _validate_nonnegative_int(value, "submitted_records")
        if value > MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT:
            raise ValueError(
                "`submitted_records` must be less than or equal to "
                f"{MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT}."
            )
        return value

    @field_validator("stored_identities", mode="before")
    @classmethod
    def copy_stored_identities(
        cls,
        value,
    ) -> list[KnowledgeEmbeddingIdentity]:
        if type(value) is not list:
            raise ValueError("`stored_identities` must be a list.")
        return [copy_knowledge_embedding_identity(identity) for identity in value]

    @model_validator(mode="after")
    def validate_stored_partition(self) -> KnowledgeEmbeddingProjectionWriteResult:
        if len(self.stored_identities) > self.submitted_records:
            raise ValueError("Stored projection identities cannot exceed submitted records.")
        identity_sha256s = {
            _knowledge_embedding_identity_sha256(identity) for identity in self.stored_identities
        }
        if len(identity_sha256s) != len(self.stored_identities):
            raise ValueError("`stored_identities` cannot contain duplicates.")
        return self


class KnowledgeEmbeddingBackfillResult(BaseModel):
    """Portable outcome from one bounded embedding repair/backfill pass."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    scanned_records: int
    indexed_records: int
    failed_records: int
    skipped_records: int
    limit: int
    refresh_existing: bool
    next_cursor: str | None = None

    @field_validator(
        "scanned_records",
        "indexed_records",
        "failed_records",
        "skipped_records",
    )
    @classmethod
    def validate_count(cls, value: int, info) -> int:
        _validate_nonnegative_int(value, info.field_name)
        return value

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: int) -> int:
        _validate_knowledge_embedding_work_record_limit(value, field_name="limit")
        return value

    @field_validator("refresh_existing", mode="before")
    @classmethod
    def validate_refresh_existing(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`refresh_existing` must be a boolean.")
        return value

    @field_validator("next_cursor")
    @classmethod
    def validate_next_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_knowledge_embedding_backfill_cursor(value, "next_cursor")

    @model_validator(mode="after")
    def validate_partition(self) -> KnowledgeEmbeddingBackfillResult:
        if self.scanned_records > self.limit:
            raise ValueError("`scanned_records` cannot exceed `limit`.")
        if self.indexed_records + self.failed_records + self.skipped_records != (
            self.scanned_records
        ):
            raise ValueError("Backfill outcomes must partition all scanned records.")
        return self


class KnowledgeIndexReadinessUpdate(BaseModel):
    """One requested state transition for an exact embedding identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    identity: KnowledgeEmbeddingIdentity
    state: KnowledgeIndexState
    attempt_id: str
    failure_code: str | None = None

    @field_validator("identity", mode="before")
    @classmethod
    def copy_identity(cls, value: KnowledgeEmbeddingIdentity) -> KnowledgeEmbeddingIdentity:
        if type(value) is not KnowledgeEmbeddingIdentity:
            raise TypeError("Index readiness requires a KnowledgeEmbeddingIdentity.")
        return value.model_copy(deep=True)

    @field_validator("attempt_id")
    @classmethod
    def validate_attempt_id(cls, value: str) -> str:
        return _bounded_knowledge_index_identity(value, "attempt_id")

    @field_validator("failure_code")
    @classmethod
    def validate_failure_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_knowledge_index_identity(value, "failure_code")

    @model_validator(mode="after")
    def validate_failure_state(self) -> KnowledgeIndexReadinessUpdate:
        if self.state is KnowledgeIndexState.FAILED and self.failure_code is None:
            raise ValueError("Failed index readiness requires `failure_code`.")
        if self.state is not KnowledgeIndexState.FAILED and self.failure_code is not None:
            raise ValueError("`failure_code` is valid only for failed index readiness.")
        return self


class KnowledgeIndexReadiness(BaseModel):
    """Immutable sequenced evidence of one derived-index state transition."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    sequence: int
    identity: KnowledgeEmbeddingIdentity
    state: KnowledgeIndexState
    attempt_id: str
    failure_code: str | None = None
    operation_id: str
    published_at: datetime

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: int) -> int:
        _validate_knowledge_index_sequence(value, "sequence", allow_zero=False)
        return value

    @field_validator("identity", mode="before")
    @classmethod
    def copy_identity(cls, value: KnowledgeEmbeddingIdentity) -> KnowledgeEmbeddingIdentity:
        if type(value) is not KnowledgeEmbeddingIdentity:
            raise TypeError("Index readiness requires a KnowledgeEmbeddingIdentity.")
        return value.model_copy(deep=True)

    @field_validator("attempt_id", "operation_id")
    @classmethod
    def validate_required_identity(cls, value: str, info) -> str:
        return _bounded_knowledge_index_identity(value, info.field_name)

    @field_validator("failure_code")
    @classmethod
    def validate_failure_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_knowledge_index_identity(value, "failure_code")

    @field_validator("published_at")
    @classmethod
    def validate_published_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`published_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_failure_state(self) -> KnowledgeIndexReadiness:
        KnowledgeIndexReadinessUpdate(
            identity=self.identity,
            state=self.state,
            attempt_id=self.attempt_id,
            failure_code=self.failure_code,
        )
        return self


class KnowledgeIndexReadinessBatch(BaseModel):
    """Bounded ordered readiness events through one captured high-water mark."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    readiness: list[KnowledgeIndexReadiness] = Field(default_factory=list)
    after_sequence: int = 0
    next_after_sequence: int = 0
    high_water_sequence: int = 0
    truncated: bool = False
    limit: int

    @field_validator("readiness", mode="before")
    @classmethod
    def copy_readiness(cls, value: list[KnowledgeIndexReadiness]) -> list[KnowledgeIndexReadiness]:
        return [copy_knowledge_index_readiness(item) for item in value]

    @field_validator("after_sequence", "next_after_sequence", "high_water_sequence")
    @classmethod
    def validate_sequences(cls, value: int, info) -> int:
        _validate_knowledge_index_sequence(value, info.field_name)
        return value

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: int) -> int:
        _validate_knowledge_index_readiness_limit(value)
        return value

    @field_validator("truncated", mode="before")
    @classmethod
    def validate_truncated(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`truncated` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_page(self) -> KnowledgeIndexReadinessBatch:
        if self.next_after_sequence < self.after_sequence:
            raise ValueError("`next_after_sequence` cannot precede `after_sequence`.")
        if len(self.readiness) > self.limit:
            raise ValueError("`readiness` cannot contain more records than `limit`.")
        sequences = [item.sequence for item in self.readiness]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("Index readiness records must have unique ascending sequences.")
        if any(sequence <= self.after_sequence for sequence in sequences):
            raise ValueError("Index readiness records fall outside the page frontier.")
        if sequences and sequences[-1] > self.high_water_sequence:
            raise ValueError("Index readiness records cannot exceed `high_water_sequence`.")
        if self.truncated and not self.readiness:
            raise ValueError("A truncated readiness page must contain a continuation record.")
        expected_next = (
            self.readiness[-1].sequence
            if self.truncated
            else max(self.after_sequence, self.high_water_sequence)
        )
        if self.next_after_sequence != expected_next:
            raise ValueError("`next_after_sequence` does not match readiness page semantics.")
        return self


class KnowledgeIndexCoverage(BaseModel):
    """Machine-readable semantic-index coverage for one search projection."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    projection_type: str
    embedding_model: str
    dimensions: int
    preprocessing_version: str
    generator: str
    generator_version: str
    index_representation_version: str
    eligible_records: int
    ready_records: int
    pending_records: int
    failed_records: int
    high_water_sequence: int
    complete: bool

    @field_validator(
        "projection_type",
        "embedding_model",
        "preprocessing_version",
        "generator",
        "generator_version",
        "index_representation_version",
    )
    @classmethod
    def validate_space_identity(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 512:
            raise ValueError(f"`{info.field_name}` must be at most 512 UTF-8 bytes.")
        return value

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: int) -> int:
        _validate_positive_int(value, "dimensions")
        if value > MAX_KNOWLEDGE_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"`dimensions` must be less than or equal to {MAX_KNOWLEDGE_EMBEDDING_DIMENSIONS}."
            )
        return value

    @field_validator(
        "eligible_records",
        "ready_records",
        "pending_records",
        "failed_records",
    )
    @classmethod
    def validate_counts(cls, value: int, info) -> int:
        _validate_nonnegative_int(value, info.field_name)
        return value

    @field_validator("high_water_sequence")
    @classmethod
    def validate_high_water_sequence(cls, value: int) -> int:
        _validate_knowledge_index_sequence(value, "high_water_sequence")
        return value

    @field_validator("complete", mode="before")
    @classmethod
    def validate_complete(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`complete` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_complete_partition(self) -> KnowledgeIndexCoverage:
        if self.ready_records + self.pending_records + self.failed_records != (
            self.eligible_records
        ):
            raise ValueError("Index coverage states must partition all eligible records.")
        if self.complete != (
            self.ready_records == self.eligible_records
            and self.pending_records == 0
            and self.failed_records == 0
        ):
            raise ValueError("`complete` must reflect complete ready index coverage.")
        return self


class KnowledgeEmbeddingWorkerResult(BaseModel):
    """Bounded outcome from consuming canonical changes into one embedding index."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    consumer_id: str
    worker_id: str
    claimed_changes: int
    acknowledged_changes: int
    indexed_records: int
    failed_records: int
    removed_records: int
    limit: int
    processed_records: int
    record_limit: int

    @field_validator("consumer_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _knowledge_change_identity(value, info.field_name)

    @field_validator(
        "claimed_changes",
        "acknowledged_changes",
        "indexed_records",
        "failed_records",
        "removed_records",
        "processed_records",
    )
    @classmethod
    def validate_count(cls, value: int, info) -> int:
        _validate_nonnegative_int(value, info.field_name)
        return value

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: int) -> int:
        _validate_knowledge_change_limit(value)
        return value

    @field_validator("record_limit")
    @classmethod
    def validate_record_limit(cls, value: int) -> int:
        _validate_knowledge_embedding_work_record_limit(value)
        return value

    @model_validator(mode="after")
    def validate_claims(self) -> KnowledgeEmbeddingWorkerResult:
        if self.claimed_changes > self.limit:
            raise ValueError("`claimed_changes` cannot exceed `limit`.")
        if self.acknowledged_changes > self.claimed_changes:
            raise ValueError("`acknowledged_changes` cannot exceed `claimed_changes`.")
        if self.processed_records > self.record_limit:
            raise ValueError("`processed_records` cannot exceed `record_limit`.")
        if self.indexed_records + self.failed_records + self.removed_records > (
            self.processed_records
        ):
            raise ValueError("Embedding outcomes cannot exceed processed records.")
        return self


def copy_knowledge_embedding_identity(
    identity: KnowledgeEmbeddingIdentity,
) -> KnowledgeEmbeddingIdentity:
    if type(identity) is not KnowledgeEmbeddingIdentity:
        raise TypeError("KnowledgeEmbeddingIdentity instances must not be subclasses.")
    return KnowledgeEmbeddingIdentity(**identity.model_dump())


def copy_knowledge_embedding_projection(
    projection: KnowledgeEmbeddingProjection,
) -> KnowledgeEmbeddingProjection:
    if type(projection) is not KnowledgeEmbeddingProjection:
        raise TypeError("KnowledgeEmbeddingProjection instances must not be subclasses.")
    return KnowledgeEmbeddingProjection(
        identity=projection.identity,
        readiness_sequence=projection.readiness_sequence,
        attempt_id=projection.attempt_id,
        vector=list(projection.vector),
    )


def _copy_knowledge_embedding_projections(
    projections: list[KnowledgeEmbeddingProjection],
) -> list[KnowledgeEmbeddingProjection]:
    if type(projections) is not list:
        raise TypeError("`projections` must be a list.")
    if len(projections) > MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT:
        raise ValueError(
            "`projections` must contain at most "
            f"{MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT} records."
        )
    copied = [copy_knowledge_embedding_projection(projection) for projection in projections]
    identity_sha256s = {
        _knowledge_embedding_identity_sha256(projection.identity) for projection in copied
    }
    if len(identity_sha256s) != len(copied):
        raise ValueError("`projections` cannot contain duplicate identities.")
    return copied


def copy_knowledge_index_readiness_update(
    update: KnowledgeIndexReadinessUpdate,
) -> KnowledgeIndexReadinessUpdate:
    if type(update) is not KnowledgeIndexReadinessUpdate:
        raise TypeError("KnowledgeIndexReadinessUpdate instances must not be subclasses.")
    return KnowledgeIndexReadinessUpdate(
        identity=copy_knowledge_embedding_identity(update.identity),
        state=update.state,
        attempt_id=update.attempt_id,
        failure_code=update.failure_code,
    )


def copy_knowledge_index_readiness(
    readiness: KnowledgeIndexReadiness,
) -> KnowledgeIndexReadiness:
    if type(readiness) is not KnowledgeIndexReadiness:
        raise TypeError("KnowledgeIndexReadiness instances must not be subclasses.")
    return KnowledgeIndexReadiness(
        sequence=readiness.sequence,
        identity=copy_knowledge_embedding_identity(readiness.identity),
        state=readiness.state,
        attempt_id=readiness.attempt_id,
        failure_code=readiness.failure_code,
        operation_id=readiness.operation_id,
        published_at=readiness.published_at,
    )


def copy_knowledge_index_coverage(
    coverage: KnowledgeIndexCoverage,
) -> KnowledgeIndexCoverage:
    if type(coverage) is not KnowledgeIndexCoverage:
        raise TypeError("KnowledgeIndexCoverage instances must not be subclasses.")
    return KnowledgeIndexCoverage(**coverage.model_dump())


def _knowledge_chunk_content_hash(chunk: KnowledgeChunk) -> str:
    return f"sha256:{sha256(chunk.text.encode('utf-8')).hexdigest()}"


def knowledge_chunk_embedding_identity(
    chunk: KnowledgeChunk,
    *,
    embedding_model: str,
    dimensions: int,
) -> KnowledgeEmbeddingIdentity:
    """Build the built-in canonical chunk-text embedding identity."""

    chunk = copy_knowledge_chunk(chunk)
    return KnowledgeEmbeddingIdentity(
        entry_id=chunk.entry_id,
        entry_revision=chunk.entry_revision,
        chunk_id=chunk.id,
        projection_type=KNOWLEDGE_CHUNK_TEXT_PROJECTION,
        projection_content_hash=_knowledge_chunk_content_hash(chunk),
        embedding_model=embedding_model,
        dimensions=dimensions,
        preprocessing_version=KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
        generator=KNOWLEDGE_CHUNK_TEXT_GENERATOR,
        generator_version=KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
        index_representation_version=KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
    )


def _validate_knowledge_index_sequence(
    value: int,
    field_name: str,
    *,
    allow_zero: bool = True,
) -> None:
    if allow_zero:
        _validate_nonnegative_int(value, field_name)
    else:
        _validate_positive_int(value, field_name)
    if value > MAX_KNOWLEDGE_CHANGE_SEQUENCE:
        raise ValueError(f"`{field_name}` must be at most {MAX_KNOWLEDGE_CHANGE_SEQUENCE}.")


def _validate_knowledge_index_readiness_limit(value: int) -> None:
    _validate_positive_int(value, "limit")
    if value > MAX_KNOWLEDGE_INDEX_READINESS_LIMIT:
        raise ValueError(
            f"`limit` must be less than or equal to {MAX_KNOWLEDGE_INDEX_READINESS_LIMIT}."
        )


def _validate_knowledge_embedding_work_record_limit(
    value: int,
    *,
    field_name: str = "record_limit",
) -> None:
    _validate_positive_int(value, field_name)
    if value > MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT:
        raise ValueError(
            f"`{field_name}` must be less than or equal to "
            f"{MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT}."
        )


def _bounded_knowledge_index_identity(value: str, field_name: str) -> str:
    return _bounded_knowledge_identity(value, field_name, max_bytes=256)


def _knowledge_embedding_identity_sha256(identity: KnowledgeEmbeddingIdentity) -> str:
    identity = copy_knowledge_embedding_identity(identity)
    return sha256(
        canonical_durable_json_bytes(
            identity.model_dump(mode="json"),
            "knowledge embedding identity",
        )
    ).hexdigest()


def _knowledge_embedding_vector_sha256(vector: list[float]) -> str:
    """Fingerprint the validated vector payload before backend representation casts."""

    return sha256(
        canonical_durable_json_bytes(
            [0.0 if component == 0.0 else component for component in vector],
            "knowledge embedding vector",
        )
    ).hexdigest()


def _bounded_knowledge_embedding_backfill_cursor(value: str, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > _MAX_KNOWLEDGE_EMBEDDING_BACKFILL_CURSOR_BYTES:
        raise ValueError(
            f"`{field_name}` must be at most "
            f"{_MAX_KNOWLEDGE_EMBEDDING_BACKFILL_CURSOR_BYTES} UTF-8 bytes."
        )
    return value


def _knowledge_index_readiness_update_sha256(
    update: KnowledgeIndexReadinessUpdate,
) -> str:
    update = copy_knowledge_index_readiness_update(update)
    return sha256(
        canonical_durable_json_bytes(
            update.model_dump(mode="json"),
            "knowledge index readiness update",
        )
    ).hexdigest()


def _validate_knowledge_index_readiness_transition(
    current: KnowledgeIndexReadiness | None,
    update: KnowledgeIndexReadinessUpdate,
    *,
    expected_sequence: int | None,
) -> None:
    if current is None:
        if expected_sequence is not None:
            raise KnowledgeIndexReadinessConflict("unknown_expected_sequence")
        if update.state is not KnowledgeIndexState.PENDING:
            raise KnowledgeIndexReadinessConflict("initial_state_must_be_pending")
        return
    if expected_sequence != current.sequence:
        raise KnowledgeIndexReadinessConflict("stale_sequence")
    if update.state is KnowledgeIndexState.PENDING:
        if update.attempt_id == current.attempt_id:
            raise KnowledgeIndexReadinessConflict("attempt_reuse")
        return
    if current.state is not KnowledgeIndexState.PENDING:
        raise KnowledgeIndexReadinessConflict("terminal_state_requires_new_attempt")
    if update.attempt_id != current.attempt_id:
        raise KnowledgeIndexReadinessConflict("stale_attempt")
