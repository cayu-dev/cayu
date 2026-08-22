from __future__ import annotations

import base64
import binascii
import hmac
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
    require_execution_unit_id,
)
from cayu.retrieval import RetrievalCandidateIdentity

RECALL_RECEIPT_VERSION = "cayu.recall_receipt.v1"
CONTEXT_EXPOSURE_VERSION = "cayu.context_exposure.v1"
RECALL_ITEM_EXPOSURE_VERSION = "cayu.recall_item_exposure.v1"

MAX_MEMORY_EVIDENCE_ID_CHARS = 256
MAX_MEMORY_EVIDENCE_REFERENCE_CHARS = 512
MAX_MEMORY_EVIDENCE_SESSION_ID_BYTES = 2_048
MAX_MEMORY_EVIDENCE_NAME_BYTES = 256
MAX_RECALL_CANDIDATE_RECORD_ID_BYTES = 4_096
MAX_RECALL_KNOWLEDGE_ENTRY_ID_BYTES = 256
MAX_RECALL_KNOWLEDGE_CHUNK_ID_BYTES = 512
MAX_RECALL_KNOWLEDGE_REVISION = 2**31 - 1
MAX_RECALL_KNOWLEDGE_CHUNK_INDEX = 2**31 - 1
MAX_RECALL_RECEIPT_ITEMS = 64
MAX_RECALL_RECEIPT_SOURCES = 32
MAX_RECALL_ITEM_MATCH_CHANNELS = 32
MAX_RECALL_LOCATOR_TEXT_PARTS = 64
MAX_RECALL_ITEM_LOCATOR_BYTES = 4_096
MAX_RECALL_RECEIPT_BYTES = 256_000
MAX_CONTEXT_EXPOSURE_RECEIPTS = 32
MAX_CONTEXT_EXPOSURE_CONTRIBUTORS = 64
MAX_CONTEXT_EXPOSURE_TRANSITIONS = 16
MAX_CONTEXT_EXPOSURE_BYTES = 128_000
MAX_RECALL_ITEM_EXPOSURE_BYTES = 16_384
MAX_MEMORY_EVIDENCE_PAGE_LIMIT = 100
MIN_MEMORY_EVIDENCE_PAGE_BYTES = MAX_RECALL_RECEIPT_BYTES + 2
MAX_MEMORY_EVIDENCE_PAGE_BYTES = 4_194_304
MAX_MEMORY_EVIDENCE_CURSOR_BYTES = 2_048

_SHA256_HEX = frozenset("0123456789abcdef")
_CURSOR_VERSION = 1
_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    hide_input_in_errors=True,
    revalidate_instances="always",
    validate_default=True,
)


class RecallSourceCoverageState(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class RecallItemAdmission(StrEnum):
    ADMITTED = "admitted"
    OFFERED = "offered"


class RecallItemSelectionReason(StrEnum):
    CALIBRATED_STRONG_MATCH = "calibrated_strong_match"
    CALIBRATED_PLAUSIBLE_MATCH = "calibrated_plausible_match"
    DUPLICATE_STRONG_REFERENCE = "duplicate_strong_reference"
    STRONG_MATCH_NOT_FOCUSED = "strong_match_not_focused"
    STRONG_MATCH_OFFERED_BY_MODE = "strong_match_offered_by_mode"
    EXPLICIT_APPLICATION_SELECTION = "explicit_application_selection"


class ContextExposureState(StrEnum):
    PLANNED = "planned"
    PREPARED = "prepared"
    DISPATCH_STARTED = "dispatch_started"
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INDETERMINATE = "indeterminate"

    @property
    def terminal(self) -> bool:
        return self in {
            ContextExposureState.COMPLETED,
            ContextExposureState.FAILED,
            ContextExposureState.CANCELLED,
            ContextExposureState.INDETERMINATE,
        }


class ContextExposureEvidenceKind(StrEnum):
    COMPOSITION_PLANNED = "composition_planned"
    REQUEST_PREPARED = "request_prepared"
    DISPATCH_INTENT_COMMITTED = "dispatch_intent_committed"
    PROVIDER_ACKNOWLEDGEMENT = "provider_acknowledgement"
    RECOVERY_ACKNOWLEDGEMENT = "recovery_acknowledgement"
    PROVIDER_COMPLETION = "provider_completion"
    RECOVERY_COMPLETION = "recovery_completion"
    CONCLUSIVE_FAILURE = "conclusive_failure"
    CONCLUSIVE_CANCELLATION = "conclusive_cancellation"
    AMBIGUOUS_TRANSPORT = "ambiguous_transport"
    RECOVERY_INDETERMINATE = "recovery_indeterminate"


class KeyedEvidenceFingerprintDomain(StrEnum):
    SITUATION = "recall_situation"
    SOURCE_CONFIGURATION = "recall_source_configuration"
    ADMISSION_POLICY = "recall_admission_policy"
    ACCESS_SCOPE = "recall_access_scope"
    FRONTIER = "recall_frontier"
    SOURCE_LOCATOR = "recall_source_locator"
    COMPOSITION = "context_composition"
    EXECUTION_PROFILE = "execution_profile"
    CONTEXT_POLICY = "context_policy"
    TOOL_EXPOSURE = "tool_exposure"
    REQUEST_CONTRACT = "provider_request_contract"


_STATE_EVIDENCE_KINDS: dict[ContextExposureState, frozenset[ContextExposureEvidenceKind]] = {
    ContextExposureState.PLANNED: frozenset({ContextExposureEvidenceKind.COMPOSITION_PLANNED}),
    ContextExposureState.PREPARED: frozenset({ContextExposureEvidenceKind.REQUEST_PREPARED}),
    ContextExposureState.DISPATCH_STARTED: frozenset(
        {ContextExposureEvidenceKind.DISPATCH_INTENT_COMMITTED}
    ),
    ContextExposureState.ACKNOWLEDGED: frozenset(
        {
            ContextExposureEvidenceKind.PROVIDER_ACKNOWLEDGEMENT,
            ContextExposureEvidenceKind.RECOVERY_ACKNOWLEDGEMENT,
        }
    ),
    ContextExposureState.COMPLETED: frozenset(
        {
            ContextExposureEvidenceKind.PROVIDER_COMPLETION,
            ContextExposureEvidenceKind.RECOVERY_COMPLETION,
        }
    ),
    ContextExposureState.FAILED: frozenset({ContextExposureEvidenceKind.CONCLUSIVE_FAILURE}),
    ContextExposureState.CANCELLED: frozenset(
        {ContextExposureEvidenceKind.CONCLUSIVE_CANCELLATION}
    ),
    ContextExposureState.INDETERMINATE: frozenset(
        {
            ContextExposureEvidenceKind.AMBIGUOUS_TRANSPORT,
            ContextExposureEvidenceKind.RECOVERY_INDETERMINATE,
        }
    ),
}

_ALLOWED_EXPOSURE_TRANSITIONS: dict[ContextExposureState, frozenset[ContextExposureState]] = {
    ContextExposureState.PLANNED: frozenset(
        {
            ContextExposureState.PREPARED,
            ContextExposureState.FAILED,
            ContextExposureState.CANCELLED,
        }
    ),
    ContextExposureState.PREPARED: frozenset(
        {
            ContextExposureState.DISPATCH_STARTED,
            ContextExposureState.FAILED,
            ContextExposureState.CANCELLED,
        }
    ),
    ContextExposureState.DISPATCH_STARTED: frozenset(
        {
            ContextExposureState.ACKNOWLEDGED,
            ContextExposureState.FAILED,
            ContextExposureState.CANCELLED,
            ContextExposureState.INDETERMINATE,
        }
    ),
    ContextExposureState.ACKNOWLEDGED: frozenset(
        {
            ContextExposureState.COMPLETED,
            ContextExposureState.FAILED,
            ContextExposureState.CANCELLED,
            ContextExposureState.INDETERMINATE,
        }
    ),
    ContextExposureState.COMPLETED: frozenset(),
    ContextExposureState.FAILED: frozenset(),
    ContextExposureState.CANCELLED: frozenset(),
    ContextExposureState.INDETERMINATE: frozenset(),
}


class RecallEvidenceConflict(RuntimeError):
    """An immutable recall-evidence identity was reused with different material."""

    def __init__(self, record_type: str, record_id: str) -> None:
        self.record_type = _bounded_text(record_type, "record_type")
        self.record_id = _identity(record_id, "record_id")
        super().__init__(f"{self.record_type} identity conflicts with durable state.")


class ContextExposureTransitionConflict(RuntimeError):
    """A context-exposure transition lost its exact state/revision fence."""

    def __init__(
        self,
        exposure_id: str,
        *,
        expected_state: ContextExposureState,
        expected_revision: int,
        actual_state: ContextExposureState,
        actual_revision: int,
    ) -> None:
        self.exposure_id = _identity(exposure_id, "exposure_id")
        self.expected_state = ContextExposureState(expected_state)
        self.expected_revision = _nonnegative_int(expected_revision, "expected_revision")
        self.actual_state = ContextExposureState(actual_state)
        self.actual_revision = _nonnegative_int(actual_revision, "actual_revision")
        super().__init__("Context exposure transition conflicts with durable state.")


class KeyedEvidenceFingerprint(BaseModel):
    """A domain-separated HMAC fingerprint without the secret key material."""

    model_config = _MODEL_CONFIG

    algorithm: Literal["hmac-sha256"] = "hmac-sha256"
    domain: KeyedEvidenceFingerprintDomain
    key_id: str
    digest: str

    @field_validator("key_id")
    @classmethod
    def validate_key_id(cls, value: str) -> str:
        return _bounded_text(value, "key_id")

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _digest(value, "digest")


class KnowledgeEntryEvidenceLocator(BaseModel):
    """Closed safe locator for one exact knowledge-entry revision."""

    model_config = _MODEL_CONFIG

    kind: Literal["knowledge_entry"] = "knowledge_entry"
    entry_id: str
    entry_revision: StrictInt = Field(ge=1, le=MAX_RECALL_KNOWLEDGE_REVISION)

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _bounded_utf8_identity(
            value,
            "entry_id",
            MAX_RECALL_KNOWLEDGE_ENTRY_ID_BYTES,
        )


class KnowledgeChunkEvidenceLocator(BaseModel):
    """Closed safe locator for one exact chunk of a knowledge revision."""

    model_config = _MODEL_CONFIG

    kind: Literal["knowledge_chunk"] = "knowledge_chunk"
    entry_id: str
    entry_revision: StrictInt = Field(ge=1, le=MAX_RECALL_KNOWLEDGE_REVISION)
    chunk_id: str
    chunk_index: StrictInt = Field(ge=0, le=MAX_RECALL_KNOWLEDGE_CHUNK_INDEX)

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _bounded_utf8_identity(
            value,
            "entry_id",
            MAX_RECALL_KNOWLEDGE_ENTRY_ID_BYTES,
        )

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        return _bounded_utf8_identity(
            value,
            "chunk_id",
            MAX_RECALL_KNOWLEDGE_CHUNK_ID_BYTES,
        )


class TranscriptMessageEvidenceLocator(BaseModel):
    """Closed safe locator for the exact indexed text parts of one message."""

    model_config = _MODEL_CONFIG

    kind: Literal["transcript_message"] = "transcript_message"
    session_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_SESSION_ID_BYTES)
    interaction_id: str | None = Field(
        default=None,
        max_length=MAX_MEMORY_EVIDENCE_ID_CHARS,
    )
    transcript_index: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    text_part_indexes: tuple[StrictInt, ...]

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return _session_identity(value)

    @field_validator("interaction_id")
    @classmethod
    def validate_interaction_id(cls, value: str | None) -> str | None:
        return None if value is None else _identity(value, "interaction_id")

    @field_validator("text_part_indexes", mode="before")
    @classmethod
    def validate_text_part_indexes(cls, value: Any) -> tuple[int, ...]:
        if type(value) not in (list, tuple) or not value:
            raise ValueError("text_part_indexes must be a non-empty list or tuple.")
        if len(value) > MAX_RECALL_LOCATOR_TEXT_PARTS:
            raise ValueError("text_part_indexes exceed their count bound.")
        indexes = tuple(value)
        if any(
            type(index) is not int or not 0 <= index <= MAX_DURABLE_JSON_INTEGER
            for index in indexes
        ):
            raise ValueError("text_part_indexes must contain durable non-negative integers.")
        if indexes != tuple(sorted(set(indexes))):
            raise ValueError("text_part_indexes must be unique and ascending.")
        return indexes


class OpaqueRecallEvidenceLocator(BaseModel):
    """Safe exact binding for a custom source locator without retaining its payload."""

    model_config = _MODEL_CONFIG

    kind: Literal["opaque"] = "opaque"
    fingerprint: KeyedEvidenceFingerprint

    @field_validator("fingerprint", mode="before")
    @classmethod
    def copy_fingerprint(cls, value: Any) -> KeyedEvidenceFingerprint:
        return KeyedEvidenceFingerprint.model_validate(
            value.model_dump(mode="python") if type(value) is KeyedEvidenceFingerprint else value
        )

    @model_validator(mode="after")
    def validate_fingerprint_domain(self) -> Self:
        if self.fingerprint.domain is not KeyedEvidenceFingerprintDomain.SOURCE_LOCATOR:
            raise ValueError("Opaque recall locator has the wrong keyed-fingerprint domain.")
        return self


RecallEvidenceLocator = Annotated[
    KnowledgeEntryEvidenceLocator
    | KnowledgeChunkEvidenceLocator
    | TranscriptMessageEvidenceLocator
    | OpaqueRecallEvidenceLocator,
    Field(discriminator="kind"),
]


class RecallSourceCoverage(BaseModel):
    model_config = _MODEL_CONFIG

    source: str
    required: StrictBool
    channels: tuple[str, ...]
    state: RecallSourceCoverageState
    failure_code: str | None = None
    inspected_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    candidate_limit: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    truncated: StrictBool = False
    continuation_available: StrictBool = False

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _bounded_text(value, "source")

    @field_validator("channels", mode="before")
    @classmethod
    def copy_channels(cls, value: Any) -> tuple[str, ...]:
        if type(value) not in (list, tuple) or not value:
            raise ValueError("Recall source channels must be a non-empty list or tuple.")
        if len(value) > MAX_RECALL_ITEM_MATCH_CHANNELS:
            raise ValueError("Recall source channels exceed their count bound.")
        channels = tuple(sorted(_bounded_text(item, "channel") for item in value))
        if len(channels) != len(set(channels)):
            raise ValueError("Recall source channels must be unique.")
        return channels

    @field_validator("failure_code")
    @classmethod
    def validate_failure_code(cls, value: str | None) -> str | None:
        return None if value is None else _bounded_text(value, "failure_code")

    @model_validator(mode="after")
    def validate_coverage(self) -> Self:
        if self.inspected_count > self.candidate_limit:
            raise ValueError("Recall source inspected count exceeds its candidate limit.")
        if self.continuation_available and not self.truncated:
            raise ValueError("A recall source continuation requires truncated coverage.")
        if self.state in {
            RecallSourceCoverageState.UNAVAILABLE,
            RecallSourceCoverageState.FAILED,
        } and (self.inspected_count != 0 or self.continuation_available or not self.truncated):
            raise ValueError(
                "Unavailable recall coverage must be explicitly incomplete and cannot "
                "report inspected candidates."
            )
        if self.state is RecallSourceCoverageState.COMPLETE and self.truncated:
            raise ValueError("Complete recall coverage cannot be truncated.")
        if self.state is RecallSourceCoverageState.PARTIAL and not self.truncated:
            raise ValueError("Partial recall coverage must be truncated.")
        if (self.state is RecallSourceCoverageState.COMPLETE) is not (self.failure_code is None):
            raise ValueError("Only incomplete recall source coverage carries a failure code.")
        return self


class RecallReceiptItem(BaseModel):
    model_config = _MODEL_CONFIG

    ordinal: StrictInt = Field(ge=0, lt=MAX_RECALL_RECEIPT_ITEMS)
    identity: RetrievalCandidateIdentity
    representation_id: str
    content_sha256: str
    locator: RecallEvidenceLocator
    admission: RecallItemAdmission
    selection_reason: RecallItemSelectionReason
    fused_rank: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    match_channels: tuple[str, ...]

    @field_validator("identity", mode="before")
    @classmethod
    def copy_identity(cls, value: Any) -> RetrievalCandidateIdentity:
        return _copy_bounded_candidate_identity(value)

    @field_validator("representation_id")
    @classmethod
    def validate_representation_id(cls, value: str) -> str:
        return _bounded_text(value, "representation_id")

    @field_validator("content_sha256")
    @classmethod
    def validate_content_sha256(cls, value: str) -> str:
        return _digest(value, "content_sha256")

    @field_validator("locator", mode="before")
    @classmethod
    def copy_locator(cls, value: Any) -> Any:
        return _copy_recall_evidence_locator_input(value)

    @field_validator("locator")
    @classmethod
    def validate_locator(cls, value: RecallEvidenceLocator) -> RecallEvidenceLocator:
        _validate_recall_evidence_locator_bytes(value)
        return value

    @field_validator("match_channels", mode="before")
    @classmethod
    def copy_match_channels(cls, value: Any) -> tuple[str, ...]:
        if type(value) not in (list, tuple) or not value:
            raise ValueError("match_channels must be a non-empty list or tuple.")
        if len(value) > MAX_RECALL_ITEM_MATCH_CHANNELS:
            raise ValueError("Recall item match channels exceed their count bound.")
        copied = tuple(_bounded_text(item, "match_channel") for item in value)
        if len(copied) != len(set(copied)):
            raise ValueError("Recall item match channels must be unique.")
        return copied

    @model_validator(mode="after")
    def validate_admission_reason(self) -> Self:
        _validate_admission_reason(self.admission, self.selection_reason)
        _validate_recall_evidence_locator_identity(
            self.locator,
            self.identity,
            self.content_sha256,
        )
        return self


class RecallReceipt(BaseModel):
    """Immutable bounded proof of one already-computed recall/admission operation."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["cayu.recall_receipt.v1"] = RECALL_RECEIPT_VERSION
    receipt_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    session_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_SESSION_ID_BYTES)
    interaction_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    model_step_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    created_at: datetime
    situation_fingerprint: KeyedEvidenceFingerprint
    engine_version: str
    source_configuration_fingerprint: KeyedEvidenceFingerprint
    admission_policy_fingerprint: KeyedEvidenceFingerprint
    access_scope_fingerprint: KeyedEvidenceFingerprint
    frontier_fingerprint: KeyedEvidenceFingerprint
    sources: tuple[RecallSourceCoverage, ...]
    inspected_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    eligible_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    admitted_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    offered_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    silent_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    omitted_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    truncated: StrictBool
    items: tuple[RecallReceiptItem, ...]

    @field_validator("receipt_id", "interaction_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _identity(value, info.field_name)

    @field_validator("model_step_id")
    @classmethod
    def validate_model_step_id(cls, value: str) -> str:
        return require_execution_unit_id(value, "model_step_id")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return _session_identity(value)

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _utc(value, "created_at")

    @field_validator(
        "situation_fingerprint",
        "source_configuration_fingerprint",
        "admission_policy_fingerprint",
        "access_scope_fingerprint",
        "frontier_fingerprint",
        mode="before",
    )
    @classmethod
    def copy_fingerprints(cls, value: Any) -> KeyedEvidenceFingerprint:
        return KeyedEvidenceFingerprint.model_validate(
            value.model_dump(mode="python") if type(value) is KeyedEvidenceFingerprint else value
        )

    @model_validator(mode="after")
    def validate_fingerprint_domains(self) -> Self:
        expected = {
            "situation_fingerprint": KeyedEvidenceFingerprintDomain.SITUATION,
            "source_configuration_fingerprint": (
                KeyedEvidenceFingerprintDomain.SOURCE_CONFIGURATION
            ),
            "admission_policy_fingerprint": KeyedEvidenceFingerprintDomain.ADMISSION_POLICY,
            "access_scope_fingerprint": KeyedEvidenceFingerprintDomain.ACCESS_SCOPE,
            "frontier_fingerprint": KeyedEvidenceFingerprintDomain.FRONTIER,
        }
        for field_name, domain in expected.items():
            if getattr(self, field_name).domain is not domain:
                raise ValueError(f"`{field_name}` has the wrong keyed-fingerprint domain.")
        return self

    @field_validator("engine_version")
    @classmethod
    def validate_names(cls, value: str, info: Any) -> str:
        return _bounded_text(value, info.field_name)

    @field_validator("sources", mode="before")
    @classmethod
    def copy_sources(cls, value: Any) -> tuple[RecallSourceCoverage, ...]:
        if type(value) not in (list, tuple) or not value:
            raise ValueError("sources must contain at least one recall source.")
        if len(value) > MAX_RECALL_RECEIPT_SOURCES:
            raise ValueError("Recall receipt sources exceed their count bound.")
        return tuple(
            RecallSourceCoverage.model_validate(
                item.model_dump(mode="python") if type(item) is RecallSourceCoverage else item
            )
            for item in value
        )

    @field_validator("items", mode="before")
    @classmethod
    def copy_items(cls, value: Any) -> tuple[RecallReceiptItem, ...]:
        if type(value) not in (list, tuple):
            raise ValueError("items must be a list or tuple.")
        if len(value) > MAX_RECALL_RECEIPT_ITEMS:
            raise ValueError("Recall receipt items exceed their count bound.")
        return tuple(
            RecallReceiptItem.model_validate(
                item.model_dump(mode="python") if type(item) is RecallReceiptItem else item
            )
            for item in value
        )

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        source_names = tuple(source.source for source in self.sources)
        if len(source_names) != len(set(source_names)):
            raise ValueError("Recall receipt source identities must be unique.")
        declared_channels = {channel for source in self.sources for channel in source.channels}
        if any(
            channel not in declared_channels
            for item in self.items
            for channel in item.match_channels
        ):
            raise ValueError("Recall receipt item channels must be declared by source coverage.")
        ordinals = tuple(item.ordinal for item in self.items)
        if ordinals != tuple(range(len(self.items))):
            raise ValueError("Recall receipt item ordinals must be contiguous from zero.")
        identities = tuple(item.identity.sort_key() for item in self.items)
        if len(identities) != len(set(identities)):
            raise ValueError("Recall receipt items must have unique canonical identities.")
        fused_ranks = tuple(item.fused_rank for item in self.items)
        if len(fused_ranks) != len(set(fused_ranks)):
            raise ValueError("Recall receipt items must have unique fused ranks.")
        if any(fused_rank > self.inspected_count for fused_rank in fused_ranks):
            raise ValueError("Recall receipt item rank exceeds its inspected candidate count.")
        if self.admitted_count != sum(
            item.admission is RecallItemAdmission.ADMITTED for item in self.items
        ):
            raise ValueError("Recall receipt admitted count does not match its items.")
        if self.offered_count != sum(
            item.admission is RecallItemAdmission.OFFERED for item in self.items
        ):
            raise ValueError("Recall receipt offered count does not match its items.")
        if self.eligible_count != (
            self.admitted_count + self.offered_count + self.silent_count + self.omitted_count
        ):
            raise ValueError("Recall receipt eligible count does not match its outcomes.")
        if self.inspected_count < self.eligible_count:
            raise ValueError("Recall receipt inspected count is smaller than eligible count.")
        if self.inspected_count != sum(source.inspected_count for source in self.sources):
            raise ValueError("Recall receipt inspected count does not match source coverage.")
        expected_truncation = self.omitted_count > 0 or any(
            source.truncated for source in self.sources
        )
        if self.truncated is not expected_truncation:
            raise ValueError("Recall receipt truncation does not match its bounded evidence.")
        _enforce_document_bytes(self, MAX_RECALL_RECEIPT_BYTES, "Recall receipt")
        return self


class ContextExposureTransition(BaseModel):
    model_config = _MODEL_CONFIG

    transition_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    revision: StrictInt = Field(ge=0, lt=MAX_CONTEXT_EXPOSURE_TRANSITIONS)
    state: ContextExposureState
    occurred_at: datetime
    evidence_kind: ContextExposureEvidenceKind
    evidence_ref: str = Field(max_length=MAX_MEMORY_EVIDENCE_REFERENCE_CHARS)
    provider_request_id: str | None = Field(
        default=None,
        max_length=MAX_MEMORY_EVIDENCE_REFERENCE_CHARS,
    )

    @field_validator("transition_id")
    @classmethod
    def validate_transition_id(cls, value: str) -> str:
        return _identity(value, "transition_id")

    @field_validator("evidence_ref")
    @classmethod
    def validate_evidence_ref(cls, value: str) -> str:
        return _reference_identity(value, "evidence_ref")

    @field_validator("provider_request_id")
    @classmethod
    def validate_optional_provider_request_id(cls, value: str | None) -> str | None:
        return None if value is None else _reference_identity(value, "provider_request_id")

    @field_validator("occurred_at")
    @classmethod
    def normalize_occurred_at(cls, value: datetime) -> datetime:
        return _utc(value, "occurred_at")

    @model_validator(mode="after")
    def validate_evidence_kind(self) -> Self:
        if self.evidence_kind not in _STATE_EVIDENCE_KINDS[self.state]:
            raise ValueError("Context exposure evidence kind does not prove its state.")
        supports_provider_request_id = self.state in {
            ContextExposureState.ACKNOWLEDGED,
            ContextExposureState.COMPLETED,
        }
        if not supports_provider_request_id and self.provider_request_id is not None:
            raise ValueError(
                "Only provider acknowledgement/completion evidence may carry a provider request id."
            )
        return self


class ContextExposure(BaseModel):
    """Durable lifecycle of one exact provider-neutral composition and attempt."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["cayu.context_exposure.v1"] = CONTEXT_EXPOSURE_VERSION
    exposure_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    session_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_SESSION_ID_BYTES)
    interaction_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    model_step_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    model_attempt_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    provider_attempt_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    provider_name: str
    model_name: str
    composition_fingerprint: KeyedEvidenceFingerprint
    execution_profile_fingerprint: KeyedEvidenceFingerprint
    context_policy_fingerprint: KeyedEvidenceFingerprint
    tool_exposure_fingerprint: KeyedEvidenceFingerprint
    request_contract_fingerprint: KeyedEvidenceFingerprint
    receipt_ids: tuple[str, ...] = ()
    contributor_ids: tuple[str, ...] = ()
    created_at: datetime
    updated_at: datetime
    state: ContextExposureState
    state_revision: StrictInt = Field(ge=0, lt=MAX_CONTEXT_EXPOSURE_TRANSITIONS)
    transitions: tuple[ContextExposureTransition, ...]

    @field_validator(
        "exposure_id",
        "interaction_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _identity(value, info.field_name)

    @field_validator("model_step_id", "model_attempt_id")
    @classmethod
    def validate_execution_unit_ids(cls, value: str, info: Any) -> str:
        return require_execution_unit_id(value, info.field_name)

    @field_validator("provider_attempt_id")
    @classmethod
    def validate_provider_attempt_id(cls, value: str) -> str:
        return _provider_attempt_identity(value)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return _session_identity(value)

    @field_validator("provider_name", "model_name")
    @classmethod
    def validate_names(cls, value: str, info: Any) -> str:
        return _bounded_text(value, info.field_name)

    @field_validator(
        "composition_fingerprint",
        "execution_profile_fingerprint",
        "context_policy_fingerprint",
        "tool_exposure_fingerprint",
        "request_contract_fingerprint",
        mode="before",
    )
    @classmethod
    def copy_fingerprints(cls, value: Any) -> KeyedEvidenceFingerprint:
        return KeyedEvidenceFingerprint.model_validate(
            value.model_dump(mode="python") if type(value) is KeyedEvidenceFingerprint else value
        )

    @model_validator(mode="after")
    def validate_fingerprint_domains(self) -> Self:
        expected = {
            "composition_fingerprint": KeyedEvidenceFingerprintDomain.COMPOSITION,
            "execution_profile_fingerprint": KeyedEvidenceFingerprintDomain.EXECUTION_PROFILE,
            "context_policy_fingerprint": KeyedEvidenceFingerprintDomain.CONTEXT_POLICY,
            "tool_exposure_fingerprint": KeyedEvidenceFingerprintDomain.TOOL_EXPOSURE,
            "request_contract_fingerprint": KeyedEvidenceFingerprintDomain.REQUEST_CONTRACT,
        }
        for field_name, domain in expected.items():
            if getattr(self, field_name).domain is not domain:
                raise ValueError(f"`{field_name}` has the wrong keyed-fingerprint domain.")
        return self

    @field_validator("receipt_ids", mode="before")
    @classmethod
    def copy_receipt_ids(cls, value: Any) -> tuple[str, ...]:
        return _copy_identity_tuple(value, "receipt_id", MAX_CONTEXT_EXPOSURE_RECEIPTS)

    @field_validator("contributor_ids", mode="before")
    @classmethod
    def copy_contributor_ids(cls, value: Any) -> tuple[str, ...]:
        return _copy_identity_tuple(
            value,
            "contributor_id",
            MAX_CONTEXT_EXPOSURE_CONTRIBUTORS,
        )

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime, info: Any) -> datetime:
        return _utc(value, info.field_name)

    @field_validator("transitions", mode="before")
    @classmethod
    def copy_transitions(cls, value: Any) -> tuple[ContextExposureTransition, ...]:
        if type(value) not in (list, tuple) or not value:
            raise ValueError("transitions must contain at least the planned transition.")
        if len(value) > MAX_CONTEXT_EXPOSURE_TRANSITIONS:
            raise ValueError("Context exposure transitions exceed their count bound.")
        return tuple(
            ContextExposureTransition.model_validate(
                transition.model_dump(mode="python")
                if type(transition) is ContextExposureTransition
                else transition
            )
            for transition in value
        )

    @model_validator(mode="after")
    def validate_exposure(self) -> Self:
        first = self.transitions[0]
        if first.revision != 0 or first.state is not ContextExposureState.PLANNED:
            raise ValueError("A context exposure must begin with revision-zero planned state.")
        for previous, current in zip(self.transitions, self.transitions[1:], strict=False):
            if current.revision != previous.revision + 1:
                raise ValueError("Context exposure transition revisions must be contiguous.")
            if current.state not in _ALLOWED_EXPOSURE_TRANSITIONS[previous.state]:
                raise ValueError("Context exposure lifecycle contains an invalid transition.")
            if current.occurred_at < previous.occurred_at:
                raise ValueError("Context exposure transition timestamps must be monotonic.")
        transition_ids = tuple(transition.transition_id for transition in self.transitions)
        if len(transition_ids) != len(set(transition_ids)):
            raise ValueError("Context exposure transition identities must be unique.")
        provider_request_ids = {
            transition.provider_request_id
            for transition in self.transitions
            if transition.provider_request_id is not None
        }
        if len(provider_request_ids) > 1:
            raise ValueError("A context exposure cannot combine different provider requests.")
        latest = self.transitions[-1]
        if self.state is not latest.state or self.state_revision != latest.revision:
            raise ValueError("Context exposure state must match its latest transition.")
        if self.created_at != first.occurred_at or self.updated_at != latest.occurred_at:
            raise ValueError("Context exposure timestamps must match its transition history.")
        _enforce_document_bytes(self, MAX_CONTEXT_EXPOSURE_BYTES, "Context exposure")
        return self

    @property
    def provider_exposure_proven(self) -> bool:
        """Whether acknowledgement/completion evidence proves provider exposure."""

        return any(
            transition.state in {ContextExposureState.ACKNOWLEDGED, ContextExposureState.COMPLETED}
            for transition in self.transitions
        )


class RecallItemExposure(BaseModel):
    """One exact recalled representation included in a context exposure plan."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["cayu.recall_item_exposure.v1"] = RECALL_ITEM_EXPOSURE_VERSION
    exposure_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    receipt_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    ordinal: StrictInt = Field(ge=0, lt=MAX_RECALL_RECEIPT_ITEMS)
    receipt_item_ordinal: StrictInt = Field(ge=0, lt=MAX_RECALL_RECEIPT_ITEMS)
    identity: RetrievalCandidateIdentity
    representation_id: str
    content_sha256: str
    locator: RecallEvidenceLocator
    admission: RecallItemAdmission
    selection_reason: RecallItemSelectionReason

    @field_validator("exposure_id", "receipt_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _identity(value, info.field_name)

    @field_validator("identity", mode="before")
    @classmethod
    def copy_identity(cls, value: Any) -> RetrievalCandidateIdentity:
        return _copy_bounded_candidate_identity(value)

    @field_validator("representation_id")
    @classmethod
    def validate_representation_id(cls, value: str) -> str:
        return _bounded_text(value, "representation_id")

    @field_validator("content_sha256")
    @classmethod
    def validate_content_sha256(cls, value: str) -> str:
        return _digest(value, "content_sha256")

    @field_validator("locator", mode="before")
    @classmethod
    def copy_locator(cls, value: Any) -> Any:
        return _copy_recall_evidence_locator_input(value)

    @field_validator("locator")
    @classmethod
    def validate_locator(cls, value: RecallEvidenceLocator) -> RecallEvidenceLocator:
        _validate_recall_evidence_locator_bytes(value)
        return value

    @model_validator(mode="after")
    def validate_document_size(self) -> Self:
        _validate_admission_reason(self.admission, self.selection_reason)
        _validate_recall_evidence_locator_identity(
            self.locator,
            self.identity,
            self.content_sha256,
        )
        _enforce_document_bytes(
            self,
            MAX_RECALL_ITEM_EXPOSURE_BYTES,
            "Recall item exposure",
        )
        return self


class ContextExposureTransitionRequest(BaseModel):
    """Compare-and-swap request for one exact, idempotent lifecycle transition."""

    model_config = _MODEL_CONFIG

    transition_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    expected_state: ContextExposureState
    expected_revision: StrictInt = Field(ge=0, lt=MAX_CONTEXT_EXPOSURE_TRANSITIONS - 1)
    state: ContextExposureState
    occurred_at: datetime
    evidence_kind: ContextExposureEvidenceKind
    evidence_ref: str = Field(max_length=MAX_MEMORY_EVIDENCE_REFERENCE_CHARS)
    provider_request_id: str | None = Field(
        default=None,
        max_length=MAX_MEMORY_EVIDENCE_REFERENCE_CHARS,
    )

    @field_validator("transition_id")
    @classmethod
    def validate_transition_id(cls, value: str) -> str:
        return _identity(value, "transition_id")

    @field_validator("evidence_ref")
    @classmethod
    def validate_evidence_ref(cls, value: str) -> str:
        return _reference_identity(value, "evidence_ref")

    @field_validator("provider_request_id")
    @classmethod
    def validate_optional_provider_request_id(cls, value: str | None) -> str | None:
        return None if value is None else _reference_identity(value, "provider_request_id")

    @field_validator("occurred_at")
    @classmethod
    def normalize_occurred_at(cls, value: datetime) -> datetime:
        return _utc(value, "occurred_at")

    @model_validator(mode="after")
    def validate_requested_transition(self) -> Self:
        if self.state not in _ALLOWED_EXPOSURE_TRANSITIONS[self.expected_state]:
            raise ValueError("Requested context exposure lifecycle transition is invalid.")
        ContextExposureTransition(
            transition_id=self.transition_id,
            revision=self.expected_revision + 1,
            state=self.state,
            occurred_at=self.occurred_at,
            evidence_kind=self.evidence_kind,
            evidence_ref=self.evidence_ref,
            provider_request_id=self.provider_request_id,
        )
        return self

    def materialize(self) -> ContextExposureTransition:
        return ContextExposureTransition(
            transition_id=self.transition_id,
            revision=self.expected_revision + 1,
            state=self.state,
            occurred_at=self.occurred_at,
            evidence_kind=self.evidence_kind,
            evidence_ref=self.evidence_ref,
            provider_request_id=self.provider_request_id,
        )


class RecallEvidenceQuery(BaseModel):
    """Explicit session-scoped keyset query for bounded receipt/exposure pages."""

    model_config = _MODEL_CONFIG

    session_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_SESSION_ID_BYTES)
    interaction_id: str | None = Field(
        default=None,
        max_length=MAX_MEMORY_EVIDENCE_ID_CHARS,
    )
    model_step_id: str | None = Field(
        default=None,
        max_length=MAX_MEMORY_EVIDENCE_ID_CHARS,
    )
    limit: StrictInt = Field(default=50, ge=1, le=MAX_MEMORY_EVIDENCE_PAGE_LIMIT)
    max_bytes: StrictInt = Field(
        default=1_000_000,
        ge=MIN_MEMORY_EVIDENCE_PAGE_BYTES,
        le=MAX_MEMORY_EVIDENCE_PAGE_BYTES,
    )
    cursor: str | None = Field(default=None, max_length=MAX_MEMORY_EVIDENCE_CURSOR_BYTES)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return _session_identity(value)

    @field_validator("interaction_id")
    @classmethod
    def validate_optional_ids(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _identity(value, info.field_name)

    @field_validator("model_step_id")
    @classmethod
    def validate_optional_model_step_id(cls, value: str | None) -> str | None:
        return None if value is None else require_execution_unit_id(value, "model_step_id")

    def fingerprint(self, record_kind: Literal["receipt", "exposure"]) -> str:
        from hashlib import sha256

        return sha256(
            canonical_durable_json_bytes(
                {
                    "record_kind": record_kind,
                    "session_id": self.session_id,
                    "interaction_id": self.interaction_id,
                    "model_step_id": self.model_step_id,
                },
                "recall evidence query",
            )
        ).hexdigest()


class RecallReceiptPage(BaseModel):
    model_config = _MODEL_CONFIG

    items: tuple[RecallReceipt, ...]
    next_cursor: str | None = Field(default=None, max_length=MAX_MEMORY_EVIDENCE_CURSOR_BYTES)
    truncated: StrictBool

    @field_validator("items", mode="before")
    @classmethod
    def copy_items(cls, value: Any) -> tuple[RecallReceipt, ...]:
        return _copy_page_items(value, RecallReceipt, "recall receipt page")

    @field_validator("next_cursor")
    @classmethod
    def validate_next_cursor(cls, value: str | None) -> str | None:
        return _optional_cursor(value)

    @model_validator(mode="after")
    def validate_page(self) -> Self:
        _validate_page_envelope(
            items=self.items,
            next_cursor=self.next_cursor,
            truncated=self.truncated,
            label="Recall receipt page",
        )
        return self


class ContextExposurePage(BaseModel):
    model_config = _MODEL_CONFIG

    items: tuple[ContextExposure, ...]
    next_cursor: str | None = Field(default=None, max_length=MAX_MEMORY_EVIDENCE_CURSOR_BYTES)
    truncated: StrictBool

    @field_validator("items", mode="before")
    @classmethod
    def copy_items(cls, value: Any) -> tuple[ContextExposure, ...]:
        return _copy_page_items(value, ContextExposure, "context exposure page")

    @field_validator("next_cursor")
    @classmethod
    def validate_next_cursor(cls, value: str | None) -> str | None:
        return _optional_cursor(value)

    @model_validator(mode="after")
    def validate_page(self) -> Self:
        _validate_page_envelope(
            items=self.items,
            next_cursor=self.next_cursor,
            truncated=self.truncated,
            label="Context exposure page",
        )
        return self


def new_recall_receipt_id() -> str:
    from uuid import uuid4

    return f"rrcp_{uuid4().hex}"


def new_context_exposure_id() -> str:
    from uuid import uuid4

    return f"cexp_{uuid4().hex}"


def new_context_exposure_transition_id() -> str:
    from uuid import uuid4

    return f"cext_{uuid4().hex}"


def new_provider_attempt_id() -> str:
    from uuid import uuid4

    return f"patt_{uuid4().hex}"


def keyed_evidence_fingerprint(
    source_sha256: str,
    *,
    domain: KeyedEvidenceFingerprintDomain,
    key_id: str,
    key: bytes,
) -> KeyedEvidenceFingerprint:
    """Blind an existing exact digest with a domain-separated HMAC key."""

    source_sha256 = _digest(source_sha256, "source_sha256")
    domain = KeyedEvidenceFingerprintDomain(domain)
    key_id = _bounded_text(key_id, "key_id")
    if type(key) is not bytes:
        raise TypeError("key must be bytes.")
    if not 32 <= len(key) <= 1_024:
        raise ValueError("key must contain between 32 and 1024 bytes.")
    material = b"cayu.memory_evidence.v1\x00" + domain.value.encode("ascii")
    material += b"\x00" + source_sha256.encode("ascii")
    return KeyedEvidenceFingerprint(
        domain=domain,
        key_id=key_id,
        digest=hmac.digest(key, material, "sha256").hex(),
    )


def append_context_exposure_transition(
    exposure: ContextExposure,
    request: ContextExposureTransitionRequest,
) -> ContextExposure:
    """Apply one validated transition after its caller has won the durable CAS fence."""

    copied = copy_context_exposure(exposure)
    copied_request = ContextExposureTransitionRequest.model_validate(
        request.model_dump(mode="python")
    )
    if (
        copied.state is not copied_request.expected_state
        or copied.state_revision != copied_request.expected_revision
    ):
        raise ContextExposureTransitionConflict(
            copied.exposure_id,
            expected_state=copied_request.expected_state,
            expected_revision=copied_request.expected_revision,
            actual_state=copied.state,
            actual_revision=copied.state_revision,
        )
    transition = copied_request.materialize()
    if transition.occurred_at < copied.updated_at:
        raise ValueError("Context exposure transition timestamp precedes durable state.")
    return ContextExposure.model_validate(
        {
            **copied.model_dump(mode="python"),
            "updated_at": transition.occurred_at,
            "state": transition.state,
            "state_revision": transition.revision,
            "transitions": (*copied.transitions, transition),
        }
    )


def context_exposure_transition_replays(
    exposure: ContextExposure,
    request: ContextExposureTransitionRequest,
) -> bool:
    """Whether a transition request exactly matches its durable fenced write."""

    transition = request.materialize()
    if transition.revision >= len(exposure.transitions):
        return False
    return (
        exposure.transitions[transition.revision] == transition
        and exposure.transitions[transition.revision - 1].state is request.expected_state
    )


def _validate_recall_evidence_locator_bytes(locator: RecallEvidenceLocator) -> None:
    if (
        len(
            canonical_durable_json_bytes(
                locator.model_dump(mode="json"),
                "recall evidence locator",
            )
        )
        > MAX_RECALL_ITEM_LOCATOR_BYTES
    ):
        raise ValueError("Recall evidence locator exceeds its byte bound.")


def _validate_recall_evidence_locator_identity(
    locator: RecallEvidenceLocator,
    identity: RetrievalCandidateIdentity,
    content_sha256: str,
) -> None:
    if isinstance(locator, KnowledgeEntryEvidenceLocator):
        matches = (
            identity.record_type == "knowledge_entry"
            and identity.record_id == locator.entry_id
            and identity.revision == str(locator.entry_revision)
        )
    elif isinstance(locator, KnowledgeChunkEvidenceLocator):
        matches = (
            identity.record_type == "knowledge_chunk"
            and identity.record_id == locator.chunk_id
            and identity.revision == str(locator.entry_revision)
        )
    elif isinstance(locator, TranscriptMessageEvidenceLocator):
        matches = (
            identity.record_type == "transcript_message"
            and identity.record_id == f"{locator.session_id}:{locator.transcript_index}"
            and identity.revision == content_sha256
        )
    else:
        matches = isinstance(locator, OpaqueRecallEvidenceLocator) and identity.record_type not in {
            "knowledge_entry",
            "knowledge_chunk",
            "transcript_message",
        }
    if not matches:
        raise ValueError("Recall evidence locator conflicts with its canonical identity.")


def recall_item_exposure_matches_receipt_item(
    exposure: RecallItemExposure,
    receipt: RecallReceipt,
) -> bool:
    if exposure.receipt_id != receipt.receipt_id:
        return False
    if exposure.receipt_item_ordinal >= len(receipt.items):
        return False
    item = receipt.items[exposure.receipt_item_ordinal]
    return (
        exposure.identity == item.identity
        and exposure.representation_id == item.representation_id
        and exposure.content_sha256 == item.content_sha256
        and exposure.locator == item.locator
        and exposure.admission is item.admission
        and exposure.selection_reason is item.selection_reason
    )


def validate_new_context_exposure(
    exposure: ContextExposure,
    item_exposures: tuple[RecallItemExposure, ...],
) -> None:
    """Validate one exact, revision-zero exposure bundle before persistence."""

    if (
        exposure.state is not ContextExposureState.PLANNED
        or exposure.state_revision != 0
        or len(exposure.transitions) != 1
    ):
        raise ValueError("A new context exposure must contain only planned state.")
    if tuple(item.ordinal for item in item_exposures) != tuple(range(len(item_exposures))):
        raise ValueError("Recall item exposure ordinals must be contiguous from zero.")
    if any(item.exposure_id != exposure.exposure_id for item in item_exposures):
        raise ValueError("Recall item exposure identity does not match its parent.")
    if any(item.receipt_id not in exposure.receipt_ids for item in item_exposures):
        raise ValueError("Recall item exposure references an unlinked receipt.")
    receipt_items = tuple((item.receipt_id, item.receipt_item_ordinal) for item in item_exposures)
    if len(receipt_items) != len(set(receipt_items)):
        raise ValueError("A receipt item may appear only once in one context exposure.")


def context_exposure_creation_matches(
    current: ContextExposure,
    requested: ContextExposure,
) -> bool:
    """Compare an original create request with a possibly advanced lifecycle."""

    current = copy_context_exposure(current)
    requested = copy_context_exposure(requested)
    validate_new_context_exposure(requested, ())
    initial = ContextExposure.model_validate(
        {
            **current.model_dump(mode="python"),
            "updated_at": current.created_at,
            "state": ContextExposureState.PLANNED,
            "state_revision": 0,
            "transitions": current.transitions[:1],
        }
    )
    return memory_evidence_document_bytes(
        initial,
        "stored context exposure creation",
    ) == memory_evidence_document_bytes(requested, "context exposure creation")


def validate_context_exposure_receipt_scope(
    exposure: ContextExposure,
    receipt: RecallReceipt,
) -> None:
    """Reject a receipt linked across interaction or model-step boundaries."""

    if (
        receipt.session_id != exposure.session_id
        or receipt.interaction_id != exposure.interaction_id
        or receipt.model_step_id != exposure.model_step_id
    ):
        raise ValueError("Context exposure receipt scope does not match its parent.")
    if receipt.created_at > exposure.created_at:
        raise ValueError("Context exposure cannot predate its linked recall receipt.")


def copy_recall_receipt(receipt: RecallReceipt) -> RecallReceipt:
    if type(receipt) is not RecallReceipt:
        raise TypeError("receipt must be a RecallReceipt.")
    return RecallReceipt.model_validate(receipt.model_dump(mode="python"))


def copy_context_exposure(exposure: ContextExposure) -> ContextExposure:
    if type(exposure) is not ContextExposure:
        raise TypeError("exposure must be a ContextExposure.")
    return ContextExposure.model_validate(exposure.model_dump(mode="python"))


def copy_recall_item_exposure(exposure: RecallItemExposure) -> RecallItemExposure:
    if type(exposure) is not RecallItemExposure:
        raise TypeError("item exposure must be a RecallItemExposure.")
    return RecallItemExposure.model_validate(exposure.model_dump(mode="python"))


def memory_evidence_document_bytes(value: BaseModel, field_name: str) -> bytes:
    return canonical_durable_json_bytes(value.model_dump(mode="json"), field_name)


def require_memory_evidence_id(value: str, field_name: str) -> str:
    """Validate one bounded public receipt/exposure identity."""

    return _identity(value, field_name)


def require_memory_evidence_session_id(value: str) -> str:
    """Validate one session identity at the SessionStore byte boundary."""

    return _session_identity(value)


def encode_recall_evidence_cursor(
    *,
    record_kind: Literal["receipt", "exposure"],
    query_fingerprint: str,
    created_at: datetime,
    record_id: str,
) -> str:
    payload = {
        "version": _CURSOR_VERSION,
        "record_kind": record_kind,
        "query_fingerprint": _digest(query_fingerprint, "query_fingerprint"),
        "created_at": _utc(created_at, "created_at").isoformat(),
        "record_id": _identity(record_id, "record_id"),
    }
    encoded = base64.urlsafe_b64encode(
        canonical_durable_json_bytes(payload, "recall evidence cursor")
    ).decode("ascii")
    if len(encoded) > MAX_MEMORY_EVIDENCE_CURSOR_BYTES:
        raise ValueError("Recall evidence cursor exceeds its byte bound.")
    return encoded


def decode_recall_evidence_cursor(
    cursor: str,
    *,
    record_kind: Literal["receipt", "exposure"],
    query_fingerprint: str,
) -> tuple[datetime, str]:
    if type(cursor) is not str or not cursor or len(cursor) > MAX_MEMORY_EVIDENCE_CURSOR_BYTES:
        raise ValueError("Recall evidence cursor is invalid.")
    try:
        raw = base64.b64decode(cursor.encode("ascii"), altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw).decode("ascii") != cursor:
            raise ValueError("Recall evidence cursor is non-canonical.")
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, binascii.Error, ValueError) as exc:
        raise ValueError("Recall evidence cursor is invalid.") from exc
    if type(payload) is not dict or set(payload) != {
        "version",
        "record_kind",
        "query_fingerprint",
        "created_at",
        "record_id",
    }:
        raise ValueError("Recall evidence cursor is invalid.")
    if type(payload["version"]) is not int or payload["version"] != _CURSOR_VERSION:
        raise ValueError("Recall evidence cursor is invalid.")
    if payload["record_kind"] != record_kind or payload["query_fingerprint"] != _digest(
        query_fingerprint,
        "query_fingerprint",
    ):
        raise ValueError("Recall evidence cursor does not match its query.")
    if type(payload["created_at"]) is not str:
        raise ValueError("Recall evidence cursor is invalid.")
    try:
        created_at = datetime.fromisoformat(payload["created_at"])
    except ValueError as exc:
        raise ValueError("Recall evidence cursor is invalid.") from exc
    return _utc(created_at, "cursor created_at"), _identity(payload["record_id"], "record_id")


def _copy_identity_tuple(value: Any, field_name: str, limit: int) -> tuple[str, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{field_name}s must be a list or tuple.")
    if len(value) > limit:
        raise ValueError(f"{field_name}s exceed their count bound.")
    copied = tuple(_identity(item, field_name) for item in value)
    if len(copied) != len(set(copied)):
        raise ValueError(f"{field_name}s must be unique.")
    return copied


def _copy_bounded_candidate_identity(value: Any) -> RetrievalCandidateIdentity:
    if isinstance(value, RetrievalCandidateIdentity):
        if type(value) is not RetrievalCandidateIdentity:
            raise TypeError("identity must not be a RetrievalCandidateIdentity subclass.")
        value = value.model_dump(mode="python")
    identity = RetrievalCandidateIdentity.model_validate(value)
    return RetrievalCandidateIdentity(
        record_type=_bounded_text(identity.record_type, "record_type"),
        record_id=_bounded_utf8_identity(
            identity.record_id,
            "record_id",
            MAX_RECALL_CANDIDATE_RECORD_ID_BYTES,
        ),
        revision=_bounded_text(identity.revision, "revision"),
    )


def _copy_recall_evidence_locator_input(value: Any) -> Any:
    locator_types = (
        KnowledgeEntryEvidenceLocator,
        KnowledgeChunkEvidenceLocator,
        TranscriptMessageEvidenceLocator,
        OpaqueRecallEvidenceLocator,
    )
    if isinstance(value, locator_types):
        if type(value) not in locator_types:
            raise TypeError("locator must not be a RecallEvidenceLocator subclass.")
        return value.model_dump(mode="python")
    return value


_EvidencePageItem = TypeVar("_EvidencePageItem", RecallReceipt, ContextExposure)


def _copy_page_items(
    value: Any,
    model_type: type[_EvidencePageItem],
    label: str,
) -> tuple[_EvidencePageItem, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{label} items must be a list or tuple.")
    if len(value) > MAX_MEMORY_EVIDENCE_PAGE_LIMIT:
        raise ValueError(f"{label} exceeds its item-count bound.")
    return tuple(
        model_type.model_validate(
            item.model_dump(mode="python") if type(item) is model_type else item
        )
        for item in value
    )


def _optional_cursor(value: str | None) -> str | None:
    if value is None:
        return None
    return require_durable_clean_nonblank(value, "next_cursor")


def _validate_page_envelope(
    *,
    items: tuple[RecallReceipt, ...] | tuple[ContextExposure, ...],
    next_cursor: str | None,
    truncated: bool,
    label: str,
) -> None:
    if (next_cursor is not None) is not truncated:
        raise ValueError(f"{label} cursor does not match its truncation state.")
    if truncated and not items:
        raise ValueError(f"{label} cannot advance an empty truncated page.")
    serialized_bytes = 2 + sum(
        len(memory_evidence_document_bytes(item, f"{label.lower()} item")) for item in items
    )
    serialized_bytes += max(0, len(items) - 1)
    if serialized_bytes > MAX_MEMORY_EVIDENCE_PAGE_BYTES:
        raise ValueError(f"{label} exceeds its serialized item-byte bound.")


def _bounded_text(value: str, field_name: str) -> str:
    result = require_durable_clean_nonblank(value, field_name)
    if len(result.encode("utf-8")) > MAX_MEMORY_EVIDENCE_NAME_BYTES:
        raise ValueError(f"`{field_name}` exceeds its UTF-8 byte bound.")
    return result


def _identity(value: str, field_name: str) -> str:
    result = require_durable_clean_nonblank(value, field_name)
    if len(result) > MAX_MEMORY_EVIDENCE_ID_CHARS:
        raise ValueError(f"`{field_name}` exceeds its character bound.")
    return result


def _reference_identity(value: str, field_name: str) -> str:
    result = require_durable_clean_nonblank(value, field_name)
    if len(result) > MAX_MEMORY_EVIDENCE_REFERENCE_CHARS:
        raise ValueError(f"`{field_name}` exceeds its character bound.")
    return result


def _bounded_utf8_identity(value: str, field_name: str, max_bytes: int) -> str:
    result = require_durable_clean_nonblank(value, field_name)
    if len(result.encode("utf-8")) > max_bytes:
        raise ValueError(f"`{field_name}` exceeds its UTF-8 byte bound.")
    return result


def _session_identity(value: str) -> str:
    result = require_durable_clean_nonblank(value, "session_id")
    if len(result.encode("utf-8")) > MAX_MEMORY_EVIDENCE_SESSION_ID_BYTES:
        raise ValueError("`session_id` exceeds its UTF-8 byte bound.")
    return result


def _provider_attempt_identity(value: str) -> str:
    result = _identity(value, "provider_attempt_id")
    if (
        len(result) != 37
        or not result.startswith("patt_")
        or any(character not in _SHA256_HEX for character in result[5:])
    ):
        raise ValueError("provider_attempt_id is not a valid Cayu provider-attempt identifier.")
    return result


def _digest(value: str, field_name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _SHA256_HEX for char in value):
        raise ValueError(f"`{field_name}` must be a lowercase SHA-256 digest.")
    return value


def _validate_admission_reason(
    admission: RecallItemAdmission,
    selection_reason: RecallItemSelectionReason,
) -> None:
    admitted_reasons = {
        RecallItemSelectionReason.CALIBRATED_STRONG_MATCH,
        RecallItemSelectionReason.EXPLICIT_APPLICATION_SELECTION,
    }
    if (selection_reason in admitted_reasons) is not (admission is RecallItemAdmission.ADMITTED):
        raise ValueError("Recall item selection reason does not match its admission.")


def _utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"`{field_name}` must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"`{field_name}` must be timezone-aware.")
    return value.astimezone(UTC)


def _nonnegative_int(value: int, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"`{field_name}` must be a non-negative integer.")
    return value


def _enforce_document_bytes(value: BaseModel, limit: int, label: str) -> None:
    if len(memory_evidence_document_bytes(value, label.lower())) > limit:
        raise ValueError(f"{label} exceeds its serialized byte bound.")


__all__ = [
    "CONTEXT_EXPOSURE_VERSION",
    "MAX_CONTEXT_EXPOSURE_TRANSITIONS",
    "MAX_MEMORY_EVIDENCE_PAGE_BYTES",
    "MAX_MEMORY_EVIDENCE_PAGE_LIMIT",
    "MAX_RECALL_RECEIPT_ITEMS",
    "RECALL_ITEM_EXPOSURE_VERSION",
    "RECALL_RECEIPT_VERSION",
    "ContextExposure",
    "ContextExposureEvidenceKind",
    "ContextExposurePage",
    "ContextExposureState",
    "ContextExposureTransition",
    "ContextExposureTransitionConflict",
    "ContextExposureTransitionRequest",
    "KeyedEvidenceFingerprint",
    "KeyedEvidenceFingerprintDomain",
    "KnowledgeChunkEvidenceLocator",
    "KnowledgeEntryEvidenceLocator",
    "OpaqueRecallEvidenceLocator",
    "RecallEvidenceConflict",
    "RecallEvidenceLocator",
    "RecallEvidenceQuery",
    "RecallItemAdmission",
    "RecallItemExposure",
    "RecallItemSelectionReason",
    "RecallReceipt",
    "RecallReceiptItem",
    "RecallReceiptPage",
    "RecallSourceCoverage",
    "RecallSourceCoverageState",
    "TranscriptMessageEvidenceLocator",
    "append_context_exposure_transition",
    "context_exposure_creation_matches",
    "copy_context_exposure",
    "copy_recall_item_exposure",
    "copy_recall_receipt",
    "decode_recall_evidence_cursor",
    "encode_recall_evidence_cursor",
    "keyed_evidence_fingerprint",
    "memory_evidence_document_bytes",
    "new_context_exposure_id",
    "new_context_exposure_transition_id",
    "new_provider_attempt_id",
    "new_recall_receipt_id",
    "recall_item_exposure_matches_receipt_item",
]
