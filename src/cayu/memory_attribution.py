"""Bounded, secret-safe public attribution for durable memory evidence."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import MAX_DURABLE_JSON_INTEGER, require_durable_clean_nonblank
from cayu.memory_evidence import (
    MAX_CONTEXT_EXPOSURE_CONTRIBUTORS,
    MAX_CONTEXT_EXPOSURE_RECEIPTS,
    MAX_CONTEXT_EXPOSURE_TRANSITIONS,
    MAX_MEMORY_EVIDENCE_ID_CHARS,
    MAX_RECALL_RECEIPT_ITEMS,
    MAX_RECALL_RECEIPT_SOURCES,
    ContextExposureEvidenceKind,
    ContextExposureState,
    RecallItemAdmission,
    RecallItemSelectionReason,
)

MEMORY_ATTRIBUTION_VERSION = "cayu.memory_attribution.v1"

MEMORY_ATTRIBUTION_DEFAULT_MAX_RECEIPTS = 100
MEMORY_ATTRIBUTION_DEFAULT_MAX_EXPOSURES = 100
MEMORY_ATTRIBUTION_DEFAULT_MAX_ITEMS = 1_000
MEMORY_ATTRIBUTION_DEFAULT_MAX_SOURCE_BYTES = 16 * 1024 * 1024
MEMORY_ATTRIBUTION_DEFAULT_MAX_PROJECTION_BYTES = 4 * 1024 * 1024

MEMORY_ATTRIBUTION_HARD_MAX_RECEIPTS = 1_000
MEMORY_ATTRIBUTION_HARD_MAX_EXPOSURES = 1_000
MEMORY_ATTRIBUTION_HARD_MAX_ITEMS = 10_000
MEMORY_ATTRIBUTION_HARD_MAX_SOURCE_BYTES = 64 * 1024 * 1024
MEMORY_ATTRIBUTION_HARD_MAX_PROJECTION_BYTES = 16 * 1024 * 1024

_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    hide_input_in_errors=True,
    revalidate_instances="always",
    validate_default=True,
)


class MemoryAttributionStatus(StrEnum):
    """Trust state of one session's projected memory evidence."""

    COMPLETE = "complete"
    TRUNCATED = "truncated"
    UNAVAILABLE = "unavailable"
    REDACTED = "redacted"
    CONTRADICTORY = "contradictory"


class MemoryAttributionUnavailableReason(StrEnum):
    """Stable reason memory attribution could not be represented."""

    STORE_UNSUPPORTED = "store_unsupported"
    EVIDENCE_READ_FAILED = "evidence_read_failed"
    ALIAS_KEY_UNAVAILABLE = "alias_key_unavailable"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"


class MemoryAttributionBounds(BaseModel):
    """Global bounds shared by every session in one evidence capture."""

    model_config = _MODEL_CONFIG

    max_receipts: StrictInt = Field(
        default=MEMORY_ATTRIBUTION_DEFAULT_MAX_RECEIPTS,
        ge=1,
        le=MEMORY_ATTRIBUTION_HARD_MAX_RECEIPTS,
    )
    max_exposures: StrictInt = Field(
        default=MEMORY_ATTRIBUTION_DEFAULT_MAX_EXPOSURES,
        ge=1,
        le=MEMORY_ATTRIBUTION_HARD_MAX_EXPOSURES,
    )
    max_items: StrictInt = Field(
        default=MEMORY_ATTRIBUTION_DEFAULT_MAX_ITEMS,
        ge=1,
        le=MEMORY_ATTRIBUTION_HARD_MAX_ITEMS,
    )
    max_source_bytes: StrictInt = Field(
        default=MEMORY_ATTRIBUTION_DEFAULT_MAX_SOURCE_BYTES,
        ge=1,
        le=MEMORY_ATTRIBUTION_HARD_MAX_SOURCE_BYTES,
    )
    max_projection_bytes: StrictInt = Field(
        default=MEMORY_ATTRIBUTION_DEFAULT_MAX_PROJECTION_BYTES,
        ge=1,
        le=MEMORY_ATTRIBUTION_HARD_MAX_PROJECTION_BYTES,
    )


class MemoryEvidenceAlias(BaseModel):
    """A domain- and session-scoped HMAC alias for a private durable identity."""

    model_config = _MODEL_CONFIG

    algorithm: Literal["hmac-sha256"] = "hmac-sha256"
    key_id: str = Field(max_length=64)
    kind: Literal["receipt", "exposure", "item", "interaction"]
    digest: str = Field(min_length=64, max_length=64)

    @field_validator("key_id")
    @classmethod
    def validate_key_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "key_id")

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if any(character not in "0123456789abcdef" for character in value):
            raise ValueError("digest must be a lowercase SHA-256 digest.")
        return value


class MemoryRecallItemAttribution(BaseModel):
    model_config = _MODEL_CONFIG

    item_alias: MemoryEvidenceAlias
    ordinal: StrictInt = Field(ge=0, lt=MAX_RECALL_RECEIPT_ITEMS)
    admission: RecallItemAdmission
    selection_reason: RecallItemSelectionReason

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        if self.item_alias.kind != "item":
            raise ValueError("Recall item attribution requires an item alias.")
        if not self.admission.matches_selection_reason(self.selection_reason):
            raise ValueError("Recall item selection reason does not match its admission.")
        return self


class MemoryRecallAttribution(BaseModel):
    """Secret-free structural facts from one recall receipt."""

    model_config = _MODEL_CONFIG

    receipt_alias: MemoryEvidenceAlias
    interaction_alias: MemoryEvidenceAlias
    projection_ordinal: StrictInt = Field(
        ge=0,
        lt=MEMORY_ATTRIBUTION_HARD_MAX_RECEIPTS,
    )
    model_step_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    created_at: datetime
    inspected_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    eligible_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    admitted_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    offered_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    silent_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    omitted_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    complete_source_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    partial_source_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    unavailable_source_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    failed_source_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    truncated: StrictBool
    items: tuple[MemoryRecallItemAttribution, ...] = Field(
        default=(), max_length=MAX_RECALL_RECEIPT_ITEMS
    )
    items_truncated: StrictBool = False
    omitted_item_count_at_least: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)

    @field_validator("model_step_id")
    @classmethod
    def validate_model_step_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "model_step_id")

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "created_at")

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        if self.receipt_alias.kind != "receipt" or self.interaction_alias.kind != "interaction":
            raise ValueError("Recall attribution aliases have the wrong domains.")
        aliases = (
            self.receipt_alias,
            self.interaction_alias,
            *(item.item_alias for item in self.items),
        )
        if len({alias.key_id for alias in aliases}) != 1:
            raise ValueError("Recall attribution aliases must use one key identity.")
        if tuple(item.ordinal for item in self.items) != tuple(range(len(self.items))):
            raise ValueError("Projected recall item ordinals must be contiguous from zero.")
        item_aliases = tuple(item.item_alias.digest for item in self.items)
        if any(item.item_alias.kind != "item" for item in self.items) or len(item_aliases) != len(
            set(item_aliases)
        ):
            raise ValueError("Projected recall item aliases must be unique item aliases.")
        source_count = sum(
            (
                self.complete_source_count,
                self.partial_source_count,
                self.unavailable_source_count,
                self.failed_source_count,
            )
        )
        if not source_count:
            raise ValueError("Recall attribution must retain at least one source state.")
        if source_count > MAX_RECALL_RECEIPT_SOURCES:
            raise ValueError("Recall attribution source count exceeds its durable bound.")
        if self.eligible_count != (
            self.admitted_count + self.offered_count + self.silent_count + self.omitted_count
        ):
            raise ValueError("Recall attribution eligible count does not match its outcomes.")
        if self.inspected_count < self.eligible_count:
            raise ValueError("Recall attribution inspected count is smaller than eligible count.")
        expected_recall_truncation = bool(
            self.omitted_count
            or self.partial_source_count
            or self.unavailable_source_count
            or self.failed_source_count
        )
        if self.truncated is not expected_recall_truncation:
            raise ValueError("Recall attribution truncation does not match its source evidence.")
        retained_admitted = sum(
            item.admission is RecallItemAdmission.ADMITTED for item in self.items
        )
        retained_offered = sum(item.admission is RecallItemAdmission.OFFERED for item in self.items)
        if self.admitted_count < retained_admitted or self.offered_count < retained_offered:
            raise ValueError("Recall attribution item outcomes exceed their declared counts.")
        missing_items = self.admitted_count + self.offered_count - len(self.items)
        if missing_items < 0:
            raise ValueError("Recall attribution retains more items than its declared outcomes.")
        if self.items_truncated is not (missing_items > 0):
            raise ValueError("Recall attribution item truncation does not match retained items.")
        if self.omitted_item_count_at_least != missing_items:
            raise ValueError("Recall attribution item omission count is inconsistent.")
        if not self.items_truncated and (
            self.admitted_count != retained_admitted or self.offered_count != retained_offered
        ):
            raise ValueError("Complete recall attribution items do not match their outcomes.")
        return self


class MemoryExposureTransitionAttribution(BaseModel):
    model_config = _MODEL_CONFIG

    revision: StrictInt = Field(ge=0, lt=MAX_CONTEXT_EXPOSURE_TRANSITIONS)
    state: ContextExposureState
    occurred_at: datetime
    evidence_kind: ContextExposureEvidenceKind

    @field_validator("occurred_at")
    @classmethod
    def normalize_occurred_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "occurred_at")

    @model_validator(mode="after")
    def validate_evidence_kind(self) -> Self:
        if not self.state.accepts_evidence_kind(self.evidence_kind):
            raise ValueError("Context exposure evidence does not prove its state.")
        return self


class MemoryExposureItemAttribution(BaseModel):
    model_config = _MODEL_CONFIG

    item_alias: MemoryEvidenceAlias
    receipt_alias: MemoryEvidenceAlias
    ordinal: StrictInt = Field(ge=0, lt=MAX_RECALL_RECEIPT_ITEMS)
    receipt_item_ordinal: StrictInt = Field(ge=0, lt=MAX_RECALL_RECEIPT_ITEMS)
    admission: RecallItemAdmission
    selection_reason: RecallItemSelectionReason

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        if self.item_alias.kind != "item" or self.receipt_alias.kind != "receipt":
            raise ValueError("Exposure item attribution aliases have the wrong domains.")
        if not self.admission.matches_selection_reason(self.selection_reason):
            raise ValueError("Recall item selection reason does not match its admission.")
        return self


class MemoryContextExposureAttribution(BaseModel):
    """Secret-free structural and lifecycle facts from one context exposure."""

    model_config = _MODEL_CONFIG

    exposure_alias: MemoryEvidenceAlias
    interaction_alias: MemoryEvidenceAlias
    projection_ordinal: StrictInt = Field(
        ge=0,
        lt=MEMORY_ATTRIBUTION_HARD_MAX_EXPOSURES,
    )
    model_step_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    model_attempt_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    provider_attempt_id: str = Field(max_length=MAX_MEMORY_EVIDENCE_ID_CHARS)
    created_at: datetime
    updated_at: datetime
    state: ContextExposureState
    state_revision: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    provider_exposure_proven: StrictBool
    receipt_aliases: tuple[MemoryEvidenceAlias, ...] = Field(
        default=(), max_length=MAX_CONTEXT_EXPOSURE_RECEIPTS
    )
    contributor_count: StrictInt = Field(ge=0, le=MAX_CONTEXT_EXPOSURE_CONTRIBUTORS)
    transitions: tuple[MemoryExposureTransitionAttribution, ...] = Field(
        default=(), max_length=MAX_CONTEXT_EXPOSURE_TRANSITIONS
    )
    items: tuple[MemoryExposureItemAttribution, ...] = Field(
        default=(), max_length=MAX_RECALL_RECEIPT_ITEMS
    )
    items_truncated: StrictBool = False
    omitted_item_count_at_least: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)

    @field_validator("model_step_id", "model_attempt_id", "provider_attempt_id")
    @classmethod
    def validate_attempt_ids(cls, value: str, info: Any) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime, info: Any) -> datetime:
        return normalize_utc_datetime(value, info.field_name)

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        if self.exposure_alias.kind != "exposure" or self.interaction_alias.kind != "interaction":
            raise ValueError("Context exposure aliases have the wrong domains.")
        if any(alias.kind != "receipt" for alias in self.receipt_aliases):
            raise ValueError("Context exposure receipt aliases have the wrong domain.")
        aliases = (
            self.exposure_alias,
            self.interaction_alias,
            *self.receipt_aliases,
            *(item.item_alias for item in self.items),
            *(item.receipt_alias for item in self.items),
        )
        if len({alias.key_id for alias in aliases}) != 1:
            raise ValueError("Context exposure aliases must use one key identity.")
        receipt_aliases = tuple((alias.key_id, alias.digest) for alias in self.receipt_aliases)
        if len(receipt_aliases) != len(set(receipt_aliases)):
            raise ValueError("Context exposure receipt aliases must be unique.")
        if tuple(transition.revision for transition in self.transitions) != tuple(
            range(len(self.transitions))
        ):
            raise ValueError("Context exposure transition revisions must be contiguous.")
        if not self.transitions:
            raise ValueError("Context exposure attribution requires lifecycle evidence.")
        if (
            self.transitions[0].revision != 0
            or self.transitions[0].state is not ContextExposureState.PLANNED
        ):
            raise ValueError("Context exposure attribution must begin in planned state.")
        for previous, current in zip(self.transitions, self.transitions[1:], strict=False):
            if not previous.state.permits_transition_to(current.state):
                raise ValueError("Context exposure attribution has an invalid transition.")
            if current.occurred_at < previous.occurred_at:
                raise ValueError("Context exposure attribution timestamps must be monotonic.")
        if any(
            not transition.state.accepts_evidence_kind(transition.evidence_kind)
            for transition in self.transitions
        ):
            raise ValueError("Context exposure attribution evidence does not prove its state.")
        latest = self.transitions[-1]
        if latest.state is not self.state or latest.revision != self.state_revision:
            raise ValueError("Context exposure lifecycle does not match its latest transition.")
        if (
            self.created_at != self.transitions[0].occurred_at
            or self.updated_at != latest.occurred_at
        ):
            raise ValueError("Context exposure timestamps do not match its lifecycle.")
        if tuple(item.ordinal for item in self.items) != tuple(range(len(self.items))):
            raise ValueError("Context exposure item ordinals must be contiguous from zero.")
        if any(
            item.item_alias.kind != "item"
            or item.receipt_alias.kind != "receipt"
            or (item.receipt_alias.key_id, item.receipt_alias.digest) not in receipt_aliases
            for item in self.items
        ):
            raise ValueError("Context exposure items must use linked typed aliases.")
        item_aliases = tuple(item.item_alias.digest for item in self.items)
        if len(item_aliases) != len(set(item_aliases)):
            raise ValueError("Context exposure item aliases must be unique.")
        receipt_items = tuple(
            (
                item.receipt_alias.key_id,
                item.receipt_alias.digest,
                item.receipt_item_ordinal,
            )
            for item in self.items
        )
        if len(receipt_items) != len(set(receipt_items)):
            raise ValueError("A recall receipt item may appear only once in one context exposure.")
        if not self.items_truncated and self.omitted_item_count_at_least:
            raise ValueError("Complete context exposure items cannot report omissions.")
        expected_provider_exposure = any(
            transition.state in {ContextExposureState.ACKNOWLEDGED, ContextExposureState.COMPLETED}
            for transition in self.transitions
        )
        if self.provider_exposure_proven is not expected_provider_exposure:
            raise ValueError("Provider exposure proof does not match the retained lifecycle.")
        return self


class MemoryAttribution(BaseModel):
    """Versioned bounded projection of one session's durable memory-use truth."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["cayu.memory_attribution.v1"] = MEMORY_ATTRIBUTION_VERSION
    status: MemoryAttributionStatus
    truncated: StrictBool
    reason: MemoryAttributionUnavailableReason | None = None
    observed_receipt_count: StrictInt = Field(
        ge=0,
        le=MEMORY_ATTRIBUTION_HARD_MAX_RECEIPTS,
    )
    observed_exposure_count: StrictInt = Field(
        ge=0,
        le=MEMORY_ATTRIBUTION_HARD_MAX_EXPOSURES,
    )
    observed_item_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    omitted_receipt_count_at_least: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    omitted_exposure_count_at_least: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    omitted_item_count_at_least: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    receipts: tuple[MemoryRecallAttribution, ...] = Field(
        default=(), max_length=MEMORY_ATTRIBUTION_HARD_MAX_RECEIPTS
    )
    exposures: tuple[MemoryContextExposureAttribution, ...] = Field(
        default=(), max_length=MEMORY_ATTRIBUTION_HARD_MAX_EXPOSURES
    )

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.status is MemoryAttributionStatus.COMPLETE:
            if self.truncated or self.reason is not None:
                raise ValueError("Complete memory attribution cannot be truncated or unavailable.")
        elif self.status is MemoryAttributionStatus.TRUNCATED:
            if not self.truncated or self.reason is not None:
                raise ValueError("Truncated memory attribution requires only a truncation marker.")
        elif self.status is MemoryAttributionStatus.UNAVAILABLE:
            if self.reason not in {
                MemoryAttributionUnavailableReason.STORE_UNSUPPORTED,
                MemoryAttributionUnavailableReason.EVIDENCE_READ_FAILED,
            }:
                raise ValueError("Unavailable memory attribution requires a read reason.")
        elif self.status is MemoryAttributionStatus.REDACTED:
            if self.reason is not MemoryAttributionUnavailableReason.ALIAS_KEY_UNAVAILABLE:
                raise ValueError(
                    "Redacted memory attribution requires unavailable alias authority."
                )
        elif (
            self.status is MemoryAttributionStatus.CONTRADICTORY
            and self.reason is not MemoryAttributionUnavailableReason.CONTRADICTORY_EVIDENCE
        ):
            raise ValueError("Contradictory memory attribution requires a contradiction reason.")
        if self.status in {
            MemoryAttributionStatus.UNAVAILABLE,
            MemoryAttributionStatus.REDACTED,
            MemoryAttributionStatus.CONTRADICTORY,
        } and (self.receipts or self.exposures):
            raise ValueError("Untrusted memory attribution cannot retain projected records.")
        if self.status in {
            MemoryAttributionStatus.UNAVAILABLE,
            MemoryAttributionStatus.REDACTED,
            MemoryAttributionStatus.CONTRADICTORY,
        } and (
            self.omitted_receipt_count_at_least < self.observed_receipt_count
            or self.omitted_exposure_count_at_least < self.observed_exposure_count
            or self.omitted_item_count_at_least < self.observed_item_count
        ):
            raise ValueError("Untrusted memory attribution must count every observed omission.")
        if self.status is MemoryAttributionStatus.COMPLETE and any(
            (
                self.omitted_receipt_count_at_least,
                self.omitted_exposure_count_at_least,
                self.omitted_item_count_at_least,
            )
        ):
            raise ValueError("Complete memory attribution cannot report omitted evidence.")
        if self.observed_receipt_count < len(self.receipts):
            raise ValueError("Observed recall receipt count is smaller than retained evidence.")
        if self.observed_exposure_count < len(self.exposures):
            raise ValueError("Observed context exposure count is smaller than retained evidence.")
        retained_item_count = sum(len(receipt.items) for receipt in self.receipts) + sum(
            len(exposure.items) for exposure in self.exposures
        )
        if retained_item_count > MEMORY_ATTRIBUTION_HARD_MAX_ITEMS:
            raise ValueError("Memory attribution retained items exceed the hard global bound.")
        if self.observed_item_count < retained_item_count:
            raise ValueError("Observed item count is smaller than retained evidence.")
        if (
            self.omitted_receipt_count_at_least < self.observed_receipt_count - len(self.receipts)
            or self.omitted_exposure_count_at_least
            < self.observed_exposure_count - len(self.exposures)
            or self.omitted_item_count_at_least < self.observed_item_count - retained_item_count
        ):
            raise ValueError("Memory attribution omission counts lose observed evidence.")
        receipt_aliases = tuple(receipt.receipt_alias.digest for receipt in self.receipts)
        exposure_aliases = tuple(exposure.exposure_alias.digest for exposure in self.exposures)
        if len(receipt_aliases) != len(set(receipt_aliases)) or len(exposure_aliases) != len(
            set(exposure_aliases)
        ):
            raise ValueError("Memory attribution record aliases must be unique.")
        receipt_item_aliases = tuple(
            item.item_alias.digest for receipt in self.receipts for item in receipt.items
        )
        if len(receipt_item_aliases) != len(set(receipt_item_aliases)):
            raise ValueError("Recall item aliases must be unique across retained receipts.")
        model_attempt_ids = tuple(exposure.model_attempt_id for exposure in self.exposures)
        provider_attempt_ids = tuple(exposure.provider_attempt_id for exposure in self.exposures)
        if len(model_attempt_ids) != len(set(model_attempt_ids)) or len(
            provider_attempt_ids
        ) != len(set(provider_attempt_ids)):
            raise ValueError("Context exposure attempt identities must be unique.")
        aliases = (
            *(
                alias
                for receipt in self.receipts
                for alias in (
                    receipt.receipt_alias,
                    receipt.interaction_alias,
                    *(item.item_alias for item in receipt.items),
                )
            ),
            *(
                alias
                for exposure in self.exposures
                for alias in (
                    exposure.exposure_alias,
                    exposure.interaction_alias,
                    *exposure.receipt_aliases,
                    *(item.item_alias for item in exposure.items),
                    *(item.receipt_alias for item in exposure.items),
                )
            ),
        )
        if aliases and len({alias.key_id for alias in aliases}) != 1:
            raise ValueError("Memory attribution aliases must use one key identity.")
        if tuple(receipt.projection_ordinal for receipt in self.receipts) != tuple(
            range(len(self.receipts))
        ) or tuple(exposure.projection_ordinal for exposure in self.exposures) != tuple(
            range(len(self.exposures))
        ):
            raise ValueError("Memory attribution record ordering must be contiguous.")
        if tuple(receipt.created_at for receipt in self.receipts) != tuple(
            sorted(receipt.created_at for receipt in self.receipts)
        ) or tuple(exposure.created_at for exposure in self.exposures) != tuple(
            sorted(exposure.created_at for exposure in self.exposures)
        ):
            raise ValueError("Memory attribution records must retain chronological order.")
        if self.status is MemoryAttributionStatus.COMPLETE and (
            self.observed_receipt_count != len(self.receipts)
            or self.observed_exposure_count != len(self.exposures)
            or self.observed_item_count != retained_item_count
        ):
            raise ValueError("Complete memory attribution must retain every observed record.")
        if self.status is MemoryAttributionStatus.COMPLETE and any(
            record.items_truncated for record in (*self.receipts, *self.exposures)
        ):
            raise ValueError("Complete memory attribution cannot contain truncated item evidence.")
        _validate_retained_linkage(self)
        return self


def _validate_retained_linkage(attribution: MemoryAttribution) -> None:
    """Reject contradictory relationships at public parse and replay boundaries."""

    receipt_by_alias = {
        (receipt.receipt_alias.key_id, receipt.receipt_alias.digest): receipt
        for receipt in attribution.receipts
    }
    receipt_items_by_alias = {
        receipt_key: {item.ordinal: item for item in receipt.items}
        for receipt_key, receipt in receipt_by_alias.items()
    }
    missing_receipt_aliases: set[tuple[str, str]] = set()
    interaction_by_receipt_alias: dict[tuple[str, str], MemoryEvidenceAlias] = {}
    item_signature_by_receipt_position: dict[
        tuple[str, str, int],
        tuple[MemoryEvidenceAlias, RecallItemAdmission, RecallItemSelectionReason],
    ] = {}
    receipt_position_by_item_alias: dict[tuple[str, str], tuple[str, str, int]] = {}
    for exposure in attribution.exposures:
        for receipt_alias in exposure.receipt_aliases:
            receipt_key = (receipt_alias.key_id, receipt_alias.digest)
            previous_interaction = interaction_by_receipt_alias.setdefault(
                receipt_key,
                exposure.interaction_alias,
            )
            if previous_interaction != exposure.interaction_alias:
                raise ValueError(
                    "Memory attribution links one receipt across different interactions."
                )
            receipt = receipt_by_alias.get(receipt_key)
            if receipt is None:
                missing_receipt_aliases.add(receipt_key)
                if attribution.status is MemoryAttributionStatus.COMPLETE:
                    raise ValueError(
                        "Complete memory attribution references an omitted recall receipt."
                    )
                continue
            if (
                receipt.interaction_alias != exposure.interaction_alias
                or receipt.created_at > exposure.created_at
            ):
                raise ValueError("Memory attribution receipt/exposure linkage is contradictory.")

        for item in exposure.items:
            receipt_key = (item.receipt_alias.key_id, item.receipt_alias.digest)
            receipt_position = (*receipt_key, item.receipt_item_ordinal)
            item_signature = (item.item_alias, item.admission, item.selection_reason)
            previous_signature = item_signature_by_receipt_position.setdefault(
                receipt_position,
                item_signature,
            )
            if previous_signature != item_signature:
                raise ValueError(
                    "Memory attribution assigns conflicting facts to one recall receipt item."
                )
            item_alias_key = (item.item_alias.key_id, item.item_alias.digest)
            previous_position = receipt_position_by_item_alias.setdefault(
                item_alias_key,
                receipt_position,
            )
            if previous_position != receipt_position:
                raise ValueError(
                    "Memory attribution reuses one item alias for different receipt items."
                )
            receipt = receipt_by_alias.get(receipt_key)
            if receipt is None:
                missing_receipt_aliases.add(receipt_key)
                if attribution.status is MemoryAttributionStatus.COMPLETE:
                    raise ValueError(
                        "Complete memory attribution item references an omitted recall receipt."
                    )
                continue
            receipt_item = receipt_items_by_alias[receipt_key].get(item.receipt_item_ordinal)
            if (
                receipt_item is None
                or receipt_item.item_alias != item.item_alias
                or receipt_item.admission is not item.admission
                or receipt_item.selection_reason is not item.selection_reason
            ):
                raise ValueError(
                    "Memory attribution receipt/exposure item linkage is contradictory."
                )
    if len(missing_receipt_aliases) > attribution.omitted_receipt_count_at_least:
        raise ValueError("Memory attribution loses known omitted recall receipts.")


__all__ = [
    "MEMORY_ATTRIBUTION_VERSION",
    "MemoryAttribution",
    "MemoryAttributionBounds",
    "MemoryAttributionStatus",
    "MemoryAttributionUnavailableReason",
    "MemoryContextExposureAttribution",
    "MemoryEvidenceAlias",
    "MemoryExposureItemAttribution",
    "MemoryExposureTransitionAttribution",
    "MemoryRecallAttribution",
    "MemoryRecallItemAttribution",
]
