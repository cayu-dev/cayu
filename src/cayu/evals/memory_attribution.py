"""Portable, bounded memory-attribution evidence for evaluation results."""

from __future__ import annotations

import hashlib
import hmac
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    compact_json_utf8_size,
    json_utf8_size_within_limit,
    require_durable_clean_nonblank,
)
from cayu.memory_attribution import (
    MEMORY_ATTRIBUTION_DEFAULT_MAX_EXPOSURES,
    MEMORY_ATTRIBUTION_DEFAULT_MAX_ITEMS,
    MEMORY_ATTRIBUTION_DEFAULT_MAX_RECEIPTS,
    MemoryAttribution,
    MemoryAttributionBounds,
    MemoryAttributionStatus,
    MemoryAttributionUnavailableReason,
)
from cayu.memory_evidence import ContextExposureState
from cayu.memory_interventions import memory_attribution_fingerprint

EVAL_MEMORY_ATTRIBUTION_SCHEMA_VERSION = 1
EVAL_MEMORY_ATTRIBUTION_POLICY_SCHEMA_VERSION = 1
EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES = 100
EVAL_MEMORY_ATTRIBUTION_MAX_SOURCE_BYTES_PER_TRIAL = 4 << 20
EVAL_MEMORY_ATTRIBUTION_MAX_PROJECTION_BYTES_PER_TRIAL = 512 << 10
EVAL_MEMORY_ATTRIBUTION_MAX_BYTES = 1 << 20
EVAL_MEMORY_ATTRIBUTION_SOURCE_BUDGET_BYTES = 32 << 20
EVAL_MEMORY_ATTRIBUTION_PROJECTION_BUDGET_BYTES = 8 << 20
EVAL_MEMORY_ATTRIBUTION_RESULT_BUDGET_BYTES = 9 << 20
EVAL_MEMORY_ATTRIBUTION_MIN_DOCUMENT_BYTES = 900
EVAL_MEMORY_ATTRIBUTION_MAX_RETAINED_SOURCES_PER_RUN = 10_000
EVAL_MEMORY_SOURCE_ALIAS_CONTEXT = b"cayu.eval.memory-attribution.source.v1"

_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    hide_input_in_errors=True,
    revalidate_instances="always",
    validate_default=True,
)
_HEX = frozenset("0123456789abcdef")


class EvalMemoryEvidenceCompleteness(StrEnum):
    COMPLETE = "complete"
    TRUNCATED = "truncated"
    UNAVAILABLE = "unavailable"


class EvalMemoryEvidenceLimitation(StrEnum):
    SOURCE_LIMIT = "source_limit"
    SOURCE_BYTES_LIMIT = "source_bytes_limit"
    PROJECTION_BYTES_LIMIT = "projection_bytes_limit"
    RUNTIME_ATTRIBUTION_TRUNCATED = "runtime_attribution_truncated"
    STORE_UNSUPPORTED = "store_unsupported"
    EVIDENCE_READ_FAILED = "evidence_read_failed"
    ALIAS_KEY_UNAVAILABLE = "alias_key_unavailable"
    MISSING = "missing"
    DELETED = "deleted"
    LEGACY = "legacy"
    CONTRADICTORY_LINEAGE = "contradictory_lineage"
    SOURCE_TREE_INCOMPLETE = "source_tree_incomplete"
    CLOSURE_CHANGED = "closure_changed"
    DEADLINE_EXPIRED = "deadline_expired"


_UNAVAILABLE_LIMITATIONS = frozenset(
    {
        EvalMemoryEvidenceLimitation.STORE_UNSUPPORTED,
        EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED,
        EvalMemoryEvidenceLimitation.ALIAS_KEY_UNAVAILABLE,
        EvalMemoryEvidenceLimitation.MISSING,
        EvalMemoryEvidenceLimitation.DELETED,
        EvalMemoryEvidenceLimitation.LEGACY,
        EvalMemoryEvidenceLimitation.CONTRADICTORY_LINEAGE,
        EvalMemoryEvidenceLimitation.SOURCE_TREE_INCOMPLETE,
        EvalMemoryEvidenceLimitation.CLOSURE_CHANGED,
        EvalMemoryEvidenceLimitation.DEADLINE_EXPIRED,
    }
)


def _sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _content_revision(value: object, field_name: str) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return "sha256:" + hashlib.sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


def eval_memory_attribution_fingerprint(attribution: MemoryAttribution) -> str:
    """Reuse the runtime/intervention projection's canonical identity."""

    return memory_attribution_fingerprint(attribution)


class EvalMemoryAttributionCapturePolicyV1(BaseModel):
    """One fixed eval-owned projection contract."""

    model_config = _MODEL_CONFIG

    schema_version: Literal[1] = EVAL_MEMORY_ATTRIBUTION_POLICY_SCHEMA_VERSION
    revision: StrictStr
    runtime_projection: Literal["cayu.memory_attribution.v1"] = "cayu.memory_attribution.v1"
    max_sources: Literal[100] = EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES
    max_receipts: Literal[100] = MEMORY_ATTRIBUTION_DEFAULT_MAX_RECEIPTS
    max_exposures: Literal[100] = MEMORY_ATTRIBUTION_DEFAULT_MAX_EXPOSURES
    max_items: Literal[1000] = MEMORY_ATTRIBUTION_DEFAULT_MAX_ITEMS
    max_source_bytes: Literal[4194304] = EVAL_MEMORY_ATTRIBUTION_MAX_SOURCE_BYTES_PER_TRIAL
    max_projection_bytes: Literal[524288] = EVAL_MEMORY_ATTRIBUTION_MAX_PROJECTION_BYTES_PER_TRIAL
    max_document_bytes: Literal[1048576] = EVAL_MEMORY_ATTRIBUTION_MAX_BYTES

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator(
        "max_sources",
        "max_receipts",
        "max_exposures",
        "max_items",
        "max_source_bytes",
        "max_projection_bytes",
        "max_document_bytes",
        mode="before",
    )
    @classmethod
    def validate_integer_policy_limits(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be an integer.")
        return value

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if not value.startswith("sha256:"):
            raise ValueError("revision must be a sha256 content revision.")
        _sha256(value.removeprefix("sha256:"), "revision")
        return value

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        values = self.model_dump(mode="json", exclude={"revision"})
        if self.revision != _content_revision(values, "eval memory attribution policy"):
            raise ValueError("Eval memory-attribution policy revision changed.")
        return self

    @classmethod
    def standard(cls) -> EvalMemoryAttributionCapturePolicyV1:
        values: dict[str, Any] = {
            "schema_version": EVAL_MEMORY_ATTRIBUTION_POLICY_SCHEMA_VERSION,
            "runtime_projection": "cayu.memory_attribution.v1",
            "max_sources": EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES,
            "max_receipts": MEMORY_ATTRIBUTION_DEFAULT_MAX_RECEIPTS,
            "max_exposures": MEMORY_ATTRIBUTION_DEFAULT_MAX_EXPOSURES,
            "max_items": MEMORY_ATTRIBUTION_DEFAULT_MAX_ITEMS,
            "max_source_bytes": EVAL_MEMORY_ATTRIBUTION_MAX_SOURCE_BYTES_PER_TRIAL,
            "max_projection_bytes": EVAL_MEMORY_ATTRIBUTION_MAX_PROJECTION_BYTES_PER_TRIAL,
            "max_document_bytes": EVAL_MEMORY_ATTRIBUTION_MAX_BYTES,
        }
        return cls(
            revision=_content_revision(values, "eval memory attribution policy"),
            **values,
        )


class EvalMemorySourceAliasV1(BaseModel):
    model_config = _MODEL_CONFIG

    algorithm: Literal["hmac-sha256"] = "hmac-sha256"
    key_id: StrictStr = Field(max_length=64)
    digest: StrictStr = Field(min_length=64, max_length=64)

    @field_validator("key_id")
    @classmethod
    def validate_key_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "key_id")

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _sha256(value, "digest")


class EvalMemorySourceReferenceV1(BaseModel):
    model_config = _MODEL_CONFIG

    role: Literal["root", "descendant"]
    tree_path: tuple[StrictInt, ...] = Field(max_length=32)
    session_alias: EvalMemorySourceAliasV1 | None = None

    @field_validator("tree_path", mode="before")
    @classmethod
    def validate_ordered_path(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("tree_path must be an ordered array.")
        return value

    @model_validator(mode="after")
    def validate_role(self) -> Self:
        if self.role == "root" and self.tree_path:
            raise ValueError("The root memory source must use an empty tree path.")
        if self.role == "descendant" and not self.tree_path:
            raise ValueError("A descendant memory source requires a tree path.")
        if any(
            type(index) is not int or not 0 <= index <= MAX_DURABLE_JSON_INTEGER
            for index in self.tree_path
        ):
            raise ValueError("Memory source tree indexes must be portable non-negative integers.")
        return self


class EvalMemoryAttributionSourceV1(BaseModel):
    model_config = _MODEL_CONFIG

    source: EvalMemorySourceReferenceV1
    terminal_status: Literal["completed", "failed", "interrupted"]
    expected_receipt_count: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    expected_exposure_count: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    attribution: MemoryAttribution | None = None
    attribution_fingerprint: StrictStr | None = None
    limitations: tuple[EvalMemoryEvidenceLimitation, ...] = ()

    @field_validator("limitations", mode="before")
    @classmethod
    def validate_ordered_limitations(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("limitations must be an ordered array.")
        return value

    @field_validator("attribution", mode="before")
    @classmethod
    def copy_attribution(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, BaseModel) and type(value) is not MemoryAttribution:
            raise TypeError("attribution must be an exact MemoryAttribution or JSON object.")
        return value

    @field_validator("attribution_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str | None) -> str | None:
        return None if value is None else _sha256(value, "attribution_fingerprint")

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        ordered_limitations = tuple(sorted(set(self.limitations), key=str))
        if self.limitations != ordered_limitations:
            raise ValueError("Memory source limitations must be unique and sorted.")
        if (self.attribution is None) != (self.attribution_fingerprint is None):
            raise ValueError("Memory source attribution and fingerprint must be present together.")
        expected_limitations: set[EvalMemoryEvidenceLimitation] = set()
        if self.attribution is not None:
            expected = eval_memory_attribution_fingerprint(self.attribution)
            if self.attribution_fingerprint != expected:
                raise ValueError("Memory source attribution fingerprint changed.")
            if self.attribution.status is MemoryAttributionStatus.TRUNCATED:
                expected_limitations.add(EvalMemoryEvidenceLimitation.RUNTIME_ATTRIBUTION_TRUNCATED)
            elif self.attribution.status is MemoryAttributionStatus.REDACTED:
                expected_limitations.add(EvalMemoryEvidenceLimitation.ALIAS_KEY_UNAVAILABLE)
            elif self.attribution.status is MemoryAttributionStatus.CONTRADICTORY:
                expected_limitations.add(EvalMemoryEvidenceLimitation.CONTRADICTORY_LINEAGE)
            elif self.attribution.status is MemoryAttributionStatus.UNAVAILABLE:
                expected_limitations.add(
                    EvalMemoryEvidenceLimitation.STORE_UNSUPPORTED
                    if self.attribution.reason
                    is MemoryAttributionUnavailableReason.STORE_UNSUPPORTED
                    else EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED
                )
            if (
                self.attribution.status
                in {MemoryAttributionStatus.COMPLETE, MemoryAttributionStatus.REDACTED}
                and not self.attribution.truncated
                and (
                    (
                        self.expected_receipt_count is not None
                        and self.attribution.observed_receipt_count < self.expected_receipt_count
                    )
                    or (
                        self.expected_exposure_count is not None
                        and self.attribution.observed_exposure_count < self.expected_exposure_count
                    )
                )
            ):
                expected_limitations.add(EvalMemoryEvidenceLimitation.DELETED)
        else:
            if len(self.limitations) != 1 or self.limitations[0] not in _UNAVAILABLE_LIMITATIONS:
                raise ValueError(
                    "A missing memory source attribution requires one unavailable limitation."
                )
            if self.limitations[0] is EvalMemoryEvidenceLimitation.DELETED:
                raise ValueError(
                    "Deleted memory evidence requires positive retained source evidence."
                )
            expected_limitations.add(self.limitations[0])
        if self.limitations != tuple(sorted(expected_limitations, key=str)):
            raise ValueError("Memory source limitations contradict its runtime evidence.")
        return self


class EvalMemoryAttributionEvidenceV1(BaseModel):
    """Complete portable memory evidence for one eval trial's session tree."""

    model_config = _MODEL_CONFIG

    schema_version: Literal[1] = EVAL_MEMORY_ATTRIBUTION_SCHEMA_VERSION
    revision: StrictStr
    policy: EvalMemoryAttributionCapturePolicyV1
    effective_bounds: MemoryAttributionBounds
    effective_source_limit: StrictInt = Field(ge=0, le=EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES)
    effective_max_bytes: StrictInt = Field(ge=1, le=EVAL_MEMORY_ATTRIBUTION_MAX_BYTES)
    completeness: EvalMemoryEvidenceCompleteness
    limitations: tuple[EvalMemoryEvidenceLimitation, ...] = ()
    # This is an eval-derived cardinality, not retained source material.  The
    # retained records and the complete document carry the actual size bounds.
    # Keeping the exact count lets a bounded projection honestly describe a
    # larger, already-retained trajectory instead of failing while producing
    # unrelated assertion evidence.
    total_source_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    retained_source_count: StrictInt = Field(ge=0, le=EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES)
    omitted_source_count_at_least: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    sources: tuple[EvalMemoryAttributionSourceV1, ...] = Field(
        default=(), max_length=EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES
    )
    proves_empty: StrictBool
    has_indeterminate_exposure: StrictBool

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if not value.startswith("sha256:"):
            raise ValueError("revision must be a sha256 content revision.")
        _sha256(value.removeprefix("sha256:"), "revision")
        return value

    @field_validator("limitations", mode="before")
    @classmethod
    def validate_ordered_limitations(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("limitations must be an ordered array.")
        return value

    @field_validator("sources", mode="before")
    @classmethod
    def validate_ordered_sources(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("sources must be an ordered array.")
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.policy != EvalMemoryAttributionCapturePolicyV1.standard():
            raise ValueError("Unsupported eval memory-attribution capture policy.")
        if self.effective_bounds.max_receipts > self.policy.max_receipts:
            raise ValueError("Effective receipt bound exceeds the capture policy.")
        if self.effective_bounds.max_exposures > self.policy.max_exposures:
            raise ValueError("Effective exposure bound exceeds the capture policy.")
        if self.effective_bounds.max_items > self.policy.max_items:
            raise ValueError("Effective item bound exceeds the capture policy.")
        if self.effective_bounds.max_source_bytes > self.policy.max_source_bytes:
            raise ValueError("Effective source-byte bound exceeds the capture policy.")
        if self.effective_bounds.max_projection_bytes > self.policy.max_projection_bytes:
            raise ValueError("Effective projection-byte bound exceeds the capture policy.")
        if self.effective_source_limit > self.policy.max_sources:
            raise ValueError("Effective source bound exceeds the capture policy.")
        if self.effective_max_bytes > self.policy.max_document_bytes:
            raise ValueError("Effective document-byte bound exceeds the capture policy.")
        expected_limitations = tuple(sorted(set(self.limitations), key=str))
        if self.limitations != expected_limitations:
            raise ValueError("Memory evidence limitations must be unique and sorted.")
        if EvalMemoryEvidenceLimitation.DELETED in self.limitations:
            raise ValueError("Deleted memory evidence requires positive retained source evidence.")
        paths = tuple(source.source.tree_path for source in self.sources)
        if len(paths) != len(set(paths)):
            raise ValueError("Memory attribution source paths must be unique.")
        if paths and paths[0] != ():
            raise ValueError("Memory attribution sources must begin with the root.")
        if paths != tuple(sorted(paths)):
            raise ValueError("Memory attribution sources must use deterministic tree order.")
        if self.retained_source_count != len(self.sources):
            raise ValueError("Retained memory source count does not match its records.")
        if self.retained_source_count > self.effective_source_limit:
            raise ValueError("Memory evidence exceeds its effective source bound.")
        if self.total_source_count < self.retained_source_count:
            raise ValueError("Total memory source count is smaller than retained evidence.")
        if self.omitted_source_count_at_least < (
            self.total_source_count - self.retained_source_count
        ):
            raise ValueError("Memory evidence loses known omitted sources.")
        source_limit_applied = EvalMemoryEvidenceLimitation.SOURCE_LIMIT in self.limitations
        if source_limit_applied != (self.total_source_count > self.effective_source_limit):
            raise ValueError("Memory evidence source-limit classification contradicts its bounds.")
        if (
            EvalMemoryEvidenceLimitation.SOURCE_TREE_INCOMPLETE not in self.limitations
            and self.omitted_source_count_at_least
            != self.total_source_count - self.retained_source_count
        ):
            raise ValueError("Memory evidence omission count requires an incomplete source tree.")
        attributions = tuple(
            source.attribution for source in self.sources if source.attribution is not None
        )
        if sum(len(item.receipts) for item in attributions) > self.effective_bounds.max_receipts:
            raise ValueError("Memory evidence exceeds its effective receipt bound.")
        if sum(len(item.exposures) for item in attributions) > self.effective_bounds.max_exposures:
            raise ValueError("Memory evidence exceeds its effective exposure bound.")
        retained_items = sum(
            len(record.items)
            for attribution in attributions
            for record in (*attribution.receipts, *attribution.exposures)
        )
        if retained_items > self.effective_bounds.max_items:
            raise ValueError("Memory evidence exceeds its effective item bound.")
        retained_projection_bytes = sum(
            compact_json_utf8_size(record.model_dump(mode="json"))
            for attribution in attributions
            for record in (*attribution.receipts, *attribution.exposures)
        )
        if retained_projection_bytes > self.effective_bounds.max_projection_bytes:
            raise ValueError("Memory evidence exceeds its effective projection-byte bound.")
        source_limitations = {item for source in self.sources for item in source.limitations}
        all_limitations = set(self.limitations) | source_limitations
        expected_completeness = (
            EvalMemoryEvidenceCompleteness.UNAVAILABLE
            if all_limitations.intersection(_UNAVAILABLE_LIMITATIONS)
            else (
                EvalMemoryEvidenceCompleteness.TRUNCATED
                if all_limitations or self.omitted_source_count_at_least
                else EvalMemoryEvidenceCompleteness.COMPLETE
            )
        )
        if self.completeness is not expected_completeness:
            raise ValueError("Memory evidence completeness contradicts its limitations.")
        if self.completeness is EvalMemoryEvidenceCompleteness.COMPLETE and (
            not self.sources
            or self.total_source_count != self.retained_source_count
            or self.omitted_source_count_at_least != 0
        ):
            raise ValueError("Complete memory evidence requires one exact retained source tree.")
        if self.completeness is EvalMemoryEvidenceCompleteness.COMPLETE:
            path_set = set(paths)
            children_by_parent: dict[tuple[int, ...], list[int]] = {}
            for path in paths[1:]:
                parent = path[:-1]
                if parent not in path_set:
                    raise ValueError("Complete memory evidence requires every source parent.")
                children_by_parent.setdefault(parent, []).append(path[-1])
            if any(indexes != list(range(len(indexes))) for indexes in children_by_parent.values()):
                raise ValueError(
                    "Complete memory evidence requires contiguous source child indexes."
                )
        expected_empty = bool(
            self.completeness is EvalMemoryEvidenceCompleteness.COMPLETE
            and self.sources
            and all(
                source.attribution is not None
                and source.attribution.status is MemoryAttributionStatus.COMPLETE
                and source.attribution.observed_receipt_count == 0
                and source.attribution.observed_exposure_count == 0
                and source.attribution.observed_item_count == 0
                for source in self.sources
            )
        )
        if self.proves_empty is not expected_empty:
            raise ValueError("Memory evidence empty proof contradicts its sources.")
        expected_indeterminate = any(
            source.attribution is not None
            and any(
                exposure.state is ContextExposureState.INDETERMINATE
                for exposure in source.attribution.exposures
            )
            for source in self.sources
        )
        if self.has_indeterminate_exposure is not expected_indeterminate:
            raise ValueError("Memory indeterminate-exposure summary changed.")
        document = self.model_dump(mode="json", exclude={"revision"})
        if self.revision != _content_revision(document, "eval memory attribution evidence"):
            raise ValueError("Eval memory-attribution evidence revision changed.")
        if not json_utf8_size_within_limit(self, self.effective_max_bytes):
            raise ValueError("Eval memory-attribution evidence exceeds its effective byte limit.")
        return self

    @classmethod
    def create(
        cls,
        *,
        effective_bounds: MemoryAttributionBounds,
        effective_source_limit: int = EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES,
        effective_max_bytes: int = EVAL_MEMORY_ATTRIBUTION_MAX_BYTES,
        completeness: EvalMemoryEvidenceCompleteness,
        limitations: tuple[EvalMemoryEvidenceLimitation, ...],
        total_source_count: int,
        sources: tuple[EvalMemoryAttributionSourceV1, ...],
        omitted_source_count_at_least: int = 0,
    ) -> EvalMemoryAttributionEvidenceV1:
        ordered_limitations = tuple(sorted(set(limitations), key=str))
        values = {
            "schema_version": EVAL_MEMORY_ATTRIBUTION_SCHEMA_VERSION,
            "policy": EvalMemoryAttributionCapturePolicyV1.standard(),
            "effective_bounds": effective_bounds,
            "effective_source_limit": effective_source_limit,
            "effective_max_bytes": effective_max_bytes,
            "completeness": completeness,
            "limitations": ordered_limitations,
            "total_source_count": total_source_count,
            "retained_source_count": len(sources),
            "omitted_source_count_at_least": omitted_source_count_at_least,
            "sources": sources,
            "proves_empty": bool(
                completeness is EvalMemoryEvidenceCompleteness.COMPLETE
                and sources
                and all(
                    source.attribution is not None
                    and source.attribution.status is MemoryAttributionStatus.COMPLETE
                    and source.attribution.observed_receipt_count == 0
                    and source.attribution.observed_exposure_count == 0
                    and source.attribution.observed_item_count == 0
                    for source in sources
                )
            ),
            "has_indeterminate_exposure": any(
                source.attribution is not None
                and any(
                    exposure.state is ContextExposureState.INDETERMINATE
                    for exposure in source.attribution.exposures
                )
                for source in sources
            ),
        }
        # Construct only to obtain Pydantic's canonical JSON projection.  Hashing the
        # Python values directly would make StrEnum and nested-model handling depend on
        # the caller's in-process representation instead of the durable document.
        document = cls.model_construct(
            _fields_set=None,
            revision="sha256:" + "0" * 64,
            **values,
        ).model_dump(mode="json", exclude={"revision"})
        return cls(
            revision=_content_revision(document, "eval memory attribution evidence"),
            **values,
        )

    @classmethod
    def unavailable(
        cls,
        reason: EvalMemoryEvidenceLimitation = EvalMemoryEvidenceLimitation.MISSING,
        *,
        effective_bounds: MemoryAttributionBounds | None = None,
        effective_source_limit: int = EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES,
        effective_max_bytes: int = EVAL_MEMORY_ATTRIBUTION_MAX_BYTES,
    ) -> EvalMemoryAttributionEvidenceV1:
        if reason not in _UNAVAILABLE_LIMITATIONS:
            raise ValueError("Unavailable memory evidence requires an unavailable reason.")
        if reason is EvalMemoryEvidenceLimitation.DELETED:
            raise ValueError("Deleted memory evidence requires positive retained source evidence.")
        return cls.create(
            effective_bounds=effective_bounds or standard_eval_memory_attribution_bounds(),
            effective_source_limit=effective_source_limit,
            effective_max_bytes=effective_max_bytes,
            completeness=EvalMemoryEvidenceCompleteness.UNAVAILABLE,
            limitations=(reason,),
            total_source_count=0,
            sources=(),
        )


def eval_memory_attribution_summary(evidence: EvalMemoryAttributionEvidenceV1) -> str:
    """Return the bounded classification used by public HTML report surfaces."""

    if type(evidence) is not EvalMemoryAttributionEvidenceV1:
        raise TypeError("evidence must be an exact EvalMemoryAttributionEvidenceV1.")
    limitations = tuple(
        sorted(
            {
                limitation.value
                for limitation in (
                    *evidence.limitations,
                    *(
                        limitation
                        for source in evidence.sources
                        for limitation in source.limitations
                    ),
                )
            }
        )
    )
    lifecycle_states = tuple(
        sorted(
            {
                exposure.state.value
                for source in evidence.sources
                if source.attribution is not None
                for exposure in source.attribution.exposures
            }
        )
    )
    summary = (
        f"{evidence.completeness.value} · {evidence.retained_source_count}/"
        f"{evidence.total_source_count} source(s) retained"
    )
    if evidence.proves_empty:
        summary += " · proven empty"
    if evidence.has_indeterminate_exposure:
        summary += " · indeterminate exposure"
    summary += " · limitations " + (", ".join(limitations) if limitations else "none")
    summary += " · lifecycle " + (", ".join(lifecycle_states) if lifecycle_states else "none")
    return summary


def standard_eval_memory_attribution_bounds(
    *,
    max_source_bytes: int = EVAL_MEMORY_ATTRIBUTION_MAX_SOURCE_BYTES_PER_TRIAL,
    max_projection_bytes: int = EVAL_MEMORY_ATTRIBUTION_MAX_PROJECTION_BYTES_PER_TRIAL,
) -> MemoryAttributionBounds:
    policy = EvalMemoryAttributionCapturePolicyV1.standard()
    return MemoryAttributionBounds(
        max_receipts=policy.max_receipts,
        max_exposures=policy.max_exposures,
        max_items=policy.max_items,
        max_source_bytes=min(max_source_bytes, policy.max_source_bytes),
        max_projection_bytes=min(max_projection_bytes, policy.max_projection_bytes),
    )


def eval_memory_attribution_bounds_for_trial_count(
    trial_count: int,
) -> MemoryAttributionBounds:
    """Partition both fresh closure reads before any trial is dispatched."""

    if type(trial_count) is not int:
        raise TypeError("trials must be an integer.")
    if trial_count < 1:
        raise ValueError("trials must be a positive integer.")
    closure_read_count = trial_count * 2
    return standard_eval_memory_attribution_bounds(
        max_source_bytes=max(
            1,
            min(
                EVAL_MEMORY_ATTRIBUTION_MAX_SOURCE_BYTES_PER_TRIAL,
                EVAL_MEMORY_ATTRIBUTION_SOURCE_BUDGET_BYTES // closure_read_count,
            ),
        ),
        max_projection_bytes=max(
            1,
            min(
                EVAL_MEMORY_ATTRIBUTION_MAX_PROJECTION_BYTES_PER_TRIAL,
                EVAL_MEMORY_ATTRIBUTION_PROJECTION_BUDGET_BYTES // closure_read_count,
            ),
        ),
    )


def eval_memory_attribution_source_limit_for_trial_count(trial_count: int) -> int:
    """Partition retained source wrappers across one published run before dispatch."""

    if type(trial_count) is not int:
        raise TypeError("trials must be an integer.")
    if trial_count < 1:
        raise ValueError("trials must be a positive integer.")
    return min(
        EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES,
        EVAL_MEMORY_ATTRIBUTION_MAX_RETAINED_SOURCES_PER_RUN // trial_count,
    )


def eval_memory_attribution_max_bytes_for_trial_count(trial_count: int) -> int:
    """Partition the final published memory section before any trial dispatches."""

    if type(trial_count) is not int:
        raise TypeError("trials must be an integer.")
    if trial_count < 1:
        raise ValueError("trials must be a positive integer.")
    selected = min(
        EVAL_MEMORY_ATTRIBUTION_MAX_BYTES,
        EVAL_MEMORY_ATTRIBUTION_RESULT_BUDGET_BYTES // trial_count,
    )
    if selected < EVAL_MEMORY_ATTRIBUTION_MIN_DOCUMENT_BYTES:
        raise ValueError("trials exceed the aggregate eval memory-attribution result budget.")
    return selected


def eval_memory_source_alias(
    *,
    session_id: str,
    key_id: str,
    key: bytes,
) -> EvalMemorySourceAliasV1:
    if type(session_id) is not str or not session_id:
        raise ValueError("session_id must be nonempty.")
    if type(key_id) is not str or not key_id.strip():
        raise ValueError("key_id must be nonblank.")
    if type(key) is not bytes or not key:
        raise ValueError("key must be nonempty bytes.")
    material = canonical_durable_json_bytes(
        {"schema_version": 1, "session_id": session_id},
        "eval memory source alias",
    )
    return EvalMemorySourceAliasV1(
        key_id=key_id,
        digest=hmac.digest(
            key,
            EVAL_MEMORY_SOURCE_ALIAS_CONTEXT + b"\x00" + material,
            "sha256",
        ).hex(),
    )


__all__ = [
    "EVAL_MEMORY_ATTRIBUTION_MAX_BYTES",
    "EVAL_MEMORY_ATTRIBUTION_PROJECTION_BUDGET_BYTES",
    "EVAL_MEMORY_ATTRIBUTION_RESULT_BUDGET_BYTES",
    "EVAL_MEMORY_ATTRIBUTION_SCHEMA_VERSION",
    "EVAL_MEMORY_ATTRIBUTION_SOURCE_BUDGET_BYTES",
    "EvalMemoryAttributionCapturePolicyV1",
    "EvalMemoryAttributionEvidenceV1",
    "EvalMemoryAttributionSourceV1",
    "EvalMemoryEvidenceCompleteness",
    "EvalMemoryEvidenceLimitation",
    "EvalMemorySourceAliasV1",
    "EvalMemorySourceReferenceV1",
    "eval_memory_attribution_bounds_for_trial_count",
    "eval_memory_attribution_fingerprint",
    "eval_memory_attribution_max_bytes_for_trial_count",
    "eval_memory_attribution_source_limit_for_trial_count",
    "eval_memory_attribution_summary",
    "eval_memory_source_alias",
    "standard_eval_memory_attribution_bounds",
]
