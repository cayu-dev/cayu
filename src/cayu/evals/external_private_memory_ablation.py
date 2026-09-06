"""Bounded application-owned execution for private memory-ablation campaigns."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import stat
import sys
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Self, cast

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

from cayu._filesystem_lock import cooperative_path_lock
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    compact_json_utf8_size,
    require_durable_clean_nonblank,
    revalidate_model_input,
)
from cayu.agent_snapshots import AgentSnapshot
from cayu.cli._guarded_tree_publication import (
    DestinationPolicy,
    GuardedTreePublicationError,
    GuardedTreeStage,
    _assert_windows_directory_dacl_is_protected,
    _capture_parent,
    _capture_stable_identity,
    _Identity,
    _is_windows_reparse_point,
    _linux_birth_time_ns,
    _pinned_parent,
    _reject_link_components,
    _sync_windows_path,
    _windows_directory_namespace_fence,
    publish_guarded_tree,
)
from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_BYTES,
    EVAL_CORPUS_MAX_JUDGE_EXPLANATION_CHARS,
    EvalCaseSpec,
    EvalCorpusDocument,
    PrivateJudgeReferenceV1,
    StructuredModelJudgeAssertionSpec,
    assertion_spec_revision,
    eval_corpus_from_json,
    eval_run_contract_for_corpus,
    eval_suite_trial_policy,
    pricing_profile_identity,
)
from cayu.evals.execution import (
    CorpusExecutionResult,
    CorpusTarget,
    EvaluationTargetIdentity,
    compile_corpus_suite,
    evaluation_target_identity,
)
from cayu.evals.execution_profiles import _eval_execution_target_material_identity
from cayu.evals.memory_reporting import (
    MEMORY_EXPERIMENT_REPORT_MAX_BYTES,
    MemoryExperimentCase,
    MemoryExperimentReport,
    MemoryExperimentReportRequest,
    MemoryExperimentTrialEvidence,
    MemoryExperimentVariant,
    MemoryPreparationOverheadEvidence,
    MemoryPublishedResultEvidence,
    MemoryTrialAvailability,
    _validate_accounting_side_authority,
    build_memory_experiment_report,
    memory_experiment_report_from_json,
    memory_experiment_report_to_json,
)
from cayu.evals.models import (
    EvalAssertionResult,
    EvalCaseResult,
    EvalOutcome,
    EvalRun,
    EvalStatus,
    EvalTrialResult,
    aggregate_eval_score,
    aggregate_eval_status,
)
from cayu.evals.published import (
    _publish_eval_run_with_trial_public_data,
    _published_assertion,
)
from cayu.evals.result_contract import (
    EvalTrialDiagnosticCode,
    EvalTrialOutputPreviewV1,
    _EvalTrialPublicData,
)
from cayu.memory_intervention_execution import (
    MEMORY_INTERVENTION_EXECUTION_MAX_RECORD_BYTES,
    MemoryInterventionExecutionStatus,
    MemoryInterventionExecutor,
    MemoryInterventionExecutorAuthority,
    MemoryInterventionExecutorStatePaths,
    MemoryInterventionProviderExecutionMode,
    MemoryInterventionTrialOutcome,
    MemoryInterventionTrialRequest,
)
from cayu.memory_interventions import MemoryInterventionKind
from cayu.runtime.cost_quality import (
    PairedCostQualityComparisonRequest,
    PairedCostQualityPair,
    PairedCostQualitySide,
    compare_paired_cost_quality,
)
from cayu.runtime.sessions import RunRequest

EXTERNAL_PRIVATE_MEMORY_ABLATION_SCHEMA_VERSION = 1
# Used only in a trusted CorpusTarget's request budget template. Preflight
# replaces this key with the exact trial causal-budget identity before freezing
# the complete request. Ordinary runtime requests never receive the marker.
EXTERNAL_PRIVATE_MEMORY_ABLATION_TRIAL_BUDGET_KEY = "cayu:private-memory:trial-budget"
EXTERNAL_PRIVATE_MEMORY_ABLATION_MAX_ARTIFACT_BYTES = 64 << 20
EXTERNAL_PRIVATE_MEMORY_ABLATION_MAX_LIMITATIONS = 32
# Even when the full intervention binding must be omitted, the complete terminal
# execution record and its JSON container must always fit. Execution records are
# independently bounded, and 256 bytes is a conservative bound for the three
# fixed object keys, separators, and two null evidence members.
EXTERNAL_PRIVATE_MEMORY_ABLATION_MIN_REPORT_EVIDENCE_BYTES_PER_TRIAL = (
    MEMORY_INTERVENTION_EXECUTION_MAX_RECORD_BYTES + 256
)
EXTERNAL_PRIVATE_MEMORY_ABLATION_MAX_REPORT_EVIDENCE_BYTES_PER_TRIAL = 4 << 20
EXTERNAL_PRIVATE_MEMORY_ABLATION_MAX_SUPPLEMENTAL_BYTES_PER_TRIAL = 64 << 10
EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME = "memory-experiment-report.json"
EXTERNAL_PRIVATE_MEMORY_ABLATION_METHODOLOGY_FILENAME = "methodology.json"
EXTERNAL_PRIVATE_MEMORY_ABLATION_COMPLETION_FILENAME = "COMPLETE"

_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    hide_input_in_errors=True,
    revalidate_instances="always",
    validate_default=True,
)
_PREPARED_AUTHORITY = object()
_PRIVATE_FILESYSTEM_IDENTITY_LOCK = Lock()
_PRIVATE_OUTPUT_PLACEHOLDER = "[external-private output omitted]"
_COST_DECIMAL_MAX_DIGITS = 64
_COST_DECIMAL_MAX_PLACES = 64
_REPORT_FIXED_EVIDENCE_BYTES_PER_TRIAL = 256 << 10
_REPORT_PUBLISHED_CONTAINER_BYTES_PER_VARIANT = 64 << 10
_REPORT_ACCOUNTING_PROJECTION_MULTIPLIER = 8
_ARTIFACT_COMPLETION_MAX_BYTES = 4 << 10
_PRIVATE_SAFE_PORTABLE_ASSERTION_KINDS = frozenset(
    {
        "child_status",
        "final_output_contains",
        "final_output_equals",
        "max_estimated_cost",
        "max_model_steps",
        "max_tool_calls",
        "max_total_tokens",
        "memory_attribution",
        "process_event",
        "process_events_in_order",
        "root_status",
        "structured_model_judge",
        "tool_called",
        "tools_called_in_order",
        "usage_recorded",
    }
)
_SUPPORTS_PRIVATE_DESCRIPTOR_TRAVERSAL = (
    os.name != "nt"
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "fchmod")
    and os.open in os.supports_dir_fd
    and os.mkdir in os.supports_dir_fd
    and os.rmdir in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
    and os.stat in os.supports_follow_symlinks
)


@dataclass(frozen=True, slots=True)
class _PrivateFilesystemIdentity:
    """Process-local private identity with a Linux inode-reuse pin."""

    device: int
    inode: int
    kind: int
    incarnation: int
    descriptor: int | None = field(default=None, compare=False, repr=False)
    _closed: bool = field(default=False, compare=False, repr=False)

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, _memo: object) -> Self:
        return self

    def require_open(self) -> None:
        with _PRIVATE_FILESYSTEM_IDENTITY_LOCK:
            if self._closed:
                raise ValueError("Private filesystem authority is closed.")
            if sys.platform.startswith("linux"):
                descriptor = self.descriptor
                if descriptor is None:
                    raise ValueError("Linux private filesystem authority lost its inode pin.")
                try:
                    observed = os.fstat(descriptor)
                except OSError as exc:
                    raise ValueError(
                        "Linux private filesystem authority lost its inode pin."
                    ) from exc
                if not self.matches(observed):
                    raise ValueError("Linux private filesystem authority inode pin changed.")

    def close(self) -> None:
        with _PRIVATE_FILESYSTEM_IDENTITY_LOCK:
            if self._closed:
                return
            descriptor = self.descriptor
            object.__setattr__(self, "descriptor", None)
            object.__setattr__(self, "_closed", True)
            if descriptor is not None:
                os.close(descriptor)

    @classmethod
    def from_guarded_identity(cls, value: _Identity) -> _PrivateFilesystemIdentity:
        if value.incarnation is None:
            raise ValueError("Private filesystem authority lacks incarnation evidence.")
        return cls(
            device=value.device,
            inode=value.inode,
            kind=value.kind,
            incarnation=value.incarnation,
        )

    @classmethod
    def capture(
        cls,
        value: os.stat_result,
        *,
        incarnation: int,
        descriptor: int | None = None,
    ) -> _PrivateFilesystemIdentity:
        observed = _Identity.capture(value)
        return cls(
            device=observed.device,
            inode=observed.inode,
            kind=observed.kind,
            incarnation=incarnation,
            descriptor=descriptor,
        )

    def matches(self, value: os.stat_result) -> bool:
        return (
            self.device == int(value.st_dev)
            and self.inode == int(value.st_ino)
            and self.kind == stat.S_IFMT(value.st_mode)
        )


def _clean(value: str, field_name: str, *, maximum: int = 256) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} characters.")
    return value


def _fingerprint(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _revision(value: str, field_name: str) -> str:
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must be a sha256 revision.")
    return value


def _utc(value: datetime, field_name: str) -> datetime:
    if type(value) is not datetime:
        raise ValueError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


def _content_revision(value: object, field_name: str) -> str:
    return "sha256:" + hashlib.sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


class ExternalPrivateMemoryAblationExecutionMode(StrEnum):
    HERMETIC = "hermetic"
    LIVE = "live"


class ExternalPrivateMemoryAblationScheduleStrategy(StrEnum):
    FIXED = "fixed"
    DETERMINISTIC_RANDOMIZED = "deterministic_randomized"
    COUNTERBALANCED = "counterbalanced"


class ExternalPrivateMemoryAblationCacheEvidencePolicy(StrEnum):
    REQUIRED = "required"
    BEST_EFFORT = "best_effort"
    UNAVAILABLE = "unavailable"


class ExternalPrivateMemoryAblationRunStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class ExternalPrivateMemoryAblationSchedulePolicy(BaseModel):
    """Versioned policy that deterministically realizes one exact trial order."""

    model_config = _MODEL_CONFIG

    schema_version: Literal[1] = EXTERNAL_PRIVATE_MEMORY_ABLATION_SCHEMA_VERSION
    strategy: ExternalPrivateMemoryAblationScheduleStrategy
    seed_fingerprint: StrictStr | None = Field(default=None, min_length=64, max_length=64)
    max_concurrency: Literal[1] = 1

    @field_validator("schema_version", "max_concurrency", mode="before")
    @classmethod
    def validate_integer_literals(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be a JSON integer.")
        return value

    @field_validator("seed_fingerprint")
    @classmethod
    def validate_seed(cls, value: str | None, info) -> str | None:
        return None if value is None else _fingerprint(value, info.field_name)

    @model_validator(mode="after")
    def validate_strategy(self) -> Self:
        seeded = self.strategy is not ExternalPrivateMemoryAblationScheduleStrategy.FIXED
        if seeded != (self.seed_fingerprint is not None):
            raise ValueError(
                "Randomized and counterbalanced schedules require exactly one seed fingerprint."
            )
        return self

    @property
    def revision(self) -> str:
        return _content_revision(
            self.model_dump(mode="json"),
            "external private memory ablation schedule policy",
        )


class ExternalPrivateMemoryAblationAuthorization(BaseModel):
    """Application-owned authority and hard ceilings checked before provider work."""

    model_config = _MODEL_CONFIG

    schema_version: Literal[1] = EXTERNAL_PRIVATE_MEMORY_ABLATION_SCHEMA_VERSION
    authorization_id: StrictStr = Field(max_length=256)
    valid_from: datetime
    valid_through: datetime
    corpus_revision: StrictStr = Field(min_length=71, max_length=71)
    suite_id: StrictStr = Field(max_length=128)
    experiment_revision: StrictStr = Field(min_length=71, max_length=71)
    schedule_policy_revision: StrictStr = Field(min_length=71, max_length=71)
    target_key: StrictStr = Field(max_length=128)
    application_release_id: StrictStr = Field(max_length=256)
    app_manifest_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    snapshot_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    evaluator_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    provider_name: StrictStr = Field(max_length=256)
    model: StrictStr = Field(max_length=256)
    provider_configuration_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    evidence_collector_fingerprint: StrictStr | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )
    evidence_policy_revision: StrictStr = Field(min_length=71, max_length=71)
    pricing_profile_fingerprint: StrictStr | None = Field(
        default=None,
        min_length=71,
        max_length=71,
    )
    redaction_policy_revision: StrictStr = Field(min_length=71, max_length=71)
    retention_policy_revision: StrictStr = Field(min_length=71, max_length=71)
    state_storage_id: StrictStr = Field(max_length=256)
    report_destination_id: StrictStr = Field(max_length=256)
    report_destination_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    execution_mode: ExternalPrivateMemoryAblationExecutionMode
    live_execution_authorization_id: StrictStr | None = Field(default=None, max_length=256)
    allowed_variant_ids: tuple[StrictStr, ...] = Field(min_length=2, max_length=128)
    allowed_variant_kinds: tuple[MemoryInterventionKind, ...] = Field(
        min_length=2,
        max_length=len(MemoryInterventionKind),
    )
    minimum_cases: StrictInt = Field(ge=1, le=1_000)
    maximum_cases: StrictInt = Field(ge=1, le=1_000)
    minimum_repetitions: StrictInt = Field(ge=1, le=1_000)
    maximum_repetitions: StrictInt = Field(ge=1, le=1_000)
    maximum_total_trials: StrictInt = Field(ge=2, le=100_000)
    maximum_concurrency: Literal[1] = 1
    maximum_timeout_seconds: StrictInt = Field(ge=1, le=86_400)
    maximum_model_steps: StrictInt = Field(ge=1, le=256)
    maximum_report_evidence_bytes_per_trial: StrictInt = Field(
        ge=EXTERNAL_PRIVATE_MEMORY_ABLATION_MIN_REPORT_EVIDENCE_BYTES_PER_TRIAL,
        le=EXTERNAL_PRIVATE_MEMORY_ABLATION_MAX_REPORT_EVIDENCE_BYTES_PER_TRIAL,
    )
    maximum_total_tokens_per_trial: StrictInt | None = Field(
        default=None,
        ge=1,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    maximum_estimated_cost_total: Decimal | None = Field(default=None, gt=Decimal(0))
    cost_currency: StrictStr | None = Field(default=None, min_length=3, max_length=16)
    maximum_supplemental_evidence_bytes_per_trial: StrictInt | None = Field(
        default=None,
        ge=1,
        le=EXTERNAL_PRIVATE_MEMORY_ABLATION_MAX_SUPPLEMENTAL_BYTES_PER_TRIAL,
    )
    cache_evidence_policy: ExternalPrivateMemoryAblationCacheEvidencePolicy

    @field_validator(
        "schema_version",
        "minimum_cases",
        "maximum_cases",
        "minimum_repetitions",
        "maximum_repetitions",
        "maximum_total_trials",
        "maximum_concurrency",
        "maximum_timeout_seconds",
        "maximum_model_steps",
        "maximum_report_evidence_bytes_per_trial",
        "maximum_total_tokens_per_trial",
        "maximum_supplemental_evidence_bytes_per_trial",
        mode="before",
    )
    @classmethod
    def validate_integer_fields(cls, value: object, info) -> object:
        if value is not None and type(value) is not int:
            raise ValueError(f"{info.field_name} must be a JSON integer.")
        return value

    @field_validator(
        "authorization_id",
        "suite_id",
        "target_key",
        "application_release_id",
        "provider_name",
        "model",
        "state_storage_id",
        "report_destination_id",
        "live_execution_authorization_id",
    )
    @classmethod
    def validate_identity_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _clean(value, info.field_name, maximum=256)

    @field_validator(
        "app_manifest_fingerprint",
        "snapshot_fingerprint",
        "evaluator_fingerprint",
        "provider_configuration_fingerprint",
        "evidence_collector_fingerprint",
        "report_destination_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str | None, info) -> str | None:
        return None if value is None else _fingerprint(value, info.field_name)

    @field_validator(
        "corpus_revision",
        "experiment_revision",
        "schedule_policy_revision",
        "evidence_policy_revision",
        "pricing_profile_fingerprint",
        "redaction_policy_revision",
        "retention_policy_revision",
    )
    @classmethod
    def validate_revisions(cls, value: str | None, info) -> str | None:
        return None if value is None else _revision(value, info.field_name)

    @field_validator("valid_from", "valid_through")
    @classmethod
    def validate_datetimes(cls, value: datetime, info) -> datetime:
        return _utc(value, info.field_name)

    @field_validator("allowed_variant_ids", mode="before")
    @classmethod
    def validate_variant_ids(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise TypeError("allowed_variant_ids must be an ordered array.")
        if any(type(item) is not str for item in value):
            raise TypeError("allowed_variant_ids must contain strings.")
        result = tuple(
            _clean(cast("str", item), "allowed_variant_ids", maximum=128) for item in value
        )
        if result != tuple(sorted(set(result))):
            raise ValueError("allowed_variant_ids must be unique and sorted.")
        return result

    @field_validator("allowed_variant_kinds", mode="before")
    @classmethod
    def validate_variant_kinds(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise TypeError("allowed_variant_kinds must be an ordered array.")
        result = tuple(MemoryInterventionKind(item) for item in value)
        if result != tuple(sorted(set(result), key=str)):
            raise ValueError("allowed_variant_kinds must be unique and sorted.")
        return result

    @field_validator("cost_currency")
    @classmethod
    def validate_currency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = _clean(value, "cost_currency", maximum=16).upper()
        if not value.isascii() or not value.isalpha():
            raise ValueError("cost_currency must contain uppercase ASCII letters only.")
        return value

    @field_validator("maximum_estimated_cost_total")
    @classmethod
    def validate_total_cost_ceiling(cls, value: Decimal | None) -> Decimal | None:
        if value is not None:
            _require_bounded_cost_decimal(value, "maximum_estimated_cost_total")
        return value

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        if self.valid_through <= self.valid_from:
            raise ValueError("valid_through must be later than valid_from.")
        if self.minimum_cases > self.maximum_cases:
            raise ValueError("minimum_cases cannot exceed maximum_cases.")
        if self.minimum_repetitions > self.maximum_repetitions:
            raise ValueError("minimum_repetitions cannot exceed maximum_repetitions.")
        if (self.maximum_estimated_cost_total is None) != (self.cost_currency is None):
            raise ValueError("A total cost ceiling requires exactly one currency.")
        if (
            self.maximum_estimated_cost_total is not None
            and self.pricing_profile_fingerprint is None
        ):
            raise ValueError("A total cost ceiling requires an exact pricing identity.")
        if (self.evidence_collector_fingerprint is None) != (
            self.maximum_supplemental_evidence_bytes_per_trial is None
        ):
            raise ValueError(
                "An evidence collector requires exactly one per-trial supplemental byte ceiling."
            )
        live = self.execution_mode is ExternalPrivateMemoryAblationExecutionMode.LIVE
        if live != (self.live_execution_authorization_id is not None):
            raise ValueError("Live execution requires exactly one explicit live authorization id.")
        if live and (
            self.maximum_total_tokens_per_trial is None
            or self.maximum_estimated_cost_total is None
            or self.pricing_profile_fingerprint is None
            or self.evidence_collector_fingerprint is None
            or self.cache_evidence_policy
            is ExternalPrivateMemoryAblationCacheEvidencePolicy.UNAVAILABLE
        ):
            raise ValueError(
                "Live execution requires token and cost ceilings, pricing identity, an evidence "
                "collector, and cache evidence policy."
            )
        if (
            self.cache_evidence_policy is ExternalPrivateMemoryAblationCacheEvidencePolicy.REQUIRED
            and self.evidence_collector_fingerprint is None
        ):
            raise ValueError("Required cache evidence needs an authorized evidence collector.")
        return self

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_durable_json_bytes(
                self.model_dump(mode="json"),
                "external private memory ablation authorization",
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class _ExternalPrivateCorpusFilesystemAuthority:
    root_identity: _PrivateFilesystemIdentity
    source_identity: _PrivateFilesystemIdentity

    def require_open(self) -> None:
        self.root_identity.require_open()
        self.source_identity.require_open()

    def close(self) -> None:
        with ExitStack() as cleanup:
            cleanup.callback(self.root_identity.close)
            cleanup.callback(self.source_identity.close)


@dataclass(frozen=True, slots=True, repr=False)
class _ExternalPrivateDestinationFilesystemAuthority:
    root_identity: _PrivateFilesystemIdentity
    state_identity: _PrivateFilesystemIdentity
    artifact_parent_identity: _PrivateFilesystemIdentity

    def require_open(self) -> None:
        self.root_identity.require_open()
        self.state_identity.require_open()
        self.artifact_parent_identity.require_open()

    def close(self) -> None:
        with ExitStack() as cleanup:
            cleanup.callback(self.root_identity.close)
            cleanup.callback(self.state_identity.close)
            cleanup.callback(self.artifact_parent_identity.close)


@dataclass(frozen=True, slots=True, repr=False)
class ExternalPrivateEvalCorpus:
    """One bounded standard Evals corpus loaded from an approved private root."""

    document: EvalCorpusDocument
    source_path: Path
    approved_private_root: Path
    source_sha256: str
    _filesystem_authority: _ExternalPrivateCorpusFilesystemAuthority | None = None

    def close(self) -> None:
        """Release process-local filesystem authority retained by this corpus."""

        if self._filesystem_authority is not None:
            self._filesystem_authority.close()

    def __enter__(self) -> Self:
        if self._filesystem_authority is None:
            raise ValueError("Private corpus lacks filesystem authority.")
        self._filesystem_authority.require_open()
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True, repr=False)
class ExternalPrivateMemoryAblationDestination:
    """Private state authority plus one not-yet-created artifact directory."""

    state_storage_id: str
    destination_id: str
    approved_private_root: Path
    state_directory: Path
    artifact_directory: Path
    _filesystem_authority: _ExternalPrivateDestinationFilesystemAuthority | None = None

    def close(self) -> None:
        """Release process-local filesystem authority retained by this destination."""

        if self._filesystem_authority is not None:
            self._filesystem_authority.close()

    def __enter__(self) -> Self:
        if self._filesystem_authority is None:
            raise ValueError("Private destination lacks filesystem authority.")
        self._filesystem_authority.require_open()
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    @property
    def fingerprint(self) -> str:
        """Bind the application identity to the exact canonical private paths."""

        return hashlib.sha256(
            canonical_durable_json_bytes(
                {
                    "state_storage_id": self.state_storage_id,
                    "destination_id": self.destination_id,
                    "approved_private_root": os.fspath(self.approved_private_root),
                    "state_directory": os.fspath(self.state_directory),
                    "artifact_directory": os.fspath(self.artifact_directory),
                },
                "external private memory ablation destination",
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class ExternalPrivateMemoryAblationTrial:
    """Process-private input for one exact experiment matrix coordinate."""

    case_id: str
    repetition: int
    variant_id: str
    request: MemoryInterventionTrialRequest

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _clean(self.case_id, "case_id", maximum=128))
        object.__setattr__(self, "variant_id", _clean(self.variant_id, "variant_id", maximum=128))
        if type(self.repetition) is not int or self.repetition < 1:
            raise ValueError("repetition must be a positive integer.")
        if type(self.request) is not MemoryInterventionTrialRequest:
            raise TypeError("request must be an exact MemoryInterventionTrialRequest.")
        object.__setattr__(
            self,
            "request",
            MemoryInterventionTrialRequest.model_validate(
                self.request.model_dump(mode="python", round_trip=True, warnings="none")
            ),
        )

    @property
    def coordinate(self) -> tuple[str, int, str]:
        return (self.case_id, self.repetition, self.variant_id)


class ExternalPrivateMemoryAblationScheduleEntry(BaseModel):
    model_config = _MODEL_CONFIG

    ordinal: StrictInt = Field(ge=1, le=100_000)
    case_id: StrictStr = Field(max_length=128)
    case_revision: StrictStr = Field(min_length=71, max_length=71)
    repetition: StrictInt = Field(ge=1, le=1_000)
    variant_id: StrictStr = Field(max_length=128)
    execution_id: StrictStr = Field(min_length=64, max_length=64)

    @field_validator("ordinal", "repetition", mode="before")
    @classmethod
    def validate_integers(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be a JSON integer.")
        return value

    @field_validator("case_id", "variant_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=128)

    @field_validator("case_revision")
    @classmethod
    def validate_case_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)

    @field_validator("execution_id")
    @classmethod
    def validate_execution_id(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)


@dataclass(frozen=True, slots=True, repr=False)
class PreparedExternalPrivateMemoryAblation:
    """Fully validated process-local authority ready for bounded execution."""

    corpus: ExternalPrivateEvalCorpus
    target: CorpusTarget
    target_identity: EvaluationTargetIdentity
    snapshot: AgentSnapshot
    experiment: MemoryExperimentReportRequest
    authorization: ExternalPrivateMemoryAblationAuthorization
    schedule_policy: ExternalPrivateMemoryAblationSchedulePolicy
    destination: ExternalPrivateMemoryAblationDestination
    scheduled_trials: tuple[ExternalPrivateMemoryAblationTrial, ...]
    schedule: tuple[ExternalPrivateMemoryAblationScheduleEntry, ...]
    executor_authority: MemoryInterventionExecutorAuthority
    executor_binding_revision: str
    state_files_revision: str
    preflight_revision: str
    _authority: object
    _executor: MemoryInterventionExecutor
    _trial_request_revisions: tuple[str, ...]

    def close(self) -> None:
        """Release the campaign-owned corpus and destination authority leases."""

        with ExitStack() as cleanup:
            cleanup.callback(self.corpus.close)
            cleanup.callback(self.destination.close)

    def _require_open(self) -> None:
        if self.corpus._filesystem_authority is None:
            raise ValueError("Prepared private corpus lacks filesystem authority.")
        if self.destination._filesystem_authority is None:
            raise ValueError("Prepared private destination lacks filesystem authority.")
        self.corpus._filesystem_authority.require_open()
        self.destination._filesystem_authority.require_open()

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


def _resolve_private_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("approved_private_root must be an existing directory.")
    return root


def _resolve_private_root_with_identity(
    value: str | Path,
) -> tuple[Path, _PrivateFilesystemIdentity]:
    """Canonicalize a root and bind the exact object observed during resolution."""

    root = _resolve_private_root(value)
    observed = root.stat(follow_symlinks=False)
    _require_private_directory(observed, "approved_private_root")
    return root, _capture_private_filesystem_identity(
        observed,
        path=root,
        retain_descriptor=True,
    )


def _require_below_root(path: Path, root: Path, field_name: str) -> None:
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"{field_name} must be below the approved private root.")


def _require_private_directory(value: os.stat_result, field_name: str) -> None:
    if (
        stat.S_ISLNK(value.st_mode)
        or _is_windows_reparse_point(value)
        or not stat.S_ISDIR(value.st_mode)
    ):
        raise ValueError(f"{field_name} must be an existing ordinary directory.")


def _require_posix_private_descriptor_support() -> None:
    if os.name == "nt":
        return
    if not _SUPPORTS_PRIVATE_DESCRIPTOR_TRAVERSAL:
        raise RuntimeError(
            "External private evaluation requires descriptor-relative no-follow filesystem "
            "operations on this platform."
        )


def _require_posix_private_artifact_publication_support() -> None:
    """Require the native atomic no-replace primitive used for final publication."""

    _require_posix_private_descriptor_support()
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        raise RuntimeError(
            "External private artifact publication requires atomic no-replace rename support."
        )
    import ctypes

    function_name = "renameat2" if sys.platform.startswith("linux") else "renameatx_np"
    if getattr(ctypes.CDLL(None), function_name, None) is None:
        raise RuntimeError(
            "External private artifact publication requires atomic no-replace rename support."
        )


def _capture_private_filesystem_identity(
    value: os.stat_result,
    *,
    path: Path | None = None,
    descriptor: int | None = None,
    retain_descriptor: bool = False,
) -> _PrivateFilesystemIdentity:
    """Capture object identity, optionally retaining a Linux anti-reuse descriptor."""

    if not sys.platform.startswith("linux"):
        return _PrivateFilesystemIdentity.from_guarded_identity(
            _capture_stable_identity(value, path=path, descriptor=descriptor)
        )

    expected = _Identity.capture(value)
    pinned = descriptor
    owned = False
    if pinned is None:
        if path is None:
            raise ValueError("Private filesystem identity requires a path or descriptor.")
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        if stat.S_ISDIR(value.st_mode):
            flags |= os.O_DIRECTORY
        else:
            flags |= getattr(os, "O_NONBLOCK", 0)
        pinned = os.open(path, flags)
        owned = True
    retained_descriptor: int | None = None
    try:
        if not expected.matches(os.fstat(pinned)):
            raise ValueError("Private filesystem object changed while its identity was captured.")
        birth_time_ns = _linux_birth_time_ns(
            value,
            descriptor=pinned,
            path=path,
        )
        if retain_descriptor:
            retained_descriptor = os.dup(pinned)
    finally:
        if owned:
            os.close(pinned)
    try:
        return _PrivateFilesystemIdentity.capture(
            value,
            incarnation=birth_time_ns,
            descriptor=retained_descriptor,
        )
    except BaseException:
        if retained_descriptor is not None:
            os.close(retained_descriptor)
        raise


def _require_current_path_identity(
    path: Path,
    *,
    expected: _PrivateFilesystemIdentity,
    field_name: str,
) -> None:
    try:
        _reject_link_components(path)
        current = path.stat(follow_symlinks=False)
        observed = _capture_private_filesystem_identity(current, path=path)
    except (GuardedTreePublicationError, OSError) as exc:
        raise ValueError(f"{field_name} changed while private data was accessed.") from exc
    if observed != expected:
        raise ValueError(f"{field_name} changed while private data was accessed.")


@contextmanager
def _open_private_directory(
    path: Path,
    *,
    field_name: str,
    expected: _PrivateFilesystemIdentity | None = None,
    retain_identity: bool = True,
    dependent_cleanup_on_error: Callable[[], None] | None = None,
) -> Iterator[tuple[int | None, _PrivateFilesystemIdentity]]:
    """Pin one directory and authenticate it before yielding any private authority."""

    _reject_link_components(path)
    if os.name == "nt":
        guarded_identity = _capture_parent(path)
        observed = _PrivateFilesystemIdentity.from_guarded_identity(guarded_identity)
        if expected is not None and observed != expected:
            raise ValueError(f"{field_name} changed after it was authorized.")
        try:
            with _pinned_parent(path, expected=guarded_identity):
                current = _PrivateFilesystemIdentity.from_guarded_identity(_capture_parent(path))
                if current != observed:
                    raise ValueError(f"{field_name} changed while it was being pinned.")
                yield None, observed
                _require_current_path_identity(
                    path,
                    expected=observed,
                    field_name=field_name,
                )
        except BaseException:
            with ExitStack() as cleanup:
                cleanup.callback(observed.close)
                if dependent_cleanup_on_error is not None:
                    cleanup.callback(dependent_cleanup_on_error)
            raise
        return

    _require_posix_private_descriptor_support()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        current = os.fstat(descriptor)
        _require_private_directory(current, field_name)
        observed = _capture_private_filesystem_identity(
            current,
            descriptor=descriptor,
            retain_descriptor=retain_identity,
        )
        if expected is not None and observed != expected:
            observed.close()
            raise ValueError(f"{field_name} changed after it was authorized.")
        try:
            yield descriptor, observed
            _require_current_path_identity(
                path,
                expected=observed,
                field_name=field_name,
            )
        except BaseException:
            with ExitStack() as cleanup:
                cleanup.callback(observed.close)
                if dependent_cleanup_on_error is not None:
                    cleanup.callback(dependent_cleanup_on_error)
            raise
    finally:
        os.close(descriptor)


def _safe_relative_path(path: Path, root: Path, field_name: str) -> Path:
    _require_below_root(path, root, field_name)
    relative = path.relative_to(root)
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{field_name} must be a canonical path below the approved root.")
    return relative


@contextmanager
def _open_relative_private_entry(
    root_descriptor: int,
    relative: Path,
    *,
    directory: bool,
) -> Iterator[int]:
    """Traverse every ancestor from a pinned root without following links."""

    descriptor = os.dup(root_descriptor)
    try:
        for component in relative.parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        if directory:
            flags |= os.O_DIRECTORY
        else:
            flags |= getattr(os, "O_NONBLOCK", 0)
        child = os.open(relative.parts[-1], flags, dir_fd=descriptor)
        try:
            yield child
        finally:
            os.close(child)
    finally:
        os.close(descriptor)


def _private_directory_identity_below_root(
    root_descriptor: int | None,
    *,
    root: Path,
    path: Path,
    field_name: str,
) -> _PrivateFilesystemIdentity:
    relative = _safe_relative_path(path, root, field_name)
    if root_descriptor is None:
        observed = path.stat(follow_symlinks=False)
        _require_private_directory(observed, field_name)
        return _capture_private_filesystem_identity(
            observed,
            path=path,
            retain_descriptor=True,
        )
    with _open_relative_private_entry(root_descriptor, relative, directory=True) as descriptor:
        observed = os.fstat(descriptor)
        _require_private_directory(observed, field_name)
        return _capture_private_filesystem_identity(
            observed,
            descriptor=descriptor,
            retain_descriptor=True,
        )


def _filesystem_identity_revision_material(identity: _PrivateFilesystemIdentity) -> list[str]:
    """Encode an object identity without exposing machine-sized integers to JSON."""

    return [
        str(identity.device),
        str(identity.inode),
        str(identity.kind),
        str(identity.incarnation),
    ]


def _open_private_windows_corpus(source: Path) -> int:
    """Open one Windows file without traversing a final reparse point."""

    import ctypes
    import msvcrt
    from ctypes import wintypes

    windows_ctypes: Any = ctypes
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    handle = create_file(
        os.fspath(source),
        0x80000000,  # GENERIC_READ
        0x1,  # FILE_SHARE_READ: fence mutation and namespace replacement.
        None,
        0x3,  # OPEN_EXISTING
        0x80 | 0x00200000,  # FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        error_code = windows_ctypes.get_last_error()
        raise OSError(error_code, "Could not securely open the external private corpus.")
    try:
        open_osfhandle = getattr(msvcrt, "open_osfhandle", None)
        if not callable(open_osfhandle):
            raise OSError("Windows file-descriptor conversion is unavailable.")
        return open_osfhandle(
            int(handle),
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
    except BaseException:
        close_handle(handle)
        raise


def _capture_destination_filesystem_authority(
    *,
    root: Path,
    state_directory: Path,
    artifact_parent: Path,
    expected: _ExternalPrivateDestinationFilesystemAuthority | None = None,
    expected_root_identity: _PrivateFilesystemIdentity | None = None,
) -> _ExternalPrivateDestinationFilesystemAuthority:
    if expected is not None:
        expected.require_open()
    expected_root = expected_root_identity if expected is None else expected.root_identity
    state_identity: _PrivateFilesystemIdentity | None = None
    artifact_parent_identity: _PrivateFilesystemIdentity | None = None
    try:
        with _open_private_directory(
            root,
            field_name="approved_private_root",
            expected=expected_root,
        ) as (root_descriptor, root_identity):
            state_identity = _private_directory_identity_below_root(
                root_descriptor,
                root=root,
                path=state_directory,
                field_name="state_directory",
            )
            artifact_parent_identity = (
                root_identity
                if artifact_parent == root
                else _private_directory_identity_below_root(
                    root_descriptor,
                    root=root,
                    path=artifact_parent,
                    field_name="artifact_directory parent",
                )
            )
            if expected is not None and (
                state_identity != expected.state_identity
                or artifact_parent_identity != expected.artifact_parent_identity
            ):
                raise ValueError("Private destination directories changed after authorization.")
            return _ExternalPrivateDestinationFilesystemAuthority(
                root_identity=root_identity,
                state_identity=state_identity,
                artifact_parent_identity=artifact_parent_identity,
            )
    except BaseException:
        with ExitStack() as cleanup:
            if state_identity is not None:
                cleanup.callback(state_identity.close)
            if artifact_parent_identity is not None:
                cleanup.callback(artifact_parent_identity.close)
        raise


def _read_private_corpus(
    source: Path,
    *,
    root: Path,
    expected: _ExternalPrivateCorpusFilesystemAuthority | None = None,
    expected_root_identity: _PrivateFilesystemIdentity | None = None,
    expected_source_identity: _PrivateFilesystemIdentity | None = None,
) -> tuple[
    bytes,
    EvalCorpusDocument,
    _ExternalPrivateCorpusFilesystemAuthority,
]:
    if expected is not None:
        expected.require_open()
    relative = _safe_relative_path(source, root, "corpus path")
    expected_root = expected_root_identity if expected is None else expected.root_identity
    expected_source = expected_source_identity if expected is None else expected.source_identity
    source_identity: _PrivateFilesystemIdentity | None = None

    def close_source_identity() -> None:
        if source_identity is not None:
            source_identity.close()

    with _open_private_directory(
        root,
        field_name="approved_private_root",
        expected=expected_root,
        dependent_cleanup_on_error=close_source_identity,
    ) as (root_descriptor, root_identity):
        if root_descriptor is None:
            _reject_link_components(source)
            source_parent = _capture_parent(source.parent)
            parent_context = _pinned_parent(source.parent, expected=source_parent)
            descriptor_path: str | Path = source
            descriptor_directory: int | None = None
        else:
            parent_context = nullcontext()
            descriptor_path = relative.parts[-1]
            descriptor_directory = None
            if len(relative.parts) > 1:
                directory_relative = Path(*relative.parts[:-1])
                directory_context = _open_relative_private_entry(
                    root_descriptor,
                    directory_relative,
                    directory=True,
                )
            else:
                directory_context = nullcontext(root_descriptor)

        descriptor: int | None = None
        try:
            with parent_context:
                if root_descriptor is None:
                    descriptor = _open_private_windows_corpus(source)
                    if (
                        _PrivateFilesystemIdentity.from_guarded_identity(_capture_parent(root))
                        != root_identity
                    ):
                        raise ValueError("approved_private_root changed while opening the corpus.")
                else:
                    with directory_context as opened_parent:
                        descriptor_directory = opened_parent
                        descriptor = os.open(
                            descriptor_path,
                            os.O_RDONLY
                            | os.O_NOFOLLOW
                            | getattr(os, "O_CLOEXEC", 0)
                            | getattr(os, "O_NONBLOCK", 0),
                            dir_fd=descriptor_directory,
                        )
            if descriptor is None:  # pragma: no cover - both platform branches assign it
                raise RuntimeError("External private corpus descriptor was not opened.")
            source_stat = os.fstat(descriptor)
            if _is_windows_reparse_point(source_stat) or not stat.S_ISREG(source_stat.st_mode):
                raise ValueError("External private corpus must be a regular file.")
            source_identity = _capture_private_filesystem_identity(
                source_stat,
                descriptor=descriptor,
                retain_descriptor=True,
            )
            if expected_source is not None and source_identity != expected_source:
                source_identity.close()
                raise ValueError("External private corpus changed after it was authorized.")
            chunks = []
            remaining = EVAL_CORPUS_MAX_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 << 10))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            _require_current_path_identity(
                source,
                expected=source_identity,
                field_name="External private corpus",
            )
        except BaseException:
            if source_identity is not None:
                source_identity.close()
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
    try:
        if len(raw) > EVAL_CORPUS_MAX_BYTES:
            raise ValueError(f"Eval corpus JSON exceeds {EVAL_CORPUS_MAX_BYTES} bytes.")
        text = raw.decode("utf-8")
        document = eval_corpus_from_json(text)
        if source_identity is None:  # pragma: no cover - successful reads always capture it
            raise RuntimeError("External private corpus identity was not captured.")
        authority = _ExternalPrivateCorpusFilesystemAuthority(
            root_identity=root_identity,
            source_identity=source_identity,
        )
        return raw, document, authority
    except UnicodeDecodeError as exc:
        with ExitStack() as cleanup:
            cleanup.callback(root_identity.close)
            if source_identity is not None:
                cleanup.callback(source_identity.close)
        raise ValueError("Eval corpus JSON must be UTF-8.") from exc
    except BaseException:
        with ExitStack() as cleanup:
            cleanup.callback(root_identity.close)
            if source_identity is not None:
                cleanup.callback(source_identity.close)
        raise


def _validate_external_private_corpus(
    corpus: ExternalPrivateEvalCorpus,
) -> ExternalPrivateEvalCorpus:
    if corpus._filesystem_authority is None:
        raise ValueError("Private corpus lacks filesystem authority from its trusted loader.")
    corpus._filesystem_authority.require_open()
    root = _resolve_private_root(corpus.approved_private_root)
    if root != corpus.approved_private_root:
        raise ValueError("Private corpus root must be canonical.")
    source = corpus.source_path.resolve(strict=True)
    if source != corpus.source_path:
        raise ValueError("Private corpus path must be canonical.")
    _require_below_root(source, root, "corpus path")
    raw, document, authority = _read_private_corpus(
        source,
        root=root,
        expected=corpus._filesystem_authority,
    )
    try:
        source_sha256 = hashlib.sha256(raw).hexdigest()
        if source_sha256 != _fingerprint(corpus.source_sha256, "source_sha256"):
            raise ValueError("Private corpus bytes changed after loading.")
        if document != corpus.document:
            raise ValueError("Private corpus document changed after loading.")
        return ExternalPrivateEvalCorpus(
            document=document,
            source_path=source,
            approved_private_root=root,
            source_sha256=source_sha256,
            _filesystem_authority=authority,
        )
    except BaseException:
        authority.close()
        raise


def _validate_private_destination(
    destination: ExternalPrivateMemoryAblationDestination,
    *,
    require_absent: bool = True,
) -> ExternalPrivateMemoryAblationDestination:
    if destination._filesystem_authority is None:
        raise ValueError("Private destination lacks trusted filesystem authority.")
    destination._filesystem_authority.require_open()
    root = _resolve_private_root(destination.approved_private_root)
    if root != destination.approved_private_root:
        raise ValueError("Private destination root must be canonical.")
    state_directory = destination.state_directory.resolve(strict=True)
    if state_directory != destination.state_directory:
        raise ValueError("Private state directory must be canonical.")
    _require_below_root(state_directory, root, "state_directory")
    if not state_directory.is_dir():
        raise ValueError("state_directory must be an existing directory.")
    artifact_path = destination.artifact_directory
    if not artifact_path.is_absolute():
        raise ValueError("Private artifact directory must be an absolute path.")
    parent = artifact_path.parent.resolve(strict=True)
    directory = parent / artifact_path.name
    if directory != artifact_path:
        raise ValueError("Private artifact directory must be canonical.")
    _require_below_root(directory, root, "artifact_directory")
    authority = _capture_destination_filesystem_authority(
        root=root,
        state_directory=state_directory,
        artifact_parent=parent,
        expected=destination._filesystem_authority,
    )
    try:
        if require_absent:
            try:
                current = directory.stat(follow_symlinks=False)
            except FileNotFoundError:
                current = None
            if current is not None:
                raise ValueError("artifact_directory must not already exist.")
        return ExternalPrivateMemoryAblationDestination(
            state_storage_id=_clean(destination.state_storage_id, "state_storage_id"),
            destination_id=_clean(destination.destination_id, "destination_id"),
            approved_private_root=root,
            state_directory=state_directory,
            artifact_directory=directory,
            _filesystem_authority=authority,
        )
    except BaseException:
        authority.close()
        raise


def load_external_private_memory_ablation_corpus(
    path: str | Path,
    *,
    approved_private_root: str | Path,
) -> ExternalPrivateEvalCorpus:
    """Load one standard bounded Evals document without publishing its private bytes."""

    root, root_identity = _resolve_private_root_with_identity(approved_private_root)
    source_identity: _PrivateFilesystemIdentity | None = None
    try:
        source = Path(path).expanduser().resolve(strict=True)
        _require_below_root(source, root, "corpus path")
        source_stat = source.stat(follow_symlinks=False)
        if _is_windows_reparse_point(source_stat) or not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("External private corpus must be a regular file.")
        source_identity = _capture_private_filesystem_identity(
            source_stat,
            path=source,
            retain_descriptor=True,
        )
        raw, document, authority = _read_private_corpus(
            source,
            root=root,
            expected_root_identity=root_identity,
            expected_source_identity=source_identity,
        )
    finally:
        root_identity.close()
        if source_identity is not None:
            source_identity.close()
    return ExternalPrivateEvalCorpus(
        document=document,
        source_path=source,
        approved_private_root=root,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        _filesystem_authority=authority,
    )


def external_private_memory_ablation_destination(
    *,
    state_storage_id: str,
    destination_id: str,
    approved_private_root: str | Path,
    state_directory: str | Path,
    artifact_directory: str | Path,
) -> ExternalPrivateMemoryAblationDestination:
    """Authorize existing private state and one new private artifact directory."""

    root, root_identity = _resolve_private_root_with_identity(approved_private_root)
    try:
        raw_state_directory = Path(state_directory).expanduser()
        if not raw_state_directory.is_absolute():
            raw_state_directory = root / raw_state_directory
        resolved_state_directory = raw_state_directory.resolve(strict=True)
        _require_below_root(resolved_state_directory, root, "state_directory")
        if not resolved_state_directory.is_dir():
            raise ValueError("state_directory must be an existing directory.")
        raw_directory = Path(artifact_directory).expanduser()
        if not raw_directory.is_absolute():
            raw_directory = root / raw_directory
        parent = raw_directory.parent.resolve(strict=True)
        directory = parent / raw_directory.name
        _require_below_root(directory, root, "artifact_directory")
        if directory.exists() or directory.is_symlink():
            raise ValueError("artifact_directory must not already exist.")
        filesystem_authority = _capture_destination_filesystem_authority(
            root=root,
            state_directory=resolved_state_directory,
            artifact_parent=parent,
            expected_root_identity=root_identity,
        )
    finally:
        root_identity.close()
    try:
        return _validate_private_destination(
            ExternalPrivateMemoryAblationDestination(
                state_storage_id=_clean(state_storage_id, "state_storage_id"),
                destination_id=_clean(destination_id, "destination_id"),
                approved_private_root=root,
                state_directory=resolved_state_directory,
                artifact_directory=directory,
                _filesystem_authority=filesystem_authority,
            )
        )
    finally:
        filesystem_authority.close()


def _schedule_sort_key(
    trial: ExternalPrivateMemoryAblationTrial,
    seed_fingerprint: str,
) -> tuple[str, tuple[str, int, str]]:
    material = {
        "seed_fingerprint": seed_fingerprint,
        "case_id": trial.case_id,
        "repetition": trial.repetition,
        "variant_id": trial.variant_id,
        "execution_id": trial.request.execution_id,
    }
    return (
        hashlib.sha256(
            canonical_durable_json_bytes(material, "memory ablation schedule coordinate")
        ).hexdigest(),
        trial.coordinate,
    )


def _realize_schedule(
    trials: tuple[ExternalPrivateMemoryAblationTrial, ...],
    policy: ExternalPrivateMemoryAblationSchedulePolicy,
    variant_ids: tuple[str, ...],
) -> tuple[ExternalPrivateMemoryAblationTrial, ...]:
    ordered = tuple(sorted(trials, key=lambda item: item.coordinate))
    if policy.strategy is ExternalPrivateMemoryAblationScheduleStrategy.FIXED:
        return ordered
    seed = policy.seed_fingerprint
    if seed is None:  # guarded by the policy model
        raise RuntimeError("Seeded schedule lost its seed fingerprint.")
    if policy.strategy is ExternalPrivateMemoryAblationScheduleStrategy.DETERMINISTIC_RANDOMIZED:
        return tuple(sorted(ordered, key=lambda item: _schedule_sort_key(item, seed)))

    grouped: dict[tuple[str, int], dict[str, ExternalPrivateMemoryAblationTrial]] = {}
    for trial in ordered:
        grouped.setdefault((trial.case_id, trial.repetition), {})[trial.variant_id] = trial
    result: list[ExternalPrivateMemoryAblationTrial] = []
    starting_offset = int.from_bytes(
        hashlib.sha256(seed.encode("ascii")).digest()[:8],
        "big",
    )
    for group_index, coordinate in enumerate(sorted(grouped)):
        group = grouped[coordinate]
        offset = (starting_offset + group_index) % len(variant_ids)
        rotated = (*variant_ids[offset:], *variant_ids[:offset])
        result.extend(group[variant_id] for variant_id in rotated)
    return tuple(result)


def _pricing_fingerprint(corpus: EvalCorpusDocument) -> str | None:
    return None if corpus.pricing_profile is None else corpus.pricing_profile.fingerprint


def _case_assertion_revisions(case: EvalCaseSpec) -> Mapping[str, str]:
    return {assertion.id: assertion_spec_revision(assertion) for assertion in case.assertions}


def _validate_private_assertion_projections(
    case: EvalCaseSpec,
    *,
    execution_mode: ExternalPrivateMemoryAblationExecutionMode,
) -> None:
    unsupported = tuple(
        assertion.kind
        for assertion in case.assertions
        if assertion.kind not in _PRIVATE_SAFE_PORTABLE_ASSERTION_KINDS
    )
    if unsupported:
        raise ValueError(
            "Private corpus uses assertion projections not approved for private reports: "
            + ", ".join(unsupported)
            + "."
        )
    if any(
        type(assertion) is StructuredModelJudgeAssertionSpec
        and type(assertion.reference) is not PrivateJudgeReferenceV1
        for assertion in case.assertions
    ):
        raise ValueError(
            "Private structured-judge assertions require a private reference so generated "
            "explanations cannot enter the portable report."
        )
    if execution_mode is ExternalPrivateMemoryAblationExecutionMode.LIVE and any(
        type(assertion) is StructuredModelJudgeAssertionSpec for assertion in case.assertions
    ):
        raise ValueError(
            "Live external private campaigns cannot use structured model judges until judge "
            "provider work participates in the campaign's hard total-cost authorization."
        )


def _app_agent_has_child_session_tools(app: Any, agent_name: str) -> bool:
    """Conservatively detect registered tools that can dispatch child sessions."""

    registered = app._get_registered_agent(agent_name)
    return any(tool.child_session_recovery is not None for tool in registered.tools.values())


def _require_public_runtime_identity(
    target: CorpusTarget,
    *,
    executor_authority: MemoryInterventionExecutorAuthority,
    trials: tuple[ExternalPrivateMemoryAblationTrial, ...],
) -> None:
    material = {
        "executor_authority": executor_authority.model_dump(mode="json"),
        "trials": [
            {
                "execution_id": trial.request.execution_id,
                "trial_id": trial.request.trial_id,
                "session_id": trial.request.session_id,
                "causal_budget_id": trial.request.causal_budget_id,
            }
            for trial in trials
        ],
    }
    try:
        redacted = target.app.redact_json(material)
    except Exception as exc:
        raise ValueError(
            "Runtime report identities could not cross the application redaction boundary."
        ) from exc
    if redacted != material:
        raise ValueError("Runtime report identities contain workload-secret material.")


def external_private_memory_ablation_experiment_revision(
    experiment: MemoryExperimentReportRequest,
) -> str:
    """Return the immutable pre-execution experiment contract revision."""

    if type(experiment) is not MemoryExperimentReportRequest:
        raise TypeError("experiment must be an exact MemoryExperimentReportRequest.")
    copied = MemoryExperimentReportRequest.model_validate(experiment.model_dump(mode="json"))
    if copied.trials or copied.published_results:
        raise ValueError("An experiment contract cannot contain execution evidence.")
    return _content_revision(
        copied.model_dump(mode="json"),
        "external private memory ablation experiment contract",
    )


def _canonical_executor_state_files(
    paths: MemoryInterventionExecutorStatePaths,
    *,
    destination: ExternalPrivateMemoryAblationDestination,
) -> tuple[dict[str, tuple[str, ...]], str]:
    if type(paths) is not MemoryInterventionExecutorStatePaths:
        raise TypeError("Executor returned invalid durable state-path evidence.")
    groups = {
        "snapshot_store": paths.snapshot_store,
        "execution_store": paths.execution_store,
        "runtime_session_store": paths.runtime_session_store,
        "runtime_budget_ledger": paths.runtime_budget_ledger,
    }
    relative: dict[str, tuple[str, ...]] = {}
    revision_groups: dict[str, list[dict[str, object]]] = {}
    authority = destination._filesystem_authority
    if authority is None:
        raise ValueError("Private destination lacks trusted filesystem authority.")
    authority.require_open()
    state_directory = destination.state_directory
    with _open_private_directory(
        state_directory,
        field_name="state_directory",
        expected=authority.state_identity,
        retain_identity=False,
    ) as (state_descriptor, _state_identity):
        for group_name, group_paths in groups.items():
            if not group_paths:
                raise ValueError(
                    f"External private execution requires local durable {group_name} files."
                )
            canonical_group: list[str] = []
            revision_group: list[dict[str, object]] = []
            for raw_path in group_paths:
                if not isinstance(raw_path, Path) or not raw_path.is_absolute():
                    raise ValueError(f"{group_name} paths must be absolute Path values.")
                path_relative = _safe_relative_path(
                    raw_path,
                    state_directory,
                    f"{group_name} path",
                )
                if state_descriptor is None:
                    parent = raw_path.parent.resolve(strict=True)
                    path = parent / raw_path.name
                    if path != raw_path:
                        raise ValueError(f"{group_name} paths must be canonical non-symlink files.")
                    guarded_parent_identity = _capture_parent(parent)
                    parent_identity = _PrivateFilesystemIdentity.from_guarded_identity(
                        guarded_parent_identity
                    )
                    with _pinned_parent(
                        parent,
                        expected=guarded_parent_identity,
                    ) as pinned_parent:
                        if pinned_parent.entry_stat(path.name) is None:
                            raise ValueError(f"{group_name} paths must identify existing files.")
                        descriptor = _open_private_windows_corpus(path)
                        try:
                            observed = os.fstat(descriptor)
                            if _is_windows_reparse_point(observed) or not stat.S_ISREG(
                                observed.st_mode
                            ):
                                raise ValueError(f"{group_name} paths must identify regular files.")
                            file_identity = _capture_private_filesystem_identity(
                                observed,
                                descriptor=descriptor,
                            )
                            current = pinned_parent.entry_stat(path.name)
                            if current is None or not file_identity.matches(current):
                                raise ValueError(
                                    f"{group_name} paths changed while being authorized."
                                )
                        finally:
                            os.close(descriptor)
                else:
                    if len(path_relative.parts) > 1:
                        parent_context = _open_relative_private_entry(
                            state_descriptor,
                            Path(*path_relative.parts[:-1]),
                            directory=True,
                        )
                    else:
                        parent_context = nullcontext(state_descriptor)
                    with parent_context as parent_descriptor:
                        parent_stat = os.fstat(parent_descriptor)
                        parent_identity = _capture_private_filesystem_identity(
                            parent_stat,
                            descriptor=parent_descriptor,
                        )
                        flags = (
                            os.O_RDONLY
                            | os.O_NOFOLLOW
                            | getattr(os, "O_CLOEXEC", 0)
                            | getattr(os, "O_NONBLOCK", 0)
                        )
                        try:
                            descriptor = os.open(
                                path_relative.parts[-1],
                                flags,
                                dir_fd=parent_descriptor,
                            )
                        except FileNotFoundError as exc:
                            raise ValueError(
                                f"{group_name} paths must identify existing files."
                            ) from exc
                        try:
                            observed = os.fstat(descriptor)
                            if _is_windows_reparse_point(observed) or not stat.S_ISREG(
                                observed.st_mode
                            ):
                                raise ValueError(f"{group_name} paths must identify regular files.")
                            file_identity = _capture_private_filesystem_identity(
                                observed,
                                descriptor=descriptor,
                            )
                            current = os.stat(
                                path_relative.parts[-1],
                                dir_fd=parent_descriptor,
                                follow_symlinks=False,
                            )
                            if not file_identity.matches(current):
                                raise ValueError(
                                    f"{group_name} paths changed while being authorized."
                                )
                        finally:
                            os.close(descriptor)
                    path = state_directory / path_relative
                canonical_group.append(path_relative.as_posix())
                revision_group.append(
                    {
                        "path": path_relative.as_posix(),
                        "parent_identity": _filesystem_identity_revision_material(parent_identity),
                        "file_identity": _filesystem_identity_revision_material(file_identity),
                    }
                )
            values = tuple(sorted(canonical_group))
            if len(values) != len(set(values)):
                raise ValueError(f"{group_name} paths must be unique.")
            relative[group_name] = values
            revision_groups[group_name] = sorted(
                revision_group,
                key=lambda item: cast("str", item["path"]),
            )
    revision = _content_revision(
        revision_groups,
        "external private memory ablation state files",
    )
    return relative, revision


def _external_private_executor_binding(
    executor: MemoryInterventionExecutor,
    *,
    authorization: ExternalPrivateMemoryAblationAuthorization,
    experiment: MemoryExperimentReportRequest,
    destination: ExternalPrivateMemoryAblationDestination,
) -> tuple[MemoryInterventionExecutorAuthority, str, str]:
    if type(executor) is not MemoryInterventionExecutor:
        raise TypeError("executor must be an exact MemoryInterventionExecutor.")
    authority = MemoryInterventionExecutorAuthority.model_validate(
        executor.execution_authority.model_dump(mode="json")
    )
    if authority.evaluator_fingerprint != authorization.evaluator_fingerprint:
        raise ValueError("Executor evaluator differs from the campaign authorization.")
    if (
        authority.provider_configuration_fingerprint
        != authorization.provider_configuration_fingerprint
    ):
        raise ValueError("Executor provider configuration differs from the campaign authorization.")
    expected_provider_execution_mode = MemoryInterventionProviderExecutionMode(
        authorization.execution_mode.value
    )
    if authority.provider_execution_mode is not expected_provider_execution_mode:
        raise ValueError(
            "Executor provider execution mode differs from the campaign authorization."
        )
    runtime_profiles = []
    for variant in experiment.variants:
        required = executor.runtime_runner.required_execution_profile_fingerprint(variant.spec)
        if required != variant.execution_profile_binding.runtime_execution_profile.fingerprint:
            raise ValueError(
                "Executor runtime profile differs from the authorized experiment variant."
            )
        runtime_profiles.append(
            {
                "variant_id": variant.variant_id,
                "spec_fingerprint": variant.spec.fingerprint,
                "runtime_execution_profile_fingerprint": required,
            }
        )
    _relative_files, state_files_revision = _canonical_executor_state_files(
        executor.durable_state_paths(),
        destination=destination,
    )
    binding_revision = _content_revision(
        {
            "executor_authority": authority.model_dump(mode="json"),
            "runtime_profiles": runtime_profiles,
            "state_storage_id": authorization.state_storage_id,
            "state_files_revision": state_files_revision,
        },
        "external private memory ablation executor binding",
    )
    return authority, binding_revision, state_files_revision


def _bind_compiled_trial_budget(
    trial: ExternalPrivateMemoryAblationTrial,
    compiled_request: RunRequest,
) -> ExternalPrivateMemoryAblationTrial:
    """Derive only an explicitly templated causal key, retaining exact inputs."""

    limits = []
    for limit in compiled_request.budget_limits:
        if limit.key == EXTERNAL_PRIVATE_MEMORY_ABLATION_TRIAL_BUDGET_KEY:
            if limit.scope != "causal":
                raise ValueError("The private trial budget template requires causal scope.")
            limit = limit.model_copy(update={"key": trial.request.causal_budget_id})
        limits.append(limit)
    expected = compiled_request.model_copy(update={"budget_limits": tuple(limits)})
    actual_material = trial.request.run_request.model_dump(mode="json")
    if actual_material not in (
        compiled_request.model_dump(mode="json"),
        expected.model_dump(mode="json"),
    ):
        raise ValueError("Trial request differs from the compiled private corpus input.")
    # Accept the original template or an exact previously resolved request.
    # Preflight, the journal, and recovery all retain the resolved form.
    return ExternalPrivateMemoryAblationTrial(
        case_id=trial.case_id,
        repetition=trial.repetition,
        variant_id=trial.variant_id,
        request=trial.request.model_copy(update={"run_request": expected}),
    )


def _matching_cost_ceiling(
    request: MemoryInterventionTrialRequest,
    authorization: ExternalPrivateMemoryAblationAuthorization,
) -> Decimal | None:
    currency = authorization.cost_currency
    pricing_fingerprint = authorization.pricing_profile_fingerprint
    candidates = tuple(
        limit
        for limit in request.run_request.budget_limits
        if limit.scope == "causal"
        and limit.key == request.causal_budget_id
        and limit.action == "interrupt"
        and not limit.allow_unpriced
        and limit.reservation is not None
        and limit.window.kind == "all_time"
        and currency is not None
        and limit.currency == currency
        and pricing_fingerprint is not None
        and pricing_profile_identity(limit.pricing).fingerprint == pricing_fingerprint
    )
    if not candidates:
        return None
    return min(item.max_estimated_cost for item in candidates)


def _matching_token_ceiling(
    request: MemoryInterventionTrialRequest,
    authorization: ExternalPrivateMemoryAblationAuthorization,
) -> int | None:
    authorized = authorization.maximum_total_tokens_per_trial
    configured = request.run_request.limits.max_total_tokens
    if (
        authorized is None
        or request.run_request.limits.scope != "session"
        or configured is None
        or configured > authorized
    ):
        return None
    return configured


def _cost_ceilings_within_authorization(
    ceilings: Iterable[Decimal],
    authorized: Decimal,
) -> bool:
    """Compare an aggregate cost ceiling without ambient Decimal rounding."""

    _require_bounded_cost_decimal(authorized, "maximum_estimated_cost_total")
    maximum = Fraction(authorized)
    total = Fraction()
    for index, ceiling in enumerate(ceilings):
        _require_bounded_cost_decimal(ceiling, f"trial cost ceiling {index + 1}")
        total += Fraction(ceiling)
        if total > maximum:
            return False
    return True


def _require_bounded_cost_decimal(value: Decimal, field_name: str) -> None:
    """Reject compact Decimal exponents that would expand into unbounded exact integers."""

    if type(value) is not Decimal or not value.is_finite() or value <= 0:
        raise ValueError(f"{field_name} must be a positive finite decimal.")
    _, raw_digits, raw_exponent = value.as_tuple()
    if not isinstance(raw_exponent, int):
        raise ValueError(f"{field_name} must be a finite decimal.")
    significant_digits = len(raw_digits)
    exponent = raw_exponent
    while significant_digits and raw_digits[significant_digits - 1] == 0:
        significant_digits -= 1
        exponent += 1
    integer_digits = significant_digits + max(exponent, 0)
    if (
        max(significant_digits, integer_digits) > _COST_DECIMAL_MAX_DIGITS
        or max(-exponent, 0) > _COST_DECIMAL_MAX_PLACES
    ):
        raise ValueError(
            f"{field_name} exceeds the bounded {_COST_DECIMAL_MAX_DIGITS}-digit, "
            f"{_COST_DECIMAL_MAX_PLACES}-decimal-place cost domain."
        )


def _copy_bounded_trials(
    trials: Sequence[ExternalPrivateMemoryAblationTrial],
    *,
    maximum: int,
) -> tuple[ExternalPrivateMemoryAblationTrial, ...]:
    """Copy at most the authorized number even if a Sequence mutates or misreports length."""

    if len(trials) > maximum:
        raise ValueError("Expanded experiment exceeds the authorized trial ceiling.")
    copied: list[ExternalPrivateMemoryAblationTrial] = []
    for index, item in enumerate(trials):
        if index >= maximum:
            raise ValueError("Expanded experiment exceeds the authorized trial ceiling.")
        if type(item) is not ExternalPrivateMemoryAblationTrial:
            raise TypeError("trials must contain exact ExternalPrivateMemoryAblationTrial values.")
        copied.append(
            ExternalPrivateMemoryAblationTrial(
                case_id=item.case_id,
                repetition=item.repetition,
                variant_id=item.variant_id,
                request=item.request,
            )
        )
    return tuple(copied)


def _trial_request_revision(trial: ExternalPrivateMemoryAblationTrial) -> str:
    return _content_revision(
        trial.request.identity_material(),
        "external private memory ablation trial input",
    )


def _report_capacity_reservation_bytes(
    *,
    missing_report: MemoryExperimentReport,
    cases: tuple[EvalCaseSpec, ...],
    repetitions: int,
    variant_count: int,
    trial_count: int,
    target_identity: EvaluationTargetIdentity,
    report_evidence_bytes_per_trial: int,
    supplemental_bytes_per_trial: int | None,
) -> int:
    """Return a conservative bound for request evidence and its derived report graph."""

    structured_criteria = sum(
        len(assertion.rubric.criteria)
        for case in cases
        for assertion in case.assertions
        if type(assertion) is StructuredModelJudgeAssertionSpec
    )
    explanation_bytes = (
        structured_criteria
        * repetitions
        * variant_count
        * EVAL_CORPUS_MAX_JUDGE_EXPLANATION_CHARS
        * 4
    )
    target_bytes = compact_json_utf8_size(target_identity.model_dump(mode="json"))
    reserved = (
        len(memory_experiment_report_to_json(missing_report).encode("utf-8"))
        + trial_count * (report_evidence_bytes_per_trial + _REPORT_FIXED_EVIDENCE_BYTES_PER_TRIAL)
        + explanation_bytes
        + variant_count * (target_bytes + _REPORT_PUBLISHED_CONTAINER_BYTES_PER_VARIANT)
    )
    if supplemental_bytes_per_trial is not None:
        # A baseline accounting side is serialized in its row and in both the
        # case-level and experiment-level report for every candidate comparison.
        accounting_copies = 1 + (2 * (variant_count - 1) * _REPORT_ACCOUNTING_PROJECTION_MULTIPLIER)
        reserved += trial_count * supplemental_bytes_per_trial * accounting_copies
    return reserved


def _validate_report_evidence_size(
    outcome: MemoryInterventionTrialOutcome,
    *,
    maximum_bytes: int,
    omit_binding: bool = False,
) -> None:
    evidence = {
        "execution": outcome.execution.model_dump(mode="json"),
        "binding": (
            None
            if omit_binding or outcome.binding is None
            else outcome.binding.model_dump(mode="json")
        ),
        "eval_result": (
            None
            if outcome.eval_result is None
            else outcome.eval_result.model_dump(mode="json", exclude={"trajectory"})
        ),
    }
    if compact_json_utf8_size(evidence) > maximum_bytes:
        raise _ExternalPrivateReportEvidenceTooLarge(
            "Trial report evidence exceeds its authorized byte ceiling."
        )


def _validate_report_binding_redaction(
    outcome: MemoryInterventionTrialOutcome,
    *,
    prepared: PreparedExternalPrivateMemoryAblation,
) -> None:
    binding = outcome.binding
    if binding is None:
        return
    public = binding.model_dump(mode="json")
    try:
        redacted = prepared.target.app.redact_json(public)
    except Exception as exc:
        raise _ExternalPrivateReportEvidenceInvalid(
            "Trial binding could not cross the application redaction boundary."
        ) from exc
    if redacted != public:
        raise _ExternalPrivateReportEvidenceInvalid(
            "Trial binding contains workload-secret material."
        )


def _validate_supplemental_evidence_size(
    evidence: ExternalPrivateMemoryAblationSupplementalEvidence,
    *,
    maximum_bytes: int,
) -> None:
    size = compact_json_utf8_size(evidence.model_dump(mode="json"))
    if size > maximum_bytes:
        raise ValueError("Supplemental evidence exceeds its authorized per-trial byte ceiling.")
    side = evidence.accounting_side
    if side is None:
        return
    projection = (
        compare_paired_cost_quality(
            PairedCostQualityComparisonRequest(
                pairs=(
                    PairedCostQualityPair(
                        pair_id="external-private-capacity-projection",
                        baseline=side,
                        candidate=None,
                    ),
                )
            )
        )
        .pairs[0]
        .baseline
    )
    if projection is None or compact_json_utf8_size(projection.model_dump(mode="json")) > (
        maximum_bytes * _REPORT_ACCOUNTING_PROJECTION_MULTIPLIER
    ):
        raise ValueError("Supplemental accounting projection exceeds its authorized byte bound.")


async def _validate_supplemental_evidence_authority(
    evidence: ExternalPrivateMemoryAblationSupplementalEvidence,
    *,
    variant: MemoryExperimentVariant,
    experiment_id: str,
    case_revision: str,
    repetition: int,
    executor: MemoryInterventionExecutor,
    outcome: MemoryInterventionTrialOutcome,
    admitted_accounting_attempt_ids: set[str],
) -> tuple[str, ...]:
    side = evidence.accounting_side
    _validate_accounting_side_authority(
        side,
        variant=variant,
        experiment_id=experiment_id,
        case_revision=case_revision,
        repetition=repetition,
    )
    if side is None:
        return ()
    attempt_ids = tuple(attempt.attempt_id for attempt in side.attempts)
    if len(attempt_ids) != len(set(attempt_ids)) or not admitted_accounting_attempt_ids.isdisjoint(
        attempt_ids
    ):
        raise ValueError("Accounting attempt identities must belong to exactly one trial row.")
    session_ids = tuple(attempt.session_id for attempt in side.attempts)
    if not await executor.runtime_runner.accounting_sessions_belong_to_trial(
        outcome.execution,
        session_ids,
    ):
        raise ValueError("Accounting attempts do not belong to the trial's causal session tree.")
    return attempt_ids


def prepare_external_private_memory_ablation(
    *,
    corpus: ExternalPrivateEvalCorpus,
    target: CorpusTarget,
    snapshot: AgentSnapshot,
    experiment: MemoryExperimentReportRequest,
    trials: Sequence[ExternalPrivateMemoryAblationTrial],
    executor: MemoryInterventionExecutor,
    authorization: ExternalPrivateMemoryAblationAuthorization,
    schedule_policy: ExternalPrivateMemoryAblationSchedulePolicy,
    destination: ExternalPrivateMemoryAblationDestination,
    now: datetime | None = None,
) -> PreparedExternalPrivateMemoryAblation:
    """Resolve the complete private campaign authority before provider work."""

    if type(corpus) is not ExternalPrivateEvalCorpus:
        raise TypeError("corpus must come from load_external_private_memory_ablation_corpus().")
    if type(target) is not CorpusTarget:
        raise TypeError("target must be an exact CorpusTarget.")
    if type(snapshot) is not AgentSnapshot:
        raise TypeError("snapshot must be an exact AgentSnapshot.")
    if type(experiment) is not MemoryExperimentReportRequest:
        raise TypeError("experiment must be an exact MemoryExperimentReportRequest.")
    if type(executor) is not MemoryInterventionExecutor:
        raise TypeError("executor must be an exact MemoryInterventionExecutor.")
    if type(authorization) is not ExternalPrivateMemoryAblationAuthorization:
        raise TypeError(
            "authorization must be an exact ExternalPrivateMemoryAblationAuthorization."
        )
    if type(schedule_policy) is not ExternalPrivateMemoryAblationSchedulePolicy:
        raise TypeError("schedule_policy must be an exact schedule policy.")
    if type(destination) is not ExternalPrivateMemoryAblationDestination:
        raise TypeError("destination must be an exact private destination.")
    if not isinstance(trials, Sequence) or isinstance(trials, str | bytes):
        raise TypeError("trials must be an ordered sequence.")
    copied_authorization = ExternalPrivateMemoryAblationAuthorization.model_validate(
        authorization.model_dump(mode="json")
    )
    copied_trials = _copy_bounded_trials(
        trials,
        maximum=copied_authorization.maximum_total_trials,
    )
    copied_snapshot = AgentSnapshot.model_validate(snapshot.model_dump(mode="json"))
    copied_experiment = MemoryExperimentReportRequest.model_validate(
        experiment.model_dump(mode="json")
    )
    copied_policy = ExternalPrivateMemoryAblationSchedulePolicy.model_validate(
        schedule_policy.model_dump(mode="json")
    )
    copied_corpus = _validate_external_private_corpus(corpus)
    try:
        copied_destination = _validate_private_destination(destination)
    except BaseException:
        copied_corpus.close()
        raise
    try:
        return _prepare_external_private_memory_ablation_validated(
            copied_corpus=copied_corpus,
            target=target,
            copied_snapshot=copied_snapshot,
            copied_experiment=copied_experiment,
            copied_trials=copied_trials,
            executor=executor,
            copied_authorization=copied_authorization,
            copied_policy=copied_policy,
            copied_destination=copied_destination,
            now=now,
        )
    except BaseException:
        with ExitStack() as cleanup:
            cleanup.callback(copied_corpus.close)
            cleanup.callback(copied_destination.close)
        raise


def _prepare_external_private_memory_ablation_validated(
    *,
    copied_corpus: ExternalPrivateEvalCorpus,
    target: CorpusTarget,
    copied_snapshot: AgentSnapshot,
    copied_experiment: MemoryExperimentReportRequest,
    copied_trials: tuple[ExternalPrivateMemoryAblationTrial, ...],
    executor: MemoryInterventionExecutor,
    copied_authorization: ExternalPrivateMemoryAblationAuthorization,
    copied_policy: ExternalPrivateMemoryAblationSchedulePolicy,
    copied_destination: ExternalPrivateMemoryAblationDestination,
    now: datetime | None,
) -> PreparedExternalPrivateMemoryAblation:
    """Complete preflight while the caller owns both validated filesystem leases."""

    observed_at = _utc(datetime.now(UTC) if now is None else now, "now")
    if not copied_authorization.valid_from <= observed_at <= copied_authorization.valid_through:
        raise ValueError("The external private campaign authorization is not currently valid.")
    if copied_corpus.approved_private_root != copied_destination.approved_private_root:
        raise ValueError("Corpus and report destination require the same approved private root.")
    if copied_destination.state_storage_id != copied_authorization.state_storage_id:
        raise ValueError("Private state storage differs from the authorized identity.")
    if copied_destination.destination_id != copied_authorization.report_destination_id:
        raise ValueError("Report destination differs from the authorized destination identity.")
    if copied_destination.fingerprint != copied_authorization.report_destination_fingerprint:
        raise ValueError("Private destination paths differ from the campaign authorization.")
    if copied_experiment.trials or copied_experiment.published_results:
        raise ValueError(
            "A preflight experiment cannot contain trial or published-result evidence."
        )
    if external_private_memory_ablation_experiment_revision(copied_experiment) != (
        copied_authorization.experiment_revision
    ):
        raise ValueError("Experiment gates or selection contract differ from authorization.")
    if copied_policy.revision != copied_authorization.schedule_policy_revision:
        raise ValueError("Schedule policy differs from the campaign authorization.")

    document = copied_corpus.document
    if document.revision != copied_authorization.corpus_revision:
        raise ValueError("Private corpus revision differs from the campaign authorization.")
    compiled = compile_corpus_suite(document, target, copied_authorization.suite_id)
    target_identity = evaluation_target_identity(target)
    target_authority = (
        target_identity.target_key,
        target_identity.application_release_id,
        target_identity.app_manifest_fingerprint,
        target.evidence_policy.revision,
    )
    expected_target_authority = (
        copied_authorization.target_key,
        copied_authorization.application_release_id,
        copied_authorization.app_manifest_fingerprint,
        copied_authorization.evidence_policy_revision,
    )
    if target_authority != expected_target_authority:
        raise ValueError("Trusted target identity differs from the campaign authorization.")
    if copied_snapshot.fingerprint != copied_authorization.snapshot_fingerprint:
        raise ValueError("AgentSnapshot differs from the campaign authorization.")
    if (
        copied_snapshot.evaluator is None
        or copied_snapshot.evaluator.identity.fingerprint
        != copied_authorization.evaluator_fingerprint
    ):
        raise ValueError("AgentSnapshot evaluator differs from the campaign authorization.")
    if _pricing_fingerprint(document) != copied_authorization.pricing_profile_fingerprint:
        raise ValueError("Corpus pricing identity differs from the campaign authorization.")

    suite_cases = tuple(case for case in document.cases if case.suite_id == compiled.suite.id)
    expected_cases = tuple(
        MemoryExperimentCase(case_id=case.id, case_revision=case.revision) for case in suite_cases
    )
    if copied_experiment.cases != expected_cases:
        raise ValueError("Experiment cases differ from the exact private corpus suite.")
    if copied_experiment.repetitions != compiled.trials:
        raise ValueError("Experiment repetitions differ from the private corpus suite.")
    case_count = len(suite_cases)
    if not copied_authorization.minimum_cases <= case_count <= copied_authorization.maximum_cases:
        raise ValueError("Private corpus case count is outside the authorized range.")
    if not (
        copied_authorization.minimum_repetitions
        <= copied_experiment.repetitions
        <= copied_authorization.maximum_repetitions
    ):
        raise ValueError("Experiment repetitions are outside the authorized range.")

    variants = copied_experiment.variants
    variant_ids = tuple(variant.variant_id for variant in variants)
    variant_kinds = tuple(sorted({variant.spec.kind for variant in variants}, key=str))
    if variant_ids != copied_authorization.allowed_variant_ids:
        raise ValueError("Experiment variants differ from the authorized variant ids.")
    if variant_kinds != copied_authorization.allowed_variant_kinds:
        raise ValueError("Experiment intervention kinds differ from the authorization.")
    snapshot_memory = copied_snapshot.memory_state
    if snapshot_memory is None:
        raise ValueError("Memory ablation requires an AgentSnapshot with memory state.")
    expected_intervention_frontier = (
        copied_snapshot.fingerprint,
        snapshot_memory.fingerprint,
        copied_snapshot.execution_profile.fingerprint,
        copied_snapshot.authority_scope_fingerprint,
    )
    if any(
        (
            variant.spec.snapshot_fingerprint,
            variant.spec.memory_state_fingerprint,
            variant.spec.execution_profile_fingerprint,
            variant.spec.authority_scope_fingerprint,
        )
        != expected_intervention_frontier
        for variant in variants
    ):
        raise ValueError("Every intervention must start from the exact authorized AgentSnapshot.")
    if any(
        variant.evaluator_fingerprint != copied_authorization.evaluator_fingerprint
        for variant in variants
    ):
        raise ValueError("Experiment evaluator identity differs from the authorization.")
    expected_target_material = _eval_execution_target_material_identity(target)
    for variant in variants:
        profile = variant.execution_profile
        if (
            profile.target_key != copied_authorization.target_key
            or profile.application_release_id != copied_authorization.application_release_id
            or profile.app_manifest_fingerprint != copied_authorization.app_manifest_fingerprint
            or profile.target_material != expected_target_material
            or profile.candidate.provider_name != copied_authorization.provider_name
            or profile.candidate.model != copied_authorization.model
            or profile.evidence_policy.revision != copied_authorization.evidence_policy_revision
        ):
            raise ValueError(
                "Experiment execution profile differs from authorized target identity."
            )
    for case in suite_cases:
        source = case.source
        if source is None or (
            source.application_release_id != copied_authorization.application_release_id
            or source.app_manifest_fingerprint != copied_authorization.app_manifest_fingerprint
        ):
            raise ValueError("Private corpus case source differs from the authorized application.")
        _validate_private_assertion_projections(
            case,
            execution_mode=copied_authorization.execution_mode,
        )
        assertions = _case_assertion_revisions(case)
        if any(
            assertions.get(binding.assertion_id) != binding.assertion_revision
            for binding in copied_experiment.metric_bindings
        ):
            raise ValueError("Metric bindings are not present identically in every private case.")

    executor_authority, executor_binding_revision, state_files_revision = (
        _external_private_executor_binding(
            executor,
            authorization=copied_authorization,
            experiment=copied_experiment,
            destination=copied_destination,
        )
    )
    _require_public_runtime_identity(
        target,
        executor_authority=executor_authority,
        trials=copied_trials,
    )

    expected_coordinates = tuple(
        (case.case_id, repetition, variant.variant_id)
        for case in copied_experiment.cases
        for repetition in range(1, copied_experiment.repetitions + 1)
        for variant in variants
    )
    observed_coordinates = tuple(item.coordinate for item in copied_trials)
    if tuple(sorted(observed_coordinates)) != expected_coordinates or len(
        observed_coordinates
    ) != len(set(observed_coordinates)):
        raise ValueError("Trials must cover the exact experiment matrix once.")
    case_specs = {case.id: case for case in suite_cases}
    compiled_cases = {case.id: case for case in compiled.suite.cases}
    variant_by_id = {variant.variant_id: variant for variant in variants}
    execution_ids: set[str] = set()
    trial_ids: set[str] = set()
    cost_ceilings: list[Decimal] = []
    trial_agent_names = {trial.request.run_request.agent_name for trial in copied_trials}
    live_child_session_agents = (
        {
            agent_name
            for agent_name in trial_agent_names
            if _app_agent_has_child_session_tools(target.app, agent_name)
        }
        if copied_authorization.execution_mode is ExternalPrivateMemoryAblationExecutionMode.LIVE
        else set()
    )
    if live_child_session_agents:
        raise ValueError(
            "Live external private campaigns cannot expose child-session tools until "
            "total-token authority is causal across the complete trial tree."
        )
    bound_trials: list[ExternalPrivateMemoryAblationTrial] = []
    for trial in copied_trials:
        case_spec = case_specs[trial.case_id]
        compiled_case = compiled_cases[trial.case_id]
        variant = variant_by_id[trial.variant_id]
        request = trial.request
        if (
            request.case.case_id != case_spec.id
            or request.case.case_revision != case_spec.revision
            or request.spec != variant.spec
            or request.candidate_id != variant.candidate_id
        ):
            raise ValueError("Trial request conflicts with its experiment matrix coordinate.")
        trial = _bind_compiled_trial_budget(trial, compiled_case.request)
        request = trial.request
        if request.timeout_seconds != compiled.timeout_seconds:
            raise ValueError("Trial timeout differs from the immutable suite timeout.")
        if request.timeout_seconds > copied_authorization.maximum_timeout_seconds:
            raise ValueError("Trial timeout exceeds the authorized ceiling.")
        if request.run_request.max_steps > copied_authorization.maximum_model_steps:
            raise ValueError("Trial model-step limit exceeds the authorized ceiling.")
        if (
            copied_authorization.maximum_total_tokens_per_trial is not None
            and _matching_token_ceiling(request, copied_authorization) is None
        ):
            raise ValueError(
                "Trial requires a session-cumulative total-token limit within authorization."
            )
        cost_ceiling = _matching_cost_ceiling(request, copied_authorization)
        cost_budget_required = (
            copied_authorization.execution_mode is ExternalPrivateMemoryAblationExecutionMode.LIVE
            or copied_authorization.maximum_estimated_cost_total is not None
        )
        if cost_budget_required and cost_ceiling is None:
            raise ValueError("Trial lacks a fail-closed budget for the authorized total cost cap.")
        if cost_ceiling is not None:
            cost_ceilings.append(cost_ceiling)
        if request.execution_id in execution_ids or request.trial_id in trial_ids:
            raise ValueError("Trial execution and trial ids must be unique across the matrix.")
        execution_ids.add(request.execution_id)
        trial_ids.add(request.trial_id)
        bound_trials.append(trial)
    authorized_cost = copied_authorization.maximum_estimated_cost_total
    if authorized_cost is not None and not _cost_ceilings_within_authorization(
        cost_ceilings,
        authorized_cost,
    ):
        raise ValueError("Summed live trial cost ceilings exceed total authorization.")
    if copied_authorization.execution_mode is ExternalPrivateMemoryAblationExecutionMode.LIVE and (
        not copied_experiment.gates.require_priced_cost
        or copied_experiment.gates.maximum_candidate_cost is None
        or copied_experiment.gates.cost_currency != copied_authorization.cost_currency
    ):
        raise ValueError("Live campaigns require a matching fail-closed priced-cost report gate.")
    try:
        missing_report = build_memory_experiment_report(copied_experiment)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "The complete missing-row matrix cannot fit the bounded memory report."
        ) from exc
    missing_public = missing_report.model_dump(mode="json")
    try:
        redacted_missing_public = target.app.redact_json(missing_public)
    except Exception as exc:
        raise ValueError(
            "The preflight report could not cross the application redaction boundary."
        ) from exc
    if redacted_missing_public != missing_public:
        raise ValueError("The preflight report contains workload-secret material.")
    reserved_report_bytes = _report_capacity_reservation_bytes(
        missing_report=missing_report,
        cases=suite_cases,
        repetitions=copied_experiment.repetitions,
        variant_count=len(variants),
        trial_count=len(copied_trials),
        target_identity=target_identity,
        report_evidence_bytes_per_trial=(
            copied_authorization.maximum_report_evidence_bytes_per_trial
        ),
        supplemental_bytes_per_trial=(
            copied_authorization.maximum_supplemental_evidence_bytes_per_trial
        ),
    )
    if reserved_report_bytes > MEMORY_EXPERIMENT_REPORT_MAX_BYTES:
        raise ValueError("The authorized campaign evidence cannot fit the bounded memory report.")

    scheduled_trials = _realize_schedule(tuple(bound_trials), copied_policy, variant_ids)
    case_revisions = {case.case_id: case.case_revision for case in copied_experiment.cases}
    schedule = tuple(
        ExternalPrivateMemoryAblationScheduleEntry(
            ordinal=index,
            case_id=trial.case_id,
            case_revision=case_revisions[trial.case_id],
            repetition=trial.repetition,
            variant_id=trial.variant_id,
            execution_id=trial.request.execution_id,
        )
        for index, trial in enumerate(scheduled_trials, start=1)
    )
    preflight_material = {
        "authorization_fingerprint": copied_authorization.fingerprint,
        "corpus_source_sha256": copied_corpus.source_sha256,
        "experiment": copied_experiment.model_dump(mode="json"),
        "schedule_policy": copied_policy.model_dump(mode="json"),
        "schedule": [item.model_dump(mode="json") for item in schedule],
        "report_destination_id": copied_destination.destination_id,
        "report_destination_fingerprint": copied_destination.fingerprint,
        "state_storage_id": copied_destination.state_storage_id,
        "executor_binding_revision": executor_binding_revision,
        "state_files_revision": state_files_revision,
    }
    prepared = PreparedExternalPrivateMemoryAblation(
        corpus=copied_corpus,
        target=target,
        target_identity=EvaluationTargetIdentity.model_validate(
            target_identity.model_dump(mode="json")
        ),
        snapshot=copied_snapshot,
        experiment=copied_experiment,
        authorization=copied_authorization,
        schedule_policy=copied_policy,
        destination=copied_destination,
        scheduled_trials=scheduled_trials,
        schedule=schedule,
        executor_authority=executor_authority,
        executor_binding_revision=executor_binding_revision,
        state_files_revision=state_files_revision,
        preflight_revision=_content_revision(
            preflight_material,
            "external private memory ablation preflight",
        ),
        _authority=_PREPARED_AUTHORITY,
        _executor=executor,
        _trial_request_revisions=tuple(
            _trial_request_revision(trial) for trial in scheduled_trials
        ),
    )
    try:
        missing_methodology = _methodology(
            prepared,
            missing_report,
            {},
            {},
            started_at=observed_at,
            completed_at=observed_at,
            failure_code=(
                ExternalPrivateMemoryAblationRunFailureCode.REQUIRED_EVIDENCE_UNAVAILABLE
            ),
            failure_ordinal=len(prepared.schedule),
            additional_limitations=frozenset(
                {
                    ExternalPrivateMemoryAblationLimitation.REPORT_EVIDENCE_FAILED,
                }
            ),
            omitted_binding_coordinates=frozenset(
                {
                    (
                        prepared.schedule[0].case_id,
                        prepared.schedule[0].repetition,
                        prepared.schedule[0].variant_id,
                    )
                }
            ),
        )
        _content_free_redaction_failure_result(
            prepared,
            started_at=observed_at,
            completed_at=observed_at,
            failure_ordinal=1,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "The complete trial schedule cannot cross the bounded content-free methodology "
            "boundary."
        ) from exc
    if (
        reserved_report_bytes
        + compact_json_utf8_size(missing_methodology.model_dump(mode="json"))
        + _ARTIFACT_COMPLETION_MAX_BYTES
        > EXTERNAL_PRIVATE_MEMORY_ABLATION_MAX_ARTIFACT_BYTES
    ):
        raise ValueError("The authorized campaign cannot fit its complete artifact tree.")
    return prepared


class ExternalPrivateMemoryAblationCacheEvidenceState(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ExternalPrivateMemoryAblationRunFailureCode(StrEnum):
    AUTHORIZATION_EXPIRED = "authorization_expired"
    EXECUTION_AUTHORITY_CHANGED = "execution_authority_changed"
    EXECUTION_FAILED = "execution_failed"
    EVIDENCE_COLLECTION_FAILED = "evidence_collection_failed"
    REPORT_EVIDENCE_FAILED = "report_evidence_failed"
    REQUIRED_EVIDENCE_UNAVAILABLE = "required_evidence_unavailable"
    TARGET_IDENTITY_CHANGED = "target_identity_changed"


class _ExternalPrivateExecutionAuthorityChanged(Exception):
    pass


class _ExternalPrivateEvaluatorEvidenceInvalid(Exception):
    pass


class _ExternalPrivateReportEvidenceTooLarge(ValueError):
    pass


class _ExternalPrivateReportEvidenceInvalid(ValueError):
    pass


class _ExternalPrivateReportRedactionFailed(ValueError):
    pass


class ExternalPrivateMemoryAblationLimitation(StrEnum):
    ACCOUNTING_EVIDENCE_UNAVAILABLE = "accounting_evidence_unavailable"
    CACHE_EVIDENCE_UNAVAILABLE = "cache_evidence_unavailable"
    INCOMPLETE_TRIAL_MATRIX = "incomplete_trial_matrix"
    MEMORY_OVERHEAD_UNAVAILABLE = "memory_overhead_unavailable"
    REPORT_EVIDENCE_FAILED = "report_evidence_failed"
    REPORT_EVIDENCE_OMITTED = "report_evidence_omitted"
    RUNNER_STOPPED = "runner_stopped"


class ExternalPrivateMemoryAblationSupplementalEvidence(BaseModel):
    """Application-projected operational evidence; never raw provider or prompt data."""

    model_config = _MODEL_CONFIG

    accounting_side: PairedCostQualitySide | None = None
    memory_overhead: MemoryPreparationOverheadEvidence | None = None
    cache_evidence_state: ExternalPrivateMemoryAblationCacheEvidenceState
    cache_read_tokens: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    cache_write_tokens: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    provider_retry_count: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)

    @field_validator(
        "cache_read_tokens",
        "cache_write_tokens",
        "provider_retry_count",
        mode="before",
    )
    @classmethod
    def validate_integer_fields(cls, value: object, info) -> object:
        if value is not None and type(value) is not int:
            raise ValueError(f"{info.field_name} must be a JSON integer.")
        return value

    @field_validator("accounting_side", mode="before")
    @classmethod
    def copy_accounting_side(cls, value: object) -> object:
        return revalidate_model_input(value, PairedCostQualitySide)

    @field_validator("memory_overhead", mode="before")
    @classmethod
    def copy_memory_overhead(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryPreparationOverheadEvidence)

    @model_validator(mode="after")
    def validate_cache_evidence(self) -> Self:
        tokens = (self.cache_read_tokens, self.cache_write_tokens)
        if self.cache_evidence_state is ExternalPrivateMemoryAblationCacheEvidenceState.AVAILABLE:
            if any(value is None for value in tokens):
                raise ValueError("Available cache evidence requires read and write token counts.")
        elif any(value is not None for value in tokens):
            raise ValueError("Unavailable cache evidence cannot carry token counts.")
        return self


class ExternalPrivateMemoryAblationEvidenceCollector(ABC):
    """Application adapter for bounded cost, cache, retry, and overhead evidence."""

    collector_fingerprint: str

    @abstractmethod
    async def collect(
        self,
        *,
        trial: ExternalPrivateMemoryAblationTrial,
        outcome: MemoryInterventionTrialOutcome,
    ) -> ExternalPrivateMemoryAblationSupplementalEvidence:
        """Project safe operational evidence for one terminal trial."""


class ExternalPrivateMemoryAblationTrialMethodology(BaseModel):
    model_config = _MODEL_CONFIG

    ordinal: StrictInt = Field(ge=1, le=100_000)
    case_id: StrictStr = Field(max_length=128)
    case_revision: StrictStr = Field(min_length=71, max_length=71)
    repetition: StrictInt = Field(ge=1, le=1_000)
    variant_id: StrictStr = Field(max_length=128)
    execution_id: StrictStr = Field(min_length=64, max_length=64)
    execution_status: MemoryInterventionExecutionStatus | None = None
    availability: MemoryTrialAvailability
    cache_evidence_state: ExternalPrivateMemoryAblationCacheEvidenceState
    cache_read_tokens: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    cache_write_tokens: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    provider_retry_count: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    accounting_evidence_available: StrictBool
    memory_overhead_available: StrictBool

    @field_validator(
        "ordinal",
        "repetition",
        "cache_read_tokens",
        "cache_write_tokens",
        "provider_retry_count",
        mode="before",
    )
    @classmethod
    def validate_integer_fields(cls, value: object, info) -> object:
        if value is not None and type(value) is not int:
            raise ValueError(f"{info.field_name} must be a JSON integer.")
        return value

    @field_validator("case_id", "variant_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=128)

    @field_validator("case_revision")
    @classmethod
    def validate_case_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)

    @field_validator("execution_id")
    @classmethod
    def validate_execution_id(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        available = self.cache_evidence_state is (
            ExternalPrivateMemoryAblationCacheEvidenceState.AVAILABLE
        )
        if available != (
            self.cache_read_tokens is not None and self.cache_write_tokens is not None
        ):
            raise ValueError("Cache methodology evidence is incomplete.")
        if (self.execution_status is None) != (
            self.availability is MemoryTrialAvailability.MISSING
        ):
            raise ValueError("Missing methodology rows cannot claim an execution status.")
        return self


class ExternalPrivateMemoryAblationMethodology(BaseModel):
    """Bounded, content-free methodology sidecar for one private campaign report."""

    model_config = _MODEL_CONFIG

    record_type: Literal["cayu.external-private-memory-ablation-methodology"] = (
        "cayu.external-private-memory-ablation-methodology"
    )
    schema_version: Literal[1] = EXTERNAL_PRIVATE_MEMORY_ABLATION_SCHEMA_VERSION
    revision: StrictStr = Field(min_length=71, max_length=71)
    preflight_revision: StrictStr = Field(min_length=71, max_length=71)
    authorization_id: StrictStr = Field(max_length=256)
    authorization_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    corpus_origin: Literal["external_private"] = "external_private"
    corpus_revision: StrictStr = Field(min_length=71, max_length=71)
    corpus_source_sha256: StrictStr = Field(min_length=64, max_length=64)
    suite_id: StrictStr = Field(max_length=128)
    experiment_id: StrictStr = Field(max_length=128)
    experiment_revision: StrictStr = Field(min_length=71, max_length=71)
    schedule_policy_revision: StrictStr = Field(min_length=71, max_length=71)
    target_key: StrictStr = Field(max_length=128)
    application_release_id: StrictStr = Field(max_length=256)
    app_manifest_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    snapshot_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    provider_name: StrictStr = Field(max_length=256)
    model: StrictStr = Field(max_length=256)
    provider_configuration_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    executor_authority_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    executor_binding_revision: StrictStr = Field(min_length=71, max_length=71)
    evidence_collector_fingerprint: StrictStr | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )
    evidence_policy_revision: StrictStr = Field(min_length=71, max_length=71)
    pricing_profile_fingerprint: StrictStr | None = Field(
        default=None,
        min_length=71,
        max_length=71,
    )
    redaction_policy_revision: StrictStr = Field(min_length=71, max_length=71)
    retention_policy_revision: StrictStr = Field(min_length=71, max_length=71)
    state_storage_id: StrictStr = Field(max_length=256)
    state_files_revision: StrictStr = Field(min_length=71, max_length=71)
    report_destination_id: StrictStr = Field(max_length=256)
    report_destination_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    execution_mode: ExternalPrivateMemoryAblationExecutionMode
    live_execution_authorization_id: StrictStr | None = Field(default=None, max_length=256)
    schedule_policy: ExternalPrivateMemoryAblationSchedulePolicy
    started_at: datetime
    completed_at: datetime
    status: ExternalPrivateMemoryAblationRunStatus
    run_failure_code: ExternalPrivateMemoryAblationRunFailureCode | None = None
    failure_ordinal: StrictInt | None = Field(default=None, ge=1, le=100_000)
    report_revision: StrictStr = Field(min_length=71, max_length=71)
    trials: tuple[ExternalPrivateMemoryAblationTrialMethodology, ...] = Field(
        min_length=2,
        max_length=100_000,
    )
    limitations: tuple[ExternalPrivateMemoryAblationLimitation, ...] = Field(
        default=(),
        max_length=EXTERNAL_PRIVATE_MEMORY_ABLATION_MAX_LIMITATIONS,
    )

    @field_validator("schema_version", "failure_ordinal", mode="before")
    @classmethod
    def validate_integer_fields(cls, value: object, info) -> object:
        if value is not None and type(value) is not int:
            raise ValueError(f"{info.field_name} must be a JSON integer.")
        return value

    @field_validator(
        "authorization_id",
        "suite_id",
        "experiment_id",
        "target_key",
        "application_release_id",
        "provider_name",
        "model",
        "state_storage_id",
        "report_destination_id",
        "live_execution_authorization_id",
    )
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _clean(value, info.field_name)

    @field_validator(
        "authorization_fingerprint",
        "corpus_source_sha256",
        "app_manifest_fingerprint",
        "snapshot_fingerprint",
        "provider_configuration_fingerprint",
        "executor_authority_fingerprint",
        "evidence_collector_fingerprint",
        "report_destination_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str | None, info) -> str | None:
        return None if value is None else _fingerprint(value, info.field_name)

    @field_validator(
        "revision",
        "preflight_revision",
        "corpus_revision",
        "experiment_revision",
        "schedule_policy_revision",
        "evidence_policy_revision",
        "pricing_profile_fingerprint",
        "redaction_policy_revision",
        "retention_policy_revision",
        "executor_binding_revision",
        "state_files_revision",
        "report_revision",
    )
    @classmethod
    def validate_revisions(cls, value: str | None, info) -> str | None:
        return None if value is None else _revision(value, info.field_name)

    @field_validator("started_at", "completed_at")
    @classmethod
    def validate_datetimes(cls, value: datetime, info) -> datetime:
        return _utc(value, info.field_name)

    @field_validator("schedule_policy", mode="before")
    @classmethod
    def copy_schedule_policy(cls, value: object) -> object:
        return revalidate_model_input(value, ExternalPrivateMemoryAblationSchedulePolicy)

    @field_validator("limitations", mode="before")
    @classmethod
    def validate_limitations(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise TypeError("limitations must be an ordered array.")
        result = tuple(ExternalPrivateMemoryAblationLimitation(item) for item in value)
        if result != tuple(sorted(set(result), key=str)):
            raise ValueError("limitations must be unique and sorted.")
        return result

    @model_validator(mode="after")
    def validate_methodology(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at.")
        failed = self.run_failure_code is not None
        if failed != (self.failure_ordinal is not None):
            raise ValueError("Run failure code and ordinal must be present together.")
        if (self.status is ExternalPrivateMemoryAblationRunStatus.COMPLETE) == failed:
            raise ValueError("Only incomplete methodology may carry a run failure.")
        expected_ordinals = tuple(range(1, len(self.trials) + 1))
        if tuple(item.ordinal for item in self.trials) != expected_ordinals:
            raise ValueError("Methodology trials must retain the exact realized order.")
        if self.failure_ordinal is not None and self.failure_ordinal > len(self.trials):
            raise ValueError("failure_ordinal must identify a retained methodology trial.")
        if self.status is ExternalPrivateMemoryAblationRunStatus.COMPLETE and any(
            item.availability is MemoryTrialAvailability.MISSING for item in self.trials
        ):
            raise ValueError("Complete methodology cannot contain an unstarted trial.")
        material = self.model_dump(mode="json", exclude={"revision"})
        if self.revision != _content_revision(
            material,
            "external private memory ablation methodology",
        ):
            raise ValueError("Methodology revision does not match its content.")
        if compact_json_utf8_size(self.model_dump(mode="json")) > (
            MEMORY_EXPERIMENT_REPORT_MAX_BYTES
        ):
            raise ValueError("External private methodology exceeds its byte bound.")
        return self

    @classmethod
    def create(cls, **values: Any) -> ExternalPrivateMemoryAblationMethodology:
        material = cls.model_construct(
            None,
            revision="sha256:" + "0" * 64,
            **values,
        ).model_dump(mode="json", exclude={"revision"})
        return cls(
            revision=_content_revision(
                material,
                "external private memory ablation methodology",
            ),
            **values,
        )


@dataclass(frozen=True, slots=True, repr=False)
class ExternalPrivateMemoryAblationResult:
    report: MemoryExperimentReport
    methodology: ExternalPrivateMemoryAblationMethodology

    def __post_init__(self) -> None:
        if type(self.report) is not MemoryExperimentReport:
            raise TypeError("report must be an exact MemoryExperimentReport.")
        if type(self.methodology) is not ExternalPrivateMemoryAblationMethodology:
            raise TypeError("methodology must be an exact external private methodology.")
        report = memory_experiment_report_from_json(memory_experiment_report_to_json(self.report))
        methodology = ExternalPrivateMemoryAblationMethodology.model_validate_json(
            external_private_memory_ablation_methodology_to_json(self.methodology)
        )
        if methodology.report_revision != report.revision:
            raise ValueError("Methodology does not describe the supplied memory report.")
        object.__setattr__(self, "report", report)
        object.__setattr__(self, "methodology", methodology)


@dataclass(frozen=True, slots=True, repr=False)
class ExternalPrivateMemoryAblationArtifactPaths:
    artifact_directory: Path
    report_path: Path
    methodology_path: Path
    completion_path: Path


def _diagnostic(status: EvalStatus) -> EvalTrialDiagnosticCode:
    return {
        EvalStatus.PASSED: EvalTrialDiagnosticCode.PASSED,
        EvalStatus.FAILED: EvalTrialDiagnosticCode.ASSERTION_FAILED,
        EvalStatus.UNAVAILABLE: EvalTrialDiagnosticCode.ASSERTION_EVIDENCE_UNAVAILABLE,
        EvalStatus.ERROR: EvalTrialDiagnosticCode.EXECUTION_FAILED,
    }[status]


def _unavailable_trial_result(
    case: EvalCaseSpec,
    repetition: int,
    *,
    session_id: str,
    started_at: datetime,
    completed_at: datetime,
) -> EvalTrialResult:
    reason = "Portable evaluation evidence was not produced for this terminal trial."
    return EvalTrialResult(
        trial_number=repetition,
        status=EvalStatus.UNAVAILABLE,
        session_id=session_id,
        score=None,
        assertions=tuple(
            EvalAssertionResult(
                name=assertion.id,
                assertion_revision=assertion_spec_revision(assertion),
                outcome=EvalOutcome.UNAVAILABLE,
                message=reason,
            )
            for assertion in case.assertions
        ),
        unavailable_reason=reason,
        evidence_complete=False,
        events_count=0,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=max(0, int((completed_at - started_at).total_seconds() * 1_000)),
    )


def _publication_times(
    trials: Sequence[EvalTrialResult],
) -> tuple[datetime, datetime, int]:
    started_at = min(item.started_at for item in trials)
    completed_at = max(item.completed_at for item in trials)
    return (
        started_at,
        completed_at,
        max(0, int((completed_at - started_at).total_seconds() * 1_000)),
    )


def _publish_variant(
    *,
    corpus: EvalCorpusDocument,
    target_identity: EvaluationTargetIdentity,
    suite_id: str,
    experiment_id: str,
    variant_id: str,
    trial_results: Mapping[str, tuple[EvalTrialResult, ...]],
) -> CorpusExecutionResult:
    suite = next(item for item in corpus.suites if item.id == suite_id)
    cases = tuple(case for case in corpus.cases if case.suite_id == suite_id)
    trial_policy = eval_suite_trial_policy(suite)
    case_results = []
    for case in cases:
        trials = trial_results[case.id]
        started_at, completed_at, _ = _publication_times(trials)
        case_results.append(
            EvalCaseResult.from_trials(
                case_id=case.id,
                trials=trials,
                started_at=started_at,
                completed_at=completed_at,
                trial_policy=trial_policy,
            )
        )
    all_trials = tuple(trial for trials in trial_results.values() for trial in trials)
    started_at, completed_at, duration_ms = _publication_times(all_trials)
    run = EvalRun(
        run_id=f"{experiment_id}-{variant_id}",
        suite_id=suite.id,
        status=aggregate_eval_status(item.status for item in case_results),
        score=aggregate_eval_score(item.score for item in case_results),
        cases=tuple(case_results),
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        run_contract=eval_run_contract_for_corpus(corpus, suite.id),
    )
    public_data = {
        case.id: tuple(
            _EvalTrialPublicData(
                diagnostic_code=_diagnostic(trial.status),
                output=(
                    EvalTrialOutputPreviewV1.from_retained_evidence(
                        _PRIVATE_OUTPUT_PLACEHOLDER,
                        "complete",
                        max_preview_bytes=len(_PRIVATE_OUTPUT_PLACEHOLDER.encode("utf-8")),
                    )
                    if trial.status in {EvalStatus.PASSED, EvalStatus.FAILED}
                    else EvalTrialOutputPreviewV1.unavailable()
                ),
            )
            for trial in trial_results[case.id]
        )
        for case in cases
    }
    return CorpusExecutionResult.create(
        target=target_identity,
        run=_publish_eval_run_with_trial_public_data(
            corpus,
            run,
            trial_public_data_by_case=public_data,
        ),
    )


def _validate_outcome(
    trial: ExternalPrivateMemoryAblationTrial,
    outcome: MemoryInterventionTrialOutcome,
    case: EvalCaseSpec,
) -> None:
    execution = outcome.execution
    request = trial.request
    if (
        execution.execution_id != request.execution_id
        or execution.trial_id != request.trial_id
        or execution.case_id != trial.case_id
        or execution.case_revision != request.case.case_revision
        or execution.candidate_id != request.candidate_id
        or execution.spec_fingerprint != request.spec.fingerprint
    ):
        raise ValueError("Executor returned evidence for another private trial.")
    if execution.status is MemoryInterventionExecutionStatus.ACTIVE:
        raise ValueError("External private campaign executor returned an active trial.")
    result = outcome.eval_result
    if result is not None and (
        result.trial_number != trial.repetition or result.session_id != execution.session_id
    ):
        raise ValueError("Evaluator result conflicts with its scheduled repetition or session.")
    if result is not None and result.status is EvalStatus.SKIPPED:
        raise ValueError("External private campaign trials cannot be skipped.")
    if result is None:
        return
    if len(result.assertions) != len(case.assertions):
        raise _ExternalPrivateEvaluatorEvidenceInvalid(
            "Evaluator assertion results do not match the private corpus contract."
        )
    try:
        for assertion_spec, assertion_result in zip(
            case.assertions,
            result.assertions,
            strict=True,
        ):
            _published_assertion(assertion_spec, assertion_result)
    except Exception as exc:
        raise _ExternalPrivateEvaluatorEvidenceInvalid(
            "Evaluator assertion results do not match the private corpus contract."
        ) from exc


def _without_eval_result(
    outcome: MemoryInterventionTrialOutcome,
) -> MemoryInterventionTrialOutcome:
    """Retain bounded terminal execution evidence without invalid evaluator output."""

    return MemoryInterventionTrialOutcome(
        execution=outcome.execution,
        receipt=outcome.receipt,
        snapshot_result=outcome.snapshot_result,
        binding=outcome.binding,
    )


def _target_matches_authorization(prepared: PreparedExternalPrivateMemoryAblation) -> bool:
    identity = evaluation_target_identity(prepared.target)
    authorization = prepared.authorization
    return (
        identity.target_key == authorization.target_key
        and identity.application_release_id == authorization.application_release_id
        and identity.app_manifest_fingerprint == authorization.app_manifest_fingerprint
    )


def _executor_matches_prepared(
    prepared: PreparedExternalPrivateMemoryAblation,
    executor: MemoryInterventionExecutor,
) -> bool:
    if executor is not prepared._executor:
        return False
    authority, binding_revision, state_files_revision = _external_private_executor_binding(
        executor,
        authorization=prepared.authorization,
        experiment=prepared.experiment,
        destination=prepared.destination,
    )
    return (
        authority == prepared.executor_authority
        and binding_revision == prepared.executor_binding_revision
        and state_files_revision == prepared.state_files_revision
    )


def _require_executor_matches_prepared(
    prepared: PreparedExternalPrivateMemoryAblation,
    executor: MemoryInterventionExecutor,
) -> None:
    try:
        matches = _executor_matches_prepared(prepared, executor)
    except Exception:
        matches = False
    if not matches:
        raise _ExternalPrivateExecutionAuthorityChanged


def _report_from_outcomes(
    prepared: PreparedExternalPrivateMemoryAblation,
    outcomes: Mapping[tuple[str, int, str], MemoryInterventionTrialOutcome],
    supplemental: Mapping[
        tuple[str, int, str],
        ExternalPrivateMemoryAblationSupplementalEvidence,
    ],
    *,
    omitted_binding_coordinates: frozenset[tuple[str, int, str]] = frozenset(),
) -> MemoryExperimentReport:
    experiment = prepared.experiment
    authorization = prepared.authorization
    cases = tuple(
        case for case in prepared.corpus.document.cases if case.suite_id == authorization.suite_id
    )
    placeholder_time = min(
        (outcome.execution.created_at for outcome in outcomes.values()),
        default=authorization.valid_from,
    )
    results_by_variant: dict[str, dict[str, list[EvalTrialResult]]] = {
        variant.variant_id: {case.id: [] for case in cases} for variant in experiment.variants
    }
    for case in cases:
        for repetition in range(1, experiment.repetitions + 1):
            for variant in experiment.variants:
                key = (case.id, repetition, variant.variant_id)
                outcome = outcomes.get(key)
                if outcome is None:
                    trial_result = _unavailable_trial_result(
                        case,
                        repetition,
                        session_id=f"unavailable-{case.id}-{repetition}-{variant.variant_id}",
                        started_at=placeholder_time,
                        completed_at=placeholder_time,
                    )
                elif outcome.eval_result is not None:
                    trial_result = outcome.eval_result
                else:
                    trial_result = _unavailable_trial_result(
                        case,
                        repetition,
                        session_id=outcome.execution.session_id,
                        started_at=outcome.execution.created_at,
                        completed_at=outcome.execution.updated_at,
                    )
                results_by_variant[variant.variant_id][case.id].append(trial_result)

    published_by_variant: dict[str, CorpusExecutionResult] = {}
    for variant in experiment.variants:
        variant_id = variant.variant_id
        if not any(
            outcomes.get((case.id, repetition, variant_id)) is not None
            and outcomes[(case.id, repetition, variant_id)].eval_result is not None
            for case in cases
            for repetition in range(1, experiment.repetitions + 1)
        ):
            continue
        published_by_variant[variant_id] = _publish_variant(
            corpus=prepared.corpus.document,
            target_identity=prepared.target_identity,
            suite_id=authorization.suite_id,
            experiment_id=experiment.experiment_id,
            variant_id=variant_id,
            trial_results={
                case_id: tuple(values) for case_id, values in results_by_variant[variant_id].items()
            },
        )

    published_results = tuple(
        sorted(
            (
                MemoryPublishedResultEvidence(
                    run_id=f"{experiment.experiment_id}-{variant_id}",
                    result=result.model_dump(mode="json"),
                )
                for variant_id, result in published_by_variant.items()
            ),
            key=lambda item: item.result.revision,
        )
    )
    trial_evidence = []
    for case in cases:
        for repetition in range(1, experiment.repetitions + 1):
            for variant in experiment.variants:
                key = (case.id, repetition, variant.variant_id)
                outcome = outcomes.get(key)
                if outcome is None:
                    continue
                binding_omitted = key in omitted_binding_coordinates
                operational = supplemental.get(key)
                published = published_by_variant.get(variant.variant_id)
                trial_evidence.append(
                    MemoryExperimentTrialEvidence(
                        case_id=case.id,
                        case_revision=case.revision,
                        repetition=repetition,
                        variant_id=variant.variant_id,
                        execution=outcome.execution,
                        intervention_binding=(None if binding_omitted else outcome.binding),
                        intervention_binding_omitted=binding_omitted,
                        published_result_revision=(
                            published.revision
                            if outcome.eval_result is not None and published is not None
                            else None
                        ),
                        accounting_side=(
                            None if operational is None else operational.accounting_side
                        ),
                        memory_overhead=(
                            None if operational is None else operational.memory_overhead
                        ),
                    )
                )
    request_document = {
        "experiment_id": experiment.experiment_id,
        "cases": [item.model_dump(mode="json") for item in experiment.cases],
        "repetitions": experiment.repetitions,
        "baseline_variant_id": experiment.baseline_variant_id,
        "variants": [item.model_dump(mode="json") for item in experiment.variants],
        "metric_bindings": [item.model_dump(mode="json") for item in experiment.metric_bindings],
        "ranking": [item.model_dump(mode="json") for item in experiment.ranking],
        "gates": experiment.gates.model_dump(mode="json"),
        "published_results": [item.model_dump(mode="json") for item in published_results],
        "trials": [item.model_dump(mode="json") for item in trial_evidence],
    }
    request = MemoryExperimentReportRequest.model_validate_json(
        json.dumps(request_document, ensure_ascii=False, separators=(",", ":"))
    )
    report = build_memory_experiment_report(request)
    public = report.model_dump(mode="json")
    try:
        redacted = prepared.target.app.redact_json(public)
    except Exception as exc:
        raise _ExternalPrivateReportRedactionFailed(
            "Private memory-ablation report could not cross the redaction boundary."
        ) from exc
    if redacted != public:
        raise _ExternalPrivateReportRedactionFailed(
            "Private memory-ablation report contains workload-secret material."
        )
    return report


def _methodology(
    prepared: PreparedExternalPrivateMemoryAblation,
    report: MemoryExperimentReport,
    outcomes: Mapping[tuple[str, int, str], MemoryInterventionTrialOutcome],
    supplemental: Mapping[
        tuple[str, int, str],
        ExternalPrivateMemoryAblationSupplementalEvidence,
    ],
    *,
    started_at: datetime,
    completed_at: datetime,
    failure_code: ExternalPrivateMemoryAblationRunFailureCode | None,
    failure_ordinal: int | None,
    additional_limitations: frozenset[ExternalPrivateMemoryAblationLimitation] = frozenset(),
    omitted_binding_coordinates: frozenset[tuple[str, int, str]] = frozenset(),
) -> ExternalPrivateMemoryAblationMethodology:
    rows_by_coordinate = {(row.case_id, row.repetition, row.variant_id): row for row in report.rows}
    trial_rows = []
    limitations = set(additional_limitations)
    for schedule_entry in prepared.schedule:
        key = (
            schedule_entry.case_id,
            schedule_entry.repetition,
            schedule_entry.variant_id,
        )
        outcome = outcomes.get(key)
        evidence = supplemental.get(key)
        row = rows_by_coordinate[key]
        cache_state = (
            ExternalPrivateMemoryAblationCacheEvidenceState.UNAVAILABLE
            if evidence is None
            else evidence.cache_evidence_state
        )
        trial_rows.append(
            ExternalPrivateMemoryAblationTrialMethodology(
                ordinal=schedule_entry.ordinal,
                case_id=schedule_entry.case_id,
                case_revision=schedule_entry.case_revision,
                repetition=schedule_entry.repetition,
                variant_id=schedule_entry.variant_id,
                execution_id=schedule_entry.execution_id,
                execution_status=(None if outcome is None else outcome.execution.status),
                availability=row.availability,
                cache_evidence_state=cache_state,
                cache_read_tokens=(None if evidence is None else evidence.cache_read_tokens),
                cache_write_tokens=(None if evidence is None else evidence.cache_write_tokens),
                provider_retry_count=(None if evidence is None else evidence.provider_retry_count),
                accounting_evidence_available=(
                    evidence is not None and evidence.accounting_side is not None
                ),
                memory_overhead_available=(
                    evidence is not None and evidence.memory_overhead is not None
                ),
            )
        )
        if outcome is None:
            limitations.add(ExternalPrivateMemoryAblationLimitation.INCOMPLETE_TRIAL_MATRIX)
        if cache_state is ExternalPrivateMemoryAblationCacheEvidenceState.UNAVAILABLE:
            limitations.add(ExternalPrivateMemoryAblationLimitation.CACHE_EVIDENCE_UNAVAILABLE)
        if evidence is None or evidence.accounting_side is None:
            limitations.add(ExternalPrivateMemoryAblationLimitation.ACCOUNTING_EVIDENCE_UNAVAILABLE)
        if evidence is None or evidence.memory_overhead is None:
            limitations.add(ExternalPrivateMemoryAblationLimitation.MEMORY_OVERHEAD_UNAVAILABLE)
        if key in omitted_binding_coordinates:
            limitations.add(ExternalPrivateMemoryAblationLimitation.REPORT_EVIDENCE_OMITTED)
    if failure_code is not None:
        limitations.add(ExternalPrivateMemoryAblationLimitation.RUNNER_STOPPED)
    status = (
        ExternalPrivateMemoryAblationRunStatus.COMPLETE
        if len(outcomes) == len(prepared.schedule) and failure_code is None
        else ExternalPrivateMemoryAblationRunStatus.INCOMPLETE
    )
    methodology = ExternalPrivateMemoryAblationMethodology.create(
        preflight_revision=prepared.preflight_revision,
        authorization_id=prepared.authorization.authorization_id,
        authorization_fingerprint=prepared.authorization.fingerprint,
        corpus_revision=prepared.corpus.document.revision,
        corpus_source_sha256=prepared.corpus.source_sha256,
        suite_id=prepared.authorization.suite_id,
        experiment_id=prepared.experiment.experiment_id,
        experiment_revision=prepared.authorization.experiment_revision,
        schedule_policy_revision=prepared.authorization.schedule_policy_revision,
        target_key=prepared.authorization.target_key,
        application_release_id=prepared.authorization.application_release_id,
        app_manifest_fingerprint=prepared.authorization.app_manifest_fingerprint,
        snapshot_fingerprint=prepared.authorization.snapshot_fingerprint,
        provider_name=prepared.authorization.provider_name,
        model=prepared.authorization.model,
        provider_configuration_fingerprint=(
            prepared.authorization.provider_configuration_fingerprint
        ),
        executor_authority_fingerprint=prepared.executor_authority.fingerprint,
        executor_binding_revision=prepared.executor_binding_revision,
        evidence_collector_fingerprint=(prepared.authorization.evidence_collector_fingerprint),
        evidence_policy_revision=prepared.authorization.evidence_policy_revision,
        pricing_profile_fingerprint=prepared.authorization.pricing_profile_fingerprint,
        redaction_policy_revision=prepared.authorization.redaction_policy_revision,
        retention_policy_revision=prepared.authorization.retention_policy_revision,
        state_storage_id=prepared.authorization.state_storage_id,
        state_files_revision=prepared.state_files_revision,
        report_destination_id=prepared.authorization.report_destination_id,
        report_destination_fingerprint=(prepared.authorization.report_destination_fingerprint),
        execution_mode=prepared.authorization.execution_mode,
        live_execution_authorization_id=(prepared.authorization.live_execution_authorization_id),
        schedule_policy=prepared.schedule_policy,
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        run_failure_code=failure_code,
        failure_ordinal=failure_ordinal,
        report_revision=report.revision,
        trials=tuple(trial_rows),
        limitations=tuple(sorted(limitations, key=str)),
    )
    public = methodology.model_dump(mode="json")
    try:
        redacted = prepared.target.app.redact_json(public)
    except Exception as exc:
        raise _ExternalPrivateReportRedactionFailed(
            "Private memory-ablation methodology could not cross the redaction boundary."
        ) from exc
    if redacted != public:
        raise _ExternalPrivateReportRedactionFailed(
            "Private memory-ablation methodology contains workload-secret material."
        )
    return methodology


def _content_free_redaction_failure_result(
    prepared: PreparedExternalPrivateMemoryAblation,
    *,
    started_at: datetime,
    completed_at: datetime,
    failure_ordinal: int,
    failure_code: ExternalPrivateMemoryAblationRunFailureCode = (
        ExternalPrivateMemoryAblationRunFailureCode.REPORT_EVIDENCE_FAILED
    ),
    additional_limitations: frozenset[ExternalPrivateMemoryAblationLimitation] = frozenset(),
) -> ExternalPrivateMemoryAblationResult:
    """Build a content-free result without discarding campaign provenance."""

    report = _report_from_outcomes(prepared, {}, {})
    methodology = _methodology(
        prepared,
        report,
        {},
        {},
        started_at=started_at,
        completed_at=completed_at,
        failure_code=failure_code,
        failure_ordinal=failure_ordinal,
        additional_limitations=additional_limitations,
    )
    return ExternalPrivateMemoryAblationResult(
        report=report,
        methodology=methodology,
    )


def _binding_can_be_omitted(outcome: MemoryInterventionTrialOutcome) -> bool:
    """Return whether the report schema can retain this binding by fingerprint."""

    return (
        outcome.execution.status is MemoryInterventionExecutionStatus.COMPLETED
        and outcome.binding is not None
        and outcome.execution.final_binding_fingerprint is not None
    )


def _portable_terminal_report_outcomes(
    prepared: PreparedExternalPrivateMemoryAblation,
    outcomes: Mapping[tuple[str, int, str], MemoryInterventionTrialOutcome],
) -> tuple[
    dict[tuple[str, int, str], MemoryInterventionTrialOutcome],
    frozenset[tuple[str, int, str]],
]:
    """Retain only terminal evidence that safely fits the portable report schema."""

    retained: dict[tuple[str, int, str], MemoryInterventionTrialOutcome] = {}
    omitted_bindings: set[tuple[str, int, str]] = set()
    for coordinate, outcome in outcomes.items():
        stripped = _without_eval_result(outcome)
        omit_binding = _binding_can_be_omitted(stripped)
        public = {
            "execution": stripped.execution.model_dump(mode="json"),
            "binding": (
                None
                if omit_binding or stripped.binding is None
                else stripped.binding.model_dump(mode="json")
            ),
        }
        try:
            redacted = prepared.target.app.redact_json(public)
        except Exception:
            continue
        if redacted != public:
            continue
        retained[coordinate] = stripped
        if omit_binding:
            omitted_bindings.add(coordinate)
    return retained, frozenset(omitted_bindings)


async def run_external_private_memory_ablation(
    prepared: PreparedExternalPrivateMemoryAblation,
    executor: MemoryInterventionExecutor,
    *,
    evidence_collector: ExternalPrivateMemoryAblationEvidenceCollector | None = None,
    clock: Callable[[], datetime] | None = None,
) -> ExternalPrivateMemoryAblationResult:
    """Execute the exact preflight schedule and retain every terminal or missing row."""

    if type(prepared) is not PreparedExternalPrivateMemoryAblation:
        raise TypeError("prepared must come from prepare_external_private_memory_ablation().")
    if prepared._authority is not _PREPARED_AUTHORITY:
        raise ValueError("prepared must come from prepare_external_private_memory_ablation().")
    if type(executor) is not MemoryInterventionExecutor:
        raise TypeError("executor must be an exact MemoryInterventionExecutor.")
    if clock is not None and not callable(clock):
        raise TypeError("clock must be callable.")
    prepared._require_open()
    now = (lambda: datetime.now(UTC)) if clock is None else clock
    authorization = prepared.authorization
    # Public trial records contain mutable RunRequest values. Bind a detached
    # execution copy before any await so later caller mutations cannot change
    # the input admitted here, including while executor admission is waiting.
    scheduled_trials = _copy_bounded_trials(
        prepared.scheduled_trials,
        maximum=authorization.maximum_total_trials,
    )
    if tuple(_trial_request_revision(trial) for trial in scheduled_trials) != (
        prepared._trial_request_revisions
    ):
        raise ValueError("Trial input differs from the campaign preflight authority.")
    try:
        executor_matches = _executor_matches_prepared(prepared, executor)
    except Exception:
        executor_matches = False
    if not executor_matches:
        raise ValueError("Executor identity or private state differs from preflight authority.")
    started_at = _utc(now(), "clock result")
    if not authorization.valid_from <= started_at <= authorization.valid_through:
        raise ValueError("The external private campaign authorization is not currently valid.")
    expected_collector = authorization.evidence_collector_fingerprint
    if evidence_collector is None:
        actual_collector = None
    elif not isinstance(evidence_collector, ExternalPrivateMemoryAblationEvidenceCollector):
        raise TypeError("evidence_collector must implement the bounded collector contract.")
    else:
        actual_collector = _fingerprint(
            evidence_collector.collector_fingerprint,
            "evidence_collector.collector_fingerprint",
        )
    if actual_collector != expected_collector:
        raise ValueError("Evidence collector differs from the campaign authorization.")
    try:
        _content_free_redaction_failure_result(
            prepared,
            started_at=started_at,
            completed_at=started_at,
            failure_ordinal=1,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "The content-free publication boundary differs from campaign preflight."
        ) from exc

    outcomes: dict[tuple[str, int, str], MemoryInterventionTrialOutcome] = {}
    omitted_binding_coordinates: set[tuple[str, int, str]] = set()
    supplemental: dict[
        tuple[str, int, str],
        ExternalPrivateMemoryAblationSupplementalEvidence,
    ] = {}
    admitted_accounting_attempt_ids: set[str] = set()
    cases_by_id = {
        case.id: case
        for case in prepared.corpus.document.cases
        if case.suite_id == authorization.suite_id
    }
    variants_by_id = {variant.variant_id: variant for variant in prepared.experiment.variants}
    failure_code = None
    failure_ordinal = None
    additional_limitations: set[ExternalPrivateMemoryAblationLimitation] = set()
    for schedule_entry, trial in zip(
        prepared.schedule,
        scheduled_trials,
        strict=True,
    ):
        observed_at = _utc(now(), "clock result")
        if not authorization.valid_from <= observed_at <= authorization.valid_through:
            failure_code = ExternalPrivateMemoryAblationRunFailureCode.AUTHORIZATION_EXPIRED
            failure_ordinal = schedule_entry.ordinal
            break
        try:
            target_matches = _target_matches_authorization(prepared)
        except Exception:
            target_matches = False
        if not target_matches:
            failure_code = ExternalPrivateMemoryAblationRunFailureCode.TARGET_IDENTITY_CHANGED
            failure_ordinal = schedule_entry.ordinal
            break
        try:
            executor_matches = _executor_matches_prepared(prepared, executor)
        except Exception:
            executor_matches = False
        if not executor_matches:
            failure_code = ExternalPrivateMemoryAblationRunFailureCode.EXECUTION_AUTHORITY_CHANGED
            failure_ordinal = schedule_entry.ordinal
            break
        try:
            outcome = await executor.execute_trial(
                trial.request,
                admission_check=lambda: _require_executor_matches_prepared(
                    prepared,
                    executor,
                ),
            )
            _validate_outcome(trial, outcome, cases_by_id[trial.case_id])
            _validate_report_evidence_size(
                outcome,
                maximum_bytes=authorization.maximum_report_evidence_bytes_per_trial,
            )
            _validate_report_binding_redaction(outcome, prepared=prepared)
        except asyncio.CancelledError:
            raise
        except _ExternalPrivateExecutionAuthorityChanged:
            failure_code = ExternalPrivateMemoryAblationRunFailureCode.EXECUTION_AUTHORITY_CHANGED
            failure_ordinal = schedule_entry.ordinal
            break
        except (
            _ExternalPrivateEvaluatorEvidenceInvalid,
            _ExternalPrivateReportEvidenceInvalid,
            _ExternalPrivateReportEvidenceTooLarge,
        ):
            sanitized_outcome = _without_eval_result(outcome)
            try:
                _validate_report_evidence_size(
                    sanitized_outcome,
                    maximum_bytes=authorization.maximum_report_evidence_bytes_per_trial,
                )
                _validate_report_binding_redaction(sanitized_outcome, prepared=prepared)
            except (
                _ExternalPrivateReportEvidenceInvalid,
                _ExternalPrivateReportEvidenceTooLarge,
            ):
                # A completed execution can retain its terminal status and exact
                # binding fingerprint without the unsafe binding body. Other
                # terminal states remain durable in the private execution store,
                # but the portable report schema cannot claim their binding was
                # omitted.
                _validate_report_evidence_size(
                    sanitized_outcome,
                    maximum_bytes=authorization.maximum_report_evidence_bytes_per_trial,
                    omit_binding=True,
                )
                if _binding_can_be_omitted(sanitized_outcome):
                    outcomes[trial.coordinate] = sanitized_outcome
                    omitted_binding_coordinates.add(trial.coordinate)
            else:
                outcomes[trial.coordinate] = sanitized_outcome
            failure_code = ExternalPrivateMemoryAblationRunFailureCode.REPORT_EVIDENCE_FAILED
            failure_ordinal = schedule_entry.ordinal
            break
        except Exception:
            failure_code = ExternalPrivateMemoryAblationRunFailureCode.EXECUTION_FAILED
            failure_ordinal = schedule_entry.ordinal
            break
        outcomes[trial.coordinate] = outcome
        if evidence_collector is not None:
            try:
                evidence = await evidence_collector.collect(trial=trial, outcome=outcome)
                if type(evidence) is not ExternalPrivateMemoryAblationSupplementalEvidence:
                    raise TypeError("Evidence collector returned an unsupported value.")
                evidence = ExternalPrivateMemoryAblationSupplementalEvidence.model_validate(
                    evidence.model_dump(mode="json")
                )
                maximum_evidence_bytes = authorization.maximum_supplemental_evidence_bytes_per_trial
                if maximum_evidence_bytes is None:
                    raise ValueError("Authorized supplemental evidence has no byte ceiling.")
                _validate_supplemental_evidence_size(
                    evidence,
                    maximum_bytes=maximum_evidence_bytes,
                )
                accounting_attempt_ids = await _validate_supplemental_evidence_authority(
                    evidence,
                    variant=variants_by_id[trial.variant_id],
                    experiment_id=prepared.experiment.experiment_id,
                    case_revision=schedule_entry.case_revision,
                    repetition=trial.repetition,
                    executor=executor,
                    outcome=outcome,
                    admitted_accounting_attempt_ids=admitted_accounting_attempt_ids,
                )
                public_evidence = evidence.model_dump(mode="json")
                if prepared.target.app.redact_json(public_evidence) != public_evidence:
                    raise ValueError("Supplemental evidence contains workload-secret material.")
            except asyncio.CancelledError:
                raise
            except Exception:
                failure_code = (
                    ExternalPrivateMemoryAblationRunFailureCode.EVIDENCE_COLLECTION_FAILED
                )
                failure_ordinal = schedule_entry.ordinal
                break
            supplemental[trial.coordinate] = evidence
            admitted_accounting_attempt_ids.update(accounting_attempt_ids)
            required_cache = authorization.cache_evidence_policy is (
                ExternalPrivateMemoryAblationCacheEvidencePolicy.REQUIRED
            )
            required_live_evidence_missing = (
                authorization.execution_mode is ExternalPrivateMemoryAblationExecutionMode.LIVE
                and evidence.accounting_side is None
            )
            if (
                required_cache
                and evidence.cache_evidence_state
                is ExternalPrivateMemoryAblationCacheEvidenceState.UNAVAILABLE
            ) or required_live_evidence_missing:
                failure_code = (
                    ExternalPrivateMemoryAblationRunFailureCode.REQUIRED_EVIDENCE_UNAVAILABLE
                )
                failure_ordinal = schedule_entry.ordinal
                break
        try:
            target_matches = _target_matches_authorization(prepared)
        except Exception:
            target_matches = False
        if not target_matches:
            failure_code = ExternalPrivateMemoryAblationRunFailureCode.TARGET_IDENTITY_CHANGED
            failure_ordinal = schedule_entry.ordinal
            break
        try:
            executor_matches = _executor_matches_prepared(prepared, executor)
        except Exception:
            executor_matches = False
        if not executor_matches:
            failure_code = ExternalPrivateMemoryAblationRunFailureCode.EXECUTION_AUTHORITY_CHANGED
            failure_ordinal = schedule_entry.ordinal
            break

    frozen_omitted_bindings = frozenset(omitted_binding_coordinates)
    try:
        report = _report_from_outcomes(
            prepared,
            outcomes,
            supplemental,
            omitted_binding_coordinates=frozen_omitted_bindings,
        )
    except _ExternalPrivateReportRedactionFailed:
        # A runtime-only value may become unsafe after preflight. Preserve safe
        # terminal journals, omit only bindings the report schema can represent
        # by fingerprint, and discard unsafe public projections. The durable
        # private execution store remains the source of the complete outcome.
        report_failure_ordinal = failure_ordinal or max(
            (
                entry.ordinal
                for entry in prepared.schedule
                if (entry.case_id, entry.repetition, entry.variant_id) in outcomes
            ),
            default=1,
        )
        outcomes, frozen_omitted_bindings = _portable_terminal_report_outcomes(
            prepared,
            outcomes,
        )
        supplemental = {}
        additional_limitations.add(ExternalPrivateMemoryAblationLimitation.REPORT_EVIDENCE_FAILED)
        if failure_code is None:
            failure_code = ExternalPrivateMemoryAblationRunFailureCode.REPORT_EVIDENCE_FAILED
            failure_ordinal = report_failure_ordinal
        try:
            report = _report_from_outcomes(
                prepared,
                outcomes,
                supplemental,
                omitted_binding_coordinates=frozen_omitted_bindings,
            )
        except _ExternalPrivateReportRedactionFailed:
            # Rebuild through the current redaction scope instead of returning
            # a value that was only safe under the pre-dispatch scope.
            completed_at = max(started_at, _utc(now(), "clock result"))
            try:
                return _content_free_redaction_failure_result(
                    prepared,
                    started_at=started_at,
                    completed_at=completed_at,
                    failure_ordinal=report_failure_ordinal,
                    failure_code=failure_code,
                    additional_limitations=frozenset(additional_limitations),
                )
            except _ExternalPrivateReportRedactionFailed as exc:
                raise ValueError(
                    "The content-free publication boundary changed after provider "
                    "dispatch; terminal evidence remains in the private execution store."
                ) from exc
    completed_at = max(started_at, _utc(now(), "clock result"))
    try:
        methodology = _methodology(
            prepared,
            report,
            outcomes,
            supplemental,
            started_at=started_at,
            completed_at=completed_at,
            failure_code=failure_code,
            failure_ordinal=failure_ordinal,
            additional_limitations=frozenset(additional_limitations),
            omitted_binding_coordinates=frozen_omitted_bindings,
        )
    except _ExternalPrivateReportRedactionFailed:
        fallback_failure_ordinal = failure_ordinal or max(
            (
                entry.ordinal
                for entry in prepared.schedule
                if (entry.case_id, entry.repetition, entry.variant_id) in outcomes
            ),
            default=1,
        )
        try:
            additional_limitations.add(
                ExternalPrivateMemoryAblationLimitation.REPORT_EVIDENCE_FAILED
            )
            return _content_free_redaction_failure_result(
                prepared,
                started_at=started_at,
                completed_at=completed_at,
                failure_ordinal=fallback_failure_ordinal,
                failure_code=(
                    failure_code
                    or ExternalPrivateMemoryAblationRunFailureCode.REPORT_EVIDENCE_FAILED
                ),
                additional_limitations=frozenset(additional_limitations),
            )
        except _ExternalPrivateReportRedactionFailed as exc:
            raise ValueError(
                "The content-free publication boundary changed after provider dispatch; "
                "terminal evidence remains in the private execution store."
            ) from exc
    return ExternalPrivateMemoryAblationResult(
        report=report,
        methodology=methodology,
    )


def external_private_memory_ablation_methodology_to_json(
    methodology: ExternalPrivateMemoryAblationMethodology,
) -> str:
    if type(methodology) is not ExternalPrivateMemoryAblationMethodology:
        raise TypeError("methodology must be an exact methodology record.")
    copied = ExternalPrivateMemoryAblationMethodology.model_validate(
        methodology.model_dump(mode="json")
    )
    encoded = json.dumps(
        copied.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded.encode("utf-8")) > MEMORY_EXPERIMENT_REPORT_MAX_BYTES:
        raise ValueError("External private methodology exceeds its byte bound.")
    return encoded


def _write_all_private_artifact_bytes(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written <= 0:
            raise OSError("Private artifact write made no progress.")
        offset += written
    os.fsync(descriptor)


def _write_posix_private_artifact(
    directory_descriptor: int,
    filename: str,
    content: bytes,
) -> tuple[_Identity, int]:
    descriptor = os.open(
        filename,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_descriptor,
    )
    identity: _Identity | None = None
    try:
        identity = _Identity.capture(os.fstat(descriptor))
        os.fchmod(descriptor, 0o600)
        _write_all_private_artifact_bytes(descriptor, content)
        return identity, descriptor
    except BaseException:
        if identity is not None:
            with suppress(OSError):
                current = os.stat(
                    filename,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if identity.matches(os.fstat(descriptor)) and identity.matches(current):
                    os.unlink(filename, dir_fd=directory_descriptor)
        with suppress(OSError):
            os.close(descriptor)
        raise


def _posix_private_artifact_stage_name(destination_name: str) -> str:
    normalized = unicodedata.normalize("NFC", destination_name).casefold()
    return f".cayu-private-stage-{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def _posix_private_artifact_pending_name(filename: str, content: bytes) -> str:
    digest = hashlib.sha256(filename.encode("ascii") + b"\0" + content).hexdigest()
    return f".cayu-private-pending-{digest}"


def _discard_interrupted_posix_private_artifact(
    directory_descriptor: int,
    pending_name: str,
) -> None:
    """Remove only an authenticated regular pending entry from an interrupted write."""

    descriptor = os.open(
        pending_name,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
        dir_fd=directory_descriptor,
    )
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or stat.S_IMODE(observed.st_mode) & ~0o600:
            raise ValueError("Private artifact pending payload has unsafe type or permissions.")
        identity = _Identity.capture(observed)
        current = os.stat(
            pending_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if not identity.matches(current):
            raise ValueError("Private artifact pending payload changed during recovery.")
        os.unlink(pending_name, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
    finally:
        os.close(descriptor)


def _bounded_posix_private_stage_names(
    directory_descriptor: int,
) -> frozenset[str]:
    names: list[str] = []
    with os.scandir(directory_descriptor) as entries:
        for entry in entries:
            names.append(entry.name)
            if len(names) > 3:
                raise ValueError("Private artifact staging directory contains unexpected entries.")
    return frozenset(names)


def _seal_posix_private_artifact_stage(
    directory_descriptor: int,
    contents: Mapping[str, bytes],
) -> None:
    """Verify the exact bounded tree and persist every staged directory entry."""

    root = os.fstat(directory_descriptor)
    if not stat.S_ISDIR(root.st_mode) or stat.S_IMODE(root.st_mode) != 0o700:
        raise ValueError("Private artifact staging directory permissions changed.")
    names = _bounded_posix_private_stage_names(directory_descriptor)
    if set(names) != set(contents):
        raise ValueError("Private artifact staging directory contains unexpected entries.")
    for name, expected in contents.items():
        descriptor = os.open(
            name,
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_descriptor,
        )
        try:
            observed = os.fstat(descriptor)
            if (
                not stat.S_ISREG(observed.st_mode)
                or stat.S_IMODE(observed.st_mode) != 0o600
                or observed.st_size != len(expected)
            ):
                raise ValueError("Private artifact staging payload changed before publication.")
            chunks: list[bytes] = []
            remaining = len(expected) + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 << 10))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            if b"".join(chunks) != expected:
                raise ValueError("Private artifact staging payload changed before publication.")
        finally:
            os.close(descriptor)
    os.fsync(directory_descriptor)


def _private_artifact_publication_fault(_phase: str) -> None:
    """Fault-injection seam for durability-order tests."""


def _rename_posix_private_entry_no_replace(
    parent_descriptor: int,
    source: str,
    destination: str,
) -> None:
    """Atomically publish a descriptor-relative directory without replacement."""

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        function_name = "renameat2"
        arguments = (
            parent_descriptor,
            os.fsencode(source),
            parent_descriptor,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
    elif sys.platform == "darwin":
        function_name = "renameatx_np"
        arguments = (
            parent_descriptor,
            os.fsencode(source),
            parent_descriptor,
            os.fsencode(destination),
            0x00000004,  # RENAME_EXCL
        )
    else:
        raise RuntimeError(
            "External private artifact publication requires atomic no-replace rename support."
        )
    try:
        rename = getattr(libc, function_name)
    except AttributeError as exc:
        raise RuntimeError(
            "External private artifact publication requires atomic no-replace rename support."
        ) from exc
    rename.restype = ctypes.c_int
    if rename(*arguments) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError("Private artifact directory already exists.")
    raise OSError(error_number, os.strerror(error_number), destination)


def _stage_posix_private_artifact(
    directory_descriptor: int,
    filename: str,
    content: bytes,
) -> tuple[_Identity, int]:
    """Write through a pending entry so an interrupted file never blocks recovery."""

    pending_name = _posix_private_artifact_pending_name(filename, content)
    identity, descriptor = _write_posix_private_artifact(
        directory_descriptor,
        pending_name,
        content,
    )
    try:
        _rename_posix_private_entry_no_replace(
            directory_descriptor,
            pending_name,
            filename,
        )
        current = os.stat(
            filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if not identity.matches(os.fstat(descriptor)) or not identity.matches(current):
            raise ValueError("Private artifact payload changed during staging.")
        return identity, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _remove_incomplete_posix_private_stage(
    parent_descriptor: int,
    stage_name: str,
    stage_descriptor: int,
    stage_identity: _Identity,
    owned_entries: Mapping[str, tuple[_Identity, int]],
) -> None:
    """Best-effort removal of only the exact bounded stage this process created."""

    try:
        current_stage = os.stat(
            stage_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        return
    if not stage_identity.matches(current_stage):
        return
    for name, (expected, descriptor) in owned_entries.items():
        try:
            observed = os.stat(name, dir_fd=stage_descriptor, follow_symlinks=False)
            if not expected.matches(os.fstat(descriptor)) or not expected.matches(observed):
                return
            os.unlink(name, dir_fd=stage_descriptor)
        except FileNotFoundError:
            pass
        except OSError:
            return
    try:
        current = os.stat(stage_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stage_identity.matches(current):
            return
        os.rmdir(stage_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    except OSError:
        return


def _reuse_exact_posix_private_artifact_tree(
    parent_descriptor: int,
    destination_name: str,
    *,
    observed: os.stat_result,
    contents: Mapping[str, bytes],
) -> None:
    """Authenticate an exact published tree after an ambiguous prior commit."""

    if not stat.S_ISDIR(observed.st_mode) or stat.S_IMODE(observed.st_mode) != 0o700:
        raise FileExistsError("Private artifact directory already exists.")
    expected = _Identity.capture(observed)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(destination_name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise FileExistsError("Private artifact directory already exists.") from exc
    try:
        if not expected.matches(os.fstat(descriptor)):
            raise FileExistsError("Private artifact directory already exists.")
        try:
            _seal_posix_private_artifact_stage(descriptor, contents)
        except (OSError, ValueError) as exc:
            raise FileExistsError("Private artifact directory already exists.") from exc
        current = os.stat(
            destination_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not expected.matches(current):
            raise FileExistsError("Private artifact directory already exists.")
    finally:
        os.close(descriptor)


def _publish_posix_private_artifact_tree(
    destination: Path,
    *,
    contents: Mapping[str, bytes],
    expected_parent_identity: _PrivateFilesystemIdentity,
) -> None:
    _require_posix_private_artifact_publication_support()
    parent = destination.parent
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        parent_stat = os.fstat(parent_descriptor)
        observed_parent_identity = _capture_private_filesystem_identity(
            parent_stat,
            descriptor=parent_descriptor,
        )
        if observed_parent_identity != expected_parent_identity:
            raise ValueError("Private artifact parent changed after authorization.")
        try:
            existing_destination = os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing_destination = None
        if existing_destination is not None:
            _reuse_exact_posix_private_artifact_tree(
                parent_descriptor,
                destination.name,
                observed=existing_destination,
                contents=contents,
            )
            os.fsync(parent_descriptor)
            return
        stage_name = _posix_private_artifact_stage_name(destination.name)
        created = False
        try:
            os.mkdir(stage_name, mode=0o700, dir_fd=parent_descriptor)
            created = True
        except FileExistsError:
            pass
        created_stage = os.stat(
            stage_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        _require_private_directory(created_stage, "Private artifact staging directory")
        if stat.S_IMODE(created_stage.st_mode) & ~0o700:
            raise ValueError("Private artifact staging directory permissions changed.")
        stage_identity = _Identity.capture(created_stage)
        directory_descriptor = os.open(
            stage_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            opened_stage_matches = stage_identity.matches(os.fstat(directory_descriptor))
        except BaseException:
            os.close(directory_descriptor)
            raise
        if not opened_stage_matches:
            os.close(directory_descriptor)
            raise ValueError("Private artifact staging directory changed during creation.")
        try:
            if created or stat.S_IMODE(os.fstat(directory_descriptor).st_mode) != 0o700:
                os.fchmod(directory_descriptor, 0o700)
            opened_stage_mode = stat.S_IMODE(os.fstat(directory_descriptor).st_mode)
        except BaseException:
            os.close(directory_descriptor)
            raise
        if opened_stage_mode != 0o700:
            os.close(directory_descriptor)
            raise ValueError("Private artifact staging directory permissions changed.")
        published = False
        owned_entries: dict[str, tuple[_Identity, int]] = {}
        try:
            current_stage = os.stat(
                stage_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not stage_identity.matches(current_stage):
                raise ValueError("Private artifact staging directory changed during creation.")
            ordered_names = (
                EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME,
                EXTERNAL_PRIVATE_MEMORY_ABLATION_METHODOLOGY_FILENAME,
                EXTERNAL_PRIVATE_MEMORY_ABLATION_COMPLETION_FILENAME,
            )
            pending_names = {
                name: _posix_private_artifact_pending_name(name, contents[name])
                for name in ordered_names
            }
            existing_names = _bounded_posix_private_stage_names(directory_descriptor)
            prefix_length: int | None = None
            interrupted_pending: str | None = None
            for length in range(len(ordered_names) + 1):
                completed_prefix = frozenset(ordered_names[:length])
                if existing_names == completed_prefix:
                    prefix_length = length
                    break
                if length < len(ordered_names):
                    pending_name = pending_names[ordered_names[length]]
                    if existing_names == completed_prefix | {pending_name}:
                        prefix_length = length
                        interrupted_pending = pending_name
                        break
            if prefix_length is None or (created and prefix_length != 0):
                raise ValueError("Private artifact staging directory contains unexpected entries.")
            if interrupted_pending is not None:
                _discard_interrupted_posix_private_artifact(
                    directory_descriptor,
                    interrupted_pending,
                )
            existing_contents = {name: contents[name] for name in ordered_names[:prefix_length]}
            _seal_posix_private_artifact_stage(
                directory_descriptor,
                existing_contents,
            )
            for name in ordered_names[prefix_length:2]:
                owned_entries[name] = _stage_posix_private_artifact(
                    directory_descriptor,
                    name,
                    contents[name],
                )
                os.fsync(directory_descriptor)
            if prefix_length < 3:
                os.fsync(directory_descriptor)
                _private_artifact_publication_fault("payload_entries_synced")
            if prefix_length < 3:
                name = EXTERNAL_PRIVATE_MEMORY_ABLATION_COMPLETION_FILENAME
                owned_entries[name] = _stage_posix_private_artifact(
                    directory_descriptor,
                    name,
                    contents[name],
                )
                os.fsync(directory_descriptor)
                _private_artifact_publication_fault("completion_entry_synced")
            _seal_posix_private_artifact_stage(directory_descriptor, contents)
            _private_artifact_publication_fault("stage_synced")
            _seal_posix_private_artifact_stage(directory_descriptor, contents)
            current_stage = os.stat(
                stage_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not stage_identity.matches(current_stage):
                raise ValueError("Private artifact staging directory changed before publication.")
            _rename_posix_private_entry_no_replace(
                parent_descriptor,
                stage_name,
                destination.name,
            )
            published = True
            _private_artifact_publication_fault("tree_renamed")
            published = os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not stage_identity.matches(published):
                raise ValueError("Private artifact directory changed during publication.")
            _seal_posix_private_artifact_stage(directory_descriptor, contents)
            os.fsync(parent_descriptor)
            published = os.stat(
                destination.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not stage_identity.matches(published):
                raise ValueError("Private artifact directory changed during publication.")
        except BaseException:
            if not published and created:
                _remove_incomplete_posix_private_stage(
                    parent_descriptor,
                    stage_name,
                    directory_descriptor,
                    stage_identity,
                    owned_entries,
                )
            raise
        finally:
            for _identity, descriptor in owned_entries.values():
                os.close(descriptor)
            os.close(directory_descriptor)
        _require_current_path_identity(
            parent,
            expected=expected_parent_identity,
            field_name="Private artifact parent",
        )
        published = os.stat(
            destination.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stage_identity.matches(published):
            raise ValueError("Private artifact directory changed during publication.")
    finally:
        os.close(parent_descriptor)


def _publish_windows_private_artifact_tree(
    destination: Path,
    *,
    contents: Mapping[str, bytes],
    expected_parent_identity: _PrivateFilesystemIdentity,
    request_digest: str,
) -> None:
    def populate(stage: GuardedTreeStage) -> None:
        observed_parent = _PrivateFilesystemIdentity.from_guarded_identity(
            _capture_parent(destination.parent)
        )
        if observed_parent != expected_parent_identity:
            raise ValueError("Private artifact parent changed after authorization.")
        try:
            destination.stat(follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Private artifact directory already exists.")
        stage.write_tree(
            contents,
            file_mode=0o600,
            directory_mode=0o700,
            root_mode=0o700,
        )
        stage_path = stage._specialized_path()
        with _windows_directory_namespace_fence(stage_path):
            stage.capture_owned_identity()
            _assert_windows_directory_dacl_is_protected(stage_path)
            for name in (
                EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME,
                EXTERNAL_PRIVATE_MEMORY_ABLATION_METHODOLOGY_FILENAME,
            ):
                _sync_windows_path(stage_path / name, directory=False)
            _sync_windows_path(stage_path, directory=True)
            _private_artifact_publication_fault("payload_entries_synced")
            _sync_windows_path(
                stage_path / EXTERNAL_PRIVATE_MEMORY_ABLATION_COMPLETION_FILENAME,
                directory=False,
            )
            _sync_windows_path(stage_path, directory=True)
            _private_artifact_publication_fault("completion_entry_synced")
            stage.capture_owned_identity()
            _assert_windows_directory_dacl_is_protected(stage_path)

    try:
        publish_guarded_tree(
            destination,
            consumer="external-private-memory-ablation",
            request_digest=request_digest,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=populate,
            preserve_windows_destination_dacl=True,
        )
    except GuardedTreePublicationError as exc:
        if exc.code == "destination_not_empty":
            raise FileExistsError("Private artifact directory already exists.") from exc
        raise


def write_external_private_memory_ablation_artifacts(
    result: ExternalPrivateMemoryAblationResult,
    destination: ExternalPrivateMemoryAblationDestination,
) -> ExternalPrivateMemoryAblationArtifactPaths:
    """Publish a complete local private artifact directory exactly once."""

    if type(result) is not ExternalPrivateMemoryAblationResult:
        raise TypeError("result must be an exact ExternalPrivateMemoryAblationResult.")
    if type(destination) is not ExternalPrivateMemoryAblationDestination:
        raise TypeError("destination must be an exact private destination.")
    validated_destination = _validate_private_destination(destination, require_absent=False)
    try:
        return _write_external_private_memory_ablation_artifacts_validated(
            result,
            validated_destination,
        )
    finally:
        validated_destination.close()


def _write_external_private_memory_ablation_artifacts_validated(
    result: ExternalPrivateMemoryAblationResult,
    destination: ExternalPrivateMemoryAblationDestination,
) -> ExternalPrivateMemoryAblationArtifactPaths:
    """Publish artifacts through one independently validated destination lease."""

    if result.methodology.report_destination_id != destination.destination_id:
        raise ValueError("Result methodology does not authorize this destination.")
    if result.methodology.state_storage_id != destination.state_storage_id:
        raise ValueError("Result methodology does not authorize this private state storage.")
    if result.methodology.report_destination_fingerprint != destination.fingerprint:
        raise ValueError("Result methodology does not authorize these private destination paths.")
    current = destination.artifact_directory

    report = (memory_experiment_report_to_json(result.report) + "\n").encode("utf-8")
    methodology = (
        external_private_memory_ablation_methodology_to_json(result.methodology) + "\n"
    ).encode("utf-8")
    completion = (
        json.dumps(
            {
                "methodology_revision": result.methodology.revision,
                "preflight_revision": result.methodology.preflight_revision,
                "report_revision": result.report.revision,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if len(report) + len(methodology) + len(completion) > (
        EXTERNAL_PRIVATE_MEMORY_ABLATION_MAX_ARTIFACT_BYTES
    ):
        raise ValueError("External private campaign artifacts exceed their byte bound.")

    authority = destination._filesystem_authority
    if authority is None:  # pragma: no cover - validated above
        raise ValueError("Private destination lacks trusted filesystem authority.")
    contents = {
        EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME: report,
        EXTERNAL_PRIVATE_MEMORY_ABLATION_METHODOLOGY_FILENAME: methodology,
        EXTERNAL_PRIVATE_MEMORY_ABLATION_COMPLETION_FILENAME: completion,
    }
    request_digest = (
        "sha256:"
        + hashlib.sha256(
            canonical_durable_json_bytes(
                {
                    "destination_fingerprint": destination.fingerprint,
                    "files": {
                        name: hashlib.sha256(content).hexdigest()
                        for name, content in sorted(contents.items())
                    },
                },
                "external private memory ablation artifact publication",
            )
        ).hexdigest()
    )
    if os.name == "nt":
        _publish_windows_private_artifact_tree(
            current,
            contents=contents,
            expected_parent_identity=authority.artifact_parent_identity,
            request_digest=request_digest,
        )
    else:
        with cooperative_path_lock(
            current.parent,
            current.name,
            lock_directory_name="cayu-private-memory-ablation-locks-v1",
        ):
            _publish_posix_private_artifact_tree(
                current,
                contents=contents,
                expected_parent_identity=authority.artifact_parent_identity,
            )
    return ExternalPrivateMemoryAblationArtifactPaths(
        artifact_directory=current,
        report_path=current / EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME,
        methodology_path=current / EXTERNAL_PRIVATE_MEMORY_ABLATION_METHODOLOGY_FILENAME,
        completion_path=current / EXTERNAL_PRIVATE_MEMORY_ABLATION_COMPLETION_FILENAME,
    )


__all__ = [
    "EXTERNAL_PRIVATE_MEMORY_ABLATION_COMPLETION_FILENAME",
    "EXTERNAL_PRIVATE_MEMORY_ABLATION_MAX_ARTIFACT_BYTES",
    "EXTERNAL_PRIVATE_MEMORY_ABLATION_MAX_REPORT_EVIDENCE_BYTES_PER_TRIAL",
    "EXTERNAL_PRIVATE_MEMORY_ABLATION_MAX_SUPPLEMENTAL_BYTES_PER_TRIAL",
    "EXTERNAL_PRIVATE_MEMORY_ABLATION_METHODOLOGY_FILENAME",
    "EXTERNAL_PRIVATE_MEMORY_ABLATION_MIN_REPORT_EVIDENCE_BYTES_PER_TRIAL",
    "EXTERNAL_PRIVATE_MEMORY_ABLATION_REPORT_FILENAME",
    "EXTERNAL_PRIVATE_MEMORY_ABLATION_SCHEMA_VERSION",
    "ExternalPrivateEvalCorpus",
    "ExternalPrivateMemoryAblationArtifactPaths",
    "ExternalPrivateMemoryAblationAuthorization",
    "ExternalPrivateMemoryAblationCacheEvidencePolicy",
    "ExternalPrivateMemoryAblationCacheEvidenceState",
    "ExternalPrivateMemoryAblationDestination",
    "ExternalPrivateMemoryAblationEvidenceCollector",
    "ExternalPrivateMemoryAblationExecutionMode",
    "ExternalPrivateMemoryAblationLimitation",
    "ExternalPrivateMemoryAblationMethodology",
    "ExternalPrivateMemoryAblationResult",
    "ExternalPrivateMemoryAblationRunFailureCode",
    "ExternalPrivateMemoryAblationRunStatus",
    "ExternalPrivateMemoryAblationScheduleEntry",
    "ExternalPrivateMemoryAblationSchedulePolicy",
    "ExternalPrivateMemoryAblationScheduleStrategy",
    "ExternalPrivateMemoryAblationSupplementalEvidence",
    "ExternalPrivateMemoryAblationTrial",
    "ExternalPrivateMemoryAblationTrialMethodology",
    "PreparedExternalPrivateMemoryAblation",
    "external_private_memory_ablation_destination",
    "external_private_memory_ablation_experiment_revision",
    "external_private_memory_ablation_methodology_to_json",
    "load_external_private_memory_ablation_corpus",
    "prepare_external_private_memory_ablation",
    "run_external_private_memory_ablation",
    "write_external_private_memory_ablation_artifacts",
]
