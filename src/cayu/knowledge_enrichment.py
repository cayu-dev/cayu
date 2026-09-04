"""Durable, opt-in orchestration for explicit knowledge curation.

The enrichment queue stores one bounded ``LearningBatch`` in Cayu's existing
``TaskStore`` and executes it through ``KnowledgeCurator`` under lease-fenced
retry authority. It does not start a scheduler, inspect transcripts, choose a
model, or create a second knowledge or task store.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import math
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
    revalidate_model_input,
    revalidate_model_inputs,
)
from cayu.knowledge_curator import (
    MAX_LEARNING_BATCH_BYTES,
    MAX_LEARNING_SIGNALS,
    KnowledgeCurator,
    KnowledgeCuratorConfig,
    LearningBatch,
    LearningBatchResult,
    _PreparedLearningBatch,
    _validate_curator_access_scope,
    validate_learning_batch,
)
from cayu.runtime._durable_worker_loop import (
    DurableWorkerStep,
    run_durable_lease_heartbeat,
    run_durable_worker_loop,
)
from cayu.runtime.invocation import (
    InvocationOriginClaim,
    InvocationOriginTrust,
    TaskExecutionSource,
)
from cayu.runtime.tasks import (
    Task,
    TaskClaimLost,
    TaskCreate,
    TaskOrder,
    TaskQuery,
    TaskRetryAttemptDisposition,
    TaskRetryPolicy,
    TaskRetrySeriesDisposition,
    TaskRetrySettlementRequest,
    TaskStatus,
    TaskStore,
    TaskTerminalizationConflict,
    TaskTerminalizationRequest,
    TaskTerminalizationRetryPolicy,
    TaskTerminalKind,
    _task_cancellation_terminalization_request,
    _task_retry_requested_cancellation_settlement,
    copy_task,
    settle_task_retry_attempt_with_retry,
    task_create_with_execution_source,
    terminalize_task_with_retry,
)
from cayu.storage.memory import (
    MAX_KNOWLEDGE_ACTIVATION_IDENTITY_BYTES,
    KnowledgeAccessScope,
    KnowledgeGovernanceMode,
    copy_knowledge_access_scope,
    knowledge_access_scope_sha256,
)

KNOWLEDGE_ENRICHMENT_SCHEMA_VERSION = 1
DEFAULT_KNOWLEDGE_ENRICHMENT_TASK_TYPE = "cayu.knowledge-enrichment.v1"
MAX_KNOWLEDGE_ENRICHMENT_IDENTITY_BYTES = 1_024
MAX_KNOWLEDGE_ENRICHMENT_TRIGGER_METADATA_BYTES = 16 * 1024
MAX_KNOWLEDGE_ENRICHMENT_REQUEST_BYTES = MAX_LEARNING_BATCH_BYTES + 256 * 1024
MAX_KNOWLEDGE_ENRICHMENT_RESULT_BYTES = 32 * 1024 * 1024
MAX_KNOWLEDGE_ENRICHMENT_FAILURE_ANNOTATION_BYTES = 4 * 1024
MAX_KNOWLEDGE_ENRICHMENT_RECLAIMS_PER_POLL = 100

# A JSON string can expand from one UTF-8 byte to six canonical bytes when a
# control character is escaped as ``\u00xx``. Candidate proposal keys are
# already bounded by the canonical candidate-batch limit; evaluator notes are
# bounded as unescaped UTF-8 and therefore need this separate multiplier.
_MAX_JSON_STRING_BYTE_EXPANSION = 6
_MAX_RESULT_SIGNAL_SCAFFOLD_BYTES = 2 * 1024
_MAX_RESULT_CANDIDATE_SCAFFOLD_BYTES = 8 * 1024
_MAX_RESULT_ENVELOPE_SCAFFOLD_BYTES = 32 * 1024

_TASK_INPUT_KEY = "knowledge_enrichment"
_TASK_CONTRACT = "cayu.knowledge-enrichment-task.v1"
_SEMANTIC_DISPATCH_INPUT_KEY = "knowledge_enrichment_semantic_dispatch"
_SEMANTIC_DISPATCH_CONTRACT = "cayu.knowledge-enrichment-semantic-dispatch.v1"
_PREPARATION_INPUT_KEY = "knowledge_enrichment_preparation"
_PREPARATION_CONTRACT = "cayu.knowledge-enrichment-preparation.v1"
_RESULT_CONTRACT = "cayu.knowledge-enrichment-result.v1"
_FAILURE_CONTRACT = "cayu.knowledge-enrichment-failure.v1"
_TERMINAL_TASK_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
}
_DIRECT_TASK_EXECUTION_SOURCES = {
    TaskExecutionSource.SDK_TASK,
    TaskExecutionSource.SCHEDULED,
    TaskExecutionSource.WEBHOOK,
}
_PREPARATION_AVAILABLE_AT = datetime(9999, 1, 1, tzinfo=UTC)


class KnowledgeEnrichmentConflict(RuntimeError):
    """Durable task evidence conflicts with the requested enrichment job."""


class KnowledgeEnrichmentJobRejected(RuntimeError):
    """A claimed malformed job was durably rejected instead of representing idle work."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(f"Enrichment task {task_id!r} was durably rejected.")


class KnowledgeEnrichmentJobStatus(StrEnum):
    """Current projection of one complete enrichment retry series."""

    PENDING = "pending"
    PROCESSING = "processing"
    RETRY_SCHEDULED = "retry_scheduled"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class KnowledgeEnrichmentFailureCategory(StrEnum):
    """Content-safe worker failure classes."""

    INVALID_JOB = "invalid_job"
    PROFILE_CONFLICT = "profile_conflict"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    SEMANTIC_OUTCOME_UNKNOWN = "semantic_outcome_unknown"
    RESULT_LIMIT_EXCEEDED = "result_limit_exceeded"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


class _EnrichmentModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
        allow_inf_nan=False,
    )


def _clean_identity(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > MAX_KNOWLEDGE_ENRICHMENT_IDENTITY_BYTES:
        raise ValueError(
            f"{field_name} must be at most {MAX_KNOWLEDGE_ENRICHMENT_IDENTITY_BYTES} UTF-8 bytes."
        )
    return value


def _validate_sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _sha256(value: object, field_name: str) -> str:
    return sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


def _bounded_json_object(value: object, field_name: str, *, max_bytes: int) -> dict[str, Any]:
    copied = copy_durable_json_object(value, field_name)
    if len(canonical_durable_json_bytes(copied, field_name)) > max_bytes:
        raise ValueError(f"{field_name} exceeds its canonical byte limit.")
    return copied


def _maximum_enrichment_result_bytes(
    batch: LearningBatch,
    *,
    config: KnowledgeCuratorConfig,
) -> int:
    """Conservatively bound the largest result allowed by one request.

    The result repeats every candidate proposal key once in the candidate
    projection and up to once per submitted signal. The complete canonical
    candidate batch therefore bounds each possible copy. The submitted batch
    itself bounds the batch/signal identities that reappear in the result.
    Fixed scaffold allowances cover result fields, hashes, enums, derived
    entry identities, arbitrary bounded decision codes, and the outer task
    envelope. This deliberately overestimates rather than sampling a provider
    result after the curator may already have published knowledge.
    """

    signal_count = len(batch.signals)
    batch_bytes = len(canonical_durable_json_bytes(batch.model_dump(mode="json"), "learning batch"))
    proposal_keys_per_projection = min(
        config.max_candidate_batch_bytes,
        config.max_candidates
        * (
            _MAX_JSON_STRING_BYTE_EXPANSION * MAX_KNOWLEDGE_ACTIVATION_IDENTITY_BYTES
            + 2  # JSON string quotes.
        ),
    )
    proposal_key_copies = (signal_count + 1) * proposal_keys_per_projection
    signal_details = signal_count * _MAX_RESULT_SIGNAL_SCAFFOLD_BYTES
    candidate_details = config.max_candidates * (
        _MAX_JSON_STRING_BYTE_EXPANSION * config.max_evaluator_notes_bytes
        + config.max_metadata_bytes
        + _MAX_RESULT_CANDIDATE_SCAFFOLD_BYTES
    )
    return (
        batch_bytes
        + proposal_key_copies
        + signal_details
        + candidate_details
        + _MAX_RESULT_ENVELOPE_SCAFFOLD_BYTES
    )


class KnowledgeEnrichmentTrigger(_EnrichmentModel):
    """Application-declared boundary that caused one batch to be submitted."""

    schema_version: Literal[1] = KNOWLEDGE_ENRICHMENT_SCHEMA_VERSION
    kind: StrictStr
    source_type: StrictStr
    source_id: StrictStr
    source_revision: StrictStr | None = None
    source_hash: StrictStr | None = None
    occurred_at: datetime
    includes_recalled_material: StrictBool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind", "source_type", "source_id", "source_revision", "source_hash")
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _clean_identity(value, info.field_name)

    @field_validator("occurred_at")
    @classmethod
    def normalize_occurred_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "occurred_at")

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: object) -> dict[str, Any]:
        return _bounded_json_object(
            value,
            "trigger metadata",
            max_bytes=MAX_KNOWLEDGE_ENRICHMENT_TRIGGER_METADATA_BYTES,
        )

    @model_validator(mode="after")
    def validate_source_frontier(self) -> KnowledgeEnrichmentTrigger:
        if self.source_revision is None and self.source_hash is None:
            raise ValueError("An enrichment trigger requires source_revision or source_hash.")
        return self

    @classmethod
    def completed_interaction(
        cls,
        *,
        session_id: str,
        interaction_id: str,
        terminal_event_id: str,
        occurred_at: datetime,
        terminal_event_hash: str | None = None,
        includes_recalled_material: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeEnrichmentTrigger:
        """Build an exact trigger without reading or copying interaction history."""

        copied_metadata = {} if metadata is None else dict(metadata)
        if "session_id" in copied_metadata:
            raise ValueError("completed-interaction metadata cannot replace session_id.")
        copied_metadata["session_id"] = _clean_identity(session_id, "session_id")
        return cls(
            kind="completed_interaction",
            source_type="cayu.interaction",
            source_id=_clean_identity(interaction_id, "interaction_id"),
            source_revision=_clean_identity(terminal_event_id, "terminal_event_id"),
            source_hash=terminal_event_hash,
            occurred_at=occurred_at,
            includes_recalled_material=includes_recalled_material,
            metadata=copied_metadata,
        )


class KnowledgeEnrichmentFeedbackAuthorization(_EnrichmentModel):
    """Application policy evidence for explicitly declared memory-derived input."""

    schema_version: Literal[1] = KNOWLEDGE_ENRICHMENT_SCHEMA_VERSION
    policy_identity: StrictStr
    policy_version: StrictStr
    independent_source_fingerprints: tuple[StrictStr, ...] = Field(
        min_length=1,
        max_length=MAX_LEARNING_SIGNALS,
    )

    @field_validator("policy_identity", "policy_version")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean_identity(value, info.field_name)

    @field_validator("independent_source_fingerprints", mode="before")
    @classmethod
    def copy_source_fingerprints(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise TypeError("independent_source_fingerprints must be an ordered array.")
        copied_items: list[str] = []
        for item in value:
            if type(item) is not str:
                raise TypeError("independent_source_fingerprints must contain strings.")
            copied_items.append(_validate_sha256(item, "independent_source_fingerprints"))
        copied = tuple(copied_items)
        if len(copied) != len(set(copied)):
            raise ValueError("independent_source_fingerprints cannot contain duplicates.")
        return copied


class KnowledgeEnrichmentProfile(_EnrichmentModel):
    """Exact application-owned curator and access policy expected by a worker."""

    schema_version: Literal[1] = KNOWLEDGE_ENRICHMENT_SCHEMA_VERSION
    curator_configuration_fingerprint: StrictStr
    access_scope_sha256: StrictStr
    candidate_generator_identity: StrictStr
    evaluator_identity: StrictStr
    candidate_policy_identity: StrictStr | None = None
    governance_mode: KnowledgeGovernanceMode
    activation_policy_identity: StrictStr | None = None
    activation_policy_version: StrictStr | None = None

    @field_validator("curator_configuration_fingerprint", "access_scope_sha256")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator(
        "candidate_generator_identity",
        "evaluator_identity",
        "candidate_policy_identity",
        "activation_policy_identity",
        "activation_policy_version",
    )
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _clean_identity(value, info.field_name)

    @model_validator(mode="after")
    def validate_governance_identity(self) -> KnowledgeEnrichmentProfile:
        has_identity = self.activation_policy_identity is not None
        has_version = self.activation_policy_version is not None
        if has_identity != has_version:
            raise ValueError("Activation policy identity and version must appear together.")
        if self.governance_mode is KnowledgeGovernanceMode.REVIEWED and has_identity:
            raise ValueError("Reviewed enrichment cannot bind an activation policy.")
        if self.governance_mode is not KnowledgeGovernanceMode.REVIEWED and not has_identity:
            raise ValueError("Automatic enrichment requires an activation policy identity.")
        return self


def knowledge_enrichment_profile(
    config: KnowledgeCuratorConfig,
    access_scope: KnowledgeAccessScope,
) -> KnowledgeEnrichmentProfile:
    """Derive the stable worker profile from existing curator contracts."""

    if type(config) is not KnowledgeCuratorConfig:
        raise TypeError("config must be a KnowledgeCuratorConfig.")
    if type(access_scope) is not KnowledgeAccessScope:
        raise TypeError("access_scope must be a KnowledgeAccessScope.")
    copied_config = KnowledgeCuratorConfig.model_validate(config.model_dump(mode="python"))
    copied_scope = copy_knowledge_access_scope(access_scope)
    _validate_curator_access_scope(copied_config, copied_scope)
    governance = copied_config.governance
    return KnowledgeEnrichmentProfile(
        curator_configuration_fingerprint=copied_config.fingerprint,
        access_scope_sha256=knowledge_access_scope_sha256(copied_scope),
        candidate_generator_identity=copied_config.candidate_generator_identity,
        evaluator_identity=copied_config.evaluator_identity,
        candidate_policy_identity=copied_config.policy_identity,
        governance_mode=governance.mode,
        activation_policy_identity=governance.policy_identity,
        activation_policy_version=governance.policy_version,
    )


class KnowledgeEnrichmentRequest(_EnrichmentModel):
    """One bounded, exact, application-submitted enrichment operation."""

    schema_version: Literal[1] = KNOWLEDGE_ENRICHMENT_SCHEMA_VERSION
    operation_id: StrictStr
    batch: LearningBatch
    trigger: KnowledgeEnrichmentTrigger
    profile: KnowledgeEnrichmentProfile
    submitted_at: datetime
    execution_source: TaskExecutionSource = TaskExecutionSource.SDK_TASK
    invocation_origin: InvocationOriginClaim | None = None
    feedback_authorization: KnowledgeEnrichmentFeedbackAuthorization | None = None

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str) -> str:
        return _clean_identity(value, "operation_id")

    @field_validator("batch", mode="before")
    @classmethod
    def copy_batch(cls, value: object) -> object:
        return revalidate_model_input(value, LearningBatch)

    @field_validator("trigger", mode="before")
    @classmethod
    def copy_trigger(cls, value: object) -> object:
        return revalidate_model_input(value, KnowledgeEnrichmentTrigger)

    @field_validator("profile", mode="before")
    @classmethod
    def copy_profile(cls, value: object) -> object:
        return revalidate_model_input(value, KnowledgeEnrichmentProfile)

    @field_validator("feedback_authorization", mode="before")
    @classmethod
    def copy_feedback_authorization(cls, value: object) -> object:
        return revalidate_model_input(value, KnowledgeEnrichmentFeedbackAuthorization)

    @field_validator("invocation_origin", mode="before")
    @classmethod
    def copy_invocation_origin(cls, value: object) -> object:
        return revalidate_model_input(value, InvocationOriginClaim)

    @field_validator("submitted_at")
    @classmethod
    def normalize_submitted_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "submitted_at")

    @field_validator("execution_source")
    @classmethod
    def validate_execution_source(cls, value: TaskExecutionSource) -> TaskExecutionSource:
        if value not in _DIRECT_TASK_EXECUTION_SOURCES:
            raise ValueError("Knowledge enrichment requires a direct SDK task source.")
        return value

    @model_validator(mode="after")
    def validate_feedback_boundary(self) -> KnowledgeEnrichmentRequest:
        if self.submitted_at < self.trigger.occurred_at:
            raise ValueError("Enrichment submission cannot predate its trigger.")
        declared = self.trigger.includes_recalled_material
        if declared != (self.feedback_authorization is not None):
            raise ValueError(
                "Memory-derived learning input requires explicit feedback authorization, "
                "and feedback authorization is invalid when no such input is declared."
            )
        authorization = self.feedback_authorization
        if authorization is None:
            return self
        available = {
            reference.fingerprint
            for signal in self.batch.signals
            for reference in signal.source_references
        }
        if not set(authorization.independent_source_fingerprints).issubset(available):
            raise ValueError(
                "Feedback authorization must name source references in the submitted batch."
            )
        return self

    @property
    def fingerprint(self) -> str:
        return _sha256(self.model_dump(mode="json"), "knowledge enrichment request")


class KnowledgeEnrichmentQueueConfig(_EnrichmentModel):
    """Queue route namespace plus finite payload and retry ceilings."""

    schema_version: Literal[1] = KNOWLEDGE_ENRICHMENT_SCHEMA_VERSION
    task_type: StrictStr = Field(
        default=DEFAULT_KNOWLEDGE_ENRICHMENT_TASK_TYPE,
        description=(
            "Application route namespace; KnowledgeEnrichmentQueue.task_type is the "
            "profile- and configuration-derived effective TaskStore route."
        ),
    )
    max_request_bytes: StrictInt = Field(default=512 * 1024, ge=1)
    max_result_bytes: StrictInt = Field(
        default=MAX_KNOWLEDGE_ENRICHMENT_RESULT_BYTES,
        ge=MAX_KNOWLEDGE_ENRICHMENT_RESULT_BYTES,
        le=MAX_KNOWLEDGE_ENRICHMENT_RESULT_BYTES,
    )
    max_reclaims_per_poll: StrictInt = Field(
        default=100,
        ge=1,
        le=MAX_KNOWLEDGE_ENRICHMENT_RECLAIMS_PER_POLL,
    )
    retry_policy: TaskRetryPolicy = Field(
        default_factory=lambda: TaskRetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=1.0,
            backoff_multiplier=2.0,
            max_backoff_seconds=30.0,
        )
    )
    terminalization_retry_policy: TaskTerminalizationRetryPolicy = Field(
        default_factory=TaskTerminalizationRetryPolicy
    )

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, value: str) -> str:
        return _clean_identity(value, "task_type")

    @field_validator("retry_policy", mode="before")
    @classmethod
    def copy_retry_policy(cls, value: object) -> object:
        return revalidate_model_input(value, TaskRetryPolicy)

    @field_validator("terminalization_retry_policy", mode="before")
    @classmethod
    def copy_terminalization_policy(cls, value: object) -> object:
        return revalidate_model_input(value, TaskTerminalizationRetryPolicy)

    @model_validator(mode="after")
    def validate_byte_ceilings(self) -> KnowledgeEnrichmentQueueConfig:
        if self.max_request_bytes > MAX_KNOWLEDGE_ENRICHMENT_REQUEST_BYTES:
            raise ValueError("max_request_bytes exceeds the enrichment contract ceiling.")
        if self.max_result_bytes > MAX_KNOWLEDGE_ENRICHMENT_RESULT_BYTES:
            raise ValueError("max_result_bytes exceeds the enrichment contract ceiling.")
        if any(
            value is not None
            for value in (
                self.retry_policy.max_elapsed_seconds,
                self.retry_policy.max_total_tokens,
                self.retry_policy.max_estimated_cost,
            )
        ):
            raise ValueError(
                "Knowledge enrichment retry policy supports attempt/backoff bounds only; "
                "component timeouts and model budgets remain owned by their configured "
                "curator components."
            )
        return self

    @property
    def fingerprint(self) -> str:
        """Bind behavior-bearing durable queue settings, excluding local I/O retries."""

        return _sha256(
            {
                "contract": "cayu.knowledge-enrichment-queue-config.v1",
                "task_type": self.task_type,
                "max_request_bytes": self.max_request_bytes,
                "max_result_bytes": self.max_result_bytes,
                "max_reclaims_per_poll": self.max_reclaims_per_poll,
                "retry_policy": self.retry_policy.model_dump(mode="json"),
            },
            "knowledge enrichment queue configuration",
        )


class KnowledgeEnrichmentFailure(_EnrichmentModel):
    """Bounded diagnostic safe for durable task error payloads."""

    schema_version: Literal[1] = KNOWLEDGE_ENRICHMENT_SCHEMA_VERSION
    request_sha256: StrictStr | None = None
    code: StrictStr
    category: KnowledgeEnrichmentFailureCategory
    retryable: StrictBool
    annotation: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_sha256(value, "request_sha256")

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _clean_identity(value, "code")

    @field_validator("annotation", mode="before")
    @classmethod
    def copy_annotation(cls, value: object) -> dict[str, Any]:
        return _bounded_json_object(
            value,
            "failure annotation",
            max_bytes=MAX_KNOWLEDGE_ENRICHMENT_FAILURE_ANNOTATION_BYTES,
        )


class KnowledgeEnrichmentFailureDecision(_EnrichmentModel):
    """Application classification for an exception that escaped the curator."""

    code: StrictStr
    category: KnowledgeEnrichmentFailureCategory
    retryable: StrictBool = False
    retry_after_seconds: StrictFloat | None = Field(default=None, ge=0, le=86_400)

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _clean_identity(value, "code")

    @model_validator(mode="after")
    def validate_retry_delay(self) -> KnowledgeEnrichmentFailureDecision:
        if self.retry_after_seconds is not None:
            if not math.isfinite(self.retry_after_seconds):
                raise ValueError("retry_after_seconds must be finite.")
            if not self.retryable:
                raise ValueError("Only retryable failures can set retry_after_seconds.")
        return self


class KnowledgeEnrichmentExceptionClassifier(Protocol):
    """Optional application policy for retrying escaped infrastructure failures."""

    def __call__(self, error: Exception) -> KnowledgeEnrichmentFailureDecision: ...


class KnowledgeEnrichmentJobResult(_EnrichmentModel):
    """Exact bounded curation result committed with task-series settlement."""

    schema_version: Literal[1] = KNOWLEDGE_ENRICHMENT_SCHEMA_VERSION
    request_sha256: StrictStr
    task_id: StrictStr
    attempt: StrictInt = Field(ge=1, le=100)
    curation: LearningBatchResult

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _validate_sha256(value, "request_sha256")

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return _clean_identity(value, "task_id")

    @field_validator("curation", mode="before")
    @classmethod
    def copy_curation(cls, value: object) -> object:
        return revalidate_model_input(value, LearningBatchResult)


class KnowledgeEnrichmentAttempt(_EnrichmentModel):
    """Bounded durable projection of one retry-series attempt."""

    task_id: StrictStr
    attempt: StrictInt = Field(ge=1, le=100)
    task_status: TaskStatus
    disposition: TaskRetrySeriesDisposition
    failure: KnowledgeEnrichmentFailure | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return _clean_identity(value, "task_id")

    @field_validator("failure", mode="before")
    @classmethod
    def copy_failure(cls, value: object) -> object:
        return revalidate_model_input(value, KnowledgeEnrichmentFailure)

    @field_validator("created_at", "updated_at", "completed_at")
    @classmethod
    def normalize_datetime(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, info.field_name)


class KnowledgeEnrichmentJob(_EnrichmentModel):
    """Current, defensively copied view of one complete enrichment retry series."""

    schema_version: Literal[1] = KNOWLEDGE_ENRICHMENT_SCHEMA_VERSION
    id: StrictStr
    current_task_id: StrictStr
    operation_id: StrictStr
    request_sha256: StrictStr
    status: KnowledgeEnrichmentJobStatus
    request: KnowledgeEnrichmentRequest
    attempts: tuple[KnowledgeEnrichmentAttempt, ...] = Field(min_length=1, max_length=100)
    result: KnowledgeEnrichmentJobResult | None = None
    failure: KnowledgeEnrichmentFailure | None = None
    next_eligible_at: datetime | None = None

    @field_validator("id", "current_task_id", "operation_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean_identity(value, info.field_name)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _validate_sha256(value, "request_sha256")

    @field_validator("request", mode="before")
    @classmethod
    def copy_request(cls, value: object) -> object:
        return revalidate_model_input(value, KnowledgeEnrichmentRequest)

    @field_validator("attempts", mode="before")
    @classmethod
    def copy_attempts(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            KnowledgeEnrichmentAttempt,
            maximum=100,
            field_name="attempts",
        )

    @field_validator("result", mode="before")
    @classmethod
    def copy_result(cls, value: object) -> object:
        return revalidate_model_input(value, KnowledgeEnrichmentJobResult)

    @field_validator("failure", mode="before")
    @classmethod
    def copy_failure(cls, value: object) -> object:
        return revalidate_model_input(value, KnowledgeEnrichmentFailure)

    @field_validator("next_eligible_at")
    @classmethod
    def normalize_next_eligible_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, "next_eligible_at")

    @model_validator(mode="after")
    def validate_terminal_shape(self) -> KnowledgeEnrichmentJob:
        has_result = self.result is not None
        has_failure = self.failure is not None
        if tuple(attempt.attempt for attempt in self.attempts) != tuple(
            range(1, len(self.attempts) + 1)
        ):
            raise ValueError("Enrichment attempts must be contiguous and start at one.")
        current_attempt = self.attempts[-1]
        if current_attempt.task_id != self.current_task_id:
            raise ValueError("The current enrichment task must be the final projected attempt.")
        if self.status is KnowledgeEnrichmentJobStatus.COMPLETED:
            if not has_result or has_failure:
                raise ValueError("A completed enrichment job requires only a result.")
        elif self.status in {
            KnowledgeEnrichmentJobStatus.FAILED,
            KnowledgeEnrichmentJobStatus.CANCELLED,
        }:
            if has_result or not has_failure:
                raise ValueError("A failed or cancelled enrichment job requires only a failure.")
        elif has_result or has_failure:
            raise ValueError("A non-terminal enrichment job cannot carry a terminal outcome.")
        if (self.next_eligible_at is not None) != (
            self.status is KnowledgeEnrichmentJobStatus.RETRY_SCHEDULED
        ):
            raise ValueError("Only retry-scheduled enrichment jobs require next_eligible_at.")
        if self.request_sha256 != self.request.fingerprint:
            raise ValueError("Enrichment request fingerprint conflicts with the job.")
        if self.result is not None and (
            self.result.request_sha256 != self.request_sha256
            or self.result.task_id != self.current_task_id
            or self.result.attempt != current_attempt.attempt
        ):
            raise ValueError("Enrichment result conflicts with its request or attempt.")
        if self.failure is not None and self.failure.request_sha256 != self.request_sha256:
            raise ValueError("Enrichment failure conflicts with its request fingerprint.")
        return self


class KnowledgeEnrichmentQueue:
    """Submit and inspect exact enrichment jobs in an existing ``TaskStore``."""

    def __init__(
        self,
        task_store: TaskStore,
        *,
        curator_config: KnowledgeCuratorConfig,
        access_scope: KnowledgeAccessScope,
        config: KnowledgeEnrichmentQueueConfig | None = None,
    ) -> None:
        if not isinstance(task_store, TaskStore):
            raise TypeError("task_store must be a TaskStore.")
        if not task_store.supports_task_retry_series:
            raise ValueError("Knowledge enrichment requires task retry-series support.")
        if not task_store.supports_delayed_availability:
            raise ValueError("Knowledge enrichment requires delayed task availability support.")
        if not task_store.supports_idempotent_terminalization:
            raise ValueError("Knowledge enrichment requires idempotent task terminalization.")
        if type(curator_config) is not KnowledgeCuratorConfig:
            raise TypeError("curator_config must be a KnowledgeCuratorConfig.")
        if type(access_scope) is not KnowledgeAccessScope:
            raise TypeError("access_scope must be a KnowledgeAccessScope.")
        if config is None:
            config = KnowledgeEnrichmentQueueConfig()
        elif type(config) is not KnowledgeEnrichmentQueueConfig:
            raise TypeError("config must be a KnowledgeEnrichmentQueueConfig.")
        self._task_store = task_store
        self._curator_config = KnowledgeCuratorConfig.model_validate(
            curator_config.model_dump(mode="python")
        )
        self._access_scope = copy_knowledge_access_scope(access_scope)
        self._profile = knowledge_enrichment_profile(
            self._curator_config,
            self._access_scope,
        )
        self._config = KnowledgeEnrichmentQueueConfig.model_validate(
            config.model_dump(mode="python")
        )
        self._execution_domain_sha256 = _sha256(
            {
                "contract": "cayu.knowledge-enrichment-execution-domain.v1",
                "profile": self._profile.model_dump(mode="json"),
                "queue_config_sha256": self._config.fingerprint,
            },
            "knowledge enrichment execution domain",
        )
        self._task_type = f"cayu.knowledge-enrichment.route.v1.{self._execution_domain_sha256}"
        self._semantic_dispatch_task_type = (
            f"cayu.knowledge-enrichment.semantic-dispatch.v1.{self._execution_domain_sha256}"
        )
        self._preparation_task_type = (
            f"cayu.knowledge-enrichment.preparation.v1.{self._execution_domain_sha256}"
        )

    @property
    def task_store(self) -> TaskStore:
        return self._task_store

    @property
    def config(self) -> KnowledgeEnrichmentQueueConfig:
        return KnowledgeEnrichmentQueueConfig.model_validate(self._config.model_dump(mode="python"))

    @property
    def profile(self) -> KnowledgeEnrichmentProfile:
        return KnowledgeEnrichmentProfile.model_validate(self._profile.model_dump(mode="python"))

    @property
    def task_type(self) -> str:
        """Return the effective execution-domain route claimed by this queue's workers."""

        return self._task_type

    async def submit(self, request: KnowledgeEnrichmentRequest) -> KnowledgeEnrichmentJob:
        """Create or exactly replay one durable enrichment retry series."""

        if type(request) is not KnowledgeEnrichmentRequest:
            raise TypeError("request must be a KnowledgeEnrichmentRequest.")
        copied = KnowledgeEnrichmentRequest.model_validate(request.model_dump(mode="python"))
        if copied.profile != self._profile:
            raise KnowledgeEnrichmentConflict(
                "Enrichment request profile conflicts with the configured queue."
            )
        copied_batch = validate_learning_batch(copied.batch, config=self._curator_config)
        copied = copied.model_copy(update={"batch": copied_batch})
        if (
            _maximum_enrichment_result_bytes(copied.batch, config=self._curator_config)
            > self._config.max_result_bytes
        ):
            raise ValueError(
                "Knowledge enrichment batch and curator bounds can exceed max_result_bytes."
            )
        request_sha256 = copied.fingerprint
        root_task_id = self._root_task_id(copied.operation_id)
        queue_config_sha256 = self._config.fingerprint
        task_input = _task_input(
            copied,
            request_sha256=request_sha256,
            queue_config_sha256=queue_config_sha256,
            execution_domain_sha256=self._execution_domain_sha256,
        )
        if (
            len(canonical_durable_json_bytes(task_input, "knowledge enrichment task input"))
            > self._config.max_request_bytes
        ):
            raise ValueError("Knowledge enrichment request exceeds max_request_bytes.")
        task_metadata = _task_metadata(
            request_sha256,
            queue_config_sha256=queue_config_sha256,
            execution_domain_sha256=self._execution_domain_sha256,
        )

        existing = await self._task_store.load_task(root_task_id)
        if existing is None:
            create = TaskCreate(
                task_id=root_task_id,
                type=self._task_type,
                title="Knowledge enrichment",
                input=task_input,
                metadata=task_metadata,
                retry_policy=self._config.retry_policy,
                invocation_origin=copied.invocation_origin,
            )
            create = task_create_with_execution_source(create, source=copied.execution_source)
            try:
                existing = await self._task_store.create_task(create)
            except Exception as publication_error:
                try:
                    existing = await self._task_store.load_task(root_task_id)
                except Exception as reconciliation_error:
                    publication_error.add_note(
                        "Knowledge enrichment submission reconciliation also failed: "
                        f"{type(reconciliation_error).__name__}."
                    )
                    raise publication_error from reconciliation_error
                if existing is None:
                    raise
        self._require_series_task(
            existing,
            request=copied,
            request_sha256=request_sha256,
            expected_task_id=root_task_id,
            expected_attempt=1,
            expected_predecessor=None,
        )
        job = await self._load_from_root(existing)
        if job is None:  # pragma: no cover - root was already loaded
            raise KnowledgeEnrichmentConflict("Enrichment root task disappeared after submission.")
        return job

    async def load(self, operation_id: str) -> KnowledgeEnrichmentJob | None:
        """Load the complete bounded retry series for one operation identity."""

        operation_id = _clean_identity(operation_id, "operation_id")
        root = await self._task_store.load_task(self._root_task_id(operation_id))
        if root is None:
            return None
        job = await self._load_from_root(root)
        if job is None:  # pragma: no cover - root was supplied
            return None
        if job.operation_id != operation_id:
            raise KnowledgeEnrichmentConflict("Enrichment operation identity conflicts with task.")
        return job

    def _root_task_id(self, operation_id: str) -> str:
        digest = _sha256(
            {
                "contract": "cayu.knowledge-enrichment-task-identity.v1",
                "execution_domain_sha256": self._execution_domain_sha256,
                "operation_id": operation_id,
            },
            "knowledge enrichment task identity",
        )
        return f"knowledge-enrichment-{digest}"

    def _preparation_task_id(self, operation_id: str) -> str:
        digest = _sha256(
            {
                "contract": "cayu.knowledge-enrichment-preparation-identity.v1",
                "execution_domain_sha256": self._execution_domain_sha256,
                "operation_id": operation_id,
            },
            "knowledge enrichment preparation identity",
        )
        return f"knowledge-enrichment-preparation-{digest}"

    def _semantic_dispatch_task_id(self, operation_id: str) -> str:
        digest = _sha256(
            {
                "contract": "cayu.knowledge-enrichment-semantic-dispatch-identity.v1",
                "execution_domain_sha256": self._execution_domain_sha256,
                "operation_id": operation_id,
            },
            "knowledge enrichment semantic dispatch identity",
        )
        return f"knowledge-enrichment-semantic-dispatch-{digest}"

    async def _load_from_root(self, root: Task) -> KnowledgeEnrichmentJob | None:
        request, request_sha256 = self._request_from_task(root)
        root_task_id = self._root_task_id(request.operation_id)
        self._require_series_task(
            root,
            request=request,
            request_sha256=request_sha256,
            expected_task_id=root_task_id,
            expected_attempt=1,
            expected_predecessor=None,
        )
        root_series = root.retry_series
        assert root_series is not None
        expected_series_id = root_series.series_id
        expected_causal_budget_id = root_series.causal_budget_id
        expected_started_at = root_series.started_at
        attempts: list[KnowledgeEnrichmentAttempt] = []
        current = copy_task(root)
        expected_task_id = root_task_id
        seen: set[str] = set()
        for expected_attempt in range(1, self._config.retry_policy.max_attempts + 1):
            if current.id in seen:
                raise KnowledgeEnrichmentConflict("Enrichment retry series contains a cycle.")
            seen.add(current.id)
            expected_predecessor = None if expected_attempt == 1 else attempts[-1].task_id
            self._require_series_task(
                current,
                request=request,
                request_sha256=request_sha256,
                expected_task_id=expected_task_id,
                expected_attempt=expected_attempt,
                expected_predecessor=expected_predecessor,
                expected_series_id=expected_series_id,
                expected_causal_budget_id=expected_causal_budget_id,
                expected_started_at=expected_started_at,
            )
            attempts.append(_attempt_from_task(current, request_sha256=request_sha256))
            series = current.retry_series
            assert series is not None
            successor_id = series.successor_task_id
            retry_scheduled = series.disposition is TaskRetrySeriesDisposition.RETRY_SCHEDULED
            if (successor_id is not None) != retry_scheduled:
                raise KnowledgeEnrichmentConflict(
                    "Enrichment retry disposition conflicts with its successor."
                )
            if successor_id is None:
                break
            attempt_failure = attempts[-1].failure
            if (
                current.status is not TaskStatus.FAILED
                or attempt_failure is None
                or not attempt_failure.retryable
            ):
                raise KnowledgeEnrichmentConflict(
                    "Enrichment retry successor lacks a retryable predecessor."
                )
            successor = await self._task_store.load_task(successor_id)
            if successor is None:
                raise KnowledgeEnrichmentConflict("Enrichment retry successor is missing.")
            expected_task_id = successor_id
            current = copy_task(successor)
        else:
            raise KnowledgeEnrichmentConflict("Enrichment retry series exceeds its policy.")

        status = _job_status(current)
        result = None
        failure = None
        if status is KnowledgeEnrichmentJobStatus.COMPLETED:
            result = _result_from_task(
                current,
                request=request,
                request_sha256=request_sha256,
                max_result_bytes=self._config.max_result_bytes,
                curator_config=self._curator_config,
            )
        elif status in {
            KnowledgeEnrichmentJobStatus.FAILED,
            KnowledgeEnrichmentJobStatus.CANCELLED,
        }:
            failure = _failure_from_task(current, request_sha256=request_sha256)
        series = current.retry_series
        assert series is not None
        return KnowledgeEnrichmentJob(
            id=root_task_id,
            current_task_id=current.id,
            operation_id=request.operation_id,
            request_sha256=request_sha256,
            status=status,
            request=request,
            attempts=tuple(attempts),
            result=result,
            failure=failure,
            next_eligible_at=(
                current.available_at
                if status is KnowledgeEnrichmentJobStatus.RETRY_SCHEDULED
                else None
            ),
        )

    def _request_from_task(self, task: Task) -> tuple[KnowledgeEnrichmentRequest, str]:
        if type(task) is not Task:
            raise TypeError("TaskStore returned a non-Task enrichment row.")
        if set(task.input) != {_TASK_INPUT_KEY}:
            raise KnowledgeEnrichmentConflict("Enrichment task input envelope is invalid.")
        envelope = task.input.get(_TASK_INPUT_KEY)
        if (
            type(envelope) is not dict
            or set(envelope)
            != {
                "contract",
                "request_sha256",
                "queue_config_sha256",
                "execution_domain_sha256",
                "request",
            }
            or envelope.get("contract") != _TASK_CONTRACT
        ):
            raise KnowledgeEnrichmentConflict("Enrichment task envelope is invalid.")
        request_document = envelope.get("request")
        request_sha256 = envelope.get("request_sha256")
        queue_config_sha256 = envelope.get("queue_config_sha256")
        execution_domain_sha256 = envelope.get("execution_domain_sha256")
        if (
            type(request_document) is not dict
            or type(request_sha256) is not str
            or type(queue_config_sha256) is not str
            or type(execution_domain_sha256) is not str
        ):
            raise KnowledgeEnrichmentConflict("Enrichment task request evidence is invalid.")
        try:
            request = KnowledgeEnrichmentRequest.model_validate(request_document)
            request_sha256 = _validate_sha256(request_sha256, "request_sha256")
            queue_config_sha256 = _validate_sha256(
                queue_config_sha256,
                "queue_config_sha256",
            )
            execution_domain_sha256 = _validate_sha256(
                execution_domain_sha256,
                "execution_domain_sha256",
            )
        except (TypeError, ValueError) as exc:
            raise KnowledgeEnrichmentConflict("Enrichment task request is malformed.") from exc
        if request.fingerprint != request_sha256:
            raise KnowledgeEnrichmentConflict("Enrichment task request fingerprint conflicts.")
        if request.profile != self._profile:
            raise KnowledgeEnrichmentConflict("Enrichment task uses another curator profile.")
        if queue_config_sha256 != self._config.fingerprint:
            raise KnowledgeEnrichmentConflict("Enrichment task uses another queue configuration.")
        if execution_domain_sha256 != self._execution_domain_sha256:
            raise KnowledgeEnrichmentConflict("Enrichment task uses another execution domain.")
        if (
            len(canonical_durable_json_bytes(task.input, "knowledge enrichment task input"))
            > self._config.max_request_bytes
        ):
            raise KnowledgeEnrichmentConflict("Enrichment task input exceeds its queue limit.")
        try:
            validate_learning_batch(request.batch, config=self._curator_config)
        except (TypeError, ValueError) as exc:
            raise KnowledgeEnrichmentConflict(
                "Enrichment task batch exceeds its curator bounds."
            ) from exc
        if (
            _maximum_enrichment_result_bytes(request.batch, config=self._curator_config)
            > self._config.max_result_bytes
        ):
            raise KnowledgeEnrichmentConflict(
                "Enrichment task can exceed its durable result limit."
            )
        if task.metadata != _task_metadata(
            request_sha256,
            queue_config_sha256=queue_config_sha256,
            execution_domain_sha256=execution_domain_sha256,
        ):
            raise KnowledgeEnrichmentConflict("Enrichment task metadata conflicts with request.")
        return request, request_sha256

    def _preparation_from_task(
        self,
        task: Task,
        *,
        request: KnowledgeEnrichmentRequest,
        request_sha256: str,
    ) -> tuple[_PreparedLearningBatch, LearningBatchResult | None]:
        expected_task_id = self._preparation_task_id(request.operation_id)
        if (
            len(
                canonical_durable_json_bytes(
                    task.input,
                    "knowledge enrichment preparation task input",
                )
            )
            > self._config.max_result_bytes
        ):
            raise KnowledgeEnrichmentConflict("Enrichment preparation exceeds its queue limit.")
        if set(task.input) != {_PREPARATION_INPUT_KEY}:
            raise KnowledgeEnrichmentConflict("Enrichment preparation input is invalid.")
        envelope = task.input.get(_PREPARATION_INPUT_KEY)
        if (
            type(envelope) is not dict
            or set(envelope)
            != {
                "contract",
                "request_sha256",
                "preparation_sha256",
                "execution_domain_sha256",
                "preparation",
            }
            or envelope.get("contract") != _PREPARATION_CONTRACT
            or envelope.get("request_sha256") != request_sha256
            or envelope.get("execution_domain_sha256") != self._execution_domain_sha256
        ):
            raise KnowledgeEnrichmentConflict("Enrichment preparation envelope is invalid.")
        preparation_document = envelope.get("preparation")
        preparation_sha256 = envelope.get("preparation_sha256")
        if type(preparation_document) is not dict or type(preparation_sha256) is not str:
            raise KnowledgeEnrichmentConflict("Enrichment preparation evidence is invalid.")
        try:
            preparation_sha256 = _validate_sha256(
                preparation_sha256,
                "preparation_sha256",
            )
            prepared = _PreparedLearningBatch.model_validate(preparation_document)
        except (TypeError, ValueError) as exc:
            raise KnowledgeEnrichmentConflict("Enrichment preparation is malformed.") from exc
        if _sha256(preparation_document, "knowledge enrichment preparation") != preparation_sha256:
            raise KnowledgeEnrichmentConflict("Enrichment preparation fingerprint conflicts.")
        expected_metadata = _preparation_task_metadata(
            request_sha256=request_sha256,
            preparation_sha256=preparation_sha256,
            execution_domain_sha256=self._execution_domain_sha256,
        )
        if (
            task.id != expected_task_id
            or task.type != self._preparation_task_type
            or task.title != "Knowledge enrichment preparation"
            or task.description is not None
            or task.session_id is not None
            or task.parent_task_id is not None
            or task.assigned_agent_name is not None
            or task.available_at != _PREPARATION_AVAILABLE_AT
            or task.worker_id is not None
            or task.lease_expires_at is not None
            or task.metadata != expected_metadata
            or task.retry_series is not None
            or task.work_contract is not None
            or task.status not in {TaskStatus.PENDING, TaskStatus.COMPLETED}
            or (task.status is TaskStatus.PENDING and task.result is not None)
            or (task.status is TaskStatus.COMPLETED and task.result is None)
            or task.error is not None
            or task.status_reason is not None
            or task.status_payload is not None
            or task.invocation.source is not request.execution_source
            or not _invocation_origin_matches(task, request.invocation_origin)
        ):
            raise KnowledgeEnrichmentConflict(
                "Durable preparation task conflicts with enrichment authority."
            )
        if (
            prepared.batch != request.batch
            or prepared.configuration_fingerprint
            != request.profile.curator_configuration_fingerprint
        ):
            raise KnowledgeEnrichmentConflict(
                "Enrichment preparation conflicts with its durable request."
            )
        if task.status is TaskStatus.PENDING:
            return prepared, None
        assert task.result is not None
        if (
            len(
                canonical_durable_json_bytes(
                    task.result,
                    "knowledge enrichment preparation result",
                )
            )
            > self._config.max_result_bytes
        ):
            raise KnowledgeEnrichmentConflict(
                "Completed enrichment preparation result exceeds its queue limit."
            )
        if (
            set(task.result) != {"contract", "preparation_sha256", "curation"}
            or task.result.get("contract") != _PREPARATION_CONTRACT
            or task.result.get("preparation_sha256") != preparation_sha256
            or type(task.result.get("curation")) is not dict
        ):
            raise KnowledgeEnrichmentConflict("Completed enrichment preparation is invalid.")
        try:
            result = LearningBatchResult.model_validate(task.result["curation"])
        except (TypeError, ValueError) as exc:
            raise KnowledgeEnrichmentConflict(
                "Completed enrichment preparation result is malformed."
            ) from exc
        _validate_curation_result(
            result,
            request=request,
            request_sha256=request_sha256,
            curator_config=self._curator_config,
        )
        return prepared, result

    def _semantic_dispatch_from_task(
        self,
        task: Task,
        *,
        request: KnowledgeEnrichmentRequest,
        request_sha256: str,
    ) -> str:
        expected_task_id = self._semantic_dispatch_task_id(request.operation_id)
        envelope = task.input.get(_SEMANTIC_DISPATCH_INPUT_KEY)
        if (
            type(envelope) is not dict
            or set(envelope)
            != {
                "contract",
                "request_sha256",
                "execution_domain_sha256",
                "dispatch_id",
            }
            or envelope.get("contract") != _SEMANTIC_DISPATCH_CONTRACT
            or envelope.get("request_sha256") != request_sha256
            or envelope.get("execution_domain_sha256") != self._execution_domain_sha256
        ):
            raise KnowledgeEnrichmentConflict("Enrichment semantic dispatch envelope is invalid.")
        dispatch_id = envelope.get("dispatch_id")
        try:
            if type(dispatch_id) is not str:
                raise TypeError
            dispatch_id = _clean_identity(dispatch_id, "dispatch_id")
        except (TypeError, ValueError) as exc:
            raise KnowledgeEnrichmentConflict(
                "Enrichment semantic dispatch identity is malformed."
            ) from exc
        if (
            task.id != expected_task_id
            or task.type != self._semantic_dispatch_task_type
            or task.title != "Knowledge enrichment semantic dispatch"
            or task.description is not None
            or task.session_id is not None
            or task.parent_task_id is not None
            or task.assigned_agent_name is not None
            or task.available_at != _PREPARATION_AVAILABLE_AT
            or task.worker_id is not None
            or task.lease_expires_at is not None
            or task.metadata
            != _semantic_dispatch_task_metadata(
                request_sha256=request_sha256,
                execution_domain_sha256=self._execution_domain_sha256,
                dispatch_id=dispatch_id,
            )
            or task.retry_series is not None
            or task.work_contract is not None
            or task.status is not TaskStatus.PENDING
            or task.result is not None
            or task.error is not None
            or task.status_reason is not None
            or task.status_payload is not None
            or task.invocation.source is not request.execution_source
            or not _invocation_origin_matches(task, request.invocation_origin)
        ):
            raise KnowledgeEnrichmentConflict(
                "Durable semantic dispatch conflicts with enrichment authority."
            )
        return dispatch_id

    def _require_series_task(
        self,
        task: Task,
        *,
        request: KnowledgeEnrichmentRequest,
        request_sha256: str,
        expected_task_id: str,
        expected_attempt: int,
        expected_predecessor: str | None,
        expected_series_id: str | None = None,
        expected_causal_budget_id: str | None = None,
        expected_started_at: datetime | None = None,
    ) -> None:
        loaded_request, loaded_sha256 = self._request_from_task(task)
        series = task.retry_series
        if (
            task.id != expected_task_id
            or task.type != self._task_type
            or task.title != "Knowledge enrichment"
            or task.description is not None
            or task.session_id is not None
            or task.parent_task_id is not None
            or task.assigned_agent_name is not None
            or task.work_contract is not None
            or loaded_request != request
            or loaded_sha256 != request_sha256
            or series is None
            or series.attempt != expected_attempt
            or series.predecessor_task_id != expected_predecessor
            or series.policy != self._config.retry_policy
            or (expected_series_id is not None and series.series_id != expected_series_id)
            or (
                expected_causal_budget_id is not None
                and series.causal_budget_id != expected_causal_budget_id
            )
            or (expected_started_at is not None and series.started_at != expected_started_at)
            or task.invocation.source is not request.execution_source
            or not _invocation_origin_matches(task, request.invocation_origin)
        ):
            raise KnowledgeEnrichmentConflict(
                "Durable task conflicts with enrichment request authority."
            )


class KnowledgeEnrichmentWorker:
    """Lease-fenced processor for one configured enrichment queue."""

    def __init__(
        self,
        queue: KnowledgeEnrichmentQueue,
        curator: KnowledgeCurator,
        *,
        exception_classifier: KnowledgeEnrichmentExceptionClassifier | None = None,
    ) -> None:
        if not isinstance(queue, KnowledgeEnrichmentQueue):
            raise TypeError("queue must be a KnowledgeEnrichmentQueue.")
        if not isinstance(curator, KnowledgeCurator):
            raise TypeError("curator must be a KnowledgeCurator.")
        worker_profile = knowledge_enrichment_profile(curator.config, curator.access_scope)
        if worker_profile != queue.profile:
            raise KnowledgeEnrichmentConflict(
                "Worker curator profile conflicts with the enrichment queue."
            )
        if exception_classifier is not None and not callable(exception_classifier):
            raise TypeError("exception_classifier must be callable.")
        self._queue = queue
        self._curator = curator
        self._config = queue.config
        self._classify = exception_classifier or _default_exception_classifier
        self._execution_lock = asyncio.Lock()

    @property
    def queue(self) -> KnowledgeEnrichmentQueue:
        return self._queue

    async def process_next(
        self,
        *,
        worker_id: str,
        lease_seconds: int = 300,
        reclaim: bool = True,
    ) -> KnowledgeEnrichmentJob | None:
        """Claim and settle at most one job, returning ``None`` only when idle.

        A malformed task on this queue's private execution route is durably
        failed and raises :class:`KnowledgeEnrichmentJobRejected` so callers do
        not mistake handled work for an empty queue.
        """

        worker_id = _clean_identity(worker_id, "worker_id")
        _validate_lease_seconds(lease_seconds)
        if type(reclaim) is not bool:
            raise TypeError("reclaim must be a bool.")
        claimed_task_id, job = await self._claim_and_process(
            worker_id,
            lease_seconds=lease_seconds,
            reclaim=reclaim,
        )
        if claimed_task_id is not None and job is None:
            raise KnowledgeEnrichmentJobRejected(claimed_task_id)
        return job

    async def _claim_and_process(
        self,
        worker_id: str,
        *,
        lease_seconds: int,
        reclaim: bool,
    ) -> tuple[str | None, KnowledgeEnrichmentJob | None]:
        async with self._execution_lock:
            task = await self._claim_next(
                worker_id,
                lease_seconds=lease_seconds,
                reclaim=reclaim,
            )
            if task is None:
                return None, None
            return task.id, await self._process_claimed(
                task,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
            )

    async def _claim_next(
        self,
        worker_id: str,
        *,
        lease_seconds: int,
        reclaim: bool,
    ) -> Task | None:
        query = TaskQuery(
            type=self._queue.task_type,
            order_by=TaskOrder.CREATED_AT_ASC,
        )
        if reclaim:
            await self._queue.task_store.reclaim_expired(
                query=query,
                max_reclaims=self._config.max_reclaims_per_poll,
            )
        task = await self._queue.task_store.claim_task(
            worker_id,
            query,
            lease_seconds=lease_seconds,
        )
        return None if task is None else copy_task(task)

    async def run(
        self,
        *,
        worker_id: str,
        lease_seconds: int = 300,
        poll_interval_s: float = 1.0,
        reclaim: bool = True,
        stop: asyncio.Event | None = None,
        max_jobs: int | None = None,
    ) -> int:
        """Process jobs until stopped and return the number of claims handled."""

        worker_id = _clean_identity(worker_id, "worker_id")
        _validate_lease_seconds(lease_seconds)
        if type(reclaim) is not bool:
            raise TypeError("reclaim must be a bool.")
        if stop is not None and not isinstance(stop, asyncio.Event):
            raise TypeError("stop must be an asyncio.Event or None.")
        if (
            isinstance(poll_interval_s, bool)
            or not isinstance(poll_interval_s, int | float)
            or not math.isfinite(poll_interval_s)
            or not 0 < poll_interval_s <= 300
        ):
            raise ValueError("poll_interval_s must be finite and between 0 and 300 seconds.")
        if max_jobs is not None and (
            isinstance(max_jobs, bool) or type(max_jobs) is not int or max_jobs < 0
        ):
            raise ValueError("max_jobs must be a non-negative integer or None.")

        async def step(_now: float, _handled: int) -> DurableWorkerStep:
            try:
                claimed_task_id, _job = await self._claim_and_process(
                    worker_id,
                    lease_seconds=lease_seconds,
                    reclaim=reclaim,
                )
            except TaskClaimLost:
                return DurableWorkerStep(handled=1, continue_immediately=True)
            if claimed_task_id is not None:
                return DurableWorkerStep(handled=1, continue_immediately=True)
            return DurableWorkerStep(idle=True)

        return await run_durable_worker_loop(
            step,
            poll_interval_s=float(poll_interval_s),
            stop=stop,
            max_handled=max_jobs,
        )

    async def _process_claimed(
        self,
        task: Task,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> KnowledgeEnrichmentJob | None:
        if task.lease_expires_at is None:
            raise TaskClaimLost("Claimed enrichment task has no worker lease.")
        stop_heartbeat = asyncio.Event()
        heartbeat = asyncio.create_task(
            _heartbeat_claim(
                self._queue.task_store,
                task.id,
                worker_id,
                lease_expires_at=task.lease_expires_at,
                lease_seconds=lease_seconds,
                stop=stop_heartbeat,
            )
        )
        operation = asyncio.create_task(
            self._execute_claimed(
                task,
                worker_id=worker_id,
            )
        )
        coordination = asyncio.create_task(
            asyncio.wait(
                (operation, heartbeat),
                return_when=asyncio.FIRST_COMPLETED,
            )
        )
        owner_cancellation: asyncio.CancelledError | None = None
        pending_error: BaseException | None = None
        try:
            owner_cancellation = await _wait_done_resisting_cancellation(coordination)
            coordination.result()
            if heartbeat.done() and not operation.done():
                operation.cancel("Knowledge enrichment heartbeat lost task authority.")
                additional = await _wait_done_resisting_cancellation(operation)
                owner_cancellation = _merge_cancellation(owner_cancellation, additional)
                try:
                    heartbeat.result()
                except Exception as heartbeat_error:
                    raise TaskClaimLost(
                        "Knowledge enrichment heartbeat lost task authority."
                    ) from heartbeat_error
                raise TaskClaimLost("Knowledge enrichment heartbeat stopped unexpectedly.")

            result = operation.result()
            if owner_cancellation is not None:
                raise owner_cancellation
            return result
        except BaseException as error:
            pending_error = error
            raise
        finally:
            stop_heartbeat.set()
            if not coordination.done():
                coordination.cancel()
                with contextlib.suppress(BaseException):
                    await coordination
            heartbeat_cancellation = await _wait_done_resisting_cancellation(heartbeat)
            if heartbeat_cancellation is not None:
                if pending_error is None:
                    raise heartbeat_cancellation
                pending_error.add_note(
                    "Knowledge enrichment heartbeat shutdown received another cancellation."
                )
            try:
                heartbeat.result()
            except Exception as heartbeat_error:
                if pending_error is None:
                    raise
                pending_error.add_note(
                    f"Knowledge enrichment heartbeat also failed: {type(heartbeat_error).__name__}."
                )

    async def _execute_claimed(
        self,
        task: Task,
        *,
        worker_id: str,
    ) -> KnowledgeEnrichmentJob | None:
        request: KnowledgeEnrichmentRequest | None = None
        request_sha256: str | None = None
        disposition: TaskRetryAttemptDisposition
        result_payload: dict[str, Any] | None = None
        error_payload: dict[str, Any] | None = None
        retry_after_seconds: float | None = None
        try:
            request, request_sha256 = self._queue._request_from_task(task)
            series = task.retry_series
            if (
                task.status is not TaskStatus.CLAIMED
                or task.worker_id != worker_id
                or series is None
                or series.disposition is not TaskRetrySeriesDisposition.ACTIVE
            ):
                raise KnowledgeEnrichmentConflict(
                    "Claimed task lacks active enrichment retry authority."
                )
            (
                prepared,
                preparation_task,
                result,
                validate_publications,
            ) = await self._load_or_create_preparation(
                request,
                request_sha256=request_sha256,
            )
            if result is None:
                result = await self._curator._commit_prepared_curation(
                    prepared,
                    replay_stable=True,
                    validate_publications=validate_publications,
                )
                _validate_curation_result(
                    result,
                    request=request,
                    request_sha256=request_sha256,
                    curator_config=self._queue._curator_config,
                )
                result = await self._complete_preparation(
                    preparation_task,
                    result=result,
                    request=request,
                    request_sha256=request_sha256,
                )
            _validate_curation_result(
                result,
                request=request,
                request_sha256=request_sha256,
                curator_config=self._queue._curator_config,
            )
            job_result = KnowledgeEnrichmentJobResult(
                request_sha256=request_sha256,
                task_id=task.id,
                attempt=series.attempt,
                curation=result,
            )
            result_payload = _result_payload(job_result)
            if (
                len(canonical_durable_json_bytes(result_payload, "enrichment result"))
                > self._config.max_result_bytes
            ):
                raise _EnrichmentResultTooLarge
            disposition = TaskRetryAttemptDisposition.SUCCEEDED
        except asyncio.CancelledError:
            raise
        except (TaskClaimLost, TaskTerminalizationConflict):
            raise
        except Exception as error:
            if request_sha256 is None:
                with contextlib.suppress(Exception):
                    request, request_sha256 = self._queue._request_from_task(task)
            decision = self._classify_exception(error)
            failure = KnowledgeEnrichmentFailure(
                request_sha256=request_sha256,
                code=decision.code,
                category=decision.category,
                retryable=decision.retryable,
            )
            disposition = (
                TaskRetryAttemptDisposition.RETRYABLE_FAILURE
                if decision.retryable
                else TaskRetryAttemptDisposition.NON_RETRYABLE_FAILURE
            )
            result_payload = None
            error_payload = _failure_payload(failure)
            retry_after_seconds = decision.retry_after_seconds
        await self._settle(
            task,
            worker_id=worker_id,
            disposition=disposition,
            result=result_payload,
            error=error_payload,
            retry_after_seconds=retry_after_seconds,
        )
        if request is None:
            return None
        return await self._queue.load(request.operation_id)

    async def _load_or_create_preparation(
        self,
        request: KnowledgeEnrichmentRequest,
        *,
        request_sha256: str,
    ) -> tuple[_PreparedLearningBatch, Task, LearningBatchResult | None, bool]:
        task_id = self._queue._preparation_task_id(request.operation_id)
        existing = await self._queue.task_store.load_task(task_id)
        locally_created_preparation: _PreparedLearningBatch | None = None
        if existing is None:
            await self._admit_semantic_dispatch(
                request,
                request_sha256=request_sha256,
            )
            prepared = await self._curator._prepare_curation(request.batch)
            preparation_document = prepared.model_dump(mode="json")
            preparation_sha256 = _sha256(
                preparation_document,
                "knowledge enrichment preparation",
            )
            task_input = _preparation_task_input(
                preparation_document,
                request_sha256=request_sha256,
                preparation_sha256=preparation_sha256,
                execution_domain_sha256=self._queue._execution_domain_sha256,
            )
            if (
                len(
                    canonical_durable_json_bytes(
                        task_input,
                        "knowledge enrichment preparation task input",
                    )
                )
                > self._config.max_result_bytes
            ):
                raise _EnrichmentResultTooLarge
            create = TaskCreate(
                task_id=task_id,
                type=self._queue._preparation_task_type,
                title="Knowledge enrichment preparation",
                available_at=_PREPARATION_AVAILABLE_AT,
                input=task_input,
                metadata=_preparation_task_metadata(
                    request_sha256=request_sha256,
                    preparation_sha256=preparation_sha256,
                    execution_domain_sha256=self._queue._execution_domain_sha256,
                ),
                invocation_origin=request.invocation_origin,
            )
            create = task_create_with_execution_source(create, source=request.execution_source)
            try:
                existing = await self._queue.task_store.create_task(create)
                locally_created_preparation = prepared
            except Exception as publication_error:
                try:
                    existing = await self._queue.task_store.load_task(task_id)
                except Exception as reconciliation_error:
                    publication_error.add_note(
                        "Knowledge enrichment preparation reconciliation also failed: "
                        f"{type(reconciliation_error).__name__}."
                    )
                    raise publication_error from reconciliation_error
                if existing is None:
                    raise _EnrichmentSemanticOutcomeUnknown from publication_error
        prepared, result = self._queue._preparation_from_task(
            existing,
            request=request,
            request_sha256=request_sha256,
        )
        if locally_created_preparation is not None and prepared != locally_created_preparation:
            raise KnowledgeEnrichmentConflict(
                "Stored enrichment preparation conflicts with the locally prepared plan."
            )
        return prepared, existing, result, locally_created_preparation is None

    async def _admit_semantic_dispatch(
        self,
        request: KnowledgeEnrichmentRequest,
        *,
        request_sha256: str,
    ) -> None:
        task_id = self._queue._semantic_dispatch_task_id(request.operation_id)
        dispatch_id = str(uuid4())
        existing = await self._queue.task_store.load_task(task_id)
        if existing is None:
            create = TaskCreate(
                task_id=task_id,
                type=self._queue._semantic_dispatch_task_type,
                title="Knowledge enrichment semantic dispatch",
                available_at=_PREPARATION_AVAILABLE_AT,
                input=_semantic_dispatch_task_input(
                    request_sha256=request_sha256,
                    execution_domain_sha256=self._queue._execution_domain_sha256,
                    dispatch_id=dispatch_id,
                ),
                metadata=_semantic_dispatch_task_metadata(
                    request_sha256=request_sha256,
                    execution_domain_sha256=self._queue._execution_domain_sha256,
                    dispatch_id=dispatch_id,
                ),
                invocation_origin=request.invocation_origin,
            )
            create = task_create_with_execution_source(create, source=request.execution_source)
            try:
                existing = await self._queue.task_store.create_task(create)
            except Exception as publication_error:
                try:
                    existing = await self._queue.task_store.load_task(task_id)
                except Exception as reconciliation_error:
                    publication_error.add_note(
                        "Knowledge enrichment semantic dispatch reconciliation also failed: "
                        f"{type(reconciliation_error).__name__}."
                    )
                    raise publication_error from reconciliation_error
                if existing is None:
                    raise
        durable_dispatch_id = self._queue._semantic_dispatch_from_task(
            existing,
            request=request,
            request_sha256=request_sha256,
        )
        if durable_dispatch_id != dispatch_id:
            raise _EnrichmentSemanticOutcomeUnknown

    async def _complete_preparation(
        self,
        task: Task,
        *,
        result: LearningBatchResult,
        request: KnowledgeEnrichmentRequest,
        request_sha256: str,
    ) -> LearningBatchResult:
        envelope = task.input.get(_PREPARATION_INPUT_KEY)
        preparation_sha256 = envelope.get("preparation_sha256") if type(envelope) is dict else None
        if type(preparation_sha256) is not str:
            raise KnowledgeEnrichmentConflict("Enrichment preparation fingerprint is missing.")
        completion = _preparation_task_result(preparation_sha256, result)
        if (
            len(
                canonical_durable_json_bytes(
                    completion,
                    "knowledge enrichment preparation result",
                )
            )
            > self._config.max_result_bytes
        ):
            raise _EnrichmentResultTooLarge
        try:
            completed = await self._queue.task_store.complete_task(task.id, completion)
        except Exception as completion_error:
            try:
                completed = await self._queue.task_store.load_task(task.id)
            except Exception as reconciliation_error:
                completion_error.add_note(
                    "Knowledge enrichment preparation completion reconciliation also failed: "
                    f"{type(reconciliation_error).__name__}."
                )
                raise completion_error from reconciliation_error
            if completed is None or completed.status is not TaskStatus.COMPLETED:
                raise
        _prepared, durable_result = self._queue._preparation_from_task(
            completed,
            request=request,
            request_sha256=request_sha256,
        )
        if durable_result is None:  # pragma: no cover - completed-task invariant
            raise KnowledgeEnrichmentConflict(
                "Completed enrichment preparation lost its durable result."
            )
        return durable_result

    def _classify_exception(self, error: Exception) -> KnowledgeEnrichmentFailureDecision:
        if isinstance(error, _EnrichmentSemanticOutcomeUnknown):
            return KnowledgeEnrichmentFailureDecision(
                code="semantic_outcome_unknown",
                category=KnowledgeEnrichmentFailureCategory.SEMANTIC_OUTCOME_UNKNOWN,
                retryable=True,
            )
        if isinstance(error, _EnrichmentResultTooLarge):
            return KnowledgeEnrichmentFailureDecision(
                code="result_limit_exceeded",
                category=KnowledgeEnrichmentFailureCategory.RESULT_LIMIT_EXCEEDED,
                retryable=False,
            )
        if isinstance(error, KnowledgeEnrichmentConflict):
            return KnowledgeEnrichmentFailureDecision(
                code="job_contract_conflict",
                category=KnowledgeEnrichmentFailureCategory.PROFILE_CONFLICT,
                retryable=False,
            )
        try:
            raw = self._classify(error)
        except Exception:
            return KnowledgeEnrichmentFailureDecision(
                code="failure_classifier_failed",
                category=KnowledgeEnrichmentFailureCategory.INTERNAL_ERROR,
                retryable=False,
            )
        if inspect.isawaitable(raw):
            if inspect.iscoroutine(raw):
                raw.close()
            return KnowledgeEnrichmentFailureDecision(
                code="failure_classifier_invalid",
                category=KnowledgeEnrichmentFailureCategory.INTERNAL_ERROR,
                retryable=False,
            )
        if type(raw) is not KnowledgeEnrichmentFailureDecision:
            return KnowledgeEnrichmentFailureDecision(
                code="failure_classifier_invalid",
                category=KnowledgeEnrichmentFailureCategory.INTERNAL_ERROR,
                retryable=False,
            )
        return KnowledgeEnrichmentFailureDecision.model_validate(raw.model_dump(mode="python"))

    async def _settle(
        self,
        task: Task,
        *,
        worker_id: str,
        disposition: TaskRetryAttemptDisposition,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        series = task.retry_series
        if series is None:
            await self._settle_malformed_non_series_task(
                task,
                worker_id=worker_id,
                disposition=disposition,
                result=result,
                error=error,
                retry_after_seconds=retry_after_seconds,
            )
            return
        request = TaskRetrySettlementRequest(
            task_id=task.id,
            worker_id=worker_id,
            causal_budget_id=series.causal_budget_id,
            idempotency_key=f"knowledge-enrichment:{series.series_id}:{series.attempt}",
            disposition=disposition,
            result=result,
            error=error,
            retry_after_seconds=retry_after_seconds,
        )
        try:
            await settle_task_retry_attempt_with_retry(
                self._queue.task_store,
                request,
                policy=self._config.terminalization_retry_policy,
            )
        except TaskTerminalizationConflict:
            current = await self._queue.task_store.load_task(task.id)
            if current is None:
                raise
            cancellation = _task_retry_requested_cancellation_settlement(
                current,
                worker_id=worker_id,
            )
            if cancellation is None:
                raise
            await settle_task_retry_attempt_with_retry(
                self._queue.task_store,
                cancellation,
                policy=self._config.terminalization_retry_policy,
            )

    async def _settle_malformed_non_series_task(
        self,
        task: Task,
        *,
        worker_id: str,
        disposition: TaskRetryAttemptDisposition,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        retry_after_seconds: float | None,
    ) -> None:
        if (
            disposition is not TaskRetryAttemptDisposition.NON_RETRYABLE_FAILURE
            or result is not None
            or error is None
            or retry_after_seconds is not None
        ):
            raise KnowledgeEnrichmentConflict(
                "An enrichment task without retry-series authority can only be rejected."
            )
        request = TaskTerminalizationRequest(
            task_id=task.id,
            worker_id=worker_id,
            kind=TaskTerminalKind.FAILED,
            error=error,
            idempotency_key=(
                "knowledge-enrichment-rejection:v1:"
                + _sha256(
                    {
                        "contract": "cayu.knowledge-enrichment-rejection.v1",
                        "task_id": task.id,
                    },
                    "knowledge enrichment rejection identity",
                )
            ),
        )
        try:
            await terminalize_task_with_retry(
                self._queue.task_store,
                request,
                policy=self._config.terminalization_retry_policy,
            )
        except TaskTerminalizationConflict:
            current = await self._queue.task_store.load_task(task.id)
            if current is None:
                raise
            cancellation = _task_cancellation_terminalization_request(
                current,
                worker_id=worker_id,
            )
            if cancellation is None:
                raise
            await terminalize_task_with_retry(
                self._queue.task_store,
                cancellation,
                policy=self._config.terminalization_retry_policy,
            )


class _EnrichmentResultTooLarge(ValueError):
    pass


class _EnrichmentSemanticOutcomeUnknown(RuntimeError):
    pass


def _default_exception_classifier(error: Exception) -> KnowledgeEnrichmentFailureDecision:
    if isinstance(error, (ConnectionError, TimeoutError)):
        return KnowledgeEnrichmentFailureDecision(
            code="dependency_unavailable",
            category=KnowledgeEnrichmentFailureCategory.DEPENDENCY_UNAVAILABLE,
            retryable=True,
        )
    if isinstance(error, (TypeError, ValueError)):
        return KnowledgeEnrichmentFailureDecision(
            code="job_invalid",
            category=KnowledgeEnrichmentFailureCategory.INVALID_JOB,
            retryable=False,
        )
    return KnowledgeEnrichmentFailureDecision(
        code="worker_internal_error",
        category=KnowledgeEnrichmentFailureCategory.INTERNAL_ERROR,
        retryable=False,
    )


def _task_input(
    request: KnowledgeEnrichmentRequest,
    *,
    request_sha256: str,
    queue_config_sha256: str,
    execution_domain_sha256: str,
) -> dict[str, Any]:
    return {
        _TASK_INPUT_KEY: {
            "contract": _TASK_CONTRACT,
            "request_sha256": request_sha256,
            "queue_config_sha256": queue_config_sha256,
            "execution_domain_sha256": execution_domain_sha256,
            "request": request.model_dump(mode="json"),
        }
    }


def _task_metadata(
    request_sha256: str,
    *,
    queue_config_sha256: str,
    execution_domain_sha256: str,
) -> dict[str, Any]:
    return {
        "contract": _TASK_CONTRACT,
        "request_sha256": request_sha256,
        "queue_config_sha256": queue_config_sha256,
        "execution_domain_sha256": execution_domain_sha256,
    }


def _preparation_task_input(
    preparation_document: dict[str, Any],
    *,
    request_sha256: str,
    preparation_sha256: str,
    execution_domain_sha256: str,
) -> dict[str, Any]:
    return {
        _PREPARATION_INPUT_KEY: {
            "contract": _PREPARATION_CONTRACT,
            "request_sha256": request_sha256,
            "preparation_sha256": preparation_sha256,
            "execution_domain_sha256": execution_domain_sha256,
            "preparation": copy_durable_json_object(
                preparation_document,
                "knowledge enrichment preparation",
            ),
        }
    }


def _semantic_dispatch_task_input(
    *,
    request_sha256: str,
    execution_domain_sha256: str,
    dispatch_id: str,
) -> dict[str, Any]:
    return {
        _SEMANTIC_DISPATCH_INPUT_KEY: {
            "contract": _SEMANTIC_DISPATCH_CONTRACT,
            "request_sha256": request_sha256,
            "execution_domain_sha256": execution_domain_sha256,
            "dispatch_id": dispatch_id,
        }
    }


def _semantic_dispatch_task_metadata(
    *,
    request_sha256: str,
    execution_domain_sha256: str,
    dispatch_id: str,
) -> dict[str, Any]:
    return {
        "contract": _SEMANTIC_DISPATCH_CONTRACT,
        "request_sha256": request_sha256,
        "execution_domain_sha256": execution_domain_sha256,
        "dispatch_id": dispatch_id,
    }


def _preparation_task_metadata(
    *,
    request_sha256: str,
    preparation_sha256: str,
    execution_domain_sha256: str,
) -> dict[str, Any]:
    return {
        "contract": _PREPARATION_CONTRACT,
        "request_sha256": request_sha256,
        "preparation_sha256": preparation_sha256,
        "execution_domain_sha256": execution_domain_sha256,
    }


def _preparation_task_result(
    preparation_sha256: str,
    result: LearningBatchResult,
) -> dict[str, Any]:
    return {
        "contract": _PREPARATION_CONTRACT,
        "preparation_sha256": preparation_sha256,
        "curation": result.model_dump(mode="json"),
    }


def _result_payload(result: KnowledgeEnrichmentJobResult) -> dict[str, Any]:
    return {
        "contract": _RESULT_CONTRACT,
        "result": result.model_dump(mode="json"),
    }


def _failure_payload(failure: KnowledgeEnrichmentFailure) -> dict[str, Any]:
    return {
        "contract": _FAILURE_CONTRACT,
        "failure": failure.model_dump(mode="json"),
    }


def _parse_failure_payload(
    value: object,
    *,
    request_sha256: str,
) -> KnowledgeEnrichmentFailure | None:
    if type(value) is not dict:
        return None
    document = cast("dict[str, object]", value)
    if document.get("contract") != _FAILURE_CONTRACT:
        return None
    if set(document) != {"contract", "failure"}:
        raise KnowledgeEnrichmentConflict("Enrichment failure envelope is invalid.")
    raw = document.get("failure")
    if type(raw) is not dict:
        raise KnowledgeEnrichmentConflict("Enrichment failure envelope is invalid.")
    try:
        failure = KnowledgeEnrichmentFailure.model_validate(raw)
    except (TypeError, ValueError) as exc:
        raise KnowledgeEnrichmentConflict("Enrichment failure envelope is malformed.") from exc
    if failure.request_sha256 not in {None, request_sha256}:
        raise KnowledgeEnrichmentConflict("Enrichment failure conflicts with its request.")
    return failure


def _failure_from_task(
    task: Task,
    *,
    request_sha256: str,
) -> KnowledgeEnrichmentFailure | None:
    failure = _parse_failure_payload(task.error, request_sha256=request_sha256)
    if failure is not None:
        series = task.retry_series
        if series is not None and failure.retryable:
            return failure.model_copy(
                update={
                    "retryable": False,
                    "annotation": {"terminal_reason": series.disposition.value},
                }
            )
        return failure
    if task.status is TaskStatus.CANCELLED:
        return KnowledgeEnrichmentFailure(
            request_sha256=request_sha256,
            code="cancelled",
            category=KnowledgeEnrichmentFailureCategory.CANCELLED,
            retryable=False,
        )
    return KnowledgeEnrichmentFailure(
        request_sha256=request_sha256,
        code="terminal_failure_unavailable",
        category=KnowledgeEnrichmentFailureCategory.INTERNAL_ERROR,
        retryable=False,
    )


def _attempt_from_task(
    task: Task,
    *,
    request_sha256: str,
) -> KnowledgeEnrichmentAttempt:
    series = task.retry_series
    if series is None:
        raise KnowledgeEnrichmentConflict("Enrichment attempt lacks retry-series evidence.")
    return KnowledgeEnrichmentAttempt(
        task_id=task.id,
        attempt=series.attempt,
        task_status=task.status,
        disposition=series.disposition,
        failure=_parse_failure_payload(task.error, request_sha256=request_sha256),
        created_at=task.created_at,
        updated_at=task.updated_at,
        completed_at=task.completed_at,
    )


def _result_from_task(
    task: Task,
    *,
    request: KnowledgeEnrichmentRequest,
    request_sha256: str,
    max_result_bytes: int,
    curator_config: KnowledgeCuratorConfig,
) -> KnowledgeEnrichmentJobResult:
    if (
        type(task.result) is not dict
        or set(task.result) != {"contract", "result"}
        or task.result.get("contract") != _RESULT_CONTRACT
    ):
        raise KnowledgeEnrichmentConflict("Completed enrichment task has no valid result.")
    if len(canonical_durable_json_bytes(task.result, "enrichment result")) > max_result_bytes:
        raise KnowledgeEnrichmentConflict("Completed enrichment result exceeds its queue limit.")
    raw = task.result.get("result")
    if type(raw) is not dict:
        raise KnowledgeEnrichmentConflict("Completed enrichment result is malformed.")
    try:
        result = KnowledgeEnrichmentJobResult.model_validate(raw)
    except (TypeError, ValueError) as exc:
        raise KnowledgeEnrichmentConflict("Completed enrichment result is invalid.") from exc
    series = task.retry_series
    if (
        series is None
        or result.request_sha256 != request_sha256
        or result.task_id != task.id
        or result.attempt != series.attempt
    ):
        raise KnowledgeEnrichmentConflict("Completed enrichment result conflicts with task.")
    _validate_curation_result(
        result.curation,
        request=request,
        request_sha256=request_sha256,
        curator_config=curator_config,
    )
    return result


def _validate_curation_result(
    result: LearningBatchResult,
    *,
    request: KnowledgeEnrichmentRequest,
    request_sha256: str,
    curator_config: KnowledgeCuratorConfig,
) -> None:
    if type(result) is not LearningBatchResult:
        raise TypeError("KnowledgeCurator returned a non-LearningBatchResult value.")
    if (
        result.batch_id != request.batch.id
        or result.batch_fingerprint != request.batch.fingerprint
        or result.configuration_fingerprint != request.profile.curator_configuration_fingerprint
        or result.scope != request.batch.scope
        or request.fingerprint != request_sha256
    ):
        raise KnowledgeEnrichmentConflict("Curator result conflicts with the durable request.")
    expected_signals = tuple((signal.id, signal.fingerprint) for signal in request.batch.signals)
    returned_signals = tuple(
        (signal.signal_id, signal.signal_fingerprint) for signal in result.signals
    )
    if returned_signals != expected_signals:
        raise KnowledgeEnrichmentConflict("Curator result signals conflict with the durable batch.")
    if result.candidate_count > curator_config.max_candidates:
        raise KnowledgeEnrichmentConflict("Curator result exceeds its configured candidate limit.")
    for candidate in result.candidates:
        decision = candidate.decision
        if decision is None:
            continue
        if (
            decision.notes is not None
            and len(decision.notes.encode("utf-8")) > curator_config.max_evaluator_notes_bytes
        ):
            raise KnowledgeEnrichmentConflict(
                "Curator result exceeds its configured evaluator-note limit."
            )
        if (
            len(canonical_durable_json_bytes(decision.metadata, "decision metadata"))
            > curator_config.max_metadata_bytes
        ):
            raise KnowledgeEnrichmentConflict(
                "Curator result exceeds its configured decision-metadata limit."
            )


def _job_status(task: Task) -> KnowledgeEnrichmentJobStatus:
    series = task.retry_series
    if series is None:
        raise KnowledgeEnrichmentConflict("Enrichment task lacks retry-series evidence.")
    if task.status is TaskStatus.PENDING:
        return (
            KnowledgeEnrichmentJobStatus.RETRY_SCHEDULED
            if series.attempt > 1
            else KnowledgeEnrichmentJobStatus.PENDING
        )
    if task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        return KnowledgeEnrichmentJobStatus.PROCESSING
    if series.disposition is TaskRetrySeriesDisposition.RETRY_SCHEDULED:
        return KnowledgeEnrichmentJobStatus.RETRY_SCHEDULED
    if task.status is TaskStatus.COMPLETED:
        return KnowledgeEnrichmentJobStatus.COMPLETED
    if task.status is TaskStatus.CANCELLED:
        return KnowledgeEnrichmentJobStatus.CANCELLED
    if task.status is TaskStatus.FAILED:
        return KnowledgeEnrichmentJobStatus.FAILED
    raise KnowledgeEnrichmentConflict("Enrichment task entered an unsupported lifecycle state.")


def _invocation_origin_matches(
    task: Task,
    claim: InvocationOriginClaim | None,
) -> bool:
    origin = task.invocation.origin
    if claim is None:
        return (
            origin.trust is InvocationOriginTrust.UNATTRIBUTED
            and origin.subject is None
            and origin.tenant is None
        )
    return (
        origin.trust is InvocationOriginTrust.HOST_ASSERTED
        and origin.subject == claim.subject
        and origin.tenant == claim.tenant
    )


def _validate_lease_seconds(value: int) -> None:
    if isinstance(value, bool) or type(value) is not int or not 1 <= value <= 86_400:
        raise ValueError("lease_seconds must be an integer between 1 and 86400.")


async def _heartbeat_claim(
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    *,
    lease_expires_at: datetime,
    lease_seconds: int,
    stop: asyncio.Event,
) -> None:
    async def heartbeat() -> Task:
        nonlocal lease_expires_at
        renewed = await task_store.heartbeat(
            task_id,
            worker_id,
            lease_expires_at=lease_expires_at,
            extend_seconds=lease_seconds,
        )
        if renewed.lease_expires_at is None:
            raise TaskClaimLost("Enrichment heartbeat returned no worker lease.")
        lease_expires_at = renewed.lease_expires_at
        return renewed

    async def reconcile_failure(heartbeat_error: Exception) -> None:
        try:
            current = await task_store.load_task(task_id)
        except Exception as reconciliation_error:
            raise heartbeat_error from reconciliation_error
        if current is None or current.status not in _TERMINAL_TASK_STATUSES:
            raise heartbeat_error
        await stop.wait()

    await run_durable_lease_heartbeat(
        heartbeat,
        lease_seconds=float(lease_seconds),
        stop=stop,
        stopped_outcome=None,
        maximum_interval_s=1.0,
        on_failure=reconcile_failure,
    )


async def _wait_done_resisting_cancellation(
    future: asyncio.Future[Any],
) -> asyncio.CancelledError | None:
    cancellation: asyncio.CancelledError | None = None
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError as error:
            cancellation = _merge_cancellation(cancellation, error)
        except Exception:
            break
    return cancellation


def _merge_cancellation(
    current: asyncio.CancelledError | None,
    additional: asyncio.CancelledError | None,
) -> asyncio.CancelledError | None:
    if additional is None:
        return current
    if current is None:
        return additional
    current.add_note("Knowledge enrichment received an additional cancellation request.")
    return current


__all__ = [
    "DEFAULT_KNOWLEDGE_ENRICHMENT_TASK_TYPE",
    "KNOWLEDGE_ENRICHMENT_SCHEMA_VERSION",
    "MAX_KNOWLEDGE_ENRICHMENT_FAILURE_ANNOTATION_BYTES",
    "MAX_KNOWLEDGE_ENRICHMENT_IDENTITY_BYTES",
    "MAX_KNOWLEDGE_ENRICHMENT_RECLAIMS_PER_POLL",
    "MAX_KNOWLEDGE_ENRICHMENT_REQUEST_BYTES",
    "MAX_KNOWLEDGE_ENRICHMENT_RESULT_BYTES",
    "MAX_KNOWLEDGE_ENRICHMENT_TRIGGER_METADATA_BYTES",
    "KnowledgeEnrichmentConflict",
    "KnowledgeEnrichmentExceptionClassifier",
    "KnowledgeEnrichmentFailure",
    "KnowledgeEnrichmentFailureCategory",
    "KnowledgeEnrichmentFailureDecision",
    "KnowledgeEnrichmentFeedbackAuthorization",
    "KnowledgeEnrichmentJob",
    "KnowledgeEnrichmentJobRejected",
    "KnowledgeEnrichmentJobResult",
    "KnowledgeEnrichmentJobStatus",
    "KnowledgeEnrichmentProfile",
    "KnowledgeEnrichmentQueue",
    "KnowledgeEnrichmentQueueConfig",
    "KnowledgeEnrichmentRequest",
    "KnowledgeEnrichmentTrigger",
    "KnowledgeEnrichmentWorker",
    "knowledge_enrichment_profile",
]
